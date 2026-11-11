from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import sys
from importlib.metadata import version
from pathlib import Path

from filelock import FileLock, Timeout

from decisionforge import __version__
from decisionforge.adapters import input_kind, read_records, shape
from decisionforge.config import Config
from decisionforge.llm import OpenAIProvider
from decisionforge.models import MappingSpec
from decisionforge.pipeline import Pipeline
from decisionforge.storage import Store, state_directory
from decisionforge.util import PipelineError, digest, json_text, now, write_json


def file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def signature(config: Config) -> str:
    files = [Path(config.input.path)]
    if files[0].suffix.lower() == ".txt":
        from decisionforge.generation.text_batching import context_path

        if context_path(files[0]).exists():
            files.append(context_path(files[0]))
    if config.input.mapping_spec:
        files.append(Path(config.input.mapping_spec))
    settings = config.model_dump(mode="json")
    # Operational changes may resume; semantic changes create a distinct run.
    settings.pop("output")
    settings["execution"].pop("budget_limit_usd")
    settings["execution"].pop("concurrency")
    settings["execution"].pop("text_batch_concurrency")
    settings["execution"].pop("llm_concurrency")
    settings["execution"].pop("resume")
    source = {
        str(p.relative_to(Path(__file__).parent)): file_hash(p)
        for p in Path(__file__).parent.rglob("*")
        if p.suffix in {".py", ".md"}
    }
    runtime = {name: version(name) for name in ("openai", "pydantic")}
    return digest(
        {
            "config": settings,
            "jev_model": config.output.jev_model,
            "runtime": runtime,
            "files": {str(p): file_hash(p) for p in files},
            "source": source,
        }
    )


async def execute(config: Config) -> dict:
    config = Config.model_validate(config.model_dump())
    kind = (
        ("pdf" if config.input.paths else input_kind(Path(config.input.path)))
        if config.input.type == "auto"
        else config.input.type
    )
    if kind == "pdf":
        from decisionforge.generation.pdf import prepare_pdf

        corpus = await prepare_pdf(config)
        config.input.path = str(corpus)
        config.input.paths = []
        config.input.type = "txt"
    # Resolve environment endpoint before persisting/cache fingerprinting. Never serialize API keys.
    if config.llm.base_url is None and os.getenv("OPENAI_BASE_URL"):
        config.llm.base_url = os.environ["OPENAI_BASE_URL"]
    config = Config.model_validate(config.model_dump())
    fingerprint = signature(config)
    directory = Path(config.output.directory)
    manifest_path = state_directory(directory) / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["signature"] != fingerprint:
            directory = directory.parent / f"{directory.name}_v_{fingerprint[:10]}"
            config.output.directory = str(directory)
            logging.info("Input/config/code changed; creating run version %s", directory)
        elif not config.execution.resume:
            raise ValueError("Run already exists and resume=false; choose a new output directory")
    state = state_directory(directory)
    state.mkdir(parents=True, exist_ok=True)
    with FileLock(str(state / ".run.lock"), timeout=0):
        store = Store(directory)
        provider = None
        try:
            manifest_path = state / "manifest.json"
            if not manifest_path.exists():
                write_json(
                    manifest_path,
                    {
                        "version": __version__,
                        "signature": fingerprint,
                        "created_at": now(),
                        "config": config.model_dump(mode="json"),
                    },
                )
            write_json(state / "config.resolved.json", config.model_dump(mode="json"))
            logging.info(
                "Run %s | models=%s | text_batches=%s | concurrency=%s | llm_concurrency=%s "
                "| max_calls/source=%s | USD budget=%s",
                directory,
                {
                    role: config.llm.model_for(role)
                    for role in ("analyzer", "generator", "verifier")
                },
                config.execution.text_batch_concurrency,
                config.execution.concurrency,
                config.execution.llm_concurrency,
                config.execution.max_llm_calls_per_record,
                config.execution.budget_limit_usd
                if config.execution.budget_limit_usd is not None
                else "unlimited",
            )
            provider = OpenAIProvider(config, store)
            result = await Pipeline(config, store, provider).run()
            return {
                "run_directory": str(directory),
                **result,
            }
        finally:
            if provider:
                await provider.close()
            store.close()


def inspect_input(config: Config) -> dict:
    """Offline schema inspection; never spends API tokens or leaks answer samples to stdout."""
    kind = (
        ("pdf" if config.input.paths else input_kind(Path(config.input.path)))
        if config.input.type == "auto"
        else config.input.type
    )
    if kind == "pdf":
        from decisionforge.generation.pdf import inspect_pdf

        return inspect_pdf(config)
    if config.input.paths:
        raise ValueError("input.paths is supported only for PDF input")
    counts, splits = {}, {}
    count = 0
    mapping = (
        MappingSpec.model_validate_json(Path(config.input.mapping_spec).read_text(encoding="utf-8"))
        if config.input.mapping_spec
        else None
    )
    for record in read_records(Path(config.input.path), kind, config.input.records_path):
        key = json_text(shape(record.payload))
        counts[key] = counts.get(key, 0) + 1
        if mapping and mapping.split:
            from decisionforge.adapters import field

            split = str(field(record.payload, mapping.split))
            splits[split] = splits.get(split, 0) + 1
        count += 1
    batching = None
    if kind == "txt":
        from decisionforge.generation.text_batching import context_path, read_text

        characters = len(read_text(Path(config.input.path)))
        settings = config.input.text_batching
        generated = context_path(Path(config.input.path)).exists()
        batching = {
            "enabled": settings.enabled,
            "characters": characters,
            "threshold_chars": settings.threshold_chars,
            "requires_model_assessment": settings.enabled
            and not generated
            and characters > settings.threshold_chars,
            "generated_batch": generated,
            "note": "build decides whether to split; inspect makes no model calls",
        }
    return {
        "records": count,
        "schemas": [{"shape": json.loads(k), "count": v} for k, v in counts.items()],
        "source_splits": splits,
        "requested_modes": config.task.modes,
        "text_batching": batching,
        "mapping": "explicit"
        if mapping
        else "deterministic field mapping; source-only extraction if required",
        "semantic_analysis": (
            "performed by build; narrative TXT may construct evidence-grounded QA; inspection is offline"
            if kind == "txt" and config.input.generate_qa_from_text
            else "performed by build; inspection is offline"
        ),
    }


def main(argv=None) -> int:
    from decisionforge.typer_cli import run

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        return run(argv)
    except KeyboardInterrupt:
        logging.error("Interrupted. Resume from the saved run directory.")
        return 130
    except sqlite3.Error as exc:
        logging.error("Checkpoint database error (%s): %s", getattr(exc, "sqlite_errorname", "SQLite"), exc)
        return 1
    except (ValueError, OSError, Timeout, PipelineError) as exc:
        logging.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
