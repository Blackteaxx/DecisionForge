"""Typer command tree for the public DecisionForge CLI."""

from __future__ import annotations

import asyncio
import json
import platform
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated

import typer
from dotenv import dotenv_values, load_dotenv
from filelock import FileLock

from decisionforge import __version__
from decisionforge.adapters import input_kind
from decisionforge.config import Config, load_config
from decisionforge.exporter import (
    export_canonical_jsonl,
    export_chat_sft,
    export_run,
    validate_file,
)
from decisionforge.generation.contracts import (
    CheckReport,
    StandardRecord,
    TextBatchContext,
    TextBatchPlan,
)
from decisionforge.models import LegacyTrainingExample, MappingSpec, TrainingExample, TrainingRecord
from decisionforge.storage import Store, state_directory

app = typer.Typer(
    name="dforge",
    help="Generate JEV data, train and evaluate decision models.",
    no_args_is_help=True,
    invoke_without_command=True,
)
data_app = typer.Typer(help="Prepare, inspect, validate and export JEV data.", no_args_is_help=True)
eval_app = typer.Typer(help="Evaluate decision models.", no_args_is_help=True)
train_app = typer.Typer(help="Train native decision models.", no_args_is_help=True)
schema_app = typer.Typer(help="Export JSON schemas.", no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(eval_app, name="eval")
app.add_typer(train_app, name="train")
app.add_typer(schema_app, name="schema")


def emit(value):
    typer.echo(json.dumps(value, indent=4, ensure_ascii=False))
    if value.get("status") in {
        "budget_exhausted",
        "authentication_failed",
        "access_denied",
        "model_failed",
    }:
        raise typer.Exit(2)


def config_with_overrides(config_path: Path, input_path: Path | None = None) -> Config:
    config = load_config(config_path)
    if input_path:
        config.input.path = str(input_path.resolve())
        config.input.paths = []
        config.input.type = input_kind(input_path)
    return config


@app.callback()
def root(
    version: Annotated[bool, typer.Option("--version", help="Show the installed version.")] = False,
):
    if version:
        typer.echo(__version__)
        raise typer.Exit()


@data_app.command("inspect")
def data_inspect(
    config: Annotated[Path, typer.Option(..., "--config", help="YAML configuration file.")],
    input_path: Annotated[
        Path | None,
        typer.Option(
            "--input", help="Input file or PDF directory; repeat this option for multiple PDFs."
        ),
    ] = None,
):
    """Inspect input schemas and counts offline; makes no model calls."""
    from decisionforge import cli

    emit(cli.inspect_input(config_with_overrides(config, input_path)))


@data_app.command("parse-pdf")
def data_parse_pdf(
    config: Annotated[Path, typer.Option(..., "--config", help="YAML configuration file.")],
    input_paths: Annotated[
        list[Path] | None,
        typer.Option(
            "--input", help="Input file or PDF directory; repeat this option for multiple PDFs."
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir", help="Override output directory, relative to the current directory."
        ),
    ] = None,
):
    """Parse PDFs with MinerU into reusable TXT and a parse manifest."""
    from decisionforge.generation.pdf import prepare_pdf

    load_dotenv(Path.cwd() / ".env", override=False)
    loaded = load_config(config)
    if input_paths:
        loaded.input.path = str(input_paths[0].resolve()) if len(input_paths) == 1 else ""
        loaded.input.paths = (
            [str(path.resolve()) for path in input_paths] if len(input_paths) > 1 else []
        )
    loaded.input.type = "pdf"
    corpus = asyncio.run(prepare_pdf(loaded, output_root=str(output_dir) if output_dir else None))
    emit({"corpus": str(corpus), "manifest": str(corpus.parent / "manifest.json")})


@data_app.command("build")
def data_build(
    config: Annotated[Path, typer.Option(..., "--config", help="YAML configuration file.")],
    input_path: Annotated[
        Path | None,
        typer.Option(
            "--input", help="Input file or PDF directory; repeat this option for multiple PDFs."
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir", help="Override output directory, relative to the current directory."
        ),
    ] = None,
    difficulty: Annotated[
        float | None,
        typer.Option(
            "--difficulty", help="Distractor difficulty from 0 (easy) to 1 (hard).", min=0, max=1
        ),
    ] = None,
    concurrency: Annotated[
        int | None,
        typer.Option("--concurrency", help="Maximum concurrent question workers.", min=1),
    ] = None,
    text_batch_concurrency: Annotated[
        int | None,
        typer.Option("--text-batch-concurrency", help="Maximum simultaneous text batches.", min=1),
    ] = None,
    llm_concurrency: Annotated[
        int | None,
        typer.Option("--llm-concurrency", help="Maximum simultaneous model requests.", min=1),
    ] = None,
    modes: Annotated[
        list[str] | None,
        typer.Option(
            "--modes",
            help="Source or explicit modes: auto, choice, noul, score. Repeat for multiple modes.",
        ),
    ] = None,
    extract_context: Annotated[
        bool | None,
        typer.Option(
            "--extract-context/--no-extract-context", help="Extract context from source records."
        ),
    ] = None,
    generate_context: Annotated[
        bool | None,
        typer.Option(
            "--generate-context/--no-generate-context",
            help="Generate context when source context is missing.",
        ),
    ] = None,
    context_relevance: Annotated[
        float | None,
        typer.Option(
            "--context-relevance", min=0, max=1, help="Relevance of generated context from 0 to 1."
        ),
    ] = None,
):
    """Generate, check and repair JEV records with configured models."""
    load_dotenv(Path.cwd() / ".env", override=False)
    loaded = config_with_overrides(config, input_path)
    if output_dir:
        loaded.output.directory = str(output_dir.resolve())
    if difficulty is not None:
        loaded.task.difficulty = difficulty
    if concurrency is not None:
        loaded.execution.concurrency = concurrency
    if text_batch_concurrency is not None:
        loaded.execution.text_batch_concurrency = text_batch_concurrency
    if llm_concurrency is not None:
        loaded.execution.llm_concurrency = llm_concurrency
    if modes:
        if any(mode not in {"auto", "choice", "noul", "score"} for mode in modes):
            raise typer.BadParameter(
                "modes must be auto, choice, noul or score", param_hint="--modes"
            )
        loaded.task.modes = modes
    if extract_context is not None:
        loaded.input.extract_context = extract_context
    if generate_context is not None:
        loaded.context.generate_if_missing = generate_context
    if context_relevance is not None:
        loaded.context.relevance = context_relevance
    from decisionforge import cli

    emit(asyncio.run(cli.execute(Config.model_validate(loaded.model_dump()))))


@data_app.command("resume")
def data_resume(
    run_dir: Annotated[
        Path,
        typer.Option(
            ..., "--run-dir", help="Run directory containing saved configuration and checkpoints."
        ),
    ],
    budget_usd: Annotated[
        float | None,
        typer.Option(
            "--budget-usd",
            help="New total run budget in USD, using configured model prices.",
            min=0,
        ),
    ] = None,
):
    """Resume a generation run from saved configuration and checkpoints."""
    load_dotenv(Path.cwd() / ".env", override=False)
    directory = run_dir.resolve()
    config = Config.model_validate_json(
        (state_directory(directory) / "config.resolved.json").read_text()
    )
    config.output.directory = str(directory)
    config.execution.resume = True
    if budget_usd is not None:
        config.execution.budget_limit_usd = budget_usd
    from decisionforge import cli

    emit(asyncio.run(cli.execute(Config.model_validate(config.model_dump()))))


@data_app.command("validate")
def data_validate(
    input_path: Annotated[
        Path,
        typer.Option(
            ...,
            "--input",
            help="Input file or PDF directory; repeat this option for multiple PDFs.",
        ),
    ],
):
    """Validate a JEV file and report its record count offline."""
    emit({"valid_records": validate_file(input_path)})


@data_app.command("export")
def data_export(
    run_dir: Annotated[
        Path,
        typer.Option(
            ..., "--run-dir", help="Run directory containing saved configuration and checkpoints."
        ),
    ],
    format: Annotated[
        str,
        typer.Option(
            "--format", help="Output format: canonical, jsonl, requests, jev-api, chat-sft, audit."
        ),
    ] = "canonical",
):
    """Export an existing run while holding its checkpoint lock."""
    if format not in {"canonical", "jsonl", "requests", "jev-api", "chat-sft", "audit"}:
        raise typer.BadParameter("unsupported format")
    directory = run_dir.resolve()
    state = state_directory(directory)
    if not any(
        (state / name).exists() for name in ("checkpoint.sqlite", "checkpoint.snapshot.sqlite")
    ):
        raise ValueError("Run checkpoint does not exist")
    with FileLock(str(state / ".run.lock"), timeout=0):
        store = Store(directory)
        try:
            result = export_run(
                store,
                directory,
                export_requests=format == "requests",
                include_audit=format == "audit",
                export_api=format == "jev-api",
            )
            if format == "chat-sft":
                result["chat_sft"] = str(export_chat_sft(directory))
            elif format == "jsonl":
                result["jsonl"] = str(export_canonical_jsonl(directory))
        finally:
            store.close()
    emit(result)


def train_execute(config: Path, run_dir: Path | None, validate_only: bool, resume: bool):
    from decisionforge.training.cli import execute as run

    return run(
        SimpleNamespace(
            command="train",
            config=str(config),
            run_dir=str(run_dir) if run_dir else None,
            validate_only=validate_only,
            resume=resume,
        )
    )


@train_app.command("run")
def train_run(
    config: Annotated[Path, typer.Option(..., "--config", help="YAML configuration file.")],
    run_dir: Annotated[
        Path | None,
        typer.Option(
            "--run-dir", help="Run directory containing saved configuration and checkpoints."
        ),
    ] = None,
    resume: Annotated[
        bool, typer.Option("--resume", help="Resume a compatible saved checkpoint.")
    ] = False,
):
    """Train a decision model using the configured optional backend."""
    load_dotenv(Path.cwd() / ".env", override=False)
    emit(train_execute(config, run_dir, False, resume))


@train_app.command("validate")
def train_validate(
    config: Annotated[Path, typer.Option(..., "--config", help="YAML configuration file.")],
    run_dir: Annotated[
        Path | None,
        typer.Option(
            "--run-dir", help="Run directory containing saved configuration and checkpoints."
        ),
    ] = None,
):
    """Validate training configuration and data offline."""
    emit(train_execute(config, run_dir, True, False))


def eval_execute(
    config: Path | None,
    run_dir: Path | None,
    validate_only: bool,
    resume: bool,
    score_only: bool,
    env_file: Path | None,
):
    from decisionforge.eval.cli import execute as run

    if config is None:
        return run(SimpleNamespace(command="eval-list"))
    return run(
        SimpleNamespace(
            command="eval",
            config=str(config),
            run_dir=str(run_dir) if run_dir else None,
            validate_only=validate_only,
            resume=resume,
            score_only=score_only,
            env_file=str(env_file) if env_file else None,
        )
    )


@eval_app.command("run")
def eval_run(
    config: Annotated[Path, typer.Option(..., "--config", help="YAML configuration file.")],
    run_dir: Annotated[
        Path | None,
        typer.Option(
            "--run-dir", help="Run directory containing saved configuration and checkpoints."
        ),
    ] = None,
    resume: Annotated[
        bool, typer.Option("--resume", help="Resume a compatible saved checkpoint.")
    ] = False,
    validate_only: Annotated[
        bool,
        typer.Option(
            "--validate-only", help="Validate configuration and data offline without model calls."
        ),
    ] = False,
    score_only: Annotated[
        bool,
        typer.Option(
            "--score-only",
            help="Rescore saved predictions offline (also available as eval rescore).",
        ),
    ] = False,
    env_file: Annotated[
        Path | None,
        typer.Option("--env-file", help="Credential environment file for online evaluation only."),
    ] = None,
):
    """Run evaluation or an offline configuration preflight."""
    if not (validate_only or score_only):
        load_dotenv(Path.cwd() / ".env", override=False)
    emit(eval_execute(config, run_dir, validate_only, resume, score_only, env_file))


@eval_app.command("rescore")
def eval_rescore(
    config: Annotated[Path, typer.Option(..., "--config", help="YAML configuration file.")],
    run_dir: Annotated[
        Path | None,
        typer.Option(
            "--run-dir", help="Run directory containing saved configuration and checkpoints."
        ),
    ] = None,
):
    """Rescore saved evaluation predictions without model calls."""
    emit(eval_execute(config, run_dir, False, False, True, None))


@eval_app.command("list")
def eval_list():
    """List available benchmarks and metrics offline."""
    emit(eval_execute(None, None, False, False, False, None))


@schema_app.command("export")
def schema_export(
    directory: Annotated[
        Path, typer.Option("--directory", help="Destination for generated JSON schemas.")
    ] = Path("schemas"),
):
    """Write public data, generation, training and evaluation schemas."""
    from decisionforge.eval.models import EvalConfig, Sample
    from decisionforge.training.config import TrainConfig

    models = {
        "standard_input.v2": StandardRecord,
        "checker.v2": CheckReport,
        "text_batch_plan.v1": TextBatchPlan,
        "text_batch_context.v1": TextBatchContext,
        "training_record.v1": TrainingRecord,
        "training_record.v2": LegacyTrainingExample,
        "training_record.v3": TrainingExample,
        "mapping.v1": MappingSpec,
        "config.v1": Config,
        "eval_sample.v1": Sample,
        "eval_config.v1": EvalConfig,
        "training_config.v1": TrainConfig,
    }
    directory.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        (directory / f"{name}.json").write_text(
            json.dumps(model.model_json_schema(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    emit({"schemas": len(models), "directory": str(directory.resolve())})


@app.command()
def doctor():
    """Check environment, optional backends, GPU and MinerU without network calls."""
    import importlib.util
    import os

    modules = {
        "pdf": (),
        "eval-data": ("pyarrow",),
        "eval-anyjev": ("anyjev", "torch", "transformers"),
        "eval-llm2jev": ("llm2jev", "sglang"),
        "train-unsloth": ("unsloth",),
        "train-unsloth-full": ("unsloth",),
        "train-swift": ("swift",),
        "train-laya": ("laya",),
        "train-openjev": ("openjev_rlcd",),
    }
    extras = {
        extra: {
            "available": all(importlib.util.find_spec(name) is not None for name in names),
            "install": f"uv sync --extra {extra}",
        }
        for extra, names in modules.items()
    }
    gpu = {
        "available": False,
        "note": "Install the selected training/evaluation extra to check GPU support.",
    }
    if importlib.util.find_spec("torch"):
        try:
            import torch

            gpu = {
                "available": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count(),
            }
        except (ImportError, OSError) as exc:
            gpu = {"available": False, "note": str(exc)}
    env_file = Path.cwd() / ".env"
    values = {**(dotenv_values(env_file) if env_file.is_file() else {}), **os.environ}
    emit(
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "env_file": env_file.is_file(),
            "extras": extras,
            "gpu": gpu,
            "mineru": {
                "api_key_set": bool(values.get("MINERU_API_KEY")),
                "note": "HTTP API; set MINERU_API_KEY for PDF parsing. No network request was made.",
            },
        }
    )


def run(argv=None) -> int:
    from typer.main import get_command

    try:
        get_command(app).main(args=argv, prog_name="dforge", standalone_mode=True)
        return 0
    except SystemExit as exc:
        return int(exc.code or 0)
