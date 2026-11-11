from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import defaultdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from filelock import FileLock

from decisionforge.eval import PROTOCOL_VERSION, VERSION
from decisionforge.eval.adapters.jev import create_adapter
from decisionforge.eval.data import load_samples, select_samples, write_jsonl
from decisionforge.eval.models import (
    AnyJevModelConfig,
    EvalConfig,
    LayaModelConfig,
    LocalModelConfig,
    OpenJevModelConfig,
    ScalarTarget,
    SoftTarget,
    SwiftModelConfig,
)
from decisionforge.eval.registry import load_registry
from decisionforge.eval.report import create_report
from decisionforge.eval.source_identity import inference_sources, source_versions
from decisionforge.eval.store import EvalStore
from decisionforge.util import digest, now, write_json


def validate_benchmark(config, samples, registry):
    benchmark = registry.benchmark(config.benchmark)
    if config.benchmark_version and config.benchmark_version != benchmark.version:
        raise ValueError("Benchmark version mismatch")
    benchmark.validate(registry, config.metrics)
    if any(s.question.type not in benchmark.supported_types for s in samples):
        raise ValueError("Unsupported question type for benchmark")
    if hasattr(benchmark, "validate_samples"):
        benchmark.validate_samples(samples)
    if benchmark.name == "jevbench" and any(
        isinstance(s.target, (SoftTarget, ScalarTarget)) for s in samples
    ):
        raise ValueError("Pinned JevBench supports hard/null references only")
    return benchmark


def preflight(config, registry=None):
    config = EvalConfig.model_validate(config.model_dump())
    samples = select_samples(load_samples(config.data.path), config.data)
    registry = registry if registry is not None else load_registry()
    benchmark = validate_benchmark(config, samples, registry)
    from decisionforge.media import reject_media

    if config.model.adapter not in {"swift_local", "swift_chat_local"}:
        for sample in samples:
            reject_media(sample.request, config.model.adapter)
    if isinstance(config.model, LocalModelConfig):
        from decisionforge.eval.checkpoint import validate_checkpoint

        validate_checkpoint(config.model.path)
    if isinstance(config.model, AnyJevModelConfig):
        from decisionforge.training_free.anyjev import validate_backbone
        from decisionforge.training_free.anyjev.processing import validate_request

        validate_backbone(config.model)
        for sample in samples:
            validate_request(sample.request)
    if isinstance(config.model, LayaModelConfig):
        from decisionforge.training.backends.laya import validate_artifact

        validate_artifact(config.model.path)
    if isinstance(config.model, OpenJevModelConfig):
        from decisionforge.training.backends.openjev import validate_artifact
        from decisionforge.training.backends.openjev_task import public_item

        validate_artifact(config.model.path)
        for sample in samples:
            public_item(sample.request.model_dump(mode="json", exclude_none=True), "decision")
    if isinstance(config.model, SwiftModelConfig) and config.model.adapter == "swift_local":
        from decisionforge.training.backends.swift import validate_artifact

        validate_artifact(config.model.path, model_type=config.model.model_type)
    if isinstance(config.model, SwiftModelConfig):
        from decisionforge.training.backends.swift_task import map_request
        from decisionforge.training.config import SwiftConfig

        for sample in samples:
            map_request(
                sample.request,
                SwiftConfig(
                    model_type=config.model.model_type,
                    usage=config.model.usage,
                    allow_none=config.model.allow_none,
                ),
            )
    return config, samples, registry, benchmark


async def execute_samples(config, samples, store, adapter):
    done = {p["id"] for p in store.records("predictions")}
    pending = [s for s in samples if s.id not in done]
    if not pending:
        return "completed"
    semaphore, auth_stop = asyncio.Semaphore(config.execution.concurrency), asyncio.Event()
    stop_reason = "completed"
    history = store.records("attempts")
    history_by_sample = defaultdict(list)
    for attempt in history:
        history_by_sample[attempt["sample_id"]].append(attempt)
    session = store.start_session({"execution": config.execution.model_dump()})
    window_start = time.monotonic()

    async def evaluate(sample):
        nonlocal stop_reason
        enqueued = window_start
        async with semaphore:
            if auth_stop.is_set():
                return
            previous = history_by_sample[sample.id]
            unknown = sum(a["status"] == "outcome_unknown" for a in previous)
            spent = sum(a["status"] == "finished" for a in previous)
            # An unknown send may have been charged. Allow an explicit recorded replay.
            budget = max(config.execution.max_attempts - spent, 1 if unknown else 0)
            if budget == 0:
                # A crash after a retryable attempt must not silently reset its budget.
                raise ValueError(
                    f"Attempt budget exhausted without terminal prediction: {sample.id}"
                )
            payload = {"model": config.model.model, **sample.request.model_dump(exclude_none=True)}
            if hasattr(adapter, "prepare_request"):
                payload = adapter.prepare_request(payload, sample.question)
            attempt_ids = [a["id"] for a in previous]
            costs = [a.get("cost_usd") for a in previous]
            if previous and previous[-1].get("retry_delay_seconds") is not None:
                last = previous[-1]
                elapsed = (
                    datetime.now(UTC) - datetime.fromisoformat(last["finished_at"])
                ).total_seconds()
                remaining = min(
                    config.execution.max_retry_delay, last["retry_delay_seconds"] - elapsed
                )
                if remaining > 0:
                    await asyncio.sleep(remaining)
            for index in range(budget):
                if auth_stop.is_set():
                    return
                identity = store.start_attempt(sample.id, session, adapter.redact(payload))
                attempt_ids.append(identity)
                outcome = await adapter.send(payload, sample.question)
                costs.append(outcome["cost_usd"])
                authentication = outcome["http_status"] in {401, 403}
                if outcome.get("fatal"):
                    auth_stop.set()
                    stop_reason = "model_failed"
                if authentication:
                    auth_stop.set()
                    stop_reason = (
                        "access_denied"
                        if outcome["http_status"] == 403
                        else "authentication_failed"
                    )
                delay = outcome["retry_after_seconds"]
                if delay is None:
                    delay = min(
                        config.execution.max_retry_delay,
                        config.execution.backoff * 2 ** (spent + index),
                    )
                retry = (
                    outcome["retryable"]
                    and index + 1 < budget
                    and delay <= config.execution.max_retry_delay
                    and not auth_stop.is_set()
                )
                outcome["retry_delay_seconds"] = delay if retry else None
                prediction = (
                    None
                    if retry
                    else {
                        **outcome,
                        "id": sample.id,
                        "session_id": session,
                        "attempt_ids": attempt_ids,
                        "replayed_unknown_attempts": unknown,
                        "sample_elapsed_ms": (time.monotonic() - enqueued) * 1000,
                        "cost_usd": sum(costs) if all(c is not None for c in costs) else None,
                        "cost_basis": "reported_usage_cost"
                        if all(c is not None for c in costs)
                        else outcome["cost_basis"],
                    }
                )
                store.finish_attempt(identity, outcome, prediction)
                if not retry:
                    return
                await asyncio.sleep(delay)

    tasks = [asyncio.create_task(evaluate(s)) for s in pending]
    try:
        await asyncio.gather(*tasks)
        status = stop_reason
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        store.finish_session(session, "interrupted", None)
        raise
    else:
        store.finish_session(session, status, time.monotonic() - window_start)
        return status


async def run_evaluation(
    config,
    *,
    resume=False,
    score_only=False,
    validate_only=False,
    transport=None,
    registry=None,
    adapter_factory=None,
):
    if sum((resume, score_only, validate_only)) > 1:
        raise ValueError("resume, score-only and validate-only are mutually exclusive")
    config = EvalConfig.model_validate(config.model_dump())
    directory = Path(config.output).resolve()
    code = source_versions()
    if score_only:
        if not (directory / ".eval/events.jsonl").exists():
            raise ValueError("Evaluation journal does not exist")
        samples = load_samples(directory / "samples.jsonl", resolve_media=False)
        registry = registry if registry is not None else load_registry()
        benchmark = validate_benchmark(config, samples, registry)
    else:
        config, samples, registry, benchmark = preflight(config, registry)
    if validate_only:
        result = {"status": "validated", "samples": len(samples), "benchmark": benchmark.name}
        if isinstance(config.model, AnyJevModelConfig):
            result["anyjev"] = {
                "selected_types": sorted({s.question.type for s in samples}),
                "choice_max_options": None,
                "label_candidate_limit": 18278,
                "choice_labels": "tokenizer_verified_A_to_ZZZ",
                "choice_catalog_capacity": "tokenizer_dependent_runtime_check",
                "label_protocol_version": 1,
                "score_levels": [2, 10],
                "runtime_checks": "deferred: CUDA, precision, tokenizer labels, full prompt length",
            }
        if isinstance(config.model, OpenJevModelConfig):
            from decisionforge.training.backends.openjev_task import STATIC_LABEL_CHECKS

            result["openjev"] = dict(STATIC_LABEL_CHECKS)
        return result
    identity = (
        {
            "dataset_sha256": hashlib.sha256(Path(config.data.path).read_bytes()).hexdigest(),
            "selection": config.data.model_dump(exclude={"path"}),
            "sample_ids": [s.id for s in samples],
            "model": config.model.model_dump(),
            "execution": config.execution.model_dump(),
            "protocol_version": PROTOCOL_VERSION,
            "inference_code": inference_sources(code),
            "runtime": {n: version(n) for n in ("httpx", "pydantic")},
        }
        if not score_only
        else None
    )
    if not score_only and config.model.adapter == "chat_api":
        from decisionforge.eval import chat_prompt

        identity["chat_protocol"] = {
            "prompt_version": chat_prompt.PROMPT_VERSION,
            "schema_version": chat_prompt.SCHEMA_VERSION,
        }
    if not score_only and isinstance(config.model, LocalModelConfig):
        from decisionforge.eval.checkpoint import checkpoint_identity, dependency_versions

        identity["local_checkpoint"] = checkpoint_identity(config.model.path)
        identity["runtime"].update(dependency_versions())
    if not score_only and isinstance(config.model, SwiftModelConfig):
        from decisionforge.training.backends.swift import artifact_identity

        identity["swift"] = {
            "model_type": config.model.model_type,
            "base_model": config.model.base_model,
            "base_revision": config.model.base_revision,
            "adapter_revision": config.model.adapter_revision,
            "checkpoint": artifact_identity(config.model.path, model_type=config.model.model_type)
            if config.model.adapter == "swift_local"
            else config.model.path,
        }
    if not score_only and isinstance(config.model, OpenJevModelConfig):
        from decisionforge.training.backends.openjev import (
            UPSTREAM_REVISION,
            artifact_identity,
            dependency_versions,
        )

        identity["openjev"] = {
            "upstream_revision": UPSTREAM_REVISION,
            **artifact_identity(config.model.path),
        }
        identity["runtime"].update(dependency_versions())
    if not score_only and isinstance(config.model, LayaModelConfig):
        from decisionforge.training.backends.laya import (
            UPSTREAM_REVISION,
            artifact_identity,
            dependency_versions,
        )

        identity["laya"] = {
            "upstream_revision": UPSTREAM_REVISION,
            **artifact_identity(config.model.path),
        }
        identity["runtime"].update(dependency_versions())
    if not score_only and isinstance(config.model, AnyJevModelConfig):
        from decisionforge.training_free.anyjev import (
            backbone_identity,
            dependency_versions,
            method_identity,
        )

        identity["anyjev"] = method_identity()
        identity["backbone"] = backbone_identity(config.model)
        identity["runtime"].update(dependency_versions())
    state = directory / ".eval"
    state.mkdir(parents=True, exist_ok=True)
    with FileLock(str(state / "run.lock"), timeout=0):
        manifest_path = directory / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not (resume or score_only):
                raise ValueError("Evaluation run exists; use --resume or a new output directory")
            if resume and manifest["fingerprint"] != digest(identity):
                raise ValueError(
                    "Resume incompatible with stored inference configuration/data/code; "
                    "use the matching old checkout to resume or a fresh run directory. "
                    "Offline --score-only remains available"
                )
            if not (state / "events.jsonl").exists():
                raise ValueError("Evaluation journal does not exist; refusing to repeat calls")
            snapshot = load_samples(directory / "samples.jsonl")
            if digest([s.model_dump() for s in snapshot]) != manifest["sample_snapshot_hash"]:
                raise ValueError("Evaluation sample snapshot has changed")
            samples = snapshot
        elif resume or score_only:
            raise ValueError("Evaluation manifest does not exist")
        else:
            if any(p.name != ".eval" for p in directory.iterdir()):
                raise ValueError("New evaluation output directory must be empty")
            manifest = {
                "version": VERSION,
                "created_at": now(),
                "fingerprint": digest(identity),
                "inference": identity,
                "config": config.model_dump(),
                "code": code,
                "sample_snapshot_hash": digest([s.model_dump() for s in samples]),
            }
            write_jsonl(directory / "samples.jsonl", [s.model_dump() for s in samples])
            write_json(manifest_path, manifest)
        store, adapter = EvalStore(directory, read_only=score_only), None
        try:
            status = "scored"
            if not score_only:
                store.recover()
                if len(store.records("predictions")) < len(samples):
                    adapter = await (adapter_factory or create_adapter)(
                        config.model, config.execution, transport
                    )
                    if isinstance(
                        config.model,
                        (
                            LocalModelConfig,
                            LayaModelConfig,
                            OpenJevModelConfig,
                            SwiftModelConfig,
                            AnyJevModelConfig,
                        ),
                    ):
                        metadata = adapter.metadata
                        previous = manifest.get("local_model")
                        if (
                            resume
                            and previous is not None
                            and (
                                previous.get("backbone") != metadata.get("backbone")
                                or previous.get("tokenizer") != metadata.get("tokenizer")
                                or previous.get("method") != metadata.get("method")
                                or previous.get("dependencies") != metadata.get("dependencies")
                                or previous.get("calibration") != metadata.get("calibration")
                                or previous.get("runtime", {}).get("precision")
                                != metadata.get("runtime", {}).get("precision")
                                or isinstance(config.model, (OpenJevModelConfig, SwiftModelConfig))
                                and (
                                    any(
                                        previous.get(key) != metadata.get(key)
                                        for key in (
                                            "model",
                                            "model_identity_source",
                                            "artifact_format",
                                            "checkpoint",
                                            "sampling",
                                            "generation_config",
                                            "public_protocol",
                                            "label_protocol",
                                            "trained_pool",
                                            "readout_policy",
                                            "upstream_repo",
                                            "upstream_revision",
                                            "source_sha256",
                                            "train_sha256",
                                            "source_verified",
                                        )
                                    )
                                    or any(
                                        previous.get("runtime", {}).get(key)
                                        != metadata.get("runtime", {}).get(key)
                                        for key in (
                                            "device",
                                            "gpu",
                                            "precision",
                                            "logits_precision",
                                            "cuda",
                                            "effective_max_seq_length",
                                            "attention_implementation",
                                        )
                                    )
                                )
                            )
                        ):
                            raise ValueError(
                                "Resume incompatible with loaded local model/runtime identity"
                            )
                        manifest["local_model"] = metadata
                        write_json(manifest_path, manifest)
                    status = await execute_samples(config, samples, store, adapter)
                else:
                    status = "completed"
                store.export()
            report = create_report(
                directory,
                store,
                samples,
                registry,
                benchmark,
                config.metrics,
                manifest,
                code,
                score_only,
            )
            return {"status": status, "run_directory": str(directory), "report": report}
        finally:
            try:
                if not score_only:
                    store.export()
            finally:
                try:
                    if adapter:
                        await adapter.close()
                finally:
                    store.close()
