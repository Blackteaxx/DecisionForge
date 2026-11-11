from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path

import yaml
from filelock import FileLock

from decisionforge.exporter import read_training_records
from decisionforge.training.backend import get_backend, preflight
from decisionforge.training.config import TrainConfig
from decisionforge.training.data import ROLES, PreparedData, counts, export_splits, prepare_data
from decisionforge.util import atomic_write, digest, json_text, now, write_json


def semantic_config(config):
    value = config.model_dump()
    value["output"].pop("directory")
    if value["output"].get("checkpoint_directory") is None:
        value["output"].pop("checkpoint_directory", None)
    value["training"].pop("logging_steps")
    return value


def latest_checkpoint(directory, backend, checkpoint_directory=None):
    complete = [
        p
        for p in Path(checkpoint_directory or directory / "checkpoints").glob("checkpoint-*")
        if (p / "dforge_checkpoint.json").is_file()
    ]
    if not complete:
        raise ValueError("No complete Trainer checkpoint; cannot resume from adapter or from zero")
    try:
        checkpoint = max(complete, key=lambda p: int(p.name.split("-")[-1]))
        backend.validate_checkpoint(checkpoint)
    except (ValueError, OSError, KeyError) as exc:
        raise ValueError(f"Corrupt complete checkpoint: {exc}") from exc
    return checkpoint


def checkpoint_progress(checkpoint, backend=None):
    if backend is not None and callable(getattr(backend, "checkpoint_progress", None)):
        return backend.checkpoint_progress(checkpoint)
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    step, horizon = state["global_step"], state["max_steps"]
    if type(step) is not int or type(horizon) is not int or horizon < 1 or not 0 < step <= horizon:
        raise ValueError("Corrupt checkpoint training progress")
    loss = next(
        (entry["loss"] for entry in reversed(state.get("log_history", [])) if "loss" in entry), None
    )
    return step >= horizon, {
        "global_step": step,
        "last_logged_loss": loss,
        "completion_source": "terminal Trainer checkpoint",
        "selected_checkpoint": "final",
    }


def check_snapshot(prepared):
    ids = {r.record_id for r in prepared.records}
    assignments = prepared.assignments
    assigned = [a.get("record_id") for a in assignments]
    if (
        len(ids) != len(prepared.records)
        or len(set(assigned)) != len(assigned)
        or set(assigned) != ids
    ):
        raise ValueError("Training snapshot/assignments record IDs differ or contain duplicates")
    records = {r.record_id: r for r in prepared.records}
    groups = {}
    public = set()
    for a in assignments:
        r = records[a["record_id"]]
        if (
            a.get("group_id") != r.group_id
            or a.get("original_split") != r.split
            or a.get("role") not in ROLES
        ):
            raise ValueError(f"Training snapshot/assignments invalid for record_id={r.record_id}")
        if r.group_id in groups and groups[r.group_id] != a["role"]:
            raise ValueError("Training snapshot group crosses roles")
        groups[r.group_id] = a["role"]
        request = json_text(r.request.model_dump())
        if request in public:
            raise ValueError("Training snapshot contains duplicate public requests")
        public.add(request)
    if counts(prepared.records) != prepared.report["total"]:
        raise ValueError("Training snapshot counts differ from manifest")


def error_message(exc):
    message = str(exc)
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "WANDB_API_KEY"):
        secret = os.environ.get(name)
        if secret:
            message = message.replace(secret, "[REDACTED]")
    return message


def write_report(directory, manifest, report):
    if "error" in manifest:
        report["error"] = manifest["error"]
    else:
        report.pop("error", None)
    report.update(
        status=manifest["status"],
        stage=manifest["stage"],
        started_at=manifest["started_at"],
        updated_at=manifest["updated_at"],
    )
    if "ended_at" in manifest:
        report["ended_at"] = manifest["ended_at"]
    else:
        report.pop("ended_at", None)
    write_json(directory / "report.json", report)
    atomic_write(
        directory / "summary.md",
        "# Decision training\n\n"
        f"Status: {report['status']}; stage: {report['stage']}.\n\n"
        f"Metrics below are {report.get('diagnostics_source', 'upstream')} diagnostics, not DecisionForge benchmark metrics. "
        "Calibration results are fitting diagnostics. Agreement with teacher soft labels "
        "does not establish human correctness.\n\n```json\n"
        + json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n```\n",
    )


def run_training(config, *, resume=False, validate_only=False, backend_factory=None):
    config = TrainConfig.model_validate(config.model_dump())
    if resume and validate_only:
        raise ValueError("resume and validate-only are mutually exclusive")
    cap = preflight(config, resume=resume)
    if not validate_only and int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Use the public training coordinator without external torchrun")
    if config.distributed.strategy == "ddp" and backend_factory is not None and not validate_only:
        raise ValueError("Injected backend_factory supports inline single-device execution only")
    if validate_only:
        prepared = prepare_data(config)
        preflight(config, data=prepared)
        result = {"status": "validated", **prepared.report}
        if config.backend == "swift":
            from decisionforge.training.backends.swift_runtime import encode_check
            from decisionforge.training.backends.swift_task import map_request

            rows = [row for record in prepared.records for row in map_request(record.request, config.swift)[0]]
            result["swift"] = encode_check(config, rows)
        if config.backend == "openjev":
            from decisionforge.training.backends.openjev_task import STATIC_LABEL_CHECKS

            result["openjev"] = dict(STATIC_LABEL_CHECKS)
        return result
    directory = Path(config.output.directory).resolve()
    if not resume and config.output.checkpoint_directory:
        checkpoints = Path(config.output.checkpoint_directory)
        if checkpoints.exists() and (not checkpoints.is_dir() or any(checkpoints.iterdir())):
            raise ValueError("New checkpoint directory must be empty; use resume or a new name")
    # Validate every input before allocating a model or creating the formal run.
    prepared = None if resume else prepare_data(config)
    if prepared is not None:
        preflight(config, data=prepared)
    if resume and not directory.is_dir():
        raise ValueError("Training run does not exist")
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory / ".train.lock"), timeout=0):
        manifest_path = directory / "manifest.json"
        if resume:
            if not manifest_path.is_file():
                raise ValueError("Training manifest does not exist")
            manifest = json.loads(manifest_path.read_text())
            saved_config = TrainConfig.model_validate(
                yaml.safe_load((directory / "config.yaml").read_text())
            )
            if semantic_config(config) != semantic_config(saved_config):
                raise ValueError(
                    "Resume incompatible with effective training configuration; use a new run"
                )
            if manifest["status"] == "completed":
                raise ValueError("A completed run cannot resume training")
            split = json.loads((directory / "splits.json").read_text())
            prepared = PreparedData(
                list(read_training_records(directory / "samples.jsonl")),
                split["assignments"],
                manifest["data"],
            )
            check_snapshot(prepared)
            verify_snapshot_identity(directory, manifest, config)
            if manifest["status"] == "running":
                manifest.setdefault("interruptions", []).append(
                    {"at": now(), "reason": "stale running state"}
                )
                manifest.update(status="interrupted", updated_at=now())
                write_json(manifest_path, manifest)
            report = json.loads((directory / "report.json").read_text())
        else:
            if any(p.name != ".train.lock" for p in directory.iterdir()):
                raise ValueError(
                    "New training directory must be empty; use --resume or a new directory"
                )
            manifest = {
                "version": "decisionforge.training.run.v1",
                "input_path": config.data.path,
                "model": config.model.model_dump(),
                "seed": config.training.seed,
                "started_at": now(),
                "updated_at": now(),
                "status": "running",
                "stage": "training",
                "python": platform.python_version(),
                "data": prepared.report,
                "capabilities": cap.as_dict(),
                "finetuning": config.finetuning,
                "distributed": config.distributed.model_dump(),
            }
            atomic_write(
                directory / "config.yaml", yaml.safe_dump(config.model_dump(), sort_keys=False)
            )
            atomic_write(
                directory / "samples.jsonl",
                "".join(
                    json.dumps(r.model_dump(), ensure_ascii=False, allow_nan=False) + "\n"
                    for r in prepared.records
                ),
            )
            write_json(
                directory / "splits.json",
                {
                    "seed": config.training.seed,
                    "auto_holdout": config.data.auto_holdout.model_dump(),
                    "assignments": prepared.assignments,
                },
            )
            write_json(directory / "data_report.json", prepared.report)
            report = {
                "diagnostics": [],
                "calibration": {"enabled": config.calibration.enabled},
                "artifacts": {},
                "diagnostics_source": cap.diagnostics_source,
            }
            manifest["identity"] = snapshot_identity(directory, config)
            write_json(manifest_path, manifest)
            write_report(directory, manifest, report)
        export_splits(prepared, directory)
        if config.distributed.strategy == "ddp":
            from decisionforge.training.distributed import launch_distributed

            # Effective logging overrides are persisted before workers read config.yaml.
            atomic_write(
                directory / "config.yaml", yaml.safe_dump(config.model_dump(), sort_keys=False)
            )
            try:
                return launch_distributed(config, directory, resume)
            except BaseException as exc:
                # torchrun and its owned children have already been reaped here.
                manifest = json.loads(manifest_path.read_text())
                report = json.loads((directory / "report.json").read_text())
                record_failure(directory, manifest, report, exc)
                raise
        return run_lifecycle(config, directory, prepared, manifest, report, resume, backend_factory)


def snapshot_identity(directory, config):
    return {
        "config": digest(semantic_config(config)),
        **{
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ("samples.jsonl", "splits.json")
        },
    }


def verify_snapshot_identity(directory, manifest, config):
    if manifest.get("identity") and manifest["identity"] != snapshot_identity(directory, config):
        raise ValueError("Resume incompatible with saved config/data snapshot identity")


def record_failure(directory, manifest, report, exc):
    if manifest["stage"] == "completed":
        # A launch/result handoff can fail after rank0 finishes its durable work.
        # Resume must reload the saved calibration and finish the export probe.
        manifest["stage"] = "exported"
    manifest.update(
        status="interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed",
        updated_at=now(),
        ended_at=now(),
        error={"kind": type(exc).__name__, "message": error_message(exc)},
    )
    write_report(directory, manifest, report)
    write_json(directory / "manifest.json", manifest)


def effective_backend_config(config, manifest, resume):
    previous = manifest.get("loaded_model", {})
    if (
        resume
        and config.model.revision is None
        and previous.get("actual_revision")
        and not Path(config.model.name).expanduser().is_dir()
    ):
        config = config.model_copy(deep=True)
        config.model.revision = previous["actual_revision"]
    return config


def verify_environment(previous, current):
    # Old manifests did not record every field; retain their single-GPU LoRA support.
    if any(
        current.get("packages", {}).get(name) != value
        for name, value in previous.get("packages", {}).items()
    ):
        raise ValueError(
            "Resume incompatible with saved environment packages; restore the original environment"
        )
    for key in (
        "precision",
        "cuda",
        "world_size",
        "devices",
        "upstream",
        "cuda_visible_devices",
    ):
        if key in previous and previous[key] != current.get(key):
            raise ValueError(
                f"Resume incompatible with saved environment {key}; restore the original environment"
            )


def run_lifecycle(
    config, directory, prepared, manifest, report, resume=False, backend_factory=None
):
    manifest_path = directory / "manifest.json"
    cap = preflight(config, resume=resume, data=prepared)
    backend = None
    try:
        backend_config = effective_backend_config(config, manifest, resume)
        previous_model = manifest.get("loaded_model", {})
        backend = (backend_factory or get_backend)(backend_config, directory)
        environment = backend.environment()
        if resume:
            verify_environment(manifest.get("environment", {}), environment)
        if "error" in manifest:
            manifest.setdefault("previous_errors", []).append(manifest.pop("error"))
        if resume:
            atomic_write(
                directory / "config.yaml", yaml.safe_dump(config.model_dump(), sort_keys=False)
            )
        manifest.pop("ended_at", None)
        manifest.update(environment=environment, status="running", updated_at=now())
        write_json(manifest_path, manifest)
        checkpoint = (
            latest_checkpoint(directory, backend, config.output.checkpoint_directory)
            if resume
            else None
        )
        encoded = backend.prepare(prepared)
        if encoded.get("model"):
            if (
                resume
                and previous_model
                and any(
                    previous_model.get(key) != encoded["model"].get(key)
                    for key in (
                        "actual_name",
                        "actual_revision",
                        "source_metadata",
                        "backbone_metadata",
                    )
                )
            ):
                raise ValueError(
                    "Resume incompatible with loaded backbone identity; restore the original "
                    "model revision/files or start a new run"
                )
            manifest["loaded_model"] = encoded["model"]
            write_json(directory / "loaded_model.json", encoded["model"])
            write_json(manifest_path, manifest)
        write_json(directory / "data_report.json", {**prepared.report, "encoding": encoded})
        if manifest["stage"] == "training":
            terminal, summary = (
                checkpoint_progress(checkpoint, backend) if checkpoint else (False, None)
            )
            if terminal:
                final = checkpoint
                backend.load_final_checkpoint(final)
                report.setdefault("training", summary)
                report["training"]["resume_action"] = "terminal checkpoint; no optimizer updates"
            else:
                report["training"] = backend.train(checkpoint)
                final = backend.save_final_checkpoint() if cap.full_resume else None
                if final is not None:
                    backend.validate_checkpoint(final)
            manifest.update(stage="trained")
            if final is not None:
                manifest["final_checkpoint"] = os.path.relpath(final, directory)
            # The outcome is durable before committing the phase; a kill between the
            # writes can then recover the terminal checkpoint without another update.
            write_report(directory, manifest, report)
            write_json(manifest_path, manifest)
        else:
            final = directory / manifest["final_checkpoint"]
            backend.validate_checkpoint(final)
            backend.load_final_checkpoint(final)
            if "training" not in report:
                _, report["training"] = checkpoint_progress(final, backend)
            if cap.calibration != "none" and manifest["stage"] in {
                "calibrated",
                "ready_to_export",
                "exported",
            }:
                backend.restore_calibration(
                    json.loads((directory / "calibration.json").read_text())
                )

        finish = getattr(backend, "finish_distributed", None)
        if finish is not None:
            finish()

        def diagnostic(role, phase):
            if not prepared.roles[role] or any(
                d["role"] == role and d["phase"] == phase for d in report["diagnostics"]
            ):
                return
            report["diagnostics"].append(backend.evaluate(role, phase))
            write_report(directory, manifest, report)

        if manifest["stage"] == "trained":
            for role in ("validation", "test"):
                diagnostic(role, "uncalibrated")
            if config.calibration.enabled:
                diagnostic("calibration", "uncalibrated")
                report["calibration"].update(
                    result=backend.calibrate(), meaning="fitting diagnostics"
                )
                diagnostic("calibration", "calibrated")
                diagnostic("test", "calibrated")
            if cap.calibration != "none":
                write_json(directory / "calibration.json", backend.calibration_state())
            report["calibration"]["method"] = cap.calibration
            manifest.update(
                stage="calibrated" if config.calibration.enabled else "ready_to_export",
                updated_at=now(),
            )
            write_report(directory, manifest, report)
            write_json(manifest_path, manifest)
        if manifest["stage"] in {"calibrated", "ready_to_export"}:
            report["artifacts"] = backend.save()
            manifest.update(stage="exported", updated_at=now())
            write_report(directory, manifest, report)
            write_json(manifest_path, manifest)
        report["reload"] = backend.reload_probe(prepared)
        completed_backend, backend = backend, None
        completed_backend.close()
        manifest.update(status="completed", stage="completed", ended_at=now(), updated_at=now())
        write_report(directory, manifest, report)
        write_json(manifest_path, manifest)
        return {
            "status": "completed",
            "run_directory": str(directory),
            "report": str(directory / "report.json"),
        }
    except BaseException as exc:
        record_failure(directory, manifest, report, exc)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        message = error_message(exc)
        if "out of memory" in message.lower():
            message += "; reduce batch size/max_seq_length, increase gradient accumulation or enable 4-bit loading"
        raise ValueError(f"Training failed ({type(exc).__name__}): {message}") from exc
    finally:
        if backend is not None:
            backend.close()
