"""Calibrate the independently selected checkpoint without resuming or retraining it."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from decisionforge.eval.checkpoint import file_identity, validate_checkpoint
from decisionforge.exporter import read_training_records
from decisionforge.training.backends.unsloth import (
    encoding_report,
    native_prediction,
    retain_unfitted_calibration,
    verify_calibration_reload,
)
from decisionforge.training.data import to_unsloth_row
from decisionforge.util import now, write_json
from examples.domain.medical_triage.scripts.v1.experiment import select_checkpoint


def calibration_records(records):
    groups, public, selected = {}, set(), []
    for record in records:
        if record.group_id in groups and groups[record.group_id] != record.split:
            raise ValueError("Source group crosses roles; calibration is not independent")
        groups[record.group_id] = record.split
        request = json.dumps(record.request.model_dump(), sort_keys=True)
        if request in public:
            raise ValueError("Duplicate public patient request across calibration input")
        public.add(request)
        if record.split == "calibration":
            selected.append(record)
    if len(selected) < 10:
        raise ValueError("Calibration needs at least 10 independent-role examples")
    catalogs = {len(q.criteria) for r in selected for q in r.request.questions.values()}
    if catalogs != {2, 4} or any(
        q.type != "choice" for r in selected for q in r.request.questions.values()
    ):
        raise ValueError("Calibration requires both joint medical Choice tasks")
    return selected


def load_model(path, max_length):
    import torch
    from unsloth import FastDecisionModel

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("Calibration requires the authorized BF16 CUDA resource")
    model, tokenizer = FastDecisionModel.from_pretrained(
        str(path),
        dtype=torch.bfloat16,
        max_seq_length=max_length,
        load_in_4bit=False,
        local_files_only=True,
        use_gradient_checkpointing=False,
    )
    model.eval()
    return FastDecisionModel, torch, model, tokenizer


def reload_model(model_path, requests_path, output, max_length):
    api, _, model, tokenizer = load_model(model_path, max_length)
    requests = json.loads(requests_path.read_text())
    predictions = [
        native_prediction(api.predict(model, tokenizer, **r), r["questions"]) for r in requests
    ]
    write_json(output, {"predictions": predictions, "decision_config": model.decision_config})


def fit(selection_path, data_path, output, max_length=8192, batch_size=2):
    selected = select_checkpoint([json.loads(selection_path.read_text())])
    checkpoint = validate_checkpoint(selected["checkpoint"])
    trainer = json.loads((checkpoint / "trainer_state.json").read_text())
    if trainer["global_step"] != selected["step"]:
        raise ValueError("Selected step differs from actual checkpoint")
    records = calibration_records(list(read_training_records(data_path)))
    source_identity = file_identity(checkpoint)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running",
        "stage": "loading",
        "started_at": now(),
        "operation": "post_selection_independent_calibration",
        "selection": selected,
        "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        "training_data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "source_checkpoint_identity": source_identity,
        "calibration_records": len(records),
        "calibration_groups": len({r.group_id for r in records}),
        "source_roles_used": ["calibration"],
        "optimizer_updates": 0,
        "max_seq_length": max_length,
        "batch_size": batch_size,
    }
    write_json(output / "manifest.json", manifest)
    write_json(output / "calibration-records.json", [r.model_dump(mode="json") for r in records])
    try:
        api, torch, model, tokenizer = load_model(checkpoint, max_length)
        rows = [to_unsloth_row(r) for r in records]
        items, raw = api.build_dataset(rows, tokenizer, model)
        encoding = encoding_report(records, items, raw, max_length)
        write_json(output / "encoding.json", encoding)
        if (
            encoding["skipped"]
            or encoding["truncated"]
            or encoding["questions"] != sum(len(r.request.questions) for r in records)
        ):
            raise ValueError("Calibration encoding skipped or truncated original questions")
        before = json.loads(json.dumps(model.decision_config))
        uncalibrated = api.evaluate(model, tokenizer, items, batch_size=batch_size)
        manifest.update(stage="calibrating", updated_at=now())
        write_json(output / "manifest.json", manifest)
        raw_fit = api.calibrate(model, tokenizer, items, batch_size=batch_size)
        from unsloth.models.decision import _laya

        adjustment = retain_unfitted_calibration(
            before,
            model.decision_config,
            raw_fit.get("fitted_types", []),
            _laya().common.clamp_temperature,
        )
        calibrated = api.evaluate(model, tokenizer, items, batch_size=batch_size)
        write_json(
            output / "calibration.json",
            {
                "before": before,
                "after": model.decision_config,
                "fit": raw_fit,
                "preservation_adjustment": adjustment,
                "uncalibrated": uncalibrated,
                "calibrated": calibrated,
                "meaning": "calibration-role fitting diagnostics, not independent test results",
            },
        )
        # One example per task, plus further non-test observations, for independent-process reload.
        probes, covered = [], set()
        for record in records:
            size = len(next(iter(record.request.questions.values())).criteria)
            if size not in covered:
                probes.append(record.request.model_dump(exclude_none=True))
                covered.add(size)
        probes.extend(r.request.model_dump(exclude_none=True) for r in records[:2])
        expected = [
            native_prediction(api.predict(model, tokenizer, **r), r["questions"]) for r in probes
        ]
        original_config = json.loads(json.dumps(model.decision_config))
        write_json(output / "reload-requests.json", probes)
        write_json(output / "reload-expected.json", expected)
        model.save_pretrained(str(output / "model"), tokenizer)
        manifest.update(stage="exported", updated_at=now())
        write_json(output / "manifest.json", manifest)
        del model, tokenizer, items
        gc.collect()
        torch.cuda.empty_cache()
        command = [
            sys.executable,
            "-m",
            "examples.domain.medical_triage.scripts.v1.calibrate",
            "reload",
            "--model",
            str(output / "model"),
            "--requests",
            str(output / "reload-requests.json"),
            "--output",
            str(output / "reload-actual.json"),
            "--max-seq-length",
            str(max_length),
        ]
        with (
            (output / "reload.stdout.log").open("w") as out,
            (output / "reload.stderr.log").open("w") as err,
        ):
            subprocess.run(command, check=True, stdout=out, stderr=err)
        actual = json.loads((output / "reload-actual.json").read_text())
        verify_calibration_reload(original_config, actual["decision_config"])
        if len(actual["predictions"]) != len(expected):
            raise ValueError("New process returned a different number of predictions")
        maximum_delta = 0.0
        for left, right in zip(expected, actual["predictions"], strict=True):
            if set(left["answers"]) != set(right["answers"]):
                raise ValueError("Reload changed question IDs")
            for key, answer in left["answers"].items():
                reloaded = right["answers"][key]
                if answer["choice"] != reloaded["choice"]:
                    raise ValueError("New process changed native decision")
                for label, probability in answer["probabilities"].items():
                    maximum_delta = max(
                        maximum_delta, abs(probability - reloaded["probabilities"][label])
                    )
        if maximum_delta > 0.002:
            raise ValueError(f"Reload probability drift exceeds 0.002: {maximum_delta}")
        if source_identity != file_identity(checkpoint):
            raise ValueError("Source checkpoint changed during calibration")
        manifest.update(
            status="completed",
            stage="completed",
            updated_at=now(),
            ended_at=now(),
            reload={
                "new_process": True,
                "questions": len(probes),
                "maximum_probability_delta": maximum_delta,
                "tolerance": 0.002,
                "same_native_decisions": True,
                "calibration_restored": True,
            },
            model_identity=file_identity(output / "model"),
        )
    except BaseException as exc:
        manifest.update(
            status="failed",
            updated_at=now(),
            error={"kind": type(exc).__name__, "message": str(exc)},
        )
        raise
    finally:
        write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit_parser = sub.add_parser("fit")
    fit_parser.add_argument("--selection", type=Path, required=True)
    fit_parser.add_argument("--data", type=Path, required=True)
    fit_parser.add_argument("--output", type=Path, required=True)
    fit_parser.add_argument("--max-seq-length", type=int, default=8192)
    fit_parser.add_argument("--batch-size", type=int, default=2)
    reload_parser = sub.add_parser("reload")
    reload_parser.add_argument("--model", type=Path, required=True)
    reload_parser.add_argument("--requests", type=Path, required=True)
    reload_parser.add_argument("--output", type=Path, required=True)
    reload_parser.add_argument("--max-seq-length", type=int, default=8192)
    args = parser.parse_args()
    if args.command == "fit":
        print(
            json.dumps(
                fit(args.selection, args.data, args.output, args.max_seq_length, args.batch_size),
                indent=2,
            )
        )
    else:
        reload_model(args.model, args.requests, args.output, args.max_seq_length)


if __name__ == "__main__":
    main()
