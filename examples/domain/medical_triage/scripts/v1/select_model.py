"""Evaluate all committed raw-training checkpoints on independent validation only."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

import yaml

from decisionforge.eval import EvalConfig, load_registry, run_evaluation
from decisionforge.eval.checkpoint import validate_checkpoint
from decisionforge.eval.data import load_samples
from decisionforge.util import write_json
from examples.domain.medical_triage.scripts.v1.experiment import TASKS, select_checkpoint


def checkpoints_for_selection(training_run):
    manifest = json.loads((training_run / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("Selection requires completed raw training")
    config = yaml.safe_load((training_run / "config.yaml").read_text())
    if config["calibration"]["enabled"]:
        raise ValueError(
            "Disable automatic last-checkpoint calibration; calibrate after validation selection"
        )
    checkpoints = sorted(
        (
            p
            for p in (training_run / "checkpoints").glob("checkpoint-*")
            if (p / "dforge_checkpoint.json").is_file()
        ),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if len(checkpoints) < 2:
        raise ValueError("At least two complete committed checkpoints are required for selection")
    return checkpoints


async def evaluate_checkpoints(training_run, validation, control_root, run_root):
    checkpoints = checkpoints_for_selection(training_run)
    samples = load_samples(validation)
    if {s.split for s in samples} != {"validation"} or {s.subset for s in samples} != set(TASKS):
        raise ValueError("Selection needs both tasks from independent validation only")
    registry = load_registry()
    benchmark = registry.benchmark("medical_decisions")
    extension_root = control_root.parents[1] / "benchmark_ext"
    extension = json.loads((extension_root / "IDENTITY.json").read_text())
    control_root.mkdir(parents=True, exist_ok=False)
    candidates = []
    for checkpoint in checkpoints:
        validate_checkpoint(checkpoint)
        step = int(checkpoint.name.split("-")[-1])
        name = f"checkpoint-{step}"
        control = control_root / name
        control.mkdir()
        output = run_root / name / "run"
        config = EvalConfig.model_validate(
            {
                "benchmark": "medical_decisions",
                "benchmark_version": "1",
                "data": {"path": str(validation), "splits": ["validation"]},
                "model": {
                    "adapter": "unsloth_local",
                    "path": str(checkpoint),
                    "load_in_4bit": False,
                    "precision": "bf16",
                    "max_seq_length": 8192,
                    "load_timeout": 600,
                    "local_files_only": True,
                },
                "execution": {"concurrency": 1, "max_attempts": 1, "timeout": 300},
                "output": str(output),
            }
        )
        (control / "config.yaml").write_text(yaml.safe_dump(config.model_dump(), sort_keys=False))
        write_json(control / "benchmark-extension-identity.json", extension)
        result = await run_evaluation(config)
        if result["status"] != "completed":
            raise ValueError(f"Checkpoint validation did not complete: {checkpoint}: {result}")
        predictions = [
            json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()
        ]
        candidate = {
            "checkpoint": str(checkpoint),
            "step": step,
            "split": "validation",
            "status": "completed",
            "tasks": {},
            "validation_run": str(output),
        }
        for task in TASKS:
            selected_samples = [s for s in samples if s.subset == task]
            ids = {s.id for s in selected_samples}
            selected_predictions = [p for p in predictions if p["id"] in ids]
            score = benchmark.evaluate(selected_samples, selected_predictions, registry, None)
            if score["partial"] or score["counts"]["strict_valid"] != len(selected_samples):
                raise ValueError(
                    "Every validation question must return a valid native decision and probabilities before selection"
                )
            identity = "".join(
                json.dumps(s.model_dump(mode="json"), sort_keys=True) for s in selected_samples
            )
            candidate["tasks"][task] = {
                "accuracy": score["metrics"]["medical_accuracy"]["value"],
                "brier": score["metrics"]["medical_brier"]["value"],
                "count": len(selected_samples),
                "data_sha256": hashlib.sha256(identity.encode()).hexdigest(),
            }
            write_json(control / f"{task}-metrics.json", score)
        candidates.append(candidate)
        write_json(control_root / "candidates.json", candidates)
    selected = select_checkpoint(candidates)
    write_json(control_root / "selection.json", selected)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(
        evaluate_checkpoints(args.training_run, args.validation, args.control_root, args.run_root)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
