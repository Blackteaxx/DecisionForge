"""Prepare independent validation questions and select an actually evaluated checkpoint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from decisionforge.eval.models import Sample
from examples.domain.medical_triage.scripts.prepare_benchmarks import (
    PAIR_CRITERIA,
    TRIAGE_CRITERIA,
)

TASKS = ("triage_action", "message_priority")


def select_checkpoint(candidates: list[dict]) -> dict:
    if not candidates:
        raise ValueError("No completed validation checkpoint candidates")
    ranked, reference = [], None
    for candidate in candidates:
        if candidate.get("split") != "validation" or candidate.get("status") != "completed":
            raise ValueError("Selection requires completed independent validation results only")
        if type(candidate.get("step")) is not int or candidate["step"] < 0:
            raise ValueError("Checkpoint step must be a nonnegative integer")
        tasks = candidate.get("tasks", {})
        if set(tasks) != set(TASKS):
            raise ValueError("Checkpoint needs both validation tasks")
        identity = []
        for task in TASKS:
            metric = tasks[task]
            if type(metric.get("count")) is not int or metric["count"] < 1:
                raise ValueError("Each validation task must have a positive fixed sample count")
            for name, limit in [("accuracy", 1), ("brier", 2)]:
                value = metric.get(name)
                if (
                    type(value) not in {int, float}
                    or not math.isfinite(value)
                    or not 0 <= value <= limit
                ):
                    raise ValueError(f"Invalid validation {name}")
            if not metric.get("data_sha256"):
                raise ValueError("Validation needs a frozen data identity")
            identity.append((task, metric["count"], metric["data_sha256"]))
        if reference is None:
            reference = identity
        elif reference != identity:
            raise ValueError("Checkpoint comparison requires the same frozen validation inputs")
        accuracy = sum(tasks[t]["accuracy"] for t in TASKS) / 2
        brier = sum(tasks[t]["brier"] for t in TASKS) / 2
        ranked.append(((-accuracy, brier, candidate["step"]), candidate))
    key, selected = min(ranked, key=lambda entry: entry[0])
    return {
        **selected,
        "selection_score": -key[0],
        "tie_break_brier": key[1],
        "selection_rule": "equal_task_accuracy_then_brier_then_earliest_step",
    }


def evaluation_samples(records: list[dict], tasks: dict[int, str], role: str) -> list[dict]:
    if role not in {"validation", "calibration"}:
        raise ValueError("Only independent validation/calibration roles are prepared here")
    samples = []
    for record in records:
        if record["split"] != role:
            continue
        identity = record["record_id"]
        task = tasks[identity]
        if task not in TASKS or len(record["request"]["questions"]) != 1:
            raise ValueError("Joint medical examples must have one known decision task")
        qid, question = next(iter(record["request"]["questions"].items()))
        criteria = TRIAGE_CRITERIA if task == "triage_action" else PAIR_CRITERIA
        if question["type"] != "choice" or set(question["criteria"].values()) != set(
            criteria.values()
        ):
            raise ValueError("Medical action catalog changed in accepted training data")
        choice = record["supervision"]["answers"][qid]["choice"]
        content = question["criteria"][choice]
        gold = next(label for label, value in criteria.items() if value == content)
        sample = {
            "schema_version": "decisionforge.eval.v1",
            "id": f"{role}:{identity}",
            "group_id": str(record["group_id"]),
            "split": role,
            "subset": task,
            "request": {
                "state": record["request"]["state"],
                "questions": {"decision": {**question, "criteria": criteria}},
            },
            "target": {"kind": "hard", "value": gold},
            "metadata": {
                "task": task,
                "target_source": "synthetic_teacher_hard_reference",
                "source_record_id": identity,
            },
        }
        Sample.model_validate(sample)
        samples.append(sample)
    if not samples:
        raise ValueError("No records in the requested independent role")
    return samples


def write_eval_configs(task_root, store, calibrated_model, backbone):
    """Write the frozen 5x3 comparison, preserving any already-created configuration."""
    import yaml

    from decisionforge.eval import EvalConfig, load_config

    task_root, store = Path(task_root).resolve(), Path(store).resolve()
    revision = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    common = {
        "backbone_model": "Qwen/Qwen3.5-9B",
        "backbone_revision": revision,
        "serving_backend": "transformers",
        "precision": "bf16",
        "dependencies": {
            "torch": "2.11.0+cu130",
            "transformers": "5.17.0",
            "fastapi": "0.153.0",
            "uvicorn": "0.54.0",
        },
    }
    models = {
        "trained": {
            "adapter": "unsloth_local",
            "path": str(Path(calibrated_model).resolve()),
            "load_in_4bit": False,
            "precision": "bf16",
            "max_seq_length": 8192,
            "load_timeout": 600,
            "local_files_only": True,
        },
        "jev": {
            "adapter": "jev",
            "base_url": "https://openrouter.ai",
            "endpoint_path": "/api/alpha/decisions",
            "model": "typesafe/jev-1.14",
            "api_key_env": "OPENROUTER_API_KEY",
        },
        "llm2jev": {
            "adapter": "jev",
            "base_url": "http://127.0.0.1:30000",
            "model": "qwen35-9b",
            "api_key_env": None,
            "provenance": {
                **common,
                "method": "llm2jev",
                "upstream_repo": "https://github.com/Yinsongxu/LLM2Jev",
                "upstream_revision": "f2bdad648e7a34bfce64401fb0f7c44bc13209ae",
                "settings": {"enable_thinking": False, "batch_size": 4},
            },
        },
        "anyjev": {
            "adapter": "anyjev_local",
            "model": str(backbone),
            "revision": revision,
            "precision": "bfloat16",
            "device": "cuda:0",
            "max_seq_length": 8192,
            "batch_size": 4,
            "load_timeout": 600,
            "shared_prefix": False,
        },
        "chat": {
            "adapter": "chat_api",
            "base_url": "http://127.0.0.1:30001",
            "model": "qwen35-9b",
            "api_key_env": None,
            "temperature": 0,
            "max_tokens": 2048,
            "response_format": "text",
            "serving": {"enable_thinking": False},
            "provenance": {
                **common,
                "method": "chat_baseline",
                "settings": {
                    "enable_thinking": False,
                    "temperature": 0,
                    "max_tokens": 2048,
                    "response_format": "text",
                },
            },
        },
    }
    paths = []
    for method, model in models.items():
        for dataset in ("care", "synth", "reddit"):
            name = f"{method}-{dataset}-full-001"
            folder = task_root / "evaluation" / name
            path = folder / "config.yaml"
            execution = {
                "concurrency": 4 if method == "jev" else 1,
                "timeout": 90 if method == "jev" else 300,
                "max_attempts": 3 if method == "jev" else 1,
            }
            if method == "llm2jev":
                execution["backoff"] = 1
            config = EvalConfig.model_validate(
                {
                    "benchmark": "medical_decisions",
                    "benchmark_version": "1",
                    "data": {
                        "path": str(task_root / f"evaluation/frozen/{dataset}.jsonl"),
                        "splits": ["test"],
                    },
                    "model": model,
                    "execution": execution,
                    "output": str(store / f"evaluation/{name}/run"),
                }
            )
            if path.exists():
                if load_config(path).model_dump() != config.model_dump():
                    raise ValueError(
                        f"Existing evaluation configuration differs; preserve and inspect: {path}"
                    )
            else:
                folder.mkdir(parents=True, exist_ok=True)
                path.write_text(yaml.safe_dump(config.model_dump(), sort_keys=False))
            identity = task_root / "benchmark_ext/IDENTITY.json"
            if identity.exists():
                destination = folder / "benchmark-extension-identity.json"
                if destination.exists() and json.loads(destination.read_text()) != json.loads(
                    identity.read_text()
                ):
                    raise ValueError("Existing run has a different benchmark extension identity")
                if not destination.exists():
                    destination.write_bytes(identity.read_bytes())
            paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    select = sub.add_parser("select")
    select.add_argument("--candidates", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    configs = sub.add_parser("configs")
    configs.add_argument("--task-root", type=Path, required=True)
    configs.add_argument("--store", type=Path, required=True)
    configs.add_argument("--calibrated-model", type=Path, required=True)
    configs.add_argument("--backbone", required=True)
    args = parser.parse_args()
    if args.command == "configs":
        paths = write_eval_configs(args.task_root, args.store, args.calibrated_model, args.backbone)
        print(json.dumps([str(p) for p in paths], indent=2))
        return
    result = select_checkpoint(json.loads(args.candidates.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
