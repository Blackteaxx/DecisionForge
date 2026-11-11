"""Require and report five real methods on every complete frozen medical benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from decisionforge.eval.data import load_samples
from decisionforge.util import now, write_json

METHODS = ("trained", "jev", "llm2jev", "anyjev", "chat")
DATASETS = {"care": 135, "synth": 869, "reddit": 1502}


def sample_identity(samples):
    value = [s.model_dump(mode="json") for s in samples]
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def validate_matrix(cells, expected):
    result = {}
    for cell in cells:
        key = (cell["method"], cell["dataset"])
        if key in result:
            raise ValueError(f"Duplicate matrix cell: {key}")
        if key[0] not in METHODS or key[1] not in DATASETS:
            raise ValueError(f"Unknown matrix cell: {key}")
        reference = expected[key[1]]
        if cell["sample_sha256"] != reference["sample_sha256"]:
            raise ValueError(f"Methods require the same frozen sample/target identity: {key}")
        counts = cell["counts"]
        if (
            cell["status"] != "completed"
            or counts["planned"] != reference["count"]
            or counts["completed"] != reference["count"]
            or counts["unattempted"]
            or counts["incomplete"]
        ):
            raise ValueError(f"All planned samples must be completed: {key}")
        result[key] = cell
    missing = {(m, d) for m in METHODS for d in DATASETS} - result.keys()
    if missing:
        raise ValueError(f"Incomplete method matrix; missing {sorted(missing)}")
    return result


def group_interval(rows, seed=3407, replicates=2000, minimum_groups=20):
    grouped = defaultdict(list)
    for group, value in rows:
        grouped[group].append(float(value))
    result = {
        "groups": len(grouped),
        "replicates": replicates,
        "seed": seed,
        "unit": "connected patient component",
        "interval": None,
        "reason": None,
    }
    if len(grouped) < minimum_groups:
        result["reason"] = (
            f"Fewer than {minimum_groups} independent patient components; pair count cannot substitute for group count"
        )
        return result
    groups = [(sum(v), len(v)) for v in grouped.values()]
    rng, bootstrap = random.Random(seed), []
    for _ in range(replicates):
        draw = [rng.choice(groups) for _ in groups]
        bootstrap.append(sum(x[0] for x in draw) / sum(x[1] for x in draw))
    bootstrap.sort()
    result["interval"] = [
        bootstrap[int(0.025 * (replicates - 1))],
        bootstrap[int(0.975 * (replicates - 1))],
    ]
    result["method"] = "95% percentile cluster bootstrap, resampling complete connected components"
    return result


def performance_view(raw):
    keys = (
        "sessions",
        "attempt_latency_ms",
        "successful_response_latency_ms",
        "sample_elapsed_ms",
        "failed_attempts",
        "timeout_attempts",
        "unknown_attempts",
        "usage",
        "reported_cost_usd",
        "missing_cost_attempts",
        "price_per_1000_decisions_usd",
        "cost_basis",
        "candidate_tasks",
    )
    return {k: raw.get(k) for k in keys}


def method_of(model):
    adapter = model["adapter"]
    if adapter == "unsloth_local":
        return "trained"
    if adapter == "anyjev_local":
        return "anyjev"
    if adapter == "chat_api":
        return "chat"
    if adapter == "jev":
        return "llm2jev" if (model.get("provenance") or {}).get("method") == "llm2jev" else "jev"
    raise ValueError("Unexpected medical comparison adapter")


def load_cell(run, dataset, expected_samples, task_root, extension):
    manifest = json.loads((run / "manifest.json").read_text())
    config = manifest["config"]
    if config["benchmark"] != "medical_decisions" or config["benchmark_version"] != "1":
        raise ValueError("Comparison requires medical_decisions v1")
    if config["data"].get("limit") is not None or config["data"].get("subsets") is not None:
        raise ValueError("Formal medical comparison cannot truncate or subset tests")
    samples = load_samples(run / "samples.jsonl")
    if sample_identity(samples) != sample_identity(expected_samples):
        raise ValueError("Saved samples do not match complete frozen official test")
    method = method_of(config["model"])
    provenance = task_root / "evaluation" / run.parent.name / "benchmark-extension-identity.json"
    if json.loads(provenance.read_text())["files"] != extension["files"]:
        raise ValueError("Methods use different benchmark extension sources")
    predictions = [
        json.loads(line) for line in (run / "predictions.jsonl").read_text().splitlines()
    ]
    if len(predictions) != len(samples) or {p["id"] for p in predictions} != {
        s.id for s in samples
    }:
        raise ValueError("Missing, duplicate or unknown terminal predictions")
    if any(not p.get("terminal", True) for p in predictions):
        raise ValueError("Formal comparison contains nonterminal outcomes")
    reports = sorted(run.glob("reports/*/metrics.json"), key=lambda p: p.stat().st_mtime)
    if not reports:
        raise ValueError("No actual evaluation report")
    path = reports[-1]
    report = json.loads(path.read_text())
    if report["partial"]:
        raise ValueError("Formal comparison requires complete reports")
    if set(report["metrics"]) != {
        "medical_accuracy",
        "medical_macro_f1",
        "medical_brier",
        "medical_ece",
    }:
        raise ValueError("Comparison requires the same full quality metric set")
    if any(metric["eligible"] != len(samples) for metric in report["metrics"].values()):
        raise ValueError("Metric eligibility differs from full planned denominator")
    by_id, confusion, errors, correctness = (
        {p["id"]: p for p in predictions},
        defaultdict(Counter),
        Counter(),
        [],
    )
    for sample in samples:
        p = by_id[sample.id]
        choice = (
            (p.get("answer") or {}).get("choice")
            if p["transport_ok"] and p["protocol_ok"]
            else None
        )
        if choice not in sample.question.labels:
            choice = "__failure__"
        confusion[str(sample.target.value)][choice] += 1
        if p.get("error"):
            errors[p["error"].get("kind", "unknown")] += 1
        correctness.append((sample.group_id, choice == sample.target.value))
    cell = {
        "method": method,
        "dataset": dataset,
        "status": "completed",
        "sample_sha256": sample_identity(samples),
        "counts": report["counts"],
        "metrics": report["metrics"],
        "confusion": dict(confusion),
        "error_kinds": dict(errors),
        "accuracy_uncertainty": group_interval(correctness),
        "performance": performance_view(report["performance"]),
        "prediction_provenance": report["prediction_provenance"],
        "artifacts": {
            "run": str(run),
            "report": str(path),
            "predictions": str(run / "predictions.jsonl"),
            "attempts": str(run / "attempts.jsonl"),
            "prediction_sha256": hashlib.sha256(
                (run / "predictions.jsonl").read_bytes()
            ).hexdigest(),
        },
    }
    return cell, {s.id: value for s, (_, value) in zip(samples, correctness, strict=True)}


def verify_selected_artifact(manifest, expected):
    """Bind scored predictions to the selected export's recorded file contents."""
    recorded = manifest.get("inference", {}).get("local_checkpoint", {}).get("checkpoint", {})
    configured_path = Path(manifest["config"]["model"]["path"]).resolve()
    expected_path = Path(expected["path"]).resolve()
    if (
        configured_path != expected_path
        or Path(recorded.get("path", "")).resolve() != expected_path
    ):
        raise ValueError("Trained evaluation did not use the selected calibrated artifact path")
    # Compare what inference recorded, rather than reading today's possibly
    # replaced files at the same path and retroactively changing provenance.
    if not expected.get("files") or recorded.get("files") != expected["files"]:
        raise ValueError(
            "Trained evaluation artifact identity differs from selected calibrated model"
        )


def build_comparison(run_roots, task_root, calibration_path, output):
    calibration = json.loads(calibration_path.read_text())
    if (
        calibration["status"] != "completed"
        or not calibration.get("reload", {}).get("new_process")
        or calibration["selection"]["split"] != "validation"
        or calibration["source_roles_used"] != ["calibration"]
        or calibration["optimizer_updates"] != 0
    ):
        raise ValueError(
            "Final model requires completed validation selection, independent calibration and new-process reload"
        )
    frozen, expected = {}, {}
    for dataset, count in DATASETS.items():
        frozen[dataset] = load_samples(task_root / f"evaluation/frozen/{dataset}.jsonl")
        if len(frozen[dataset]) != count:
            raise ValueError("Official frozen benchmark row count changed")
        expected[dataset] = {"count": count, "sample_sha256": sample_identity(frozen[dataset])}
    extension = json.loads((task_root / "benchmark_ext/IDENTITY.json").read_text())
    cells, correct = [], {}
    for root in run_roots:
        run = Path(root).resolve()
        samples = load_samples(run / "samples.jsonl")
        matches = [d for d in DATASETS if sample_identity(samples) == expected[d]["sample_sha256"]]
        if len(matches) != 1:
            raise ValueError("Run does not uniquely match a frozen complete benchmark")
        cell, values = load_cell(run, matches[0], frozen[matches[0]], task_root, extension)
        if cell["method"] == "trained":
            manifest = json.loads((run / "manifest.json").read_text())
            verify_selected_artifact(manifest, calibration["model_identity"])
        cells.append(cell)
        correct[(cell["method"], cell["dataset"])] = values
    matrix = validate_matrix(cells, expected)
    differences = []
    for dataset in DATASETS:
        for method in METHODS[1:]:
            rows = [
                (
                    s.group_id,
                    float(correct[("trained", dataset)][s.id])
                    - float(correct[(method, dataset)][s.id]),
                )
                for s in frozen[dataset]
            ]
            differences.append(
                {
                    "dataset": dataset,
                    "contrast": "trained minus " + method,
                    "accuracy_difference": sum(v for _, v in rows) / len(rows),
                    "uncertainty": group_interval(rows),
                }
            )
    result = {
        "schema_version": "medical.complete-comparison.v1",
        "status": "completed",
        "created_at": now(),
        "cells": list(matrix.values()),
        "expected": expected,
        "calibration_manifest": str(calibration_path),
        "selection": calibration["selection"],
        "benchmark_extension": extension,
        "accuracy_differences": differences,
        "conditions": [
            "Public tests used for final reporting only; model selection uses independent generated validation.",
            "Jev is an external-system comparison; local methods use the frozen Qwen3.5-9B BF16 backbone.",
            "Accuracy and macro-F1 retain every planned row; failed decisions count as errors. Probability metrics separately report valid coverage.",
            "Latency is measured at the adapter; HTTP/local software paths and binary candidate counts differ. GPU and API costs without verified prices remain unknown.",
            "Connected-patient components define resampling units. Synth/Reddit have too few components for reliable independent-patient intervals.",
        ],
        "protocol_deviations": [],
        "execution_recoveries": [],
    }
    deviation = task_root / "evaluation/llm2jev-full-001/sequence-deferral.json"
    if deviation.exists():
        result["protocol_deviations"].append(json.loads(deviation.read_text()))
        result["conditions"][0] = (
            "LLM2Jev baseline inference began before final model selection; existing outcomes were retained and later submissions deferred. No training or checkpoint-selection choices use those scores; see protocol_deviations."
        )
    recovery = task_root / "evaluation/execution-recovery-001.json"
    if recovery.exists():
        result["execution_recoveries"].append(json.loads(recovery.read_text()))
        result["conditions"].append(
            "AnyJev uses the upstream shared_prefix=false complete-prompt path for Qwen3.5 hybrid-cache compatibility. L0/content-free/full rotations remain unchanged. The original partial failed run, its quality/failures, elapsed time and unknown cost are retained separately in execution_recoveries."
        )
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "comparison.json", result)
    lines = [
        "# Medical triage: full method comparison",
        "",
        "All 15 method/dataset cells are complete. See comparison.json for identities, failures, confusion matrices, grouped intervals, latency, usage, costs and original artifacts.",
        "",
    ]
    for dataset in DATASETS:
        lines.extend(
            [
                f"## {dataset} ({DATASETS[dataset]} samples)",
                "",
                "| Method | Accuracy | Macro-F1 | Brier | ECE | Probability coverage | Transport/protocol failures |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method in METHODS:
            cell = matrix[(method, dataset)]
            m, c = cell["metrics"], cell["counts"]

            def value(key):
                v = m[key]["value"]
                return "unknown" if v is None else f"{v:.4f}"

            lines.append(
                f"| {method} | {value('medical_accuracy')} | {value('medical_macro_f1')} | {value('medical_brier')} | {value('medical_ece')} | {m['medical_brier']['used']}/{m['medical_brier']['eligible']} | {c['transport_failed']}/{c['protocol_failed']} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation and execution conditions",
            "",
            *["- " + c for c in result["conditions"]],
            "",
        ]
    )
    if result["protocol_deviations"]:
        lines.extend(
            [
                "Execution deviation: LLM2Jev baseline inference was started before checkpoint selection; retained honestly, with remaining submissions deferred. Training configuration and selection use generated validation only.",
                "",
            ]
        )
    if result["execution_recoveries"]:
        lines.extend(
            [
                "Execution recovery: the first AnyJev CARE run stopped after 21 valid outcomes and one hybrid-cache error, leaving 113 unattempted. Complete corrected runs are used in the table; the original partial quality, failures and performance remain separately available in comparison.json. No original outcomes were deleted.",
                "",
            ]
        )
    (output / "REPORT.md").write_text("\n".join(lines))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_comparison(args.runs, args.task_root, args.calibration_manifest, args.output)
    print(
        json.dumps(
            {"status": result["status"], "cells": len(result["cells"]), "output": str(args.output)}
        )
    )


if __name__ == "__main__":
    main()
