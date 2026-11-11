"""Offline comparison of immutable evaluation samples and selected reports."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from decisionforge.eval.comparison_contracts import validate_report
from decisionforge.eval.data import load_samples, parse_json
from decisionforge.eval.provenance import method_metadata
from decisionforge.util import digest


def read_json(path):
    value = parse_json(Path(path).read_text(encoding="utf-8"))
    json.dumps(value, allow_nan=False)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def load_run(path):
    path = Path(path).resolve()
    if path.name == "metrics.json":
        report_path, directory = path, path.parent.parent.parent
    elif (path / "metrics.json").is_file():
        report_path, directory = path / "metrics.json", path.parent.parent
    else:
        directory = path
        reports = [
            (read_json(p).get("created_at", ""), str(p), p)
            for p in directory.glob("reports/*/metrics.json")
        ]
        if not reports:
            raise ValueError(f"No evaluation reports: {path}")
        report_path = max(reports)[2]
    manifest = read_json(directory / "manifest.json")
    samples = load_samples(directory / "samples.jsonl")
    snapshot = digest([s.model_dump() for s in samples])
    if snapshot != manifest["sample_snapshot_hash"]:
        raise ValueError("Evaluation sample snapshot has changed")
    if digest(manifest["inference"]) != manifest["fingerprint"]:
        raise ValueError("Evaluation inference manifest fingerprint is invalid")
    if manifest["config"]["model"] != manifest["inference"]["model"]:
        raise ValueError("Configured model differs from inference identity")
    report = read_json(report_path)
    if report.get("inference_manifest") != manifest:
        raise ValueError("Report does not belong to this evaluation manifest")
    if report.get("probability_vector_policy") == "keys_and_bounds_only" and report.get(
        "max_probability_vector_size"
    ) != max((len(s.question.labels) for s in samples), default=0):
        raise ValueError("Report probability vector size differs from the sample snapshot")
    validate_report(report, len(samples))
    return directory, manifest, samples, report, report_path


def scoring_identity(report):
    keys = (
        "benchmark",
        "benchmark_version",
        "decision_rule",
        "numeric_prediction_rule",
        "probability_sum_tolerance",
        "probability_vector_policy",
        "request_protocol",
    )
    result = {key: report.get(key) for key in keys}
    result["metrics"] = {
        name: {key: metric.get(key) for key in ("version", "parameters", "direction", "unit")}
        for name, metric in report["metrics"].items()
    }
    code = report["scoring_code"]
    if any("/" in name for name in code):
        result["scoring_layout"] = "qualified.v1"
        result["scoring_code"] = dict(code)
    else:
        result["scoring_layout"] = "legacy.flat"
        result["scoring_code"] = {
            name: code.get(name) for name in (
                "benchmarks.py", "metrics.py", "numbers.py", "jevbench.py",
                "dataset_benchmarks.py", "classification_metrics.py",
            )
        }
    return result


def controlled_model(model):
    declared = model.get("provenance") or {}
    if not all(
        declared.get(key)
        for key in (
            "backbone_model",
            "backbone_revision",
            "precision",
            "serving_backend",
        )
    ):
        raise ValueError("Paired comparison needs declared frozen backbone and serving identity")
    thinking = declared.get("settings", {}).get("enable_thinking")
    if type(thinking) is not bool or declared["serving_backend"] != "sglang":
        raise ValueError("Main paired comparison needs explicit thinking settings on SGLang")
    if model.get("adapter") == "chat_api" and (
        model.get("serving", {}).get("enable_thinking") != thinking
    ):
        raise ValueError("Declared thinking differs from actual Chat configuration")
    if declared.get("method") == "llm2jev" and thinking:
        raise ValueError("Pinned LLM2Jev disables thinking")
    return {
        **{
            key: declared[key]
            for key in (
                "backbone_model",
                "backbone_revision",
                "precision",
                "serving_backend",
                "dependencies",
            )
        },
        "enable_thinking": thinking,
        "settings": declared.get("settings", {}),
        "base_url": model["base_url"],
        "model": model["model"],
    }


def run_row(directory, manifest, report, report_path):
    performance, provenance = report["performance"], report["prediction_provenance"]
    return {
        "run_directory": str(directory),
        "report": str(report_path.relative_to(directory)),
        **method_metadata(manifest["config"]["model"]),
        "configured_model": manifest["config"]["model"].get("model"),
        "actual_models": provenance["actual_models"],
        "unresolved_model_predictions": provenance["unresolved_model_predictions"],
        "probability_sources": provenance["probability_sources"],
        "partial": report["partial"],
        "counts": report["counts"],
        "valid_coverage": report["counts"]["scoring_valid"] / report["counts"]["planned"],
        "failure_count": sum(
            report["counts"][key]
            for key in (
                "transport_failed",
                "protocol_failed",
                "scoring_invalid",
            )
        ),
        "metrics": report["metrics"],
        "sample_latency_ms": performance["sample_elapsed_ms"],
        "usage": performance["usage"],
        "reported_cost_usd": performance["reported_cost_usd"],
        "missing_cost_attempts": performance["missing_cost_attempts"],
        "cost_basis": performance["cost_basis"],
        "candidate_tasks": performance.get("candidate_tasks"),
        "breakdowns": {
            key: report[key]
            for key in (
                "by_type",
                "by_tier",
                "by_family",
                "groups",
                "by_workflow",
                "by_question",
                "oos",
                "in_scope_macro_f1",
            )
            if key in report
        },
    }


def compare_runs(run_directories, paired=False):
    if len(run_directories) < 2:
        raise ValueError("Comparison requires at least two runs")
    groups = defaultdict(list)
    for path in run_directories:
        loaded = load_run(path)
        report = loaded[3]
        groups[(report["benchmark"], report["benchmark_version"])].append(loaded)
    benchmarks = []
    for (name, version), runs in groups.items():
        baseline = runs[0]
        if paired:
            if len(runs) != 2:
                raise ValueError("Each paired benchmark requires exactly two runs")
            for run in runs[1:]:
                if run[1]["sample_snapshot_hash"] != baseline[1]["sample_snapshot_hash"] or digest(
                    scoring_identity(run[3])
                ) != digest(scoring_identity(baseline[3])):
                    raise ValueError("Paired comparison data or scoring parameters differ")
                if digest(controlled_model(run[1]["config"]["model"])) != digest(
                    controlled_model(baseline[1]["config"]["model"])
                ):
                    raise ValueError("Paired comparison frozen model or serving settings differ")
        complete_pair = paired and not any(run[3]["partial"] for run in runs)
        deltas = None
        if complete_pair:
            deltas = {}
            for metric, first in baseline[3]["metrics"].items():
                last = runs[1][3]["metrics"][metric]
                deltas[metric] = (
                    last["value"] - first["value"]
                    if (first["value"] is not None and last["value"] is not None)
                    else None
                )
        benchmarks.append(
            {
                "benchmark": name,
                "benchmark_version": version,
                "paired": complete_pair,
                "comparison_basis": "declared_same_frozen_backbone_and_service"
                if paired
                else "system_comparison",
                "metric_deltas": deltas,
                "runs": [run_row(run[0], run[1], run[3], run[4]) for run in runs],
            }
        )
    return {
        "schema_version": "decisionforge.eval.comparison.v1",
        "benchmarks": benchmarks,
        "notes": [
            "Metric deltas are second minus first, per benchmark; no pooled score.",
            "Declared provenance is not independent verification of served weights.",
            "Partial reports do not receive paired deltas; unknown cost remains null.",
        ],
    }


def comparison_markdown(comparison):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ") if value is not None else "unknown"

    lines = ["# Evaluation comparison", ""]
    for group in comparison["benchmarks"]:
        lines += [
            f"## {cell(group['benchmark'])} ({cell(group['benchmark_version'])})",
            "",
            f"Paired: {group['paired']}; basis: {group['comparison_basis']}",
            "",
            "| Method | Configured / actual model | Probability source | Valid / planned | Failures | Partial | Sample p50/p95 ms | Known input / output tokens | Missing usage / cost attempts | Known USD |",
            "|---|---|---|---:|---:|---|---|---|---|---:|",
        ]
        for row in group["runs"]:
            latency = row["sample_latency_ms"]
            lines.append(
                "| "
                + " | ".join(
                    cell(value)
                    for value in (
                        row["method"],
                        f"{row['configured_model']} / {', '.join(row['actual_models']) or 'unresolved'}",
                        ", ".join(row["probability_sources"]),
                        f"{row['counts']['scoring_valid']} / {row['counts']['planned']}",
                        row["failure_count"],
                        row["partial"],
                        f"{latency['p50_ms']} / {latency['p95_ms']}",
                        f"{row['usage']['input_tokens']} / {row['usage']['output_tokens']}",
                        f"{row['usage']['missing_attempts']} / {row['missing_cost_attempts']}",
                        row["reported_cost_usd"],
                    )
                )
                + " |"
            )
        lines.append("")
        for row in group["runs"]:
            if row["declared"]:
                declared = row["declared"]
                lines.append(
                    f"Declared backbone for {cell(row['method'])}: "
                    f"{cell(declared.get('backbone_model'))} @ "
                    f"{cell(declared.get('backbone_revision'))}; "
                    f"precision: {cell(declared.get('precision'))}."
                )
        lines += ["", "| Method | Metric | Value | Used / eligible |", "|---|---|---:|---:|"]
        for row in group["runs"]:
            for name, metric in row["metrics"].items():
                lines.append(
                    f"| {cell(row['method'])} | {cell(name)} | {cell(metric['value'])} | "
                    f"{cell(metric.get('used'))} / {cell(metric.get('eligible'))} |"
                )
        lines.append("")
    lines.extend(comparison["notes"])
    return "\n".join(lines) + "\n"
