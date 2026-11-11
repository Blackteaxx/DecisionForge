"""Archive a complete real three-arm ScienceQA-IMG comparison; never invent results."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import yaml

from decisionforge.eval.data import load_samples
from decisionforge.eval.metrics import argmax
from decisionforge.eval.store import EvalStore
from decisionforge.training.backends.swift_task import diagnostics
from decisionforge.util import write_json


def display(value):
    if value is None:
        return "N/A"
    return f"{value:.6g}" if isinstance(value, float) else str(value)


def collect(root, label, destination):
    run = root / "evaluation" / label / "run"
    manifest = json.loads((run / "manifest.json").read_text())
    samples = load_samples(run / "samples.jsonl", resolve_media=False)
    if len(samples) != 2017 or any(s.split != "test" for s in samples):
        raise ValueError(f"{label}: full 2017-question official image test split required")
    store = EvalStore(run, read_only=True)
    predictions = list(store.records("predictions"))
    by_id = {p["id"]: p for p in predictions}
    if set(by_id) != {s.id for s in samples} or any(
        not p.get("terminal", True) for p in predictions
    ):
        raise ValueError(f"{label}: incomplete or duplicate question coverage")
    pairs, correct, abstain, seconds, memory, records = [], 0, [], [], [], []
    for sample in samples:
        p = by_id[sample.id]
        raw = p.get("raw_response") or {}
        valid = bool(p["transport_ok"] and p["protocol_ok"])
        if valid:
            dist = p["answer"]["probabilities"]
            values = [dist[k] for k in sample.question.labels]
            if abs(sum(values) - 1) > 1e-6:
                raise ValueError(
                    "A purportedly valid prediction violates the shared probability contract"
                )
            target = [float(k == sample.target.value) for k in sample.question.labels]
            pairs.append((values, target))
            correct += argmax(dist) == sample.target.value
        detail = raw.get("swift") or raw.get("swift_chat") or {}
        if "seconds" in detail:
            seconds.append(detail["seconds"])
        if "peak_memory_bytes" in detail:
            memory.append(detail["peak_memory_bytes"])
        if label != "chat":
            for column in detail.get("columns", []):
                if column.get("abstain_probability") is not None:
                    abstain.append(column["abstain_probability"])
        else:
            abstain.append(0.0)
        records.append(
            {
                "id": sample.id,
                "reference": sample.target.model_dump(),
                "transport_ok": p["transport_ok"],
                "protocol_ok": p["protocol_ok"],
                "answer": p.get("answer"),
                "raw_response": raw,
                "error": p.get("error"),
                "attempt_latency_ms": p.get("attempt_latency_ms"),
                "model": p.get("model"),
                "model_identity_source": p.get("model_identity_source"),
                "probability_source": p.get("probability_source"),
            }
        )
    quality = diagnostics(pairs)
    quality["accuracy"] = correct / len(samples)
    quality.update(
        planned=len(samples),
        predictions=len(predictions),
        valid=len(pairs),
        failures=sum(not p["transport_ok"] for p in predictions),
        illegal_outputs=sum(p["transport_ok"] and not p["protocol_ok"] for p in predictions),
        macro_f1=None,
        macro_f1_reason="Options are question-local answer positions, not shared semantic classes",
        mean_abstain_probability=sum(abstain) / len(abstain) if abstain else None,
        abstain_denominator=len(abstain),
        inference_seconds=sum(seconds),
        inference_qps=len(seconds) / sum(seconds) if seconds and sum(seconds) else None,
        peak_memory_bytes=max(memory, default=None),
        probability_metric_denominator="valid predictions; failures count as wrong for accuracy",
        inference_qps_basis="timed native/generation responses including illegal outputs; excludes IPC/model loading",
        peak_memory_basis="torch.cuda.max_memory_allocated",
        provenance=manifest.get("local_model"),
    )
    path = destination / f"{label}.predictions.jsonl"
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in records)
    )
    quality["raw_predictions"] = {
        "file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    write_json(destination / f"{label}.summary.json", quality)
    # Eval runs persist their effective config in the immutable identity manifest.
    (destination / f"{label}.yaml").write_text(
        yaml.safe_dump(manifest["config"], sort_keys=False, allow_unicode=True)
    )
    shutil.copy2(run / "manifest.json", destination / f"{label}.manifest.json")
    return quality


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    training = args.task_root / "training/formal/run"
    manifest = json.loads((training / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("Training and export/reload acceptance must have completed")
    report = json.loads((training / "report.json").read_text())
    if report["reload"]["status"] != "passed":
        raise ValueError("Missing actual GPU export/reload validation")
    args.output.mkdir(parents=True, exist_ok=False)
    summaries = {
        name: collect(args.task_root, name, args.output)
        for name in ("original", "continued", "chat")
    }
    for source, target in (
        (training / "config.yaml", "training.yaml"),
        (training / "report.json", "training.report.json"),
        (training / "swift_command.json", "training.command.json"),
        (training / "data_report.json", "data-report.json"),
        (
            training / "calibration.uncalibrated.predictions.json",
            "calibration.before.predictions.json",
        ),
        (
            training / "calibration.calibrated.predictions.json",
            "calibration.after.predictions.json",
        ),
        (args.task_root / "resource.json", "resource.json"),
    ):
        shutil.copy2(source, args.output / target)
    metadata_dir = args.output / "artifact-metadata"
    metadata_dir.mkdir()
    for name in (
        "dforge_swift_artifact.json",
        "swift_lineage.json",
        "scoring_head.json",
        "NOTICE",
        "LICENSE",
        "QWEN_LICENSE",
    ):
        shutil.copy2(training / "model" / name, metadata_dir / name)
    (args.output / ".gitattributes").write_text(
        "# Preserve archived checksums, upstream license bytes and test-run evidence.\n"
        "* -text\n"
        "artifact-metadata/QWEN_LICENSE whitespace=cr-at-eol\n"
        "*.log whitespace=-blank-at-eol\n"
    )
    lines = [
        "# ms-swift OmniJev: ScienceQA-IMG",
        "",
        "Real single-GPU runs. All three arms cover the full 2,017-question official image test split; no option or test filtering. Calibration uses only the independently held-out training role. Failures count as wrong in accuracy; ECE/Brier/NLL use valid distributions and state their denominators.",
        "",
        "| Model | Accuracy | ECE (15 bins) | Brier | NLL | Valid / 2017 | Failures / illegal | QPS | Peak GPU GiB | Mean abstain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, s in summaries.items():
        values = [
            name,
            f"{s['accuracy']:.2%}",
            s.get("ece"),
            s.get("brier"),
            s.get("nll"),
            s["valid"],
            f"{s['failures']} / {s['illegal_outputs']}",
            s["inference_qps"],
            s["peak_memory_bytes"] / 2**30 if s["peak_memory_bytes"] else None,
            s["mean_abstain_probability"],
        ]
        lines.append("| " + " | ".join(display(v) for v in values) + " |")
    lines += [
        "",
        "Macro-F1 is not a semantic class metric here: answer positions refer to different options in each question. Raw per-question outputs, failure details, provenance and measured GPU memory/latency are preserved in the JSONL and summary files.",
        "",
        "Base: Qwen/Qwen3.5-4B @ 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a. Adapter: tinnel123/OmniJev @ 91248107b7f5230bd5715db2e62e83c9a72fcd81. ms-swift @ 680df5c670d7e8add906ac49381d603869410612. ScienceQA mirror @ f18b0a70359ebfb41f658fd564208d0355b013f4.",
        "",
        "OmniJev code/weights and Qwen base are Apache-2.0. ScienceQA data is treated under the upstream CC-BY-NC-SA-4.0 license (Pan Lu et al., NeurIPS 2022). The original OmniJev training manifest remains under audit; this report does not assert uncontaminated upstream pretraining or endorse the withdrawn historical genmcq aggregate.",
    ]
    (args.output / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
