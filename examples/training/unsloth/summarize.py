"""Summarize existing reports without importing training or inference dependencies."""

import argparse
import json
import os
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument(
        "--results", type=Path, required=True, help="Directory of eval CLI JSON results"
    )
    args = parser.parse_args(argv)
    output = args.results.parent
    training = json.loads(args.training_report.read_text())
    summary = {
        "training": {"status": training["status"], **training.get("training", {})},
        "benchmarks": {},
    }
    lines = [
        "# Training and benchmark results",
        "",
        f"Training: {training['status']}; steps: {training.get('training', {}).get('global_step')}; "
        f"loss: {training.get('training', {}).get('training_loss')}.",
        "",
        "Typed Decisions uses one question per request and measures teacher agreement. "
        "Demo/generic are protocol examples, separate from the four public datasets.",
        "",
        "| Benchmark | Completed / planned | Strict valid / planned |",
        "| --- | ---: | ---: | ---: |",
    ]
    valid = training["status"] == "completed"
    for file in sorted(args.results.glob("*.json")):
        result = json.loads(file.read_text())
        report_dir = Path(result["run_directory"]) / result["report"]
        report = json.loads((report_dir / "metrics.json").read_text())
        name, counts = file.stem, report["counts"]
        relative = os.path.relpath(report_dir, output)
        summary["benchmarks"][name] = {
            "counts": counts,
            "metrics": report["metrics"],
            "report": relative,
        }
        valid &= counts["planned"] == counts["completed"] == counts["strict_valid"]
        lines.append(
            f"| {name} | {counts['completed']} / {counts['planned']} | {counts['strict_valid']} / {counts['planned']} |"
        )
    for name, report in summary["benchmarks"].items():
        lines += [
            "",
            f"## {name}",
            "",
            "| Metric | Value | Used / eligible |",
            "| --- | ---: | ---: |",
        ]
        for name_metric, metric in report["metrics"].items():
            value = "n/a" if metric["value"] is None else f"{metric['value']:.6f}"
            lines.append(f"| {name_metric} | {value} | {metric['used']} / {metric['eligible']} |")
        lines += [
            "",
            f"[Full metrics]({report['report']}/metrics.json) · [Report]({report['report']}/summary.md)",
        ]
    valid &= bool(summary["benchmarks"])
    summary["all_predictions_valid"] = bool(valid)
    lines += [
        "",
        f"All predictions strictly valid: {bool(valid)}.",
        "GPU cost is not estimated.",
        "",
    ]
    (output / "results.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (output / "report.md").write_text("\n".join(lines))
    print(output / "report.md")
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
