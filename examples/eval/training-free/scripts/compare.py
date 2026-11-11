"""Compare recorded runs offline; no provider credentials or GPU dependencies."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from decisionforge.eval.compare import compare_runs, comparison_markdown  # noqa: E402
from decisionforge.util import atomic_write, write_json  # noqa: E402

OUTPUT = ROOT / "outputs/eval/training-free"
BENCHMARKS = ("jevbench-public", "typed-decisions", "banking77", "clinc150-oos")


def headline_markdown(comparison):
    names = {
        "jevbench": "JevBench public",
        "typed_decisions": "Typed Decisions test",
        "banking77": "BANKING77 test",
        "clinc150_oos": "CLINC150+OOS test",
    }
    methods = {"chat_baseline": "Chat", "llm2jev": "LLM2Jev"}

    def percent(metric):
        value = metric.get("value")
        return f"{value:.2%}" if value is not None else "—"

    lines = [
        "# Training-free evaluation results",
        "",
        "| Benchmark | Method | Valid / planned | Accuracy | Macro-F1 | Partial |",
        "|---|---|---:|---:|---:|---|",
    ]
    for group in comparison["benchmarks"]:
        benchmark = group["benchmark"]
        metric = {
            "typed_decisions": "gold_accuracy",
            "banking77": "native_accuracy",
            "clinc150_oos": "native_accuracy",
        }.get(benchmark, "accuracy")
        for row in group["runs"]:
            counts = row["counts"]
            lines.append(
                f"| {names.get(benchmark, benchmark)} | {methods.get(row['method'], row['method'])} "
                f"| {counts['scoring_valid']:,} / {counts['planned']:,} "
                f"| {percent(row['metrics'].get(metric, {}))} "
                f"| {percent(row['metrics'].get('macro_f1', {}))} | {row['partial']} |"
            )
    lines += [
        "",
        "Typed Decisions accuracy uses the source teacher modal label. Intent benchmarks "
        "use the returned native Choice; JevBench uses argmax.",
        "",
        "Model outputs retain their raw values; probability totals are not checked.",
        "",
        "See [comparison.md](comparison.md) and [comparison.json](comparison.json) for "
        "metric denominators, failures, model provenance and paired deltas.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs", nargs="*", help="Optional run/report paths; defaults to all four benchmark pairs"
    )
    parser.add_argument(
        "--paired", action="store_true", help="Enforce identical data/scoring/frozen model"
    )
    parser.add_argument("--output", default=str(OUTPUT / "comparison"), help="Output directory")
    args = parser.parse_args()
    try:
        runs = args.runs or [
            OUTPUT / f"{benchmark}-{route}"
            for benchmark in BENCHMARKS
            for route in ("chat", "llm2jev")
        ]
        result = compare_runs(runs, paired=args.paired or not args.runs)
    except ValueError as exc:
        parser.error(str(exc))
    output = Path(args.output).resolve()
    result["run_directory_base"] = "comparison_file_directory"
    for group in result["benchmarks"]:
        for row in group["runs"]:
            row["run_directory"] = Path(os.path.relpath(row["run_directory"], output)).as_posix()
    write_json(output / "comparison.json", result)
    text = comparison_markdown(result)
    atomic_write(output / "comparison.md", text)
    headline = headline_markdown(result)
    atomic_write(output / "report.md", headline)
    print(headline)


if __name__ == "__main__":
    main()
