#!/usr/bin/env bash
# Read this recipe top to bottom; change model/training settings in the YAML files.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
export PYTHON_BIN=${PYTHON_BIN:-python3}
export UNSLOTH_COMPILE_DISABLE=${UNSLOTH_COMPILE_DISABLE:-1}

case "${1:-}" in
    ""|--score-only) ;;
    *) echo 'Usage: bash examples/training/unsloth/run_full.sh [--score-only]' >&2; exit 2 ;;
esac

if [[ "${1:-}" != --score-only ]]; then
    # 1. Prepare full Typed Decisions training data and the other benchmark datasets.
    bash examples/training/prepare_data.sh
    for benchmark in jevbench banking77 clinc150_oos; do
        "$PYTHON_BIN" "data/benchmarks/$benchmark/prepare.py"
    done

    # 2. Train two epochs; save the adapter, decision head, calibration and merged model.
    "$PYTHON_BIN" -m decisionforge train run --config examples/training/unsloth/qwen35-4b-full.yaml
fi

# 3. Evaluate every benchmark. Keep each CLI's exact report path for aggregation.
results=outputs/eval/qwen35-4b-full/results
mkdir -p "$results"
for config in examples/training/unsloth/eval/*.yaml; do
    benchmark=$(basename -- "$config" .yaml)
    "$PYTHON_BIN" -m decisionforge eval run --config "$config" "$@" > "$results/$benchmark.json"
done

# 4. Build one summary from the training report and the benchmark reports.
"$PYTHON_BIN" examples/training/unsloth/summarize.py \
    --training-report outputs/train/qwen35-4b-full/report.json \
    --results "$results"
