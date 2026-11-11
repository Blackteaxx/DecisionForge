#!/usr/bin/env bash
# Explicitly prepare full Typed Decisions; training never downloads datasets implicitly.
set -euo pipefail
TRAINING_REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
source "$TRAINING_REPO_ROOT/scripts/common.sh"

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    printf '%s\n' 'Usage: bash examples/training/prepare_data.sh [--convert-only] [--train-limit N] [--test-limit N]' \
        'Default: fetch pinned train/test and convert all 1,600 cases to data/local/typed-decisions-training.jsonl.' \
        '--convert-only: use already prepared raw Parquet files without downloading.'
    exit 0
fi
TRAINING_FETCH_ARGS=()
if [[ "${1:-}" == --convert-only ]]; then
    TRAINING_FETCH_ARGS=(--convert-only)
    shift
fi
"$PYTHON_BIN" "$TRAINING_REPO_ROOT/data/benchmarks/typed_decisions/prepare.py" --split train "${TRAINING_FETCH_ARGS[@]}"
"$PYTHON_BIN" "$TRAINING_REPO_ROOT/data/benchmarks/typed_decisions/prepare.py" --split test "${TRAINING_FETCH_ARGS[@]}"
exec "$PYTHON_BIN" "$TRAINING_REPO_ROOT/examples/training/prepare_typed_decisions.py" \
    --source "$TRAINING_REPO_ROOT/data/benchmarks/typed_decisions" \
    --output "$TRAINING_REPO_ROOT/data/local/typed-decisions-training.jsonl" "$@"
