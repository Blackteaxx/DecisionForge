#!/usr/bin/env bash
# Activate your Python environment first, or set PYTHON_BIN.
set -euo pipefail
TRAINING_REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}

# A later --config supplied by the caller overrides this default; preserve their cwd.
subcommand=run
args=()
validate_only=false
resume=false
for arg in "$@"; do
    if [[ "$arg" == --validate-only ]]; then
        subcommand=validate
        validate_only=true
    else
        args+=("$arg")
    fi
    if [[ "$arg" == --resume ]]; then resume=true; fi
done
if [[ "$validate_only" == true && "$resume" == true ]]; then
    echo "--validate-only and --resume are mutually exclusive" >&2
    exit 2
fi
command=("$PYTHON_BIN" -m decisionforge train "$subcommand" --config "$TRAINING_REPO_ROOT/examples/training/unsloth/qwen35-4b-smoke.yaml")
if (( ${#args[@]} )); then command+=("${args[@]}"); fi
exec "${command[@]}"
