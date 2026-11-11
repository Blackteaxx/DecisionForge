#!/usr/bin/env bash
set -euo pipefail
EVAL_REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -n "${VIRTUAL_ENV:-}${CONDA_PREFIX:-}" ]] && command -v python >/dev/null 2>&1; then
        PYTHON_BIN=python
    elif [[ -x "$EVAL_REPO_ROOT/.venv/bin/python" ]]; then
        PYTHON_BIN="$EVAL_REPO_ROOT/.venv/bin/python"
    else
        PYTHON_BIN=python3
    fi
fi
EVAL_ARGS=("$@")
EVAL_HAS_CONFIG=false
for arg in "$@"; do
    if [[ "$arg" == --config || "$arg" == --config=* ]]; then EVAL_HAS_CONFIG=true; fi
done
if [[ "$EVAL_HAS_CONFIG" == false ]]; then
    EVAL_ARGS=(--config "$EVAL_REPO_ROOT/examples/eval/unsloth/qwen35-4b.yaml" "${EVAL_ARGS[@]}")
fi
exec "$PYTHON_BIN" -m decisionforge eval run "${EVAL_ARGS[@]}"
