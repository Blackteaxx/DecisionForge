#!/usr/bin/env bash
set -euo pipefail
if (( $# != 1 )); then
  echo 'Usage: serve.sh data/local/models/Qwen3-4B-Instruct-2507' >&2
  exit 2
fi
python_cmd=${DFORGE_PYTHON:-python}
exec "$python_cmd" -m llm2jev.server.server --backend sglang \
  --model-path "$1" --served-model-name qwen3-4b-instruct-2507 \
  --host 127.0.0.1 --port 30000 --dtype bfloat16 \
  --context-length 32768 --mem-fraction-static 0.8
