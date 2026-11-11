#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../../.." && pwd)
python_cmd=${DFORGE_PYTHON:-python}
cd -- "$repo_root"
for dataset in jevbench typed_decisions banking77 clinc150_oos; do
  split=test
  if [[ "$dataset" == jevbench ]]; then split=public; fi
  "$python_cmd" "data/benchmarks/$dataset/prepare.py" --split "$split" "$@"
done
