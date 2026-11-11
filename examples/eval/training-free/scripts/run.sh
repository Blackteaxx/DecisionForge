#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../../.." && pwd)
config_dir="$script_dir/../configs"
python_cmd=${DFORGE_PYTHON:-python}
route=both
if (( $# > 0 )) && [[ "$1" != --* ]]; then route=$1; shift; fi
case "$route" in
  both) routes=(chat llm2jev) ;;
  chat|llm2jev) routes=("$route") ;;
  *) echo 'Usage: run.sh [both|chat|llm2jev] [--validate-only|--resume|--score-only]' >&2; exit 2 ;;
esac
validate_only=false
for arg in "$@"; do
  case "$arg" in
    --validate-only) validate_only=true ;;
    --run-dir|--run-dir=*)
      echo 'Batch evaluation cannot share one --run-dir; edit the per-benchmark YAML outputs.' >&2
      exit 2 ;;
  esac
done
cd -- "$repo_root"
for benchmark in jevbench-public typed-decisions banking77 clinc150-oos; do
  for selected in "${routes[@]}"; do
    "$python_cmd" -m decisionforge eval run --config "$config_dir/$selected/$benchmark.yaml" "$@"
  done
done
if [[ "$route" == both && "$validate_only" == false ]]; then
  "$python_cmd" "$script_dir/compare.py"
fi
