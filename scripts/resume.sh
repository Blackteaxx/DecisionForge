#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
if [ "$#" -lt 1 ]; then
    printf '%s\n' 'Usage: bash scripts/resume.sh RUN_DIRECTORY [--budget-usd N]' >&2
    exit 1
fi
RUN=$1
shift
exec uv run dforge data resume --run-dir "$RUN" "$@"
