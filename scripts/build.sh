#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
CONFIG=${1:-configs/default.yaml}
exec uv run dforge data build --config "$CONFIG"
