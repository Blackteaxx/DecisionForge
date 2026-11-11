#!/usr/bin/env bash
# Resolve every helper through the project's locked uv environment.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
command -v uv >/dev/null 2>&1 || { printf '%s\n' 'Install uv before running this script.' >&2; exit 1; }
