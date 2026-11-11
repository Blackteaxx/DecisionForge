#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
uv run ruff check src tests scripts data examples/eval extensions
uv run pytest -q "$@"
