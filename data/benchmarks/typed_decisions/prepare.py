"""Prepare this benchmark's fixed source data and canonical JSONL."""

import sys
from pathlib import Path

# Dataset scripts run from any cwd; only the repository tools need this path.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from data.benchmarks._shared.prepare import main

if __name__ == "__main__":
    raise SystemExit(main("typed_decisions", Path(__file__).resolve().parent))
