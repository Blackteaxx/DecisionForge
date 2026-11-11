"""Qualified source identities for inference compatibility and offline scoring."""

import hashlib
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCORING_HELPERS = {
    "media.py",
    "eval/metrics.py",
    "eval/numbers.py",
    "eval/models.py",
    "eval/data.py",
    "eval/benchmarks.py",
    "eval/jevbench.py",
    "eval/dataset_benchmarks.py",
    "eval/classification_metrics.py",
    "eval/registry.py",
    "eval/report.py",
    "eval/source_identity.py",
}


def source_versions():
    paths = set(PACKAGE_ROOT.glob("label_tokens.py"))
    paths.update(PACKAGE_ROOT.glob("media.py"))
    for directory in ("eval", "benchmarks", "training_free", "training/backends"):
        paths.update((PACKAGE_ROOT / directory).rglob("*.py"))
    return {
        p.relative_to(PACKAGE_ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(paths)
    }


def scoring_sources(code):
    return {
        name: value
        for name, value in code.items()
        if name.startswith("benchmarks/") or name in SCORING_HELPERS
    }


def inference_sources(code):
    helpers = {
        "media.py",
        "label_tokens.py",
        "eval/models.py",
        "eval/adapter.py",
        "eval/numbers.py",
        "eval/runner.py",
        "eval/store.py",
        "eval/checkpoint.py",
        "eval/local_adapter.py",
        "eval/local_worker.py",
        "eval/chat_adapter.py",
        "eval/chat_prompt.py",
        "eval/provenance.py",
        "eval/data.py",
        "eval/metrics.py",
        "eval/source_identity.py",
        "benchmarks/generic/metrics.py",
    }
    return {
        name: value
        for name, value in code.items()
        if name in helpers
        or name.startswith(("eval/adapters/", "training_free/", "training/backends/"))
    }
