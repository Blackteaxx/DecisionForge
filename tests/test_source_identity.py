"""Source migration must preserve inference safety and offline scoring."""

import importlib
import json

import httpx
import pytest

from decisionforge.eval import source_identity
from decisionforge.eval.compare import scoring_identity
from decisionforge.eval.runner import run_evaluation
from decisionforge.util import digest
from tests.test_eval_runner import config_at, response


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source",
    [
        "eval/adapters/jev.py",
        "training_free/chat_baseline/processing.py",
        "training_free/llm2jev/__init__.py",
        "benchmarks/generic/metrics.py",
    ],
)
async def test_real_moved_source_change_invalidates_resume(tmp_path, monkeypatch, source):
    config = config_at(tmp_path)
    await run_evaluation(config, transport=httpx.MockTransport(response))
    await run_evaluation(config, resume=True, transport=httpx.MockTransport(response))
    root = tmp_path / "source"
    # Read real implementation files into an isolated source tree; never mutate the checkout.
    for name in source_identity.source_versions():
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((source_identity.PACKAGE_ROOT / name).read_bytes())
    monkeypatch.setattr(source_identity, "PACKAGE_ROOT", root)
    before = source_identity.source_versions()
    path = root / source
    path.write_text(path.read_text() + "\n# changed implementation\n")
    assert source_identity.source_versions()[source] != before[source]
    with pytest.raises(ValueError, match="matching old checkout"):
        await run_evaluation(config, resume=True)
    # Old source identities do not restrict offline re-scoring saved records.
    manifest_path = tmp_path / "run/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["inference"]["inference_code"] = {"adapter.py": "legacy"}
    manifest["fingerprint"] = digest(manifest["inference"])
    manifest_path.write_text(json.dumps(manifest))
    result = await run_evaluation(config, score_only=True)
    assert result


def test_moved_scoring_sources_and_layout_are_compared(tmp_path, monkeypatch):
    root = tmp_path / "source"
    metric = root / "benchmarks/generic/metrics.py"
    metric.parent.mkdir(parents=True)
    metric.write_text("original")
    monkeypatch.setattr(source_identity, "PACKAGE_ROOT", root)
    before = source_identity.source_versions()
    metric.write_text("changed")
    after = source_identity.source_versions()
    report = {"metrics": {}, "scoring_code": before}
    assert scoring_identity(report) != scoring_identity({**report, "scoring_code": after})
    assert scoring_identity(report) != scoring_identity(
        {**report, "scoring_code": {"metrics.py": "old"}}
    )


@pytest.mark.parametrize(
    "legacy,canonical",
    [
        ("eval.adapter", "eval.adapters.jev"),
        ("eval.chat_adapter", "eval.adapters.chat"),
        ("eval.chat_prompt", "training_free.chat_baseline.prompt"),
        ("eval.local_adapter", "eval.adapters.local"),
        ("eval.local_worker", "eval.adapters.local_worker"),
        ("training.unsloth_backend", "training.backends.unsloth"),
    ],
)
def test_legacy_modules_share_canonical_state(legacy, canonical):
    assert importlib.import_module("decisionforge." + legacy) is importlib.import_module(
        "decisionforge." + canonical
    )


def test_qualified_scoring_identity_retains_new_helper_keys():
    report = {"metrics": {}, "scoring_code": {"benchmarks/new/helper.py": "first"}}
    changed = {**report, "scoring_code": {"benchmarks/new/helper.py": "second"}}
    assert scoring_identity(report) != scoring_identity(changed)
