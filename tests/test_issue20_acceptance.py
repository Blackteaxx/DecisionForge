"""Portable acceptance stays on public CLIs and proves base-only contracts."""

import json
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "tests/acceptance/method_acceptance.py"


def test_shared_fixtures_preserve_public_requests_and_full_soft_targets():
    from decisionforge.exporter import read_training_records

    source = {r.record_id: r for r in read_training_records(ROOT / "examples/training/data/typed-decisions-sample.jsonl")}
    cns = list(read_training_records(ROOT / "examples/training/data/acceptance/common-cns.v3.jsonl"))
    cn = list(read_training_records(ROOT / "examples/training/data/acceptance/common-cn.v3.jsonl"))
    for record, subset in zip(cns, cn, strict=True):
        assert record.request == source[record.record_id].request
        assert record.supervision == source[record.record_id].supervision
        assert subset.record_id == record.record_id and subset.group_id == record.group_id
        assert subset.split == record.split and subset.request.state == record.request.state
        supported = {k for k, q in record.request.questions.items() if q.type != "score"}
        assert set(subset.request.questions) == supported
        for key in supported:
            assert subset.request.questions[key] == record.request.questions[key]
            assert subset.supervision.answers[key] == record.supervision.answers[key]


def test_acceptance_templates_use_supported_typed_configs():
    from decisionforge.eval.models import load_config as load_eval
    from decisionforge.training.config import load_config as load_train

    for path in (
        ROOT / "examples/training/unsloth/eval-full.yaml",
        ROOT / "examples/training/laya/eval-acceptance.yaml",
        ROOT / "examples/training/openjev/eval-acceptance.yaml",
        ROOT / "examples/eval/training-free/anyjev/acceptance.yaml",
    ):
        load_eval(path)
    for path in (
        ROOT / "examples/training/unsloth/full-acceptance.yaml",
        ROOT / "examples/training/laya/acceptance.yaml",
        ROOT / "examples/training/openjev/acceptance.yaml",
    ):
        config = load_train(path)
        assert Path(config.data.path).is_file()


def test_static_matrix_accepts_full_catalogs_and_rejects_unsupported_capabilities(tmp_path):
    result = subprocess.run(
        [sys.executable, str(DRIVER), "static", "--work-dir", str(tmp_path / "acceptance")],
        cwd=ROOT, capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "acceptance/static.json").read_text())
    assert report["gpu_acceptance"] is False
    assert report["optional_import_guard"] is True
    shared = report["shared_cn"]
    assert shared["schema"] == "decisionforge.training.v3"
    consumers = report["cn_consumers"]
    assert set(consumers) == {"full", "laya", "openjev", "canonical_eval"}
    assert all(c["data_sha256"] == shared["sha256"] for c in consumers.values())
    assert all(c["data_path"] == shared["path"] for c in consumers.values())
    assert report["group_roles_disjoint"] is True
    assert set(report["rejections"]) == {
        "openjev_score", "laya_resume", "openjev_resume"
    }
    assert all(value["returncode"] != 0 for value in report["rejections"].values())
    assert set(report["full_catalogs"]) == {"anyjev_27", "anyjev_77", "anyjev_151"}
    assert all(value["returncode"] == 0 and value["guarded"] for value in report["full_catalogs"].values())
    assert not list((tmp_path / "acceptance/runs").glob("*"))


@pytest.mark.asyncio
async def test_recorder_score_only_reads_returned_versioned_report(tmp_path):
    import yaml
    from test_eval_runner import config_at, response

    from decisionforge.eval.runner import run_evaluation

    config = config_at(tmp_path)
    result = await run_evaluation(config, transport=httpx.MockTransport(response))
    first_report = result["report"]
    journal = (Path(config.output) / ".eval/events.jsonl").read_bytes()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config.model_dump()))
    completed = subprocess.run(
        [sys.executable, str(DRIVER), "record", "--work-dir", str(tmp_path / "record"),
         "eval", "--config", str(config_path), "--run-dir", config.output, "--score-only"],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    recorded = json.loads((tmp_path / "record/cli.command.json").read_text())
    assert recorded["guarded"] and recorded["result"]["report"] != first_report
    metrics = json.loads((tmp_path / "record/metrics.json").read_text())
    assert metrics["performance_source"] == "existing_attempts"
    assert metrics["counts"]["strict_valid"] == 1
    assert (Path(config.output) / ".eval/events.jsonl").read_bytes() == journal
