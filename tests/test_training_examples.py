"""Exercise the public example entrypoint without training dependencies or model downloads."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples/training"


def launch(*args, cwd):
    bash = shutil.which("bash")
    if not bash and Path("C:/Program Files/Git/bin/bash.exe").is_file():
        bash = "C:/Program Files/Git/bin/bash.exe"
    if not bash:
        pytest.skip("Bash is required for the shell entrypoint")
    return subprocess.run(
        [bash, str(EXAMPLES / "unsloth/train.sh"), *args],
        cwd=cwd,
        env={**os.environ, "PYTHON_BIN": sys.executable},
        capture_output=True,
        text=True,
    )


def test_bundled_example_validates_from_another_directory(tmp_path):
    result = launch("--validate-only", cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "validated"
    assert report["total"]["records"] == 120
    assert report["total"]["questions"] == 600
    assert {role: stats["records"] for role, stats in report["roles"].items()} == {
        "train": 80,
        "validation": 10,
        "calibration": 10,
        "test": 20,
    }
    assert set(report["total"]["types"]) == {"choice", "noul", "score"}
    assert not list(tmp_path.iterdir())


def test_example_accepts_caller_relative_config_and_run_dir(tmp_path):
    config = tmp_path / "my config.yaml"
    config.write_text(
        f"data: {{path: '{EXAMPLES / 'data/typed-decisions-sample.jsonl'}'}}\n"
        "model: {name: absent-model}\noutput: {directory: no-run}\n"
    )
    result = launch(
        "--config", config.name, "--run-dir", "relative run", "--validate-only", cwd=tmp_path
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "validated"
    assert not (tmp_path / "relative run").exists()
    assert not (tmp_path / "no-run").exists()


def test_example_delegates_mutually_exclusive_mode_validation(tmp_path):
    result = launch("--validate-only", "--resume", cwd=tmp_path)
    assert result.returncode == 2
    assert "mutually exclusive" in result.stderr
    assert not list(tmp_path.iterdir())


def test_example_provenance_and_public_prediction_request():
    from decisionforge.exporter import read_training_records
    from decisionforge.models import JEVInput

    records = list(read_training_records(EXAMPLES / "data/typed-decisions-sample.jsonl"))
    source = json.loads((EXAMPLES / "data/typed-decisions-sample.source.json").read_text())
    assert source["revision"] == "e039ebffcc280174dd354227424fb2b249f191de"
    assert source["license"] == "Apache-2.0"
    assert {r.record_id for r in records} == {r["record_id"] for r in source["records"]}
    assert [(r.record_id, r.split) for r in records] == [
        (r["record_id"], r["source_split"]) for r in source["records"]
    ]
    request = json.loads((EXAMPLES / "data/public-request.json").read_text())
    assert set(request) == {"state", "questions"}
    JEVInput.model_validate(request)
    assert request == records[0].request.model_dump(exclude_none=True)
