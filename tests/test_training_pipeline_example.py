"""Check the readable shell recipe and its report-only Python helper on CPU."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples/training/unsloth"


def example_repo(tmp_path):
    repo = tmp_path / "a checkout with spaces"
    examples = repo / "examples/training/unsloth"
    examples.mkdir(parents=True)
    shutil.copy2(EXAMPLES / "run_full.sh", examples / "run_full.sh")
    shutil.copytree(EXAMPLES / "eval", examples / "eval")
    if (EXAMPLES / "summarize.py").exists():
        shutil.copy2(EXAMPLES / "summarize.py", examples / "summarize.py")
    (examples.parent / "prepare_data.sh").write_text(
        '#!/usr/bin/env bash\nset -eu\n"$PYTHON_BIN" prepare_training_data.py\n'
    )
    fake = repo / "fake_python"
    fake.write_text(
        f"#!{sys.executable}\n"
        + """import json, os, sys
from pathlib import Path
args = sys.argv[1:]
if args[:2] == ['-m', 'decisionforge']:
    args = ['dforge', *args[2:]]
with Path('calls.jsonl').open('a') as stream:
    stream.write(json.dumps(args) + '\\n')
if args[0].endswith('summarize.py'):
    os.execv(sys.executable, [sys.executable, *args])
if len(args) > 1 and args[1] == os.environ.get('FAIL_STAGE'):
    sys.exit(2)
if len(args) > 1 and args[1] == 'train':
    root = Path('outputs/train/qwen35-4b-full')
    root.mkdir(parents=True, exist_ok=True)
    (root/'report.json').write_text(json.dumps({'status':'completed','training':{'global_step':240,'training_loss':0.8}}))
if len(args) > 1 and args[1] == 'eval':
    name = Path(args[args.index('--config')+1]).stem
    root = Path('outputs/eval/qwen35-4b-full')/name
    report = root/'reports/current'
    report.mkdir(parents=True, exist_ok=True)
    data = {'counts':{'planned':3,'completed':3,'strict_valid':3},'metrics':{'accuracy':{'value':0.75,'used':3,'eligible':3}}}
    (report/'metrics.json').write_text(json.dumps(data))
    (report/'summary.md').write_text('report')
    print(json.dumps({'status':'completed','run_directory':str(root.resolve()),'report':'reports/current'}))
"""
    )
    fake.chmod(0o755)
    return repo, fake


def launch(repo, fake, *args, **extra_env):
    return subprocess.run(
        ["bash", str(repo / "examples/training/unsloth/run_full.sh"), *args],
        cwd=repo.parent,
        env={**os.environ, "PYTHON_BIN": str(fake), **extra_env},
        capture_output=True,
        text=True,
    )


@pytest.mark.skipif(os.name != "posix", reason="Recipe fixture uses a POSIX executable shebang")
def test_readable_recipe_runs_prepare_train_eval_then_report(tmp_path):
    repo, fake = example_repo(tmp_path)
    result = launch(repo, fake)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in (repo / "calls.jsonl").read_text().splitlines()]
    assert calls[0] == ["prepare_training_data.py"]
    assert sum(len(c) > 1 and c[1] == "train" for c in calls) == 1
    assert sum(len(c) > 1 and c[1] == "eval" for c in calls) == 6
    assert calls[-1][0] == "examples/training/unsloth/summarize.py"
    report = repo / "outputs/eval/qwen35-4b-full/report.md"
    assert "0.750000" in report.read_text()
    assert str(repo) not in report.read_text()


@pytest.mark.skipif(os.name != "posix", reason="Recipe fixture uses a POSIX executable shebang")
def test_score_only_skips_preparation_and_training(tmp_path):
    repo, fake = example_repo(tmp_path)
    training = repo / "outputs/train/qwen35-4b-full"
    training.mkdir(parents=True)
    (training / "report.json").write_text('{"status":"completed","training":{"global_step":240}}')
    result = launch(repo, fake, "--score-only")
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in (repo / "calls.jsonl").read_text().splitlines()]
    evaluations = [c for c in calls if len(c) > 1 and c[1] == "eval"]
    assert len(evaluations) == 6 and all("--score-only" in c for c in evaluations)
    assert not any(
        c[0] == "prepare_training_data.py" or (len(c) > 1 and c[1] == "train") for c in calls
    )


@pytest.mark.skipif(os.name != "posix", reason="Recipe fixture uses a POSIX executable shebang")
def test_training_failure_stops_before_evaluation(tmp_path):
    repo, fake = example_repo(tmp_path)
    result = launch(repo, fake, FAIL_STAGE="train")
    assert result.returncode == 2
    calls = [json.loads(line) for line in (repo / "calls.jsonl").read_text().splitlines()]
    assert not any(len(c) > 1 and c[1] == "eval" for c in calls)
    assert not (repo / "outputs/eval/qwen35-4b-full/report.md").exists()


def test_report_helper_keeps_failures_and_uses_cli_report_path(tmp_path):
    path = EXAMPLES / "summarize.py"
    assert path.is_file(), "The small report-only helper is missing"
    spec = importlib.util.spec_from_file_location("training_example_summary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "results"
    cli = root / "results"
    cli.mkdir(parents=True)
    current = root / "typed_decisions/reports/current"
    current.mkdir(parents=True)
    metrics = {
        "counts": {"planned": 3, "completed": 3, "strict_valid": 2},
        "metrics": {"accuracy": {"value": 0.75, "used": 2, "eligible": 3}},
    }
    (current / "metrics.json").write_text(json.dumps(metrics))
    stale = root / "typed_decisions/reports/stale"
    stale.mkdir()
    (stale / "metrics.json").write_text("{}")
    os.utime(stale / "metrics.json", (2_000_000_000, 2_000_000_000))
    (cli / "typed_decisions.json").write_text(
        json.dumps({"run_directory": str(current.parents[1]), "report": "reports/current"})
    )
    training = tmp_path / "training.json"
    training.write_text('{"status":"completed","training":{"global_step":240}}')
    assert module.main(["--results", str(cli), "--training-report", str(training)]) == 2
    report = (root / "report.md").read_text()
    assert "0.750000" in report and "2 / 3" in report
    assert str(tmp_path) not in report


def test_published_training_configs_and_reports_have_no_machine_paths():
    import yaml

    for config in EXAMPLES.glob("qwen35-*.yaml"):
        model = yaml.safe_load(config.read_text())["model"]["name"]
        assert not Path(model).is_absolute()
    for path in (EXAMPLES.parent / "reports").iterdir():
        if path.is_file():
            text = path.read_text()
            assert not any(prefix in text for prefix in ("/mnt/", "/root/", "/home/"))
