from click import unstyle
from typer.testing import CliRunner

from decisionforge.typer_cli import app

runner = CliRunner()


def test_public_command_tree_exposes_expected_groups_and_commands():
    cases = {
        ("--help",): ("data", "eval", "train", "schema", "doctor"),
        ("data", "--help"): ("inspect", "parse-pdf", "build", "resume", "validate", "export"),
        ("train", "--help"): ("run", "validate"),
        ("eval", "--help"): ("run", "rescore", "list"),
        ("schema", "--help"): ("export",),
    }
    for args, commands in cases.items():
        result = runner.invoke(app, list(args))
        assert result.exit_code == 0, result.output
        for command in commands:
            assert command in result.output


def test_every_public_subcommand_has_standalone_help():
    for args in (
        ("data", "inspect", "--help"),
        ("data", "parse-pdf", "--help"),
        ("data", "build", "--help"),
        ("data", "resume", "--help"),
        ("data", "validate", "--help"),
        ("data", "export", "--help"),
        ("train", "run", "--help"),
        ("train", "validate", "--help"),
        ("eval", "run", "--help"),
        ("eval", "rescore", "--help"),
        ("eval", "list", "--help"),
        ("schema", "export", "--help"),
        ("doctor", "--help"),
    ):
        result = runner.invoke(app, list(args))
        assert result.exit_code == 0, result.output

    evaluation_help = runner.invoke(app, ["eval", "run", "--help"])
    assert "--score-only" in unstyle(evaluation_help.output)


def test_version_is_available_without_loading_business_credentials():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == "0.1.0"


def test_build_overrides_preserve_context_and_failure_status(monkeypatch):
    from decisionforge import cli

    monkeypatch.setenv("OPENAI_MODEL", "offline-test")
    captured = []

    async def execute(config):
        captured.append(config)
        return {"status": "budget_exhausted"}

    monkeypatch.setattr(cli, "execute", execute)
    result = runner.invoke(
        app,
        [
            "data",
            "build",
            "--config",
            "configs/default.yaml",
            "--no-extract-context",
            "--generate-context",
            "--context-relevance",
            "0.7",
            "--modes",
            "choice",
            "--modes",
            "score",
        ],
    )
    assert result.exit_code == 2, result.output
    config = captured[0]
    assert config.input.extract_context is False
    assert config.context.generate_if_missing is True
    assert config.context.relevance == 0.7
    assert config.task.modes == ["choice", "score"]


def test_public_wrapper_reports_usage_errors_without_traceback(capsys):
    from decisionforge.cli import main

    assert main(["data", "inspect", "--bad-option"]) == 2
    assert "Traceback" not in capsys.readouterr().err


def test_export_rejects_missing_checkpoint_without_creating_run(tmp_path):
    directory = tmp_path / "missing"
    result = runner.invoke(app, ["data", "export", "--run-dir", str(directory)])
    assert result.exit_code != 0
    assert not directory.exists()


def test_training_and_evaluation_parameters_dispatch_without_online_work(monkeypatch, tmp_path):
    from decisionforge import typer_cli

    calls = []
    monkeypatch.setattr(
        typer_cli, "train_execute", lambda *args: calls.append(args) or {"status": "validated"}
    )
    monkeypatch.setattr(
        typer_cli, "eval_execute", lambda *args: calls.append(args) or {"status": "validated"}
    )
    cases = [
        (
            ["train", "run", "--config", "train.yaml", "--run-dir", str(tmp_path), "--resume"],
            (False, True),
        ),
        (
            ["train", "validate", "--config", "train.yaml", "--run-dir", str(tmp_path)],
            (True, False),
        ),
        (
            ["eval", "run", "--config", "eval.yaml", "--run-dir", str(tmp_path), "--validate-only"],
            (True, False, False),
        ),
        (
            ["eval", "rescore", "--config", "eval.yaml", "--run-dir", str(tmp_path)],
            (False, False, True),
        ),
    ]
    for args, flags in cases:
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
        assert calls[-1][1] == tmp_path
        assert calls[-1][2 : 2 + len(flags)] == flags
    assert runner.invoke(app, ["eval", "list"]).exit_code == 0
    assert calls[-1][0] is None


def test_pdf_parameters_and_schema_export(monkeypatch, tmp_path):
    import json

    from decisionforge.generation import pdf

    captured = []

    async def parse(config, output_root):
        captured.append((config, output_root))
        return tmp_path / "corpus.txt"

    monkeypatch.setattr(pdf, "prepare_pdf", parse)
    result = runner.invoke(
        app,
        [
            "data",
            "parse-pdf",
            "--config",
            "configs/offline.yaml",
            "--input",
            "a.pdf",
            "--input",
            "b.pdf",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured[0][0].input.type == "pdf"
    assert [__import__("pathlib").Path(path).name for path in captured[0][0].input.paths] == [
        "a.pdf",
        "b.pdf",
    ]
    assert captured[0][1] == str(tmp_path)
    destination = tmp_path / "schemas"
    result = runner.invoke(app, ["schema", "export", "--directory", str(destination)])
    assert result.exit_code == 0, result.output
    assert (
        json.loads((destination / "training_record.v3.json").read_text())["title"]
        == "TrainingExample"
    )


def test_offline_inspection_validation_and_resume_parameters(monkeypatch, tmp_path):
    import json

    from decisionforge import cli
    from decisionforge.config import Pricing, load_config

    assert (
        runner.invoke(
            app,
            [
                "data",
                "inspect",
                "--config",
                "configs/offline.yaml",
                "--input",
                "examples/standard.json",
            ],
        ).exit_code
        == 0
    )
    result = runner.invoke(
        app, ["data", "validate", "--input", "examples/training/data/typed-decisions-sample.jsonl"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["valid_records"] == 120
    config = load_config("configs/offline.yaml")
    config.llm.pricing = {config.llm.model: Pricing(input_per_million=0.1, output_per_million=0.2)}
    state = tmp_path / ".dforge"
    state.mkdir()
    (state / "config.resolved.json").write_text(config.model_dump_json())
    captured = []

    async def execute(config):
        captured.append(config)
        return {"status": "complete"}

    monkeypatch.setattr(cli, "execute", execute)
    result = runner.invoke(app, ["data", "resume", "--run-dir", str(tmp_path), "--budget-usd", "3"])
    assert result.exit_code == 0, result.output
    assert captured[0].execution.resume is True
    assert captured[0].execution.budget_limit_usd == 3
