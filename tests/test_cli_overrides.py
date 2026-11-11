import json
from pathlib import Path


def test_cli_build_input_and_output_overrides(tmp_path, monkeypatch, capsys):
    from decisionforge import cli

    monkeypatch.setenv("OPENAI_MODEL", "offline-fixture")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "input.txt").write_text("English grid records")
    observed = []

    async def execute(config):
        observed.append(config)
        return {"status": "complete", "run_directory": config.output.directory + "_v_fixture"}

    monkeypatch.setattr(cli, "execute", execute)
    result = cli.main(
        [
            "data",
            "build",
            "--config",
            str(Path(__file__).resolve().parents[1] / "configs/default.yaml"),
            "--input",
            "input.txt",
            "--output-dir",
            "out",
        ]
    )
    assert result == 0
    assert observed[0].input.path == str(tmp_path / "input.txt")
    assert observed[0].input.type == "txt"
    assert observed[0].output.directory == str(tmp_path / "out")
    assert json.loads(capsys.readouterr().out)["run_directory"].endswith("out_v_fixture")
