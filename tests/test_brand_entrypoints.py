import subprocess
import sys

from click import unstyle


def test_decisionforge_module_entrypoint_uses_dforge_program_name():
    result = subprocess.run(
        [sys.executable, "-m", "decisionforge", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Usage: dforge " in unstyle(result.stdout)


def test_console_script_aliases_are_both_installed():
    for command in ("dforge", "decisionforge"):
        result = subprocess.run(
            [command, "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "Usage: dforge " in unstyle(result.stdout)
