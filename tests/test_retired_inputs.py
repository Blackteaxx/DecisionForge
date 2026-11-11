"""Retired input modes fail before any network or generation work."""

import pytest
from typer.testing import CliRunner

from decisionforge.config import Config, load_config
from decisionforge.typer_cli import app


def test_retired_search_input_explains_migration(tmp_path):
    path = tmp_path / "legacy.yaml"
    path.write_text("input:\n  type: deep_search\ndeep_search:\n  prompt: research\n")
    with pytest.raises(ValueError, match="Deep Search has been removed"):
        load_config(path)
    with pytest.raises(ValueError, match="Deep Search has been removed"):
        Config(input={"type": "deep_search"})


@pytest.mark.parametrize("command", [("ui",), ("deep-search",), ("build", "--config", "x")])
def test_retired_command_surface_is_unavailable(command):
    result = CliRunner().invoke(app, list(command))
    assert result.exit_code != 0
