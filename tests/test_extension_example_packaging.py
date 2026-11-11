"""Deterministic checks for the contributor layout and distributable example."""

import tomllib
from pathlib import Path

from decisionforge.eval import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_example_distribution_and_config_paths():
    example = ROOT / "extensions/example_plugin"
    project = tomllib.loads((example / "pyproject.toml").read_text())["project"]
    assert project["name"] == "dforge-eval-example"
    assert project["entry-points"] == {
        "decisionforge.eval.metrics": {"unanimous_groups": "jev_eval_example:group_metric"},
        "decisionforge.eval.benchmarks": {"custom_groups": "jev_eval_example:benchmark"},
    }
    config = load_config(example / "config.yaml")
    assert Path(config.data.path) == ROOT / "examples/eval/samples.jsonl"
    assert Path(config.output) == ROOT / "outputs/eval/custom-groups"
    assert not (ROOT / "examples/eval/plugin/pyproject.toml").exists()


def test_new_installed_benchmark_package_is_discovered_without_core_edit(tmp_path, monkeypatch):
    import decisionforge.benchmarks as benchmarks
    from decisionforge.eval import Registry

    package = tmp_path / "task3_probe"
    package.mkdir()
    (package / "__init__.py").write_text(
        "from decisionforge.eval import BaseBenchmark\n"
        "class Contribution(BaseBenchmark):\n"
        "    name = 'discovery_probe'\n"
        "def register(registry):\n"
        "    registry.register_benchmark(Contribution())\n"
    )
    monkeypatch.setattr(benchmarks, "__path__", [str(tmp_path)])
    import sys

    registry = Registry()
    try:
        benchmarks.register_benchmarks(registry)
        assert registry.benchmark("discovery_probe").name == "discovery_probe"
    finally:
        sys.modules.pop("decisionforge.benchmarks.task3_probe", None)
