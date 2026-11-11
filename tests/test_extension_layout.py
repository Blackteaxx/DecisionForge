"""Behavioral contracts for installed benchmark contributions and legacy imports."""

import importlib
from types import SimpleNamespace

import pytest

from decisionforge.eval.registry import load_registry


def test_added_package_is_discovered_in_sorted_order(monkeypatch):
    import decisionforge.benchmarks as packages

    calls = []
    modules = {
        f"{packages.__name__}.{name}": SimpleNamespace(register=lambda r, n=name: calls.append(n))
        for name in ("zeta", "alpha")
    }
    monkeypatch.setattr(
        packages,
        "iter_modules",
        lambda path: [
            SimpleNamespace(name="zeta", ispkg=True),
            SimpleNamespace(name="ignored", ispkg=False),
            SimpleNamespace(name="alpha", ispkg=True),
        ],
    )
    monkeypatch.setattr(packages, "import_module", modules.__getitem__)
    packages.register_benchmarks(object())
    assert calls == ["alpha", "zeta"]


def test_independent_registry_loads():
    first, second = load_registry(), load_registry()
    assert first is not second
    assert first.metrics.keys() == second.metrics.keys()
    assert set(first.benchmarks) == {
        "generic",
        "demo",
        "jevbench",
        "banking77",
        "clinc150_oos",
        "typed_decisions",
    }
    assert all(first.benchmarks[n] is not second.benchmarks[n] for n in first.benchmarks)


@pytest.mark.parametrize("kind,name", [("metrics", "accuracy"), ("benchmarks", "generic")])
def test_plugin_builtin_collision_is_rejected(monkeypatch, kind, name):
    import decisionforge.eval.registry as module

    builtin = getattr(load_registry(), kind)[name]
    entry = SimpleNamespace(load=lambda: lambda: builtin)
    monkeypatch.setattr(
        module, "entry_points", lambda *, group: [entry] if group.endswith(kind) else []
    )
    with pytest.raises(ValueError, match=f"duplicate {kind[:-1]} name: {name}"):
        load_registry()


@pytest.mark.parametrize(
    "canonical,legacy,names",
    [
        ("generic.benchmark", "benchmarks", ["GenericBenchmark"]),
        ("demo.benchmark", "benchmarks", ["DemoBenchmark"]),
        ("generic.metrics", "metrics", ["Metric", "calculate", "builtin_metrics"]),
        ("jevbench.benchmark", "jevbench", ["JevBench"]),
        ("banking77.benchmark", "dataset_benchmarks", ["Banking77"]),
        ("clinc150_oos.benchmark", "dataset_benchmarks", ["Clinc150OOS"]),
        ("typed_decisions.benchmark", "dataset_benchmarks", ["TypedDecisions"]),
        ("typed_decisions.metrics", "classification_metrics", ["teacher_metric"]),
    ],
)
def test_legacy_exports_are_canonical_objects(canonical, legacy, names):
    new = importlib.import_module(f"decisionforge.benchmarks.{canonical}")
    old = importlib.import_module(f"decisionforge.eval.{legacy}")
    for name in names:
        assert getattr(new, name) is getattr(old, name)


def test_package_import_failures_surface(monkeypatch):
    import decisionforge.benchmarks as packages

    monkeypatch.setattr(
        packages,
        "iter_modules",
        lambda path: [
            SimpleNamespace(name="broken", ispkg=True),
        ],
    )

    def broken(name):
        raise ModuleNotFoundError("missing benchmark dependency")

    monkeypatch.setattr(packages, "import_module", broken)
    with pytest.raises(ModuleNotFoundError, match="missing benchmark dependency"):
        packages.register_benchmarks(object())


def test_real_installed_child_package_hook(tmp_path, monkeypatch):
    import sys

    import decisionforge.benchmarks as packages
    from decisionforge.eval.runner import preflight
    from tests.test_eval_runner import config_at

    package_root = tmp_path / "installed"
    child = package_root / "contributed"
    child.mkdir(parents=True)
    (child / "__init__.py").write_text(
        "def register(registry):\n"
        "    from .benchmark import Contributed\n"
        "    from .metrics import custom_metric\n"
        "    registry.register_metric(custom_metric())\n"
        "    registry.register_benchmark(Contributed())\n"
    )
    (child / "benchmark.py").write_text(
        "from decisionforge.eval import BaseBenchmark\n"
        "class Contributed(BaseBenchmark):\n"
        '    name = "contributed"\n'
        '    default_metrics = {"contributed_metric": {}}\n'
    )
    (child / "metrics.py").write_text(
        "from decisionforge.eval import Metric\n"
        "def custom_metric():\n"
        '    return Metric("contributed_metric", "1", "higher", "ratio", '
        'lambda rows, params: {"value": 1})\n'
    )
    # Model another installed package location; discovery never searches the CWD.
    monkeypatch.setattr(packages, "__path__", [*packages.__path__, str(package_root)])
    importlib.invalidate_caches()
    try:
        registry = load_registry()
        config = config_at(tmp_path)
        config.benchmark = "contributed"
        _, samples, selected, benchmark = preflight(config, registry)
        assert selected is registry
        assert benchmark is registry.benchmark("contributed")
        assert (
            benchmark.evaluate(samples, [], registry, {})["metrics"]["contributed_metric"]["value"]
            == 1
        )
    finally:
        for name in tuple(sys.modules):
            if name.startswith("decisionforge.benchmarks.contributed"):
                sys.modules.pop(name)


def test_public_base_benchmark_and_metric_identity():
    from decisionforge import eval as api
    from decisionforge.benchmarks.generic.benchmark import GenericBenchmark
    from decisionforge.benchmarks.generic.metrics import Metric

    assert api.BaseBenchmark is api.GenericBenchmark is GenericBenchmark
    assert api.Metric is Metric
