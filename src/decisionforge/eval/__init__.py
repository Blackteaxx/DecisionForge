"""Independent evaluation of native JEV decision models."""

VERSION = "1"
PROTOCOL_VERSION = "typesafe.systemone.questions.2026-10-10"


def __getattr__(name):
    """Load the supported API lazily, keeping module constants free of import cycles."""
    from importlib import import_module

    exports = {
        "BaseBenchmark": ("decisionforge.benchmarks.generic.benchmark", "GenericBenchmark"),
        "GenericBenchmark": ("decisionforge.benchmarks.generic.benchmark", "GenericBenchmark"),
        "Metric": ("decisionforge.benchmarks.generic.metrics", "Metric"),
        "Registry": ("registry", "Registry"),
        "load_registry": ("registry", "load_registry"),
        "EvalConfig": ("models", "EvalConfig"),
        "load_config": ("models", "load_config"),
        "Sample": ("models", "Sample"),
        "run_evaluation": ("runner", "run_evaluation"),
    }
    if name not in exports:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = exports[name]
    value = getattr(import_module(module if module.startswith("decisionforge.") else f"{__name__}.{module}"), attribute)
    globals()[name] = value
    return value


__all__ = [
    "BaseBenchmark",
    "GenericBenchmark",
    "Metric",
    "Registry",
    "load_registry",
    "EvalConfig",
    "load_config",
    "Sample",
    "run_evaluation",
]
