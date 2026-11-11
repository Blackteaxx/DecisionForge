from importlib.metadata import entry_points

from decisionforge.benchmarks import register_benchmarks
from decisionforge.benchmarks._registration import register_metrics
from decisionforge.eval.metrics import Metric


class Registry:
    def __init__(self):
        self.metrics = {}
        self.benchmarks = {}

    def register_metric(self, metric):
        if not isinstance(metric, Metric) or not metric.name or not callable(metric.compute):
            raise ValueError("Invalid metric descriptor")
        if metric.name in self.metrics:
            raise ValueError(f"duplicate metric name: {metric.name}")
        self.metrics[metric.name] = metric

    def register_benchmark(self, benchmark):
        if (
            not getattr(benchmark, "name", None)
            or not getattr(benchmark, "version", None)
            or not callable(getattr(benchmark, "evaluate", None))
            or not callable(getattr(benchmark, "validate", None))
            or not callable(getattr(benchmark, "prepare", None))
            or not getattr(benchmark, "supported_types", None)
        ):
            raise ValueError("Invalid benchmark descriptor")
        if benchmark.name in self.benchmarks:
            raise ValueError(f"duplicate benchmark name: {benchmark.name}")
        self.benchmarks[benchmark.name] = benchmark

    def metric(self, name):
        if name not in self.metrics:
            raise ValueError(f"Unknown metric: {name}")
        return self.metrics[name]

    def benchmark(self, name):
        if name not in self.benchmarks:
            raise ValueError(f"Unknown benchmark: {name}")
        return self.benchmarks[name]


def load_registry():
    registry = Registry()
    register_metrics(registry)
    register_benchmarks(registry)
    for group, register in (
        ("decisionforge.eval.metrics", registry.register_metric),
        ("decisionforge.eval.benchmarks", registry.register_benchmark),
    ):
        for entry in entry_points(group=group):
            plugin = entry.load()
            register(plugin() if callable(plugin) else plugin)
    return registry
