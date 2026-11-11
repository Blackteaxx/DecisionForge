"""Example benchmark using a shared accuracy metric and its own group metric."""

from decisionforge.eval import BaseBenchmark


class CustomBenchmark(BaseBenchmark):
    name = "custom_groups"
    version = "1"
    default_metrics = {"accuracy": {}, "unanimous_groups": {}}


def benchmark():
    return CustomBenchmark()
