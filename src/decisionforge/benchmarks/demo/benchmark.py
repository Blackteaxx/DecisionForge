from decisionforge.benchmarks.generic.benchmark import GenericBenchmark


class DemoBenchmark(GenericBenchmark):
    name = "demo"
    default_metrics = {**GenericBenchmark.default_metrics, "group_agreement": {}}
