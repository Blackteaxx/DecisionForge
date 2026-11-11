"""Register historical global metric aliases once, independently of package order."""

from decisionforge.benchmarks.banking77.metrics import intent_metrics
from decisionforge.benchmarks.generic.metrics import builtin_metrics
from decisionforge.benchmarks.typed_decisions.metrics import teacher_metrics


def register_metrics(registry):
    for metric in builtin_metrics() + intent_metrics() + teacher_metrics():
        registry.register_metric(metric)
