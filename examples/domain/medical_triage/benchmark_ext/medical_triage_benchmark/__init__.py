"""Task-owned external benchmark for medical action and priority decisions."""

from .benchmark import MedicalDecisions
from .metrics import accuracy_metric, brier_metric, ece_metric, macro_f1_metric


def benchmark():
    return MedicalDecisions()


def register(registry):
    """Support explicit Python registration without installing the distribution."""
    for factory in (accuracy_metric, macro_f1_metric, brier_metric, ece_metric):
        registry.register_metric(factory())
    registry.register_benchmark(benchmark())
