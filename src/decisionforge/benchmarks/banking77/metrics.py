"""Shared fixed-catalog intent metrics."""

from decisionforge.benchmarks._classification import (
    label_parameters,
    macro_f1,
    native_accuracy,
)
from decisionforge.benchmarks.generic.metrics import Metric


def intent_metrics():
    return [
        Metric("macro_f1", "1", "higher", "ratio", macro_f1, label_parameters),
        Metric("native_accuracy", "1", "higher", "ratio", native_accuracy),
    ]
