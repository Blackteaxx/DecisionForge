"""Stable exports used by the distribution entry points."""

from .benchmark import benchmark
from .metrics import group_metric

__all__ = ["benchmark", "group_metric"]
