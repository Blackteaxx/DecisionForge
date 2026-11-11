"""Compatibility alias for the canonical implementation."""
import sys
from importlib import import_module

sys.modules[__name__] = import_module("decisionforge.training.backends.unsloth")
