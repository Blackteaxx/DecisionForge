"""Installed benchmark packages, each exposing a register(registry) hook."""

from importlib import import_module
from pkgutil import iter_modules


def register_benchmarks(registry):
    """Load child packages from this installed package, in stable name order."""
    for module in sorted(iter_modules(__path__), key=lambda module: module.name):
        if module.ispkg and not module.name.startswith("_"):
            import_module(f"{__name__}.{module.name}").register(registry)
