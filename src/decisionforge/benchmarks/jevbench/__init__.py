"""jevbench benchmark contribution."""


def register(registry):
    from .benchmark import JevBench

    registry.register_benchmark(JevBench())
