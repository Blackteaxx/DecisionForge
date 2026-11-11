"""generic benchmark contribution."""


def register(registry):
    from .benchmark import GenericBenchmark

    registry.register_benchmark(GenericBenchmark())
