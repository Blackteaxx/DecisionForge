"""demo benchmark contribution."""


def register(registry):
    from .benchmark import DemoBenchmark

    registry.register_benchmark(DemoBenchmark())
