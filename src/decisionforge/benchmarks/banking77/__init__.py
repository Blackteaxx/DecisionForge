"""banking77 benchmark contribution."""


def register(registry):
    from .benchmark import Banking77

    registry.register_benchmark(Banking77())
