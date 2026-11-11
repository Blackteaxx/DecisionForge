"""clinc150_oos benchmark contribution."""


def register(registry):
    from .benchmark import Clinc150OOS

    registry.register_benchmark(Clinc150OOS())
