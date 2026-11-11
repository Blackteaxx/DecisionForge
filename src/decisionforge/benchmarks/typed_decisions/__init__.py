"""typed_decisions benchmark contribution."""


def register(registry):
    from .benchmark import TypedDecisions

    registry.register_benchmark(TypedDecisions())
