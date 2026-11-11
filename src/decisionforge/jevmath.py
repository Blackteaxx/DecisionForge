"""JEV confidence statistics shared by generation and exported-record validation."""


def choice_confidence(probabilities: list[float]) -> float:
    count = len(probabilities)
    return max(0.0, min(1.0, (max(probabilities) - 1 / count) / (1 - 1 / count)))


def score_confidence(probabilities: list[float]) -> float:
    count = len(probabilities)
    peak = probabilities.index(max(probabilities))
    spread = sum(p * abs(i - peak) for i, p in enumerate(probabilities))
    uniform_spread = sum(abs(i - (count - 1) / 2) for i in range(count)) / count
    return max(0.0, min(1.0, 1 - spread / uniform_spread))
