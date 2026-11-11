import math


def number(value, low=-math.inf, high=math.inf):
    """Validate finite JSON numbers, including integers too large for float conversion."""
    try:
        return type(value) in (int, float) and low <= value <= high and math.isfinite(value)
    except OverflowError:
        return False
