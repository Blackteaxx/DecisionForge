"""JevBench latency aggregation."""


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    point = (len(values) - 1) * q
    lower = int(point)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (point - lower)


def latency_summary(values):
    values = [v for v in values if v is not None]
    return {"n": len(values), "p50_s": percentile(values, 0.5), "p95_s": percentile(values, 0.95)}
