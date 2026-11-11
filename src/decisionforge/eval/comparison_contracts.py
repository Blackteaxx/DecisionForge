"""Ordinary consistency checks for recorded reports, independent of scoring plugins."""

from __future__ import annotations

import math

from decisionforge.eval.numbers import number


def validate_report(report, sample_count):
    counts = report.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("Missing report counts")
    fields = (
        "planned",
        "attempted",
        "completed",
        "incomplete",
        "unknown_outcomes",
        "unattempted",
        "transport_failed",
        "protocol_failed",
        "scoring_invalid",
        "scorable",
        "excluded",
        "scoring_valid",
        "strict_valid",
        "renormalized",
    )
    if any(type(counts.get(key)) is not int or counts[key] < 0 for key in fields):
        raise ValueError("Invalid report count")
    planned, attempted = counts["planned"], counts["attempted"]
    failures = sum(
        counts[key] for key in ("transport_failed", "protocol_failed", "scoring_invalid")
    )
    if (
        planned != sample_count
        or attempted + counts["unattempted"] != planned
        or counts["completed"] + counts["incomplete"] != attempted
        or counts["unknown_outcomes"] > counts["incomplete"]
        or counts["scoring_valid"] > counts["completed"]
        or counts["strict_valid"] > counts["scoring_valid"]
        or counts["renormalized"] > counts["scoring_valid"]
        or counts["scorable"] > attempted
        or counts["excluded"] > planned
        or failures + counts["scoring_valid"] > attempted
    ):
        raise ValueError("Inconsistent report count relationships")
    partial = counts["completed"] < planned
    if type(report.get("partial")) is not bool or report["partial"] != partial:
        raise ValueError("Inconsistent report partial/completion status")
    coverage = counts.get("coverage")
    if not number(coverage, 0, 1) or not math.isclose(coverage, attempted / planned, abs_tol=1e-12):
        raise ValueError("Inconsistent report coverage")
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("Missing report metrics")
    bounds = {
        name: (0, 1)
        for name in (
            "accuracy",
            "native_accuracy",
            "gold_accuracy",
            "soft_accuracy",
            "ece",
            "gold_ece",
            "soft_ece",
            "macro_f1",
            "group_agreement",
        )
    }
    bounds.update(brier=(0, 2), soft_brier=(0, 2), mae=(0, 9), soft_kl=(0, float("inf")))
    if report.get("probability_vector_policy") == "keys_and_bounds_only":
        size = report.get("max_probability_vector_size")
        if type(size) is not int or size < 0 or (sample_count and size == 0):
            raise ValueError("Invalid report probability vector size")
        bounds.update(
            brier=(0, size),
            soft_brier=(0, size),
            mae=(0, max(size * (size - 1) / 2, size - 1)),
        )
    for name, metric in metrics.items():
        if not isinstance(metric, dict):
            raise ValueError(f"Metric {name} must be an object")
        for field in ("eligible", "used", "failed"):
            if type(metric.get(field)) is not int or metric[field] < 0:
                raise ValueError(f"Metric {name} has an invalid denominator")
        if metric["used"] > metric["eligible"] or metric["failed"] > metric["eligible"]:
            raise ValueError(f"Metric {name} has inconsistent denominators")
        value = metric.get("value")
        if value is not None:
            if not number(value):
                raise ValueError(f"Metric {name} has an invalid value")
            if name in bounds:
                lo, hi = bounds[name]
                if not lo - 1e-6 <= value <= hi + 1e-6:
                    raise ValueError(f"Metric {name} value is outside its valid range")
    return partial
