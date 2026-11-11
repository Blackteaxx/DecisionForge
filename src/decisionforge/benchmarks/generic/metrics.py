from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

from decisionforge.eval.models import HardTarget, SoftTarget
from decisionforge.eval.numbers import number


def distribution(values, labels, tolerance=None):
    """Accept raw predictions; explicit tolerance is for imported reference data."""
    if not isinstance(values, dict) or set(values) != set(labels):
        raise ValueError("Probability keys must cover exactly the requested labels")
    if any(not number(v, 0, 1) for v in values.values()):
        raise ValueError("Invalid probability value")
    if tolerance is not None and abs(sum(values.values()) - 1) > tolerance:
        raise ValueError("Reference probability sum outside tolerance")
    return {k: float(values[k]) for k in labels}


def argmax(values):
    return max(sorted(values), key=values.get)


def reference(sample):
    target = sample.target
    if isinstance(target, SoftTarget):
        return target.probabilities
    if isinstance(target, HardTarget):
        label = str(target.value).lower() if sample.question.type == "noul" else str(target.value)
        return {k: float(k == label) for k in sample.question.labels}
    return None


@dataclass(frozen=True)
class Metric:
    name: str
    version: str
    direction: str
    unit: str
    compute: Callable
    validate_parameters: Callable | None = None

    def parameters(self, params):
        if self.validate_parameters:
            return self.validate_parameters(params)
        if params:
            raise ValueError(f"Metric {self.name} accepts no parameters")
        return {}

    def evaluate(self, rows, params):
        params = self.parameters(params)
        return {
            "name": self.name,
            "version": self.version,
            "direction": self.direction,
            "unit": self.unit,
            "parameters": params,
            **self.compute(rows, params),
        }


def ece_parameters(params):
    bins = params.get("bins", 10)
    if set(params) - {"bins"} or type(bins) is not int or not 1 <= bins <= 10000:
        raise ValueError("ECE bins must be an integer in 1..10000")
    return {"bins": bins}


def calculate(name, rows, params):
    eligible, values, pairs = [], [], []
    for row in rows:
        sample, target = row["sample"], row["sample"].target
        if row["excluded"] or row["outcome"] is None or target is None:
            continue
        wanted = (
            sample.question.type == "score"
            if name == "mae"
            else isinstance(target, SoftTarget)
            if name.startswith("soft_")
            else isinstance(target, HardTarget)
        )
        if not wanted:
            continue
        eligible.append(row)
        valid = (
            row.get("score_valid_for_metrics", row["scoring_valid"])
            if name == "mae"
            else row["scoring_valid"]
        )
        p, q = row["probabilities"], reference(sample)
        if name in {"accuracy", "soft_accuracy"}:
            values.append(q[row["label"]] if valid else 0.0)
        elif name == "mae":
            predicted = row.get(
                "score_for_metrics", (row["outcome"].get("answer") or {}).get("score")
            )
            if valid and number(predicted):
                expected = (
                    sum(int(k) * v for k, v in q.items())
                    if isinstance(target, SoftTarget)
                    else target.value
                )
                values.append(abs(predicted - expected))
        elif valid and p is not None and q is not None:
            if name in {"brier", "soft_brier"}:
                values.append(sum((p[k] - q[k]) ** 2 for k in p))
            else:
                pairs.append((p[row["label"]], q[row["label"]]))
    used = len(pairs) if name in {"ece", "soft_ece"} else len(values)
    result = {
        "value": sum(values) / used if used else None,
        "eligible": len(eligible),
        "used": used,
        "failed": sum(
            not r.get("score_valid_for_metrics", r["scoring_valid"])
            if name == "mae"
            else not r["scoring_valid"]
            for r in eligible
        ),
        "incomplete": sum(not r["outcome"].get("terminal", True) for r in eligible),
        "unknown": sum(r["outcome"].get("outcome_unknown", False) for r in eligible),
        "excluded": len(rows) - len(eligible),
        "reason": None if used else "no applicable predictions",
        "denominator": "attempted_applicable" if name.endswith("accuracy") else "usable_applicable",
    }
    if name in {"ece", "soft_ece"}:
        bins = [
            {
                "lo": i / params["bins"],
                "hi": (i + 1) / params["bins"],
                "n": 0,
                "probability_sum": 0.0,
                "correctness_sum": 0.0,
            }
            for i in range(params["bins"])
        ]
        for confidence, correct in pairs:
            b = bins[min(int(confidence * len(bins)), len(bins) - 1)]
            b["n"] += 1
            b["probability_sum"] += confidence
            b["correctness_sum"] += correct
        total = 0.0
        for b in bins:
            n = b["n"]
            probability, correctness = b.pop("probability_sum"), b.pop("correctness_sum")
            b["mean_probability"] = probability / n if n else None
            b["mean_correctness"] = correctness / n if n else None
            if n:
                total += n * abs(b["mean_probability"] - b["mean_correctness"])
        result.update(value=total / used if used else None, bins=bins)
    return result


def group_agreement(rows, params):
    groups = defaultdict(list)
    for row in rows:
        if row["sample"].group_id and not row["excluded"]:
            groups[row["sample"].group_id].append(row)
    eligible = [g for g in groups.values() if len(g) >= 2]
    usable = [g for g in eligible if all(r["scoring_valid"] for r in g)]
    return {
        "value": sum(len({r["label"] for r in g}) == 1 for g in usable) / len(usable)
        if usable
        else None,
        "eligible": len(eligible),
        "used": len(usable),
        "failed": len(eligible) - len(usable),
        "unit_of_aggregation": "group",
    }


def builtin_metrics():
    names = ["accuracy", "soft_accuracy", "brier", "soft_brier", "ece", "soft_ece", "mae"]
    result = [
        Metric(
            name,
            "1",
            "higher" if name.endswith("accuracy") else "lower",
            "levels" if name == "mae" else "ratio",
            lambda rows, params, name=name: calculate(name, rows, params),
            ece_parameters if name in {"ece", "soft_ece"} else None,
        )
        for name in names
    ]
    result.append(Metric("group_agreement", "1", "higher", "ratio", group_agreement))
    return result
