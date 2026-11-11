"""Teacher-distribution metrics for Typed Decisions."""

import math

from decisionforge.benchmarks._classification import accounting, applicable
from decisionforge.benchmarks.generic.metrics import Metric, argmax, ece_parameters
from decisionforge.eval.models import SoftTarget
from decisionforge.eval.numbers import number


def kl_parameters(params):
    epsilon = params.get("epsilon", 1e-12)
    if set(params) - {"epsilon"} or not number(epsilon, 1e-15, 0.01):
        raise ValueError("KL epsilon must be a finite number in 1e-15..0.01")
    return {"epsilon": epsilon}


def teacher_metric(name, rows, params):
    eligible = applicable(rows, SoftTarget)
    usable = [r for r in eligible if r["scoring_valid"]]
    values, pairs = [], []
    for row in usable:
        q, p = row["sample"].target.probabilities, row["probabilities"]
        expected = row["sample"].metadata.get("gold_label", argmax(q))
        if name == "soft_kl":
            smoothed = {k: max(v, params["epsilon"]) for k, v in p.items()}
            total = sum(smoothed.values())
            values.append(sum(v * math.log(v / (smoothed[k] / total)) for k, v in q.items() if v))
        else:
            correct = row["label"] == expected
            values.append(float(correct))
            pairs.append((p[row["label"]], float(correct)))
    if name == "gold_accuracy":
        value = sum(values) / len(eligible) if eligible else None
        used = len(eligible)
    elif name == "gold_ece":
        bins = [[0, 0.0, 0.0] for _ in range(params["bins"])]
        for probability, correct in pairs:
            b = bins[min(int(probability * len(bins)), len(bins) - 1)]
            b[0] += 1
            b[1] += probability
            b[2] += correct
        value = (
            sum(abs(prob - correct) for n, prob, correct in bins) / len(pairs) if pairs else None
        )
        used = len(pairs)
    else:
        value, used = (sum(values) / len(values) if values else None), len(values)
    return {
        "value": value,
        **accounting(
            rows,
            eligible,
            used,
            len(eligible) - len(usable),
            "attempted_applicable" if name == "gold_accuracy" else "usable_applicable",
        ),
        "reference_rule": "source_teacher_modal_label" if name != "soft_kl" else "soft_reference",
    }


def teacher_metrics():
    result = []
    for name in ("gold_accuracy", "gold_ece", "soft_kl"):
        result.append(
            Metric(
                name,
                "1",
                "higher" if name == "gold_accuracy" else "lower",
                "nats" if name == "soft_kl" else "ratio",
                lambda rows, params, name=name: teacher_metric(name, rows, params),
                kl_parameters
                if name == "soft_kl"
                else ece_parameters
                if name == "gold_ece"
                else None,
            )
        )
    return result
