"""Medical quality metrics: native decisions and explicit probability coverage."""

from __future__ import annotations

from decisionforge.benchmarks.generic.metrics import ece_parameters
from decisionforge.eval import Metric
from decisionforge.eval.models import HardTarget


def eligible(rows):
    return [r for r in rows if isinstance(r["sample"].target, HardTarget)]


def accounting(rows, used, denominator):
    return {
        "eligible": len(rows),
        "used": used,
        "failed": len(rows) - used,
        "coverage": used / len(rows) if rows else None,
        "denominator": denominator,
        "reason": None if used else "no usable predictions",
    }


def accuracy(rows, params):
    rows = eligible(rows)
    valid = [r for r in rows if r["decision_valid"]]
    correct = sum(r["label"] == r["sample"].target.value for r in valid)
    return {
        "value": correct / len(rows) if rows else None,
        "correct": correct,
        **accounting(rows, len(valid), "all_planned_labeled_samples"),
        "prediction_rule": "native_choice",
    }


def macro_f1(rows, params):
    rows = eligible(rows)
    labels = sorted({k for r in rows for k in r["sample"].question.labels})
    classes = {k: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for k in labels}
    for row in rows:
        truth = row["sample"].target.value
        predicted = row["label"] if row["decision_valid"] else None
        classes[truth]["support"] += 1
        if predicted == truth:
            classes[truth]["tp"] += 1
        else:
            classes[truth]["fn"] += 1
            if predicted in classes:
                classes[predicted]["fp"] += 1
    for c in classes.values():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        c.update(
            precision=tp / (tp + fp) if tp + fp else 0.0,
            recall=tp / (tp + fn) if tp + fn else 0.0,
            f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        )
    return {
        "value": sum(c["f1"] for c in classes.values()) / len(classes)
        if rows and classes
        else None,
        "per_class": classes,
        "labels": labels,
        **accounting(
            rows,
            sum(r["decision_valid"] for r in rows),
            "all_planned_labeled_samples_fixed_classes",
        ),
        "prediction_rule": "native_choice",
        "zero_division": 0,
    }


def brier(rows, params):
    rows = eligible(rows)
    usable = [r for r in rows if r["probability_valid"]]
    values = [
        sum((p - float(k == r["sample"].target.value)) ** 2 for k, p in r["probabilities"].items())
        for r in usable
    ]
    return {
        "value": sum(values) / len(values) if values else None,
        **accounting(rows, len(values), "valid_probability_predictions"),
    }


def ece(rows, params):
    rows = eligible(rows)
    usable = [r for r in rows if r["probability_valid"] and r["decision_valid"]]
    bins = [{"n": 0, "probability_sum": 0.0, "correctness_sum": 0.0} for _ in range(params["bins"])]
    for r in usable:
        confidence = r["probabilities"][r["label"]]
        bucket = bins[min(int(confidence * len(bins)), len(bins) - 1)]
        bucket["n"] += 1
        bucket["probability_sum"] += confidence
        bucket["correctness_sum"] += float(r["label"] == r["sample"].target.value)
    total = 0.0
    for i, bucket in enumerate(bins):
        n = bucket["n"]
        probability, correctness = bucket.pop("probability_sum"), bucket.pop("correctness_sum")
        bucket.update(
            lo=i / len(bins),
            hi=(i + 1) / len(bins),
            mean_probability=probability / n if n else None,
            mean_correctness=correctness / n if n else None,
        )
        if n:
            total += abs(probability - correctness)
    return {
        "value": total / len(usable) if usable else None,
        "bins": bins,
        **accounting(rows, len(usable), "valid_native_choice_and_probability"),
        "confidence_rule": "probability_of_native_choice",
    }


def accuracy_metric():
    return Metric("medical_accuracy", "1", "higher", "ratio", accuracy)


def macro_f1_metric():
    return Metric("medical_macro_f1", "1", "higher", "ratio", macro_f1)


def brier_metric():
    return Metric("medical_brier", "1", "lower", "ratio", brier)


def ece_metric():
    return Metric("medical_ece", "1", "lower", "ratio", ece, ece_parameters)
