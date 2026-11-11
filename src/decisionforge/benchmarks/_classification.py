"""Shared intent-classification formulas and prediction accounting."""

from decisionforge.eval.models import HardTarget


def applicable(rows, target_type, choice_only=False):
    return [
        r
        for r in rows
        if not r["excluded"]
        and r["outcome"] is not None
        and isinstance(r["sample"].target, target_type)
        and (not choice_only or r["sample"].question.type == "choice")
    ]


def native_label(row):
    outcome = row["outcome"]
    if outcome and outcome["transport_ok"] and outcome["protocol_ok"]:
        answer = outcome.get("answer") or {}
        label = answer.get("choice")
        if label in row["sample"].question.labels:
            return label
    return None


def accounting(rows, eligible, used, failed, denominator):
    return {
        "eligible": len(eligible),
        "used": used,
        "failed": failed,
        "excluded": len(rows) - len(eligible),
        "incomplete": sum(not r["outcome"].get("terminal", True) for r in eligible),
        "unknown": sum(r["outcome"].get("outcome_unknown", False) for r in eligible),
        "denominator": denominator,
        "reason": None if used else "no applicable predictions",
    }


def label_parameters(params):
    if set(params) - {"labels"}:
        raise ValueError("macro_f1 accepts only labels")
    labels = params.get("labels")
    if labels is not None and (
        not isinstance(labels, list)
        or not labels
        or any(not isinstance(k, str) or not k for k in labels)
        or len(set(labels)) != len(labels)
    ):
        raise ValueError("macro_f1 labels must be unique nonempty strings")
    return {} if labels is None else {"labels": labels}


def macro_f1(rows, params):
    eligible = applicable(rows, HardTarget, True)
    labels = params.get("labels") or sorted(
        {
            k
            for r in rows
            if r["sample"].question.type == "choice"
            for k in r["sample"].question.labels
        }
    )
    classes = {k: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for k in labels}
    failed = 0
    for row in eligible:
        expected, predicted = row["sample"].target.value, native_label(row)
        if predicted is None:
            failed += 1
        if expected in classes:
            classes[expected]["support"] += 1
        if predicted == expected and expected in classes:
            classes[expected]["tp"] += 1
        else:
            if expected in classes:
                classes[expected]["fn"] += 1
            if predicted in classes:
                classes[predicted]["fp"] += 1
    for stats in classes.values():
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        stats.update(
            precision=tp / (tp + fp) if tp + fp else 0,
            recall=tp / (tp + fn) if tp + fn else 0,
            f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
        )
    return {
        "value": sum(s["f1"] for s in classes.values()) / len(labels)
        if eligible and labels
        else None,
        **accounting(rows, eligible, len(eligible), failed, "attempted_applicable_fixed_labels"),
        "per_class": classes,
        "labels": labels,
        "zero_division": 0,
        "prediction_rule": "native_choice",
    }


def native_accuracy(rows, params):
    eligible = applicable(rows, HardTarget, True)
    return {
        "value": sum(native_label(r) == r["sample"].target.value for r in eligible) / len(eligible)
        if eligible
        else None,
        **accounting(
            rows,
            eligible,
            len(eligible),
            sum(native_label(r) is None for r in eligible),
            "attempted_applicable",
        ),
        "prediction_rule": "native_choice",
    }
