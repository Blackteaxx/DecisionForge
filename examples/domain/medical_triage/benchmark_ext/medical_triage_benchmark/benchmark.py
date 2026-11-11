"""Retain native Choice independently of strict probability-vector validity."""

from __future__ import annotations

from decisionforge.benchmarks.generic.metrics import argmax, distribution
from decisionforge.eval import BaseBenchmark
from decisionforge.eval.models import HardTarget


class MedicalDecisions(BaseBenchmark):
    name, version = "medical_decisions", "1"
    supported_types = {"choice"}
    default_metrics = {
        "medical_accuracy": {},
        "medical_macro_f1": {},
        "medical_brier": {},
        "medical_ece": {"bins": 10},
    }

    def prepare(self, sample, outcome):
        if sample.question.type != "choice" or not isinstance(sample.target, HardTarget):
            raise ValueError("Medical decisions require Choice and an official hard target")
        if sample.metadata.get("exclude_reason"):
            raise ValueError("Medical benchmark exclusions cannot reduce the planned denominator")
        row = super().prepare(sample, outcome)
        row.update(
            label=None,
            probabilities=None,
            decision_valid=False,
            probability_valid=False,
            strict_valid=False,
            scoring_valid=False,
            argmax_disagreement=False,
        )
        if outcome is None or not outcome["transport_ok"] or not outcome["protocol_ok"]:
            return row
        answer = outcome.get("answer") or {}
        choice = answer.get("choice")
        if choice in sample.question.labels:
            row.update(label=choice, decision_valid=True, scoring_valid=True)
        try:
            probabilities = distribution(
                answer.get("probabilities"), sample.question.labels, tolerance=1e-6
            )
        except ValueError as exc:
            row["error"] = str(exc)
            return row
        row.update(
            probabilities=probabilities,
            probability_valid=True,
            strict_valid=row["decision_valid"],
            argmax_disagreement=row["decision_valid"] and choice != argmax(probabilities),
        )
        return row

    def evaluate(self, samples, predictions, registry, metrics):
        result = super().evaluate(samples, predictions, registry, metrics)
        by_id = {p["id"]: p for p in predictions}
        rows = [self.prepare(s, by_id.get(s.id)) for s in samples]
        result.update(
            decision_rule="native_choice",
            probability_sum_tolerance=1e-6,
            probability_vector_policy="exact_labels_bounds_sum_no_renormalization",
        )
        result["counts"].update(
            probability_valid=sum(r["probability_valid"] for r in rows),
            choice_argmax_disagreements=sum(r["argmax_disagreement"] for r in rows),
        )
        return result
