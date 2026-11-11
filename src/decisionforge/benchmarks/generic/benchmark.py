from __future__ import annotations

from collections import Counter

from decisionforge.eval.numbers import number

from .metrics import argmax, distribution


class GenericBenchmark:
    name = "generic"
    version = "1"
    supported_types = {"choice", "noul", "score"}
    default_metrics = {
        "accuracy": {},
        "soft_accuracy": {},
        "brier": {},
        "soft_brier": {},
        "ece": {},
        "soft_ece": {},
        "mae": {},
    }

    def validate(self, registry, metrics):
        selected = metrics or self.default_metrics
        for name, params in selected.items():
            registry.metric(name).parameters(params)
        return selected

    def prepare(self, sample, outcome):
        row = {
            "sample": sample,
            "outcome": outcome,
            "probabilities": None,
            "label": None,
            "strict_valid": False,
            "scoring_valid": False,
            "renormalized": False,
            "error": None,
            "score_for_metrics": None,
            "score_valid_for_metrics": False,
            "excluded": bool(sample.metadata.get("exclude_reason")),
        }
        if outcome is None or not outcome["transport_ok"] or not outcome["protocol_ok"]:
            return row
        answer = outcome["answer"]
        if sample.question.type == "score" and number(
            answer.get("score"), 0, len(sample.question.labels) - 1
        ):
            row.update(score_for_metrics=answer["score"], score_valid_for_metrics=True)
        raw = (
            {"false": 1 - answer["noul"], "true": answer["noul"]}
            if sample.question.type == "noul"
            else answer.get("probabilities")
        )
        try:
            p = distribution(raw, sample.question.labels)
        except ValueError as exc:
            row["error"] = str(exc)
            return row
        row.update(probabilities=p, label=argmax(p), strict_valid=True, scoring_valid=True)
        return row

    def evaluate(self, samples, predictions, registry, metrics):
        selected = self.validate(registry, metrics)
        by_id = {p["id"]: p for p in predictions}
        if len(by_id) != len(predictions) or set(by_id) - {s.id for s in samples}:
            raise ValueError("Duplicate or unknown prediction id")
        rows = [self.prepare(s, by_id.get(s.id)) for s in samples]
        attempted = [r for r in rows if r["outcome"] is not None]
        counts = {
            "planned": len(samples),
            "attempted": len(attempted),
            "completed": sum(r["outcome"].get("terminal", True) for r in attempted),
            "incomplete": sum(not r["outcome"].get("terminal", True) for r in attempted),
            "unknown_outcomes": sum(r["outcome"].get("outcome_unknown", False) for r in attempted),
            "unattempted": len(samples) - len(attempted),
            "transport_failed": sum(
                not r["outcome"]["transport_ok"] and not r["outcome"].get("outcome_unknown", False)
                for r in attempted
            ),
            "protocol_failed": sum(
                r["outcome"]["transport_ok"] and not r["outcome"]["protocol_ok"] for r in attempted
            ),
            "scoring_invalid": sum(
                r["outcome"]["protocol_ok"] and not r["scoring_valid"] for r in attempted
            ),
            "scorable": sum(
                r["sample"].target is not None and not r["excluded"] for r in attempted
            ),
            "excluded": sum(r["excluded"] for r in rows),
            "coverage": len(attempted) / len(samples) if samples else None,
            "scoring_valid": sum(r["scoring_valid"] for r in rows),
            "strict_valid": sum(r["strict_valid"] for r in rows),
            "renormalized": sum(r["renormalized"] for r in rows),
        }
        return {
            "benchmark": self.name,
            "benchmark_version": self.version,
            "metrics": {n: registry.metric(n).evaluate(rows, p) for n, p in selected.items()},
            "counts": counts,
            "partial": counts["completed"] < counts["planned"],
            "decision_rule": "argmax_lexicographic_ties",
            "numeric_prediction_rule": "native_score",
            "probability_sum_tolerance": None,
            "probability_vector_policy": "keys_and_bounds_only",
            "max_probability_vector_size": max(
                (len(s.question.labels) for s in samples), default=0
            ),
            "target_kinds": dict(Counter(s.target.kind if s.target else "null" for s in samples)),
            "target_sources": dict(Counter(str(s.metadata.get("target_source")) for s in samples)),
            "diagnostics": {
                r["sample"].id: {
                    **{
                        k: r[k]
                        for k in (
                            "strict_valid",
                            "scoring_valid",
                            "score_valid_for_metrics",
                            "renormalized",
                            "error",
                        )
                    },
                    "outcome_status": "unattempted"
                    if r["outcome"] is None
                    else r["outcome"].get("status", "completed"),
                }
                for r in rows
            },
        }
