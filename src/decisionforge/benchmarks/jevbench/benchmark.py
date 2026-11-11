"""JevBench policy pinned to b6b8fff7; upstream is a test oracle, not a dependency."""

from collections import defaultdict

from decisionforge.benchmarks.generic.benchmark import GenericBenchmark
from decisionforge.benchmarks.generic.metrics import argmax, calculate, distribution
from decisionforge.eval.models import HardTarget

from .metrics import latency_summary

JEVBENCH_PIN = "b6b8fff7e345b98c060ad26c13308860ddc67004"


class JevBench(GenericBenchmark):
    name = "jevbench"
    version = JEVBENCH_PIN
    default_metrics = {"accuracy": {}, "brier": {}, "ece": {}, "mae": {}, "group_agreement": {}}

    def prepare(self, sample, outcome):
        row = super().prepare(sample, None)
        row["outcome"] = outcome
        if outcome is None or not outcome["transport_ok"] or not outcome["protocol_ok"]:
            return row
        answer = outcome["answer"]
        raw = (
            {"false": 1 - answer["noul"], "true": answer["noul"]}
            if sample.question.type == "noul"
            else answer.get("probabilities")
        )
        try:
            p = distribution(raw, sample.question.labels)
            row["strict_valid"] = True
        except ValueError as exc:
            row["error"] = str(exc)
            return row
        row.update(probabilities=p, label=argmax(p), scoring_valid=True)
        if sample.question.type == "score":
            row["score_for_metrics"] = sum(int(k) * v for k, v in p.items())
            row["score_valid_for_metrics"] = True
            if sample.target is None:
                row["label"] = None
        return row

    def evaluate(self, samples, predictions, registry, metrics):
        if any(s.target is not None and not isinstance(s.target, HardTarget) for s in samples):
            raise ValueError(
                "Pinned JevBench evaluates hard references; use another benchmark for soft"
            )
        result = super().evaluate(samples, predictions, registry, metrics)
        by_id = {p["id"]: p for p in predictions}
        rows = [
            self.prepare(s, by_id.get(s.id) if by_id.get(s.id, {}).get("terminal", True) else None)
            for s in samples
        ]
        result["upstream_summary"] = self.summary(rows)
        result["upstream_summary_record_basis"] = "terminal_predictions"
        result["numeric_prediction_rule"] = "raw_weighted_grade_sum"
        result["by_tier"] = {
            tier: self.summary(
                [r for r in rows if r["sample"].metadata.get("tier", "unspecified") == tier]
            )
            for tier in sorted({s.metadata.get("tier", "unspecified") for s in samples})
        }
        return result

    def summary(self, rows):
        def aggregate(group):
            attempted = [r for r in group if r["outcome"] is not None]
            valid = [r for r in attempted if r["scoring_valid"]]
            scorable = [
                r for r in attempted if r["sample"].target is not None and not r["excluded"]
            ]
            probs = [r for r in scorable if r["probabilities"]]
            correct = sum(bool(self.correct(r)) for r in scorable)
            brier = calculate("brier", group, {})
            ece = calculate("ece", group, {"bins": 10})
            mae = calculate("mae", group, {})
            costs = [r["outcome"].get("cost_usd") for r in attempted]
            times = [
                r["outcome"]["attempt_latency_ms"] / 1000
                if r["outcome"].get("attempt_latency_ms") is not None
                else None
                for r in attempted
            ]
            return {
                "n_planned": len(group),
                "n_attempted": len(attempted),
                "n_scorable": len(scorable),
                "n_valid": len(valid),
                "n_correct": correct,
                "accuracy": correct / len(scorable) if scorable else None,
                "coverage": len(attempted) / len(group) if group else None,
                "schema_validity": len(valid) / len(attempted) if attempted else None,
                "schema_validity_strict": sum(r["strict_valid"] for r in attempted) / len(attempted)
                if attempted
                else None,
                "n_renormalized": sum(r["renormalized"] for r in attempted),
                "operational_success": sum(
                    r["outcome"]["transport_ok"] and r["outcome"]["protocol_ok"] for r in attempted
                )
                / len(attempted)
                if attempted
                else None,
                "calibration_n": len(probs),
                "brier_mean": brier["value"],
                "ece": {
                    "ece": ece["value"],
                    "n": ece["used"],
                    "bins": [
                        {
                            "lo": b["lo"],
                            "hi": b["hi"],
                            "n": b["n"],
                            "mean_confidence": b["mean_probability"],
                            "accuracy": b["mean_correctness"],
                        }
                        for b in ece["bins"]
                    ],
                }
                if probs
                else None,
                "ordinal_mae": mae["value"],
                "latency": latency_summary(times),
                "latency_failures": latency_summary(
                    [
                        t
                        for t, r in zip(times, attempted, strict=True)
                        if not (r["outcome"]["transport_ok"] and r["outcome"]["protocol_ok"])
                    ]
                ),
                "price_per_1000_decisions_usd": sum(costs) * 1000 / len(attempted)
                if attempted and all(c is not None for c in costs)
                else None,
                "cost_basis": sorted(
                    {r["outcome"].get("cost_basis", "unknown") for r in attempted}
                ),
            }

        families = {
            s: aggregate(
                [r for r in rows if r["sample"].metadata.get("family", r["sample"].subset) == s]
            )
            for s in sorted({r["sample"].metadata.get("family", r["sample"].subset) for r in rows})
        }
        accuracies = [f["accuracy"] for f in families.values() if f["accuracy"] is not None]
        attempted = [r for r in rows if r["outcome"] is not None]
        result = aggregate(rows)
        groups = defaultdict(list)
        for row in rows:
            if row["sample"].group_id:
                groups[row["sample"].group_id].append(row)
        pairs = [g for g in groups.values() if len(g) == 2]
        valid_pairs = [g for g in pairs if all(r["scoring_valid"] for r in g)]
        result.update(
            version="jevbench-v1",
            macro_accuracy=sum(accuracies) / len(accuracies) if accuracies else None,
            per_family=families,
            complete=len(attempted) == len(rows),
            paraphrase_consistency={
                "pairs": len(pairs),
                "both_valid": len(valid_pairs),
                "agree": sum(g[0]["label"] == g[1]["label"] for g in valid_pairs),
                "agreement": sum(g[0]["label"] == g[1]["label"] for g in valid_pairs)
                / len(valid_pairs)
                if valid_pairs
                else None,
                "both_correct_rate_all_pairs": sum(
                    all(self.correct(r) for r in g) for g in valid_pairs
                )
                / len(pairs)
                if pairs
                else None,
            },
            model_identities=sorted({r["outcome"].get("model", "unknown") for r in attempted}),
            probability_sources=sorted(
                {r["outcome"].get("probability_source", "native") for r in attempted}
            ),
            splits={
                s: aggregate([r for r in rows if r["sample"].split == s])
                for s in sorted({r["sample"].split for r in rows})
            },
            ledger_charged_usd=None,
        )
        return result

    @staticmethod
    def correct(row):
        if not row["scoring_valid"]:
            return False
        target = row["sample"].target
        if target is None:
            return None
        expected = (
            str(target.value).lower()
            if row["sample"].question.type == "noul"
            else str(target.value)
        )
        return row["label"] == expected
