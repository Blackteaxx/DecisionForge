from collections import defaultdict

from decisionforge.benchmarks.generic.benchmark import GenericBenchmark
from decisionforge.eval.models import SoftTarget


class TypedDecisions(GenericBenchmark):
    name, version = "typed_decisions", "1-single-question"
    default_metrics = {
        "gold_accuracy": {},
        "soft_accuracy": {},
        "soft_brier": {},
        "soft_kl": {},
        "gold_ece": {},
        "soft_ece": {},
        "mae": {},
    }

    def validate_samples(self, samples):
        if any(
            not isinstance(s.target, SoftTarget)
            or not s.metadata.get("question_name")
            or s.metadata.get("gold_label") not in s.question.labels
            for s in samples
        ):
            raise ValueError(
                "Typed Decisions requires teacher soft targets and original question metadata"
            )

    def evaluate(self, samples, predictions, registry, metrics):
        result = super().evaluate(samples, predictions, registry, metrics)
        result["request_protocol"] = "single_question"
        result["official_request_protocol"] = "five_questions_per_case"
        result["reference_basis"] = "teacher_mean_of_three_temperature_0.7"
        by_id = {p["id"]: p for p in predictions}
        for field, key in (
            ("by_workflow", lambda s: s.subset),
            ("by_question", lambda s: f"{s.subset}/{s.metadata['question_name']}"),
            ("by_type", lambda s: s.question.type),
        ):
            groups = defaultdict(list)
            for s in samples:
                groups[key(s)].append(s)
            result[field] = {}
            for name, group in groups.items():
                scored = GenericBenchmark.evaluate(
                    self, group, [by_id[s.id] for s in group if s.id in by_id], registry, metrics
                )
                result[field][name] = {k: scored[k] for k in ("counts", "metrics", "partial")}
        return result
