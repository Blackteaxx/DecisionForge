"""Shared fixed-catalog intent benchmark policy."""

from decisionforge.benchmarks.generic.benchmark import GenericBenchmark
from decisionforge.eval.models import HardTarget


class IntentBenchmark(GenericBenchmark):
    supported_types = {"choice"}
    default_metrics = {"macro_f1": {}, "native_accuracy": {}}

    def validate(self, registry, metrics):
        selected = super().validate(registry, metrics)
        if "labels" in selected.get("macro_f1", {}):
            raise ValueError(
                "Intent macro_f1 uses the complete fixed label catalog; labels cannot be overridden"
            )
        return selected

    def evaluate(self, samples, predictions, registry, metrics):
        result = super().evaluate(samples, predictions, registry, metrics)
        result["decision_rule"] = "native_choice_for_classification"
        return result

    def validate_samples(self, samples):
        catalogs = {tuple(s.question.labels) for s in samples}
        if (
            len(catalogs) != 1
            or len(next(iter(catalogs))) != self.class_count
            or any(not isinstance(s.target, HardTarget) for s in samples)
        ):
            raise ValueError(
                "Intent benchmark requires hard targets and one complete fixed label set"
            )
        if self.name == "clinc150_oos" and "oos" not in samples[0].question.labels:
            raise ValueError("CLINC150+OOS requires the oos class")
