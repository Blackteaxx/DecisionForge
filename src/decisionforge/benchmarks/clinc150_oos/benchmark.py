from decisionforge.benchmarks._intent import IntentBenchmark

from .metrics import macro_f1


class Clinc150OOS(IntentBenchmark):
    name, version, class_count = "clinc150_oos", "1", 151

    def evaluate(self, samples, predictions, registry, metrics):
        result = super().evaluate(samples, predictions, registry, metrics)
        by_id = {p["id"]: p for p in predictions}
        rows = [self.prepare(s, by_id.get(s.id)) for s in samples]
        result["oos"] = macro_f1(rows, {"labels": ["oos"]})["per_class"]["oos"]
        result["in_scope_macro_f1"] = macro_f1(
            rows, {"labels": [k for k in samples[0].question.labels if k != "oos"]}
        )
        result["classification_rule"] = "native_choice_independent_of_probability_validity"
        return result
