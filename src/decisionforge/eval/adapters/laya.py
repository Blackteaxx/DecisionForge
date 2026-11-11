"""Laya inference uses the shared owned-worker transport and unmodified public requests."""

import copy
import json

from decisionforge.eval.adapter import parse_answer
from decisionforge.eval.adapters.laya_worker import worker_main
from decisionforge.eval.adapters.local import LocalAdapter
from decisionforge.eval.metrics import distribution


class LayaAdapter(LocalAdapter):
    def __init__(self, model, execution, *, worker_target=worker_main):
        super().__init__(model, execution, worker_target=worker_target)
        self.result_probability_source = "laya_upstream_calibrated_softmax_rounded4"

    def response_answer(self, raw, question):
        # The raw native response remains untouched in the journal. Laya serializes
        # structured rubric levels in its legend; the common scoring view uses the
        # original native JSON rubric after verifying that exact upstream rendering.
        if not isinstance(raw, dict) or not isinstance(raw.get("answers"), dict):
            raise ValueError("Missing Laya answers object")
        view = copy.deepcopy(raw)
        if question.type == "score":
            answer = view["answers"].get("decision")
            if not isinstance(answer, dict):
                raise ValueError("Missing Laya Score decision answer")
            expected = {
                str(i): level
                if isinstance(level, str)
                else json.dumps(level, ensure_ascii=False, separators=(", ", ": "))
                for i, level in enumerate(question.criteria)
            }
            if answer.get("legend") != expected:
                raise ValueError("Laya Score legend differs from requested rubric")
            answer["legend"] = {str(i): level for i, level in enumerate(question.criteria)}
        answer = parse_answer(view, question)
        if question.type != "noul":
            values = distribution(answer["probabilities"], question.labels)
            if abs(sum(values.values()) - 1) > len(values) * 0.00005 + 1e-8:
                raise ValueError(
                    "Laya probability mass exceeds official four-decimal rounding bound"
                )
        return answer

    async def send(self, payload, question):
        result = await super().send(payload, question)
        if question.type == "noul":
            result["probability_source"] = "derived_from_laya_upstream_noul_rounded4"
        return result
