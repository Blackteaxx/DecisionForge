"""ms-swift provenance over the shared isolated local-worker transport."""

from decisionforge.eval.adapters.local import LocalAdapter
from decisionforge.eval.adapters.swift_worker import worker_main
from decisionforge.training.backends.swift_task import PROBABILITY_CONTRACT, readout


class SwiftAdapter(LocalAdapter):
    supports_media = True

    def __init__(self, model, execution, *, worker_target=worker_main):
        super().__init__(model, execution, worker_target=worker_target)
        self.result_probability_source = (
            "ms_swift_scoring_head_masked_softmax_condition_non_abstain"
            if model.adapter == "swift_local"
            else "llm_self_reported"
        )
        if model.adapter == "swift_local" and model.model_type != "omnijev":
            self.result_probability_source = f"ms_swift_{model.model_type}_native_columns"
        self.model_identity_source = "loaded_swift_checkpoint"

    def response_answer(self, raw, question):
        if self.model.adapter == "swift_chat_local":
            from decisionforge.eval.metrics import distribution
            from decisionforge.training_free.chat_baseline.processing import ChatProcessing

            answer = ChatProcessing.response_answer(self, raw, question, {})
            if question.type != "noul":
                distribution(answer["probabilities"], question.labels, tolerance=1e-6)
            return answer
        if self.model.adapter == "swift_local":
            if self.model.model_type != "omnijev":
                from decisionforge.training.backends.swift_routes import get_route

                route = get_route(self.model.model_type)
                detail = raw.get("swift", {})
                if (
                    detail.get("model_type") != route.name
                    or detail.get("probability_contract") != route.probability_contract
                ):
                    raise ValueError("Swift route/probability contract mismatch")
                columns = detail["columns"]
                if len(columns) != 1 or columns[0]["question_id"] != "decision":
                    raise ValueError("Swift returned unexpected question mapping")
                expected, extra = route.readout(
                    question, columns[0]["native"][0]["probs"], allow_none=self.model.allow_none
                )
                if (
                    raw["answers"]["decision"] != expected
                    or columns[0]["extra_choice_probability"] != extra
                ):
                    raise ValueError("Swift route answer differs from raw native columns")
                return super().response_answer(raw, question)
            if raw.get("swift", {}).get("probability_contract") != PROBABILITY_CONTRACT:
                raise ValueError("swift probability contract mismatch")
            columns = raw["swift"]["columns"]
            if len(columns) != 1 or columns[0]["question_id"] != "decision":
                raise ValueError("swift returned unexpected question mapping")
            expected, abstain = readout(question, columns[0]["native"][0]["probs"])
            if (
                raw["answers"]["decision"] != expected
                or columns[0]["abstain_probability"] != abstain
            ):
                raise ValueError("swift mapped answer differs from its raw probability columns")
        return super().response_answer(raw, question)
