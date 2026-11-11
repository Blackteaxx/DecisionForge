"""AnyJev uses the existing owned worker lifecycle with distinct result provenance."""

from decisionforge.eval.adapters.anyjev_worker import worker_main
from decisionforge.eval.adapters.local import LocalAdapter


class AnyJevAdapter(LocalAdapter):
    def __init__(self, model, execution, *, worker_target=worker_main):
        super().__init__(model, execution, worker_target=worker_target)
        self.model_identity_source = "loaded_backbone"
        self.result_probability_source = "anyjev_l0_label_free_corrected_next_token_logits"
