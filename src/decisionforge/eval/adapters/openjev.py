"""OpenJev-specific provenance over the shared isolated local-worker transport."""

from decisionforge.eval.adapters.local import LocalAdapter
from decisionforge.eval.adapters.openjev_worker import worker_main
from decisionforge.training.backends.openjev_task import SAMPLING


class OpenJevAdapter(LocalAdapter):
    def __init__(self, model, execution, *, worker_target=worker_main):
        super().__init__(model, execution, worker_target=worker_target)
        self.result_probability_source = SAMPLING["probability_source"]
        self.model_identity_source = "loaded_openjev_stage2_weights"
