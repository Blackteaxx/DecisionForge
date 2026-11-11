"""General method declarations; never proof of loaded service weights."""

from decisionforge.training_free.llm2jev import (
    LLM2JEV_REPO as LLM2JEV_REPO,
)
from decisionforge.training_free.llm2jev import (
    LLM2JEV_REVISION as LLM2JEV_REVISION,
)
from decisionforge.training_free.llm2jev import (
    candidate_task_count as candidate_task_count,
)


def method_metadata(model):
    declared = (
        model.get("provenance")
        if isinstance(model, dict)
        else (model.provenance.model_dump() if getattr(model, "provenance", None) else None)
    )
    adapter = model.get("adapter", "jev") if isinstance(model, dict) else model.adapter
    if adapter == "anyjev_local":
        from decisionforge.training_free.anyjev import ANYJEV_REPO, ANYJEV_REVISION, method_identity

        values = model if isinstance(model, dict) else model.model_dump()
        return {
            "method": "anyjev_l0",
            "declaration_verified": False,
            "confidence_basis": "maximum_candidate_probability",
            "declared": {
                "upstream_repo": ANYJEV_REPO,
                "upstream_revision": ANYJEV_REVISION,
                "backbone_model": values["model"],
                "backbone_revision": values.get("revision"),
                "training_free": values.get("training_free", True),
                "precision": values["precision"],
                "settings": method_identity(shared_prefix=values.get("shared_prefix", "auto"))[
                    "settings"
                ],
            },
        }
    method = (
        declared["method"]
        if declared
        else {
            "jev": "native",
            "chat_api": "chat_baseline",
            "unsloth_local": "trained_unsloth",
            "laya_local": "trained_laya_rlcd",
            "openjev_local": "trained_openjev_rlcd_two_stage",
            "swift_local": "swift_omnijev_decision",
            "swift_chat_local": "chat_baseline",
        }[adapter]
    )
    return {"method": method, "declared": declared, "declaration_verified": False}
