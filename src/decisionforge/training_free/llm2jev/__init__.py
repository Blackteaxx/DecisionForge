"""Pinned optional external LLM2Jev integration, without server dependencies."""

LLM2JEV_REPO = "https://github.com/Yinsongxu/LLM2Jev"
LLM2JEV_REVISION = "f2bdad648e7a34bfce64401fb0f7c44bc13209ae"


def candidate_task_count(question):
    """Pinned LLM2Jev binary compiler task count; not a measured GPU kernel count."""
    if question.type == "noul":
        return 2 if question.criteria else 1
    return len(question.criteria)


def validate_provenance(value):
    if value.upstream_repo != LLM2JEV_REPO or not all(
        (
            value.upstream_revision,
            value.backbone_model,
            value.backbone_revision,
            value.serving_backend,
            value.precision,
            value.dependencies,
        )
    ):
        raise ValueError(
            "llm2jev requires its source repository, fixed revisions and serving environment"
        )
