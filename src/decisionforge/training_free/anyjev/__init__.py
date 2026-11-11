"""Pinned, label-free AnyJev L0 integration (safe without optional dependencies)."""

ANYJEV_REPO = "https://github.com/nokia-applied-research/AnyJev"
ANYJEV_REVISION = "35311c58955716cb8955ce1368a665988bc63735"
SYSTEM = (
    "You are a decision function. You will be given a state and one question. "
    "Reply with the answer label only: no words, no punctuation, no explanation."
)
SETTINGS = {
    "level": "L0",
    "prior": "content_free",
    "prior_strength": 1.0,
    "adaptive_shifts": False,
    "max_permutations": None,
    "combine": "logmean",
    "cf_probes": ("N/A", "", "[MASK]"),
    "record_content_free": False,
    "system": SYSTEM,
    "shared_prefix": "auto",
    "shared_min_prefix_tokens": 256,
    "canonical_order": False,
}
PACKAGES = (
    "anyjev",
    "torch",
    "transformers",
    "numpy",
    "tokenizers",
    "huggingface_hub",
    "safetensors",
)


def method_identity(*, shared_prefix="auto"):
    return {
        "repo": ANYJEV_REPO,
        "revision": ANYJEV_REVISION,
        "settings": {
            **SETTINGS,
            "cf_probes": list(SETTINGS["cf_probes"]),
            "shared_prefix": shared_prefix,
        },
        "render_version": "decisionforge.anyjev.v2",
        "label_protocol": {
            "schema_version": 1,
            "large_choice_extension": "official_decider_labels_for_hook",
            "encoding_style": "bare_then_space",
            "candidate_pool": "A-ZZZ",
            "legacy_choice_max": 26,
            "large_choice_instruction_version": 1,
            "large_choice_diagnostics": "gzip+base64.v1",
            "large_choice_tokenization": "per_predict_verified_cache.v1",
        },
    }


def dependency_versions():
    from importlib.metadata import PackageNotFoundError, version

    result = {}
    for name in PACKAGES:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def validate_backbone(model):
    from pathlib import Path

    path = Path(model.model).expanduser()
    if path.is_dir():
        if any(
            (path / name).exists()
            for name in ("adapter_config.json", "unsloth_decision_config.json")
        ):
            raise ValueError("AnyJev requires an ordinary backbone, not trained decision artifacts")
    elif path.is_absolute() or model.model.startswith((".", "~")):
        raise ValueError(f"AnyJev backbone directory does not exist: {path}")
    return path


def backbone_identity(model):
    from decisionforge.eval.checkpoint import file_identity

    path = validate_backbone(model)
    if path.is_dir():
        return {"local_content": file_identity(path)}
    return {"model": model.model, "revision": model.revision}


def validate_installation():
    """A version string alone cannot distinguish commits sharing upstream 0.3.0."""
    import json
    from importlib.metadata import distribution

    installed = distribution("anyjev")
    source = json.loads(installed.read_text("direct_url.json") or "{}")
    commit = source.get("vcs_info", {}).get("commit_id")
    if installed.version != "0.3.0" or commit != ANYJEV_REVISION:
        raise ValueError("AnyJev must come from the pinned source; run uv sync --extra eval-anyjev")
    return {"version": installed.version, "commit_id": commit}
