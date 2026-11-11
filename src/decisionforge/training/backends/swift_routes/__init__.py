"""Explicit, source-pinned decision routes. Importing this registry loads no ML SDK."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from decisionforge.training.backends.swift_task import (
    ADAPTER_REVISION,
    BASE_REVISION,
    PROBABILITY_CONTRACT,
    SWIFT_REVISION,
)

RouteName = Literal["omnijev", "startlux", "clef", "thx02", "kev", "jev"]


@dataclass(frozen=True)
class Route:
    name: str
    model: str
    revision: str
    adapter: str | None = None
    adapter_revision: str | None = None
    adapter_subdir: str = ""
    max_options: int = 255
    max_score_levels: int = 10
    score_levels: int | None = None
    max_images: int = 32
    max_pixels: int = 768 * 28 * 28
    loss: str = "scoring"
    calibration: bool = True
    gradient_checkpointing: bool = True
    noncommercial: bool = False
    # (source owner, source relative path, exported loader asset name, env var)
    assets: tuple[tuple[str, str, str, str], ...] = ()

    @property
    def merged(self):
        return self.adapter is None

    @property
    def template(self):
        return f"dforge_{self.name}" if self.name != "omnijev" else self.name

    @property
    def probability_contract(self):
        if self.name == "omnijev":
            return PROBABILITY_CONTRACT
        contract = {
            "version": f"decisionforge.swift.{self.name}.probabilities.v1",
            "choice_supervision": "original_K_distribution_in_native_column_order",
            "choice_readout": "native_columns_mapped_to_original_keys",
            "extra_choice_column": None,
            "noul_columns": ["false", "true"],
            "score_columns": "ordered_original_level_indices",
        }
        if self.name == "startlux":
            contract.update(
                choice_supervision="original_K_plus_zero_none_when_allow_none",
                choice_readout="condition_on_original_K_when_allow_none",
                extra_choice_column="optional_none_of_the_above",
                noul_columns=["true", "false"],
            )
        return contract

    def convert(self, request, settings, **kwargs):
        if self.name == "omnijev":
            from decisionforge.training.backends.swift_task import map_request

            return map_request(request, settings, **kwargs)
        from .formats import CONVERTERS

        return CONVERTERS[self.name](request, settings, **kwargs)

    def readout(self, question, probabilities, mapping=None, *, allow_none=False):
        if self.name == "omnijev":
            from decisionforge.training.backends.swift_task import readout

            return readout(question, probabilities)
        from .formats import readout

        return readout(
            question, probabilities, mapping, model_type=self.name, allow_none=allow_none
        )


ROUTES = {
    "omnijev": Route(
        "omnijev",
        "Qwen/Qwen3.5-4B",
        BASE_REVISION,
        "tinnel123/OmniJev",
        ADAPTER_REVISION,
        loss="ce_rps",
        gradient_checkpointing=False,
        max_images=64,
        assets=(
            ("adapter", "head.pt", "head.pt", "OMNIJEV_HEAD"),
            ("adapter", "ord.pt", "ord.pt", "OMNIJEV_ORD"),
            ("adapter", "head_meta.json", "head_meta.json", "OMNIJEV_HEAD_META"),
            ("adapter", "new_tok_emb.pt", "new_tok_emb.pt", "OMNIJEV_NEW_TOK_EMB"),
        ),
    ),
    "clef": Route(
        "clef",
        "Cloudflare/clef",
        "cfddda043dd190ef3b01cff449d5ee8cff1e8ec6",
        loss="smooth_brier",
        calibration=False,
        assets=(
            (
                "model",
                "joint_head.safetensors",
                "joint_head.safetensors",
                "CLEF_JOINT_HEAD_WEIGHTS",
            ),
            ("model", "joint_head_config.json", "joint_head_config.json", "CLEF_JOINT_HEAD_CONFIG"),
        ),
    ),
    "startlux": Route(
        "startlux",
        "startlux-models/StartLux-Decision-v1.1-4B-AutoThink",
        "23dd4c513ef387705bb13fd8caadc2d5c863082f",
        max_options=26,
        max_images=8,
        loss="distill_rps",
        noncommercial=True,
        assets=(
            ("model", "decision_config.json", "decision_config.json", "STARTLUX_DECISION_CONFIG"),
        ),
    ),
    "kev": Route(
        "kev",
        "Qwen/Qwen3.5-0.8B-Base",
        "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
        "jaredpalmer/kev-0.8b",
        "bf75a6a8848ea6960ff2ed108d9ed44c2941174f",
        max_images=0,
        calibration=False,
        assets=(("adapter", "head.pt", "head.pt", "KEV_HEAD"),),
    ),
    "jev": Route(
        "jev",
        "autotrust/JEV-27B-VL",
        "9e53b8a809d2ab98e3e6a55b2b2dbf94b4c1e84a",
        "autotrust/JEV-27B-VL",
        "9e53b8a809d2ab98e3e6a55b2b2dbf94b4c1e84a",
        adapter_subdir="adapter_vllm",
        max_options=16,
        max_score_levels=6,
        score_levels=6,
        loss="distill_rps",
        assets=(
            ("adapter", "decision_head.json", "decision_head.json", "JEV_DECISION_HEAD"),
            ("model", "calibration.json", "calibration.json", "JEV_DECIDE_CALIBRATION"),
        ),
    ),
}

# Known native names are intentionally distinct from enabled routes. No fallback
# or arbitrary model_type can bypass an unresolved source/license gate.
BLOCKED_ROUTES = {
    "thx02": "THX-02 checkpoint/license cannot be verified: official HF source returns401 and ModelScope404",
}


def get_route(name: str) -> Route:
    if name in BLOCKED_ROUTES:
        raise ValueError(f"Swift route {name!r} is not enabled: {BLOCKED_ROUTES[name]}")
    if name not in ROUTES:
        raise ValueError(
            f"Unknown/unsupported Swift model_type {name!r}; enabled: {', '.join(ROUTES)}"
        )
    return ROUTES[name]


__all__ = ["BLOCKED_ROUTES", "ROUTES", "Route", "RouteName", "SWIFT_REVISION", "get_route"]
