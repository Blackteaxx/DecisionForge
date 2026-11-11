from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol

from decisionforge.training.config import TrainConfig
from decisionforge.training.data import PreparedData


@dataclass(frozen=True)
class BackendCapabilities:
    question_types: tuple[str, ...]
    devices: tuple[str, ...]
    finetuning_modes: tuple[str, ...]
    distributed: bool
    full_resume: bool
    calibration: Literal["independent_temperature", "none"]
    calibration_min_questions: int
    artifact_format: str
    diagnostics_source: str

    def as_dict(self):
        return asdict(self)


def capabilities(backend: str, *, model_type="omnijev") -> BackendCapabilities:
    # Static built-in declarations: importing these never imports a model SDK.
    if backend == "unsloth":
        return BackendCapabilities(
            ("choice", "noul", "score"),
            ("cuda",),
            ("lora", "full"),
            True,
            True,
            "independent_temperature",
            10,
            "unsloth_clef",
            "Unsloth FastDecisionModel",
        )
    if backend == "laya":
        return BackendCapabilities(
            ("choice", "noul", "score"),
            ("cuda",),
            ("full",),
            True,
            False,
            "independent_temperature",
            10,
            "laya",
            "Laya upstream diagnostics",
        )
    if backend == "openjev":
        return BackendCapabilities(
            ("choice", "noul"),
            ("cuda",),
            ("full",),
            False,
            False,
            "none",
            0,
            "openjev_rlcd",
            "OpenJev upstream diagnostics",
        )
    if backend == "swift":
        from decisionforge.training.backends.swift_routes import get_route

        route = get_route(model_type)
        return BackendCapabilities(
            ("choice", "noul", "score"),
            ("cuda",),
            ("lora", "full") if route.merged else ("lora",),
            False,
            False,
            "independent_temperature" if route.calibration else "none",
            10 if route.calibration else 0,
            "swift_omnijev" if route.name == "omnijev" else "swift_decision",
            "ms-swift decision diagnostics",
        )
    raise ValueError(f"Unsupported training backend: {backend}")


def preflight(config: TrainConfig, *, resume=False, data: PreparedData | None = None):
    cap = (
        capabilities(config.backend, model_type=config.swift.model_type)
        if config.swift
        else capabilities(config.backend)
    )
    if resume and not cap.full_resume:
        raise ValueError(
            f"{config.backend} does not support complete state resume; initialize a new run"
        )
    if config.finetuning not in cap.finetuning_modes:
        raise ValueError(f"{config.backend} requires finetuning in {cap.finetuning_modes}")
    if config.distributed.strategy == "ddp" and not cap.distributed:
        raise ValueError(f"{config.backend} does not support distributed DDP training")
    if config.calibration.enabled and cap.calibration == "none":
        raise ValueError(
            f"{config.backend} has no independent calibration; set calibration.enabled=false"
        )
    if config.backend == "openjev":
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError("OpenJev requires single-GPU execution without DDP")
        if config.openjev.init_checkpoint is not None:
            from decisionforge.training.backends.openjev import validate_artifact

            validate_artifact(config.openjev.init_checkpoint)
        if data is not None and any(not data.roles[role] for role in ("validation", "test")):
            raise ValueError(
                "OpenJev requires nonempty validation and test roles for upstream dev/test evaluation"
            )
    if config.backend == "laya" and config.laya.init_checkpoint is not None:
        from decisionforge.training.backends.laya import validate_artifact

        validate_artifact(config.laya.init_checkpoint)
    if data is not None:
        from decisionforge.media import reject_media

        for record in data.records:
            if config.backend != "swift":
                reject_media(record.request, config.backend)
            for key, question in record.request.questions.items():
                if question.type not in cap.question_types:
                    raise ValueError(
                        f"{config.backend}: record_id={record.record_id}, question={key}: "
                        f"unsupported question type {question.type}"
                    )
                if (
                    config.backend == "openjev"
                    and question.type == "choice"
                    and len(question.criteria) < 2
                ):
                    raise ValueError(
                        f"openjev: record_id={record.record_id}, question={key}: Choice needs at least 2 options"
                    )
        if config.backend == "swift":
            from decisionforge.training.backends.swift import validate_data

            validate_data(config, data)
    return cap


class TrainingBackend(Protocol):
    def environment(self) -> dict: ...
    def prepare(self, data: PreparedData) -> dict: ...
    def train(self, checkpoint: Path | None = None) -> dict: ...
    def evaluate(self, role: str, phase: str) -> dict: ...
    def save(self) -> dict: ...
    def reload_probe(self, data: PreparedData) -> dict: ...
    def close(self) -> None: ...


class ResumableTrainingBackend(TrainingBackend, Protocol):
    def validate_checkpoint(self, checkpoint: Path) -> int: ...
    def save_final_checkpoint(self) -> Path: ...
    def load_final_checkpoint(self, checkpoint: Path) -> None: ...


class CalibratedTrainingBackend(TrainingBackend, Protocol):
    def calibrate(self) -> dict: ...
    def calibration_state(self) -> dict: ...
    def restore_calibration(self, state: dict) -> None: ...


def get_backend(config: TrainConfig, directory: Path) -> TrainingBackend:
    if config.backend == "unsloth":
        from decisionforge.training.backends.unsloth import UnslothBackend

        return UnslothBackend(config, directory)
    if config.backend == "laya":
        from decisionforge.training.backends.laya import LayaBackend

        return LayaBackend(config, directory)
    if config.backend == "openjev":
        from decisionforge.training.backends.openjev import OpenJevBackend

        return OpenJevBackend(config, directory)
    if config.backend == "swift":
        from decisionforge.training.backends.swift import SwiftBackend

        return SwiftBackend(config, directory)
    raise ValueError(f"Unsupported training backend: {config.backend}")
