from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from decisionforge.training.backends.swift_routes import RouteName, get_route

Role = Literal["train", "validation", "calibration", "test"]


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class HoldoutConfig(StrictConfig):
    validation_fraction: float = Field(default=0.1, ge=0, lt=1)
    calibration_fraction: float = Field(default=0.1, ge=0, lt=1)

    @model_validator(mode="after")
    def fractions(self):
        if self.validation_fraction + self.calibration_fraction >= 1:
            raise ValueError("Holdout fractions must sum to less than one")
        return self


class DataConfig(StrictConfig):
    path: str = Field(min_length=1)
    split_map: dict[str, Role] = Field(
        default_factory=lambda: {r: r for r in ("train", "validation", "calibration", "test")}
    )
    auto_holdout: HoldoutConfig = Field(default_factory=HoldoutConfig)
    on_skipped: Literal["error", "allow"] = "error"
    on_truncated: Literal["error", "allow"] = "error"

    @model_validator(mode="after")
    def preserve_canonical_roles(self):
        for role in ("train", "validation", "calibration", "test"):
            if role in self.split_map and self.split_map[role] != role:
                raise ValueError(f"Canonical split {role!r} must retain its role; map aliases only")
        return self


class ModelConfig(StrictConfig):
    name: str = Field(default="unsloth/Qwen3.5-4B", min_length=1)
    revision: str | None = None
    max_seq_length: int = Field(default=2048, ge=32)
    load_in_4bit: bool = True


class LoraConfig(StrictConfig):
    rank: int = Field(default=16, ge=1)
    alpha: int = Field(default=16, ge=1)
    dropout: float = Field(default=0.0, ge=0, lt=1)


class TrainingConfig(StrictConfig):
    seed: int = Field(default=3407, ge=0, le=2**32 - 1)
    num_train_epochs: float | None = Field(default=2.0, gt=0)
    max_steps: int | None = Field(default=None, ge=1)
    batch_size: int = Field(default=2, ge=1)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    learning_rate: float = Field(default=0.0002, gt=0)
    head_learning_rate: float = Field(default=0.0001, gt=0)
    warmup_steps: int = Field(default=10, ge=0)
    weight_decay: float = Field(default=0.01, ge=0)
    lr_scheduler_type: Literal["linear", "cosine", "constant", "constant_with_warmup"] = "cosine"
    precision: Literal["auto", "bf16", "fp16"] = "auto"
    gradient_checkpointing: bool = True
    logging_steps: int = Field(default=10, ge=1)
    save_steps: int | None = Field(default=100, ge=1)

    @model_validator(mode="after")
    def termination(self):
        if (self.num_train_epochs is None) == (self.max_steps is None):
            raise ValueError("Exactly one of num_train_epochs / max_steps must be non-null")
        return self


class CalibrationConfig(StrictConfig):
    enabled: bool = True


class OutputConfig(StrictConfig):
    directory: str = "outputs/train/my-run"
    checkpoint_directory: str | None = None
    save_merged: bool = False


class DistributedConfig(StrictConfig):
    strategy: Literal["single", "ddp"] = "single"
    world_size: int = Field(default=1, ge=1)
    timeout_seconds: int = Field(default=1800, ge=1)

    @model_validator(mode="after")
    def single_host(self):
        if (self.strategy == "single") != (self.world_size == 1):
            raise ValueError("single requires world_size=1; ddp requires world_size>=2")
        return self


class LayaConfig(StrictConfig):
    init_checkpoint: str | None = Field(default=None, min_length=1)
    head_max_length: int = Field(default=256, ge=32)
    rl_samples: int = Field(default=4, ge=2)
    sigma_start: float = Field(default=0.4, gt=0)
    sigma_end: float = Field(default=0.1, gt=0)
    w_sph: float = Field(default=0.75, ge=0)
    w_rps: float = Field(default=1.0, ge=0)
    min_learning_rate: float = Field(default=1e-6, ge=0)
    gradient_clip: float = Field(default=1.0, gt=0)


class OpenJevConfig(StrictConfig):
    init_checkpoint: str | None = Field(default=None, min_length=1)
    reinforce_steps: int | None = Field(default=None, ge=1)
    reinforce_seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    m: int = Field(default=4, ge=2)
    m_eval: int = Field(default=1, ge=1)
    max_new_tokens: int = Field(default=128, ge=1)
    eval_every: int | None = Field(default=None, ge=1)
    dev_cap: int = Field(default=199, ge=1)
    test_cap: int = Field(default=400, ge=1)


class SwiftConfig(StrictConfig):
    """Pinned ms-swift decision integration. Only explicitly implemented model types pass."""

    model_type: RouteName = "omnijev"
    task_type: Literal["decision"] = "decision"
    adapter: str | None = None
    adapter_revision: str | None = None
    swift_revision: Literal["680df5c670d7e8add906ac49381d603869410612"] = (
        "680df5c670d7e8add906ac49381d603869410612"
    )
    max_options: int = Field(default=255, ge=2, le=255)
    max_score_levels: int = Field(default=10, ge=2, le=10)
    max_images: int = Field(default=32, ge=0, le=64)
    max_pixels: int = Field(default=768 * 28 * 28, ge=1)
    decision_subsample_options: bool = False
    gradient_checkpointing: bool = False
    freeze_vit: bool = True
    init_checkpoint: str | None = None
    allow_none: bool = False
    usage: Literal["noncommercial_research"] | None = None
    packing: Literal[False] = False
    padding_free: Literal[False] = False

    @model_validator(mode="before")
    @classmethod
    def route_defaults(cls, value):
        if not isinstance(value, dict):
            return value
        value = dict(value)
        route = get_route(value.get("model_type", "omnijev"))
        for key, default in {
            "adapter": route.adapter,
            "adapter_revision": route.adapter_revision,
            "max_options": route.max_options,
            "max_score_levels": route.max_score_levels,
            "max_images": min(32, route.max_images),
            "max_pixels": route.max_pixels,
            "gradient_checkpointing": route.gradient_checkpointing,
        }.items():
            value.setdefault(key, default)
        return value

    @model_validator(mode="after")
    def restrictions(self):
        route = get_route(self.model_type)
        if self.decision_subsample_options:
            raise ValueError("Swift routes keep decision_subsample_options disabled")
        for key in ("max_options", "max_score_levels", "max_images", "max_pixels"):
            if getattr(self, key) > getattr(route, key):
                raise ValueError(f"{route.name}: {key} exceeds the route limit")
        if (self.adapter, self.adapter_revision) != (route.adapter, route.adapter_revision):
            raise ValueError(f"{route.name} requires its fixed published adapter and revision")
        if route.noncommercial and self.usage != "noncommercial_research":
            raise ValueError("StartLux CC-BY-NC-4.0 requires explicit usage=noncommercial_research")
        if self.allow_none and route.name != "startlux":
            raise ValueError("allow_none is only supported by the StartLux route")
        return self


class TrainConfig(StrictConfig):
    backend: Literal["unsloth", "laya", "openjev", "swift"] = "unsloth"
    finetuning: Literal["lora", "full"] = "lora"
    data: DataConfig
    model: ModelConfig = Field(default_factory=ModelConfig)
    lora: LoraConfig | None = None
    distributed: DistributedConfig = Field(default_factory=DistributedConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    laya: LayaConfig | None = None
    openjev: OpenJevConfig | None = None
    swift: SwiftConfig | None = None

    @model_validator(mode="before")
    @classmethod
    def backend_defaults(cls, value):
        if isinstance(value, dict) and value.get("backend") == "swift":
            value = dict(value)
            for name in ("model", "training"):
                if isinstance(value.get(name), BaseModel):
                    value[name] = value[name].model_dump()
                if name in value and not isinstance(value[name], dict):
                    return value
            value.setdefault("swift", {})
            settings = value["swift"]
            if isinstance(settings, BaseModel):
                settings = settings.model_dump()
            if not isinstance(settings, dict):
                return value
            route = get_route(settings.get("model_type", "omnijev"))
            settings = dict(settings)
            training = value.get("training", {})
            gradient_checkpointing = training.get(
                "gradient_checkpointing",
                settings.get("gradient_checkpointing", route.gradient_checkpointing),
            )
            settings.setdefault("gradient_checkpointing", gradient_checkpointing)
            value["swift"] = settings
            value["model"] = {
                "name": route.model,
                "revision": route.revision,
                "load_in_4bit": False,
                "max_seq_length": 4096,
                **value.get("model", {}),
            }
            value["training"] = {
                "gradient_checkpointing": gradient_checkpointing,
                "batch_size": 1,
                "precision": "bf16",
                "learning_rate": 1e-4,
                "head_learning_rate": 1e-4,
                **value.get("training", {}),
            }
            if route.merged and value.get("finetuning", "lora") == "lora":
                value.setdefault("lora", {"rank": 8, "alpha": 32, "dropout": 0.0})
            if not route.calibration:
                value.setdefault("calibration", {"enabled": False})
            return value
        if not isinstance(value, dict) or value.get("backend") not in {"laya", "openjev"}:
            return value
        value = dict(value)
        value.setdefault("finetuning", "full")
        for name in ("model", "training"):
            if isinstance(value.get(name), BaseModel):
                value[name] = value[name].model_dump()
            if name in value and not isinstance(value[name], dict):
                return value  # Let Pydantic report the strict child type error.
        default_name = (
            "Qwen/Qwen3-0.6B" if value["backend"] == "openjev" else "convaiinnovations/laya"
        )
        if value.get("model", {}).get("name", default_name) != default_name:
            value["model"].setdefault("revision", None)
        if value["backend"] == "openjev":
            value["model"] = {
                "name": "Qwen/Qwen3-0.6B",
                "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
                "load_in_4bit": False,
                "max_seq_length": 2048,
                **value.get("model", {}),
            }
            training = value.get("training", {})
            value["training"] = {
                "num_train_epochs": None,
                "max_steps": 400,
                "batch_size": 8,
                "gradient_accumulation_steps": 1,
                "learning_rate": 2e-6,
                "head_learning_rate": training.get("learning_rate", 2e-6),
                "warmup_steps": 0,
                "weight_decay": 0.0,
                "lr_scheduler_type": "constant",
                "precision": "bf16",
                "save_steps": None,
                **training,
            }
            value.setdefault("calibration", {"enabled": False})
            value.setdefault("openjev", {})
            return value
        value["model"] = {
            "name": "convaiinnovations/laya",
            "revision": "7b928d828b7b0e022f929d9bd2e44165aa270148",
            "load_in_4bit": False,
            "max_seq_length": 1024,
            **value.get("model", {}),
        }
        value["training"] = {
            "seed": 3407,
            "num_train_epochs": 4.0,
            "batch_size": 8,
            "gradient_accumulation_steps": 8,
            "learning_rate": 2.5e-5,
            "warmup_steps": 0,
            "precision": "fp16",
            "save_steps": None,
            **value.get("training", {}),
        }
        value.setdefault("laya", {})
        return value

    @model_validator(mode="after")
    def finetuning_settings(self):
        if self.backend == "laya":
            t = self.training
            if (
                self.finetuning != "full"
                or self.model.load_in_4bit
                or self.lora is not None
                or t.max_steps is not None
                or t.num_train_epochs is None
                or not t.num_train_epochs.is_integer()
                or t.precision == "bf16"
                or t.warmup_steps != 0
                or t.lr_scheduler_type != "cosine"
                or t.save_steps is not None
                or self.output.save_merged
                or self.laya is None
            ):
                raise ValueError(
                    "Laya requires full unquantized RLCD, integer epochs, fp16/auto, "
                    "cosine without warmup, epoch saves (save_steps=null), no merged export"
                )
        elif self.laya is not None:
            raise ValueError("laya settings require backend=laya")
        if self.backend == "openjev":
            t = self.training
            if (
                self.finetuning != "full"
                or self.model.load_in_4bit
                or self.lora is not None
                or self.openjev is None
                or t.num_train_epochs is not None
                or t.max_steps is None
                or t.gradient_accumulation_steps != 1
                or t.head_learning_rate != t.learning_rate
                or t.warmup_steps != 0
                or t.weight_decay != 0
                or t.lr_scheduler_type != "constant"
                or t.precision != "bf16"
                or not t.gradient_checkpointing
                or t.logging_steps != 10
                or t.save_steps is not None
                or self.calibration.enabled
                or self.output.save_merged
            ):
                raise ValueError(
                    "OpenJev requires full unquantized BF16, step budgets, accumulation=1, "
                    "one learning rate, zero warmup/weight_decay, constant scheduler, "
                    "gradient checkpointing, logging_steps=10, save_steps=null, "
                    "no independent calibration or merged export"
                )
            if self.openjev.eval_every is not None and any(
                steps % self.openjev.eval_every
                for steps in (t.max_steps, self.openjev.reinforce_steps or t.max_steps)
            ):
                raise ValueError("OpenJev eval_every must divide both stage step budgets")
        elif self.openjev is not None:
            raise ValueError("openjev settings require backend=openjev")
        if self.backend == "swift":
            if self.swift is None:
                raise ValueError("swift settings require backend=swift")
            route = get_route(self.swift.model_type)
            if self.model.load_in_4bit:
                raise ValueError("Swift routes require unquantized checkpoints")
            if not route.merged and (self.finetuning != "lora" or self.lora is not None):
                raise ValueError(
                    f"{route.name} continues the published LoRA; fresh LoRA/full are unsupported"
                )
            if route.merged and self.finetuning == "lora" and self.lora is None:
                raise ValueError("Merged Swift checkpoints require explicit new-LoRA settings")
            if self.output.save_merged or self.output.checkpoint_directory:
                raise ValueError(
                    "swift uses native checkpoints inside the run; merged export unsupported"
                )
            if (
                self.training.head_learning_rate != self.training.learning_rate
                or self.training.precision != "bf16"
            ):
                raise ValueError("swift requires BF16 and one shared adapter/head learning rate")
            if self.model.name != route.model or self.model.revision != route.revision:
                raise ValueError(f"{route.name} requires its pinned published checkpoint")
            if self.distributed.strategy != "single":
                raise ValueError("Swift routes currently support single-GPU execution only")
            if route.name == "omnijev" and (
                self.training.gradient_checkpointing or self.swift.gradient_checkpointing
            ):
                raise ValueError("swift OmniJev branch cache requires gradient_checkpointing=false")
            if self.training.gradient_checkpointing != self.swift.gradient_checkpointing:
                raise ValueError("Swift/training gradient_checkpointing settings disagree")
            if self.calibration.enabled and not route.calibration:
                raise ValueError(
                    f"{route.name}: pinned native head has no supported independent calibration callback"
                )
        elif self.swift is not None:
            raise ValueError("swift settings require backend=swift")
        if self.backend == "unsloth" and self.training.save_steps is None:
            raise ValueError("unsloth requires save_steps")
        if self.finetuning == "full":
            if self.model.load_in_4bit or self.lora is not None:
                raise ValueError("full finetuning requires load_in_4bit=false and no lora settings")
        elif self.lora is None and self.backend != "swift":
            self.lora = LoraConfig()
        return self


def load_config(path: str | Path) -> TrainConfig:
    path = Path(path).resolve()
    config = TrainConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    if config.output.checkpoint_directory:
        config.output.checkpoint_directory = str(
            (path.parent / config.output.checkpoint_directory).resolve()
        )
    for owner, key in ((config.data, "path"), (config.output, "directory")):
        value = Path(getattr(owner, key)).expanduser()
        setattr(
            owner, key, str((path.parent / value).resolve() if not value.is_absolute() else value)
        )
    name = config.model.name
    if name.startswith((".", "/", "~")) or (path.parent / name).is_dir():
        value = Path(name).expanduser()
        config.model.name = str(
            (path.parent / value).resolve() if not value.is_absolute() else value
        )
    for settings in (config.laya, config.openjev, config.swift):
        if settings is not None and settings.init_checkpoint is not None:
            checkpoint = Path(settings.init_checkpoint).expanduser()
            settings.init_checkpoint = str(
                (path.parent / checkpoint).resolve() if not checkpoint.is_absolute() else checkpoint
            )
    return config
