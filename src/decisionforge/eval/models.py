from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from decisionforge.media import MediaRequest


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


Content = str | dict[str, JsonValue] | list[JsonValue]


class Question(StrictModel):
    type: Literal["choice", "noul", "score"]
    instructions: Content | None = None
    criteria: dict[str, JsonValue] | list[JsonValue] | None = None

    @model_validator(mode="after")
    def native_shape(self):
        if self.type == "choice":
            if not isinstance(self.criteria, dict) or not 2 <= len(self.criteria) <= 255:
                raise ValueError("Choice needs 2–255 criteria")
            if any(not k.strip() for k in self.criteria):
                raise ValueError("Empty option key")
        elif self.type == "score":
            if not isinstance(self.criteria, list) or not 2 <= len(self.criteria) <= 10:
                raise ValueError("Score needs 2–10 ordered criteria")
        elif self.criteria is not None:
            if not isinstance(self.criteria, dict) or not set(self.criteria) <= {"false", "true"}:
                raise ValueError("Noul criteria must describe false/true")
        descriptions = (
            self.criteria.values() if isinstance(self.criteria, dict) else self.criteria or []
        )
        if any(v is not None and not isinstance(v, (str, dict, list)) for v in descriptions):
            raise ValueError("Descriptions must be native JSON content or null")
        if self.type == "score" and any(v is None for v in self.criteria):
            raise ValueError("Score descriptions cannot be null")
        if (
            self.type == "noul"
            and self.instructions is None
            and not any(v is not None for v in descriptions)
        ):
            raise ValueError("Noul requires instructions or described criteria")
        return self

    @property
    def labels(self) -> list[str]:
        if self.type == "noul":
            return ["false", "true"]
        if self.type == "score":
            return [str(i) for i in range(len(self.criteria))]
        return list(self.criteria)


class Request(MediaRequest, StrictModel):
    state: Content
    questions: dict[str, Question]

    @model_validator(mode="after")
    def single_decision(self):
        if set(self.questions) != {"decision"}:
            raise ValueError("questions must contain exactly decision")
        return self


class HardTarget(StrictModel):
    kind: Literal["hard"]
    value: str | bool | int


class SoftTarget(StrictModel):
    kind: Literal["soft"]
    probabilities: dict[str, float]


class ScalarTarget(StrictModel):
    kind: Literal["scalar"]
    value: float


Target = Annotated[HardTarget | SoftTarget | ScalarTarget, Field(discriminator="kind")]


class Sample(StrictModel):
    schema_version: Literal["decisionforge.eval.v1"]
    id: str = Field(min_length=1)
    group_id: str | None
    split: str = Field(min_length=1)
    subset: str = Field(min_length=1)
    request: Request
    target: Target | None
    metadata: dict[str, JsonValue]

    @property
    def question(self) -> Question:
        return self.request.questions["decision"]

    @model_validator(mode="after")
    def reference_contract(self):
        json.dumps(self.model_dump(), allow_nan=False)
        if any(not v.strip() for v in (self.id, self.split, self.subset)):
            raise ValueError("Identifiers cannot be blank")
        if self.group_id is not None and not self.group_id.strip():
            raise ValueError("group_id cannot be blank")
        target, question = self.target, self.question
        if isinstance(target, HardTarget):
            v = target.value
            valid = (
                type(v) is str and v in question.labels
                if question.type == "choice"
                else type(v) is bool
                if question.type == "noul"
                else type(v) is int and 0 <= v < len(question.labels)
            )
            if not valid:
                raise ValueError("Hard reference does not match question type/labels")
        elif isinstance(target, SoftTarget):
            p = target.probabilities
            if set(p) != set(question.labels):
                raise ValueError("Soft reference keys must cover all labels")
            if any(not math.isfinite(v) or not 0 <= v <= 1 for v in p.values()):
                raise ValueError("Invalid soft reference probability")
            if abs(sum(p.values()) - 1) > 1e-6:
                raise ValueError("Soft reference probabilities must sum to one")
            if not self.metadata.get("target_source"):
                raise ValueError("Soft reference requires metadata.target_source")
        elif isinstance(target, ScalarTarget):
            if question.type != "score" or not 0 <= target.value < len(question.labels):
                raise ValueError("Scalar reference requires Score within rubric bounds")
            if target.value > len(question.labels) - 1:
                raise ValueError("Scalar reference exceeds highest grade")
        return self


class DataConfig(StrictModel):
    path: str
    splits: list[str] | None = None
    subsets: list[str] | None = None
    limit: int | None = Field(default=None, ge=1)


class ModelProvenance(StrictModel):
    """User declarations, never proof of the service's loaded weights."""

    method: Literal["native", "chat_baseline", "llm2jev"]
    upstream_repo: str | None = None
    upstream_revision: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    backbone_model: str | None = Field(default=None, min_length=1)
    backbone_revision: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    serving_backend: str | None = Field(default=None, min_length=1)
    dependencies: dict[str, str] = Field(default_factory=dict)
    precision: str | None = Field(default=None, min_length=1)
    settings: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def pinned_method(self):
        if self.method == "llm2jev":
            from decisionforge.training_free.llm2jev import validate_provenance

            validate_provenance(self)
        return self


class ModelConfig(StrictModel):
    adapter: Literal["jev"] = "jev"
    base_url: str = "https://api.typesafe.ai"
    endpoint_path: Literal["/v1/systemone", "/api/alpha/decisions"] = "/v1/systemone"
    model: str = Field(default="jev-latest", min_length=1)
    api_key_env: str | None = "TYPESAFE_API_KEY"
    provenance: ModelProvenance | None = None

    @model_validator(mode="after")
    def endpoint(self):
        parsed = urlsplit(self.base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("base_url must be an HTTP service root without credentials/query")
        if self.api_key_env is not None and not self.api_key_env.strip():
            raise ValueError("api_key_env must be a name or null")
        if self.provenance:
            allowed = {"chat_baseline"} if self.adapter == "chat_api" else {"native", "llm2jev"}
            if self.provenance.method not in allowed:
                raise ValueError("Method metadata does not match adapter")
            if self.provenance.method == "llm2jev" and self.endpoint_path != "/v1/systemone":
                raise ValueError("llm2jev requires /v1/systemone")
        self.base_url = self.base_url.rstrip("/")
        return self


class LocalModelConfig(StrictModel):
    adapter: Literal["unsloth_local"]
    path: str = Field(min_length=1)
    load_in_4bit: bool = True
    precision: Literal["auto", "bf16", "fp16"] = "auto"
    max_seq_length: int = Field(default=16384, ge=32, le=32768)
    load_timeout: float = Field(default=600, gt=0, le=3600)
    local_files_only: bool = True

    @property
    def model(self):
        return self.path

    @model_validator(mode="after")
    def local_path(self):
        if not self.path.strip() or self.path.startswith(("http://", "https://")):
            raise ValueError("unsloth_local requires a local checkpoint directory")
        return self


class AnyJevModelConfig(StrictModel):
    adapter: Literal["anyjev_local"]
    model: str = Field(min_length=1)
    revision: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    precision: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    device: str = Field(default="cuda:0", pattern=r"^cuda(?::[0-9]+)?$")
    max_seq_length: int = Field(default=16384, ge=32, le=131072)
    batch_size: int = Field(default=16, ge=1, le=256)
    load_timeout: float = Field(default=600, gt=0, le=3600)
    level: Literal["L0"] = "L0"
    prior: Literal["content_free"] = "content_free"
    adaptive_shifts: Literal[False] = False
    # Official AnyJev option: complete-prompt scoring for caches without batch replication.
    shared_prefix: Literal["auto", False] = "auto"
    max_permutations: None = None
    combine: Literal["logmean"] = "logmean"
    training_free: Literal[True] = True

    @model_validator(mode="after")
    def ordinary_backbone(self):
        if not self.model.strip() or self.model.startswith(("http://", "https://")):
            raise ValueError("AnyJev requires an ordinary HF backbone or local model directory")
        if any(word in self.model.lower() for word in ("tacit", "clef", "decision-head")):
            raise ValueError(
                "AnyJev training-free requires an ordinary backbone, not a decision checkpoint"
            )
        local = self.model.startswith(("/", ".", "~")) or Path(self.model).is_dir()
        if not local and self.revision is None:
            raise ValueError("HF backbone revision must be a pinned 40-character commit")
        return self


class LayaModelConfig(StrictModel):
    adapter: Literal["laya_local"]
    path: str = Field(min_length=1)
    device: str = Field(default="cuda:0", pattern=r"^cuda(?::[0-9]+)?$")
    precision: Literal["auto", "fp16"] = "auto"
    load_timeout: float = Field(default=600, gt=0, le=3600)

    @property
    def model(self):
        return self.path

    @model_validator(mode="after")
    def local_path(self):
        if not self.path.strip() or self.path.startswith(("http://", "https://")):
            raise ValueError("laya_local requires a local Laya checkpoint directory")
        return self


class OpenJevModelConfig(StrictModel):
    adapter: Literal["openjev_local"]
    path: str = Field(min_length=1)
    device: str = Field(default="cuda:0", pattern=r"^cuda(?::[0-9]+)?$")
    precision: Literal["bf16"] = "bf16"
    max_seq_length: int | None = Field(default=None, ge=32)
    max_new_tokens: int | None = Field(default=None, ge=1)
    seed: int = Field(default=3407, ge=0, le=2**32 - 1)
    load_timeout: float = Field(default=600, gt=0, le=3600)

    @property
    def model(self):
        return self.path

    @model_validator(mode="after")
    def local_path(self):
        if not self.path.strip() or self.path.startswith(("http://", "https://")):
            raise ValueError("openjev_local requires a local two-stage OpenJev artifact directory")
        return self


class SwiftModelConfig(StrictModel):
    adapter: Literal["swift_local", "swift_chat_local"]
    path: str = Field(min_length=1)
    model_type: Literal["omnijev", "startlux", "clef", "thx02", "kev", "jev"] = "omnijev"
    base_model: str | None = None
    base_revision: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    adapter_revision: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    usage: Literal["noncommercial_research"] | None = None
    allow_none: bool = False
    device: str = Field(default="cuda:0", pattern=r"^cuda(?::[0-9]+)?$")
    precision: Literal["bf16"] = "bf16"
    max_seq_length: int = Field(default=4096, ge=32)
    calibrated: bool = True
    load_timeout: float = Field(default=600, gt=0, le=3600)

    @model_validator(mode="before")
    @classmethod
    def route_defaults(cls, value):
        if not isinstance(value, dict):
            return value
        from decisionforge.training.backends.swift_routes import get_route

        value = dict(value)
        route = get_route(value.get("model_type", "omnijev"))
        value.setdefault("base_model", route.model)
        value.setdefault("base_revision", route.revision)
        value.setdefault("adapter_revision", route.adapter_revision)
        return value

    @property
    def model(self):
        return self.path

    @model_validator(mode="after")
    def pinned_models(self):
        from decisionforge.training.backends.swift_routes import get_route

        route = get_route(self.model_type)
        if (self.base_model, self.base_revision, self.adapter_revision) != (
            route.model,
            route.revision,
            route.adapter_revision,
        ):
            raise ValueError(f"swift {route.name} requires its pinned model/adapter revisions")
        if route.noncommercial and self.usage != "noncommercial_research":
            raise ValueError("StartLux CC-BY-NC-4.0 requires explicit usage=noncommercial_research")
        if self.allow_none and route.name != "startlux":
            raise ValueError("allow_none is only supported by StartLux")
        if self.adapter == "swift_chat_local" and route.name != "omnijev":
            raise ValueError("The existing Swift Chat baseline is Qwen3.5-4B only")
        if self.adapter == "swift_chat_local" and self.path != self.base_model:
            raise ValueError("swift_chat_local path must be the pinned base model ID")
        return self


class ChatServingConfig(StrictModel):
    enable_thinking: bool | None = None


class ChatModelConfig(ModelConfig):
    adapter: Literal["chat_api"] = "chat_api"
    base_url: str = "https://api.openai.com"
    endpoint_path: Literal["/v1/chat/completions"] = "/v1/chat/completions"
    model: str = Field(min_length=1)
    api_key_env: str | None = "OPENAI_API_KEY"
    max_tokens: int = Field(default=4096, ge=1, le=131072)
    temperature: float = Field(default=0, ge=0, le=2)
    response_format: Literal["text", "json_object", "json_schema"] = "text"
    serving: ChatServingConfig = Field(default_factory=ChatServingConfig)


EvaluationModel = Annotated[
    ModelConfig
    | LocalModelConfig
    | LayaModelConfig
    | OpenJevModelConfig
    | SwiftModelConfig
    | AnyJevModelConfig
    | ChatModelConfig,
    Field(discriminator="adapter"),
]


class ExecutionConfig(StrictModel):
    timeout: float = Field(default=60, gt=0)
    concurrency: int = Field(default=1, ge=1, le=256)
    max_attempts: int = Field(default=1, ge=1, le=20)
    backoff: float = Field(default=0.5, ge=0, le=60)
    max_retry_delay: float = Field(default=60, ge=0, le=3600)


class EvalConfig(StrictModel):
    benchmark: str = "generic"
    benchmark_version: str | None = None
    data: DataConfig
    model: EvaluationModel = Field(default_factory=ModelConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    metrics: dict[str, dict[str, JsonValue]] = Field(default_factory=dict)
    output: str = "outputs/eval"

    @model_validator(mode="before")
    @classmethod
    def legacy_http_adapter(cls, value):
        if isinstance(value, dict) and isinstance(value.get("model"), dict):
            value = {**value, "model": {"adapter": "jev", **value["model"]}}
        return value

    @model_validator(mode="after")
    def local_execution(self):
        if isinstance(
            self.model,
            (
                LocalModelConfig,
                LayaModelConfig,
                OpenJevModelConfig,
                SwiftModelConfig,
                AnyJevModelConfig,
            ),
        ) and (self.execution.concurrency != 1 or self.execution.max_attempts != 1):
            raise ValueError(f"{self.model.adapter} requires concurrency=1 and max_attempts=1")
        return self


def load_config(path: str | Path) -> EvalConfig:
    path = Path(path).resolve()
    config = EvalConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    for owner, name in ((config.data, "path"), (config, "output")):
        value = Path(getattr(owner, name)).expanduser()
        setattr(
            owner, name, str((path.parent / value).resolve() if not value.is_absolute() else value)
        )
    swift_published = False
    if isinstance(config.model, SwiftModelConfig) and config.model.model_type != "omnijev":
        from decisionforge.training.backends.swift_routes import get_route
        from decisionforge.training.backends.swift_routes.artifacts import published

        swift_published = published(config.model.path, get_route(config.model.model_type))
    if (
        isinstance(
            config.model, (LocalModelConfig, LayaModelConfig, OpenJevModelConfig, SwiftModelConfig)
        )
        and config.model.adapter != "swift_chat_local"
        and not swift_published
    ):
        checkpoint = Path(config.model.path).expanduser()
        config.model.path = str(
            (path.parent / checkpoint).resolve() if not checkpoint.is_absolute() else checkpoint
        )
    if isinstance(config.model, AnyJevModelConfig) and config.model.model.startswith(
        ("/", ".", "~")
    ):
        backbone = Path(config.model.model).expanduser()
        config.model.model = str(
            (path.parent / backbone).resolve() if not backbone.is_absolute() else backbone
        )
    return config
