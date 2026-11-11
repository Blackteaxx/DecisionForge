from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProjectConfig(ConfigModel):
    name: str = "decisionforge"
    schema_version: Literal["decisionforge.training.v1"] = "decisionforge.training.v1"
    language: str = "preserve_source"


class TextBatchingConfig(ConfigModel):
    enabled: bool = True
    threshold_chars: int = Field(default=10000, ge=100)
    max_batches: int = Field(default=64, ge=2, le=4096)


class InputConfig(ConfigModel):
    type: Literal["auto", "json", "txt", "pdf"] = "auto"
    path: str = ""
    paths: list[str] = Field(default_factory=list)
    records_path: str | None = None
    mapping_spec: str | None = None
    extract_context: bool = True
    generate_qa_from_text: bool = True
    max_generated_questions: int = Field(default=5, ge=1, le=50)
    text_batching: TextBatchingConfig = Field(default_factory=TextBatchingConfig)
    preserve_source_split: bool = True
    default_split: str = "train"

    @model_validator(mode="after")
    def check_paths(self):
        if self.paths:
            if self.path:
                raise ValueError("Use either input.path or input.paths, not both")
            if self.type not in {"auto", "pdf"}:
                raise ValueError("input.paths is supported only for PDF input")
            if any(not p.strip() for p in self.paths):
                raise ValueError("PDF paths must not be empty")
        return self


class ContextConfig(ConfigModel):
    generate_if_missing: bool = False
    relevance: float = Field(default=0.5, ge=0, le=1, strict=True, allow_inf_nan=False)


class TaskConfig(ConfigModel):
    domain_profile: str = "general_qa"
    modes: list[Literal["auto", "choice", "noul", "score"]] = ["auto"]
    difficulty: float = Field(
        default=0.5,
        ge=0,
        le=1,
        strict=True,
        allow_inf_nan=False,
        description="Target difficulty of generated distractors and erroneous content; 0 easy, 1 hard.",
    )

    @model_validator(mode="after")
    def check_modes(self):
        if len(set(self.modes)) != len(self.modes):
            raise ValueError("Modes must be unique")
        if not self.modes or ("auto" in self.modes and len(self.modes) != 1):
            raise ValueError("Use [auto] or explicit modes")
        return self


class Capabilities(ConfigModel):
    structured_outputs: bool = True
    temperature: bool = False
    reasoning_effort: bool = False


class Pricing(ConfigModel):
    input_per_million: float = Field(ge=0)
    output_per_million: float = Field(ge=0)


LLM_ROLES = (
    "text_batch_planner",
    "extractor",
    "text_assessor",
    "qa_constructor",
    "context",
    "choice",
    "noul",
    "score",
    "checker",
)
ModuleRole = Literal[
    "analyzer",
    "generator",
    "verifier",
    "text_batch_planner",
    "extractor",
    "text_assessor",
    "qa_constructor",
    "context",
    "choice",
    "noul",
    "score",
    "checker",
]


class LLMOverride(ConfigModel):
    """Unset fields inherit the global configuration; secrets stay in the environment."""

    model: str | None = Field(default=None, min_length=1)
    base_url: str | None = None
    api_key_env: str | None = Field(default=None, min_length=1)
    api: Literal["responses", "chat_completions"] | None = None
    structured_output: Literal["strict", "json", "prompt"] | None = None
    chat_token_limit_param: Literal["max_completion_tokens", "max_tokens"] | None = None
    capabilities: Capabilities | None = None
    temperature: float | None = None
    reasoning_effort: str | None = None
    max_output_tokens: int | None = Field(default=None, ge=256)
    timeout_seconds: float | None = Field(default=None, gt=0)


class MinerUConfig(ConfigModel):
    base_url: str = "https://mineru.net/api/v4"
    api_key_env: str = Field(default="MINERU_API_KEY", min_length=1)
    model_version: Literal["pipeline", "vlm"] = "vlm"
    language: str = "ch"
    is_ocr: bool = False
    enable_formula: bool = True
    enable_table: bool = True
    batch_size: int = Field(default=50, ge=1, le=50)
    timeout_seconds: float = Field(default=120, gt=0)
    poll_interval_seconds: float = Field(default=5, gt=0)
    max_wait_seconds: float = Field(default=1800, gt=0)

    @model_validator(mode="after")
    def check_url(self):
        parsed = urlparse(self.base_url)
        if (
            parsed.scheme not in {"https", "http"}
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "MinerU base_url must be an HTTP(S) URL without credentials/query/fragment"
            )
        return self


class LLMConfig(ConfigModel):
    provider: Literal["openai"] = "openai"
    api: Literal["responses", "chat_completions"] = "responses"
    model: str = "gpt-4.1-mini"
    analyzer_model: str | None = None
    generator_model: str | None = None
    verifier_model: str | None = None
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    structured_output: Literal["strict", "json", "prompt"] = "json"
    chat_token_limit_param: Literal["max_completion_tokens", "max_tokens"] = "max_completion_tokens"
    capabilities: Capabilities = Field(default_factory=Capabilities)
    temperature: float | None = None
    reasoning_effort: str | None = None
    max_output_tokens: int = Field(default=6000, ge=256)
    max_input_chars: int = Field(default=100000, ge=1000)
    timeout_seconds: float = Field(default=90, gt=0)
    pricing: dict[str, Pricing] = Field(default_factory=dict)
    modules: dict[ModuleRole, LLMOverride] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def environment_defaults(cls, value):
        if isinstance(value, dict):
            value = dict(value)
            for field, variable in (
                ("api", "OPENAI_API"),
                ("structured_output", "OPENAI_STRUCTURED_OUTPUT"),
                ("chat_token_limit_param", "OPENAI_CHAT_TOKEN_LIMIT_PARAM"),
            ):
                if field not in value and os.getenv(variable):
                    value[field] = os.environ[variable]
        return value

    @model_validator(mode="after")
    def check_capabilities(self):
        if self.structured_output == "strict" and not self.capabilities.structured_outputs:
            raise ValueError("strict output requires declared structured_outputs capability")
        for key in ("temperature", "reasoning_effort"):
            if getattr(self, key) is not None and not getattr(self.capabilities, key):
                raise ValueError(f"Unsupported capability: {key}")
        if self.base_url:
            parsed = urlparse(self.base_url)
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError("base_url cannot contain credentials/query/fragment")
        if self.modules:
            for role in LLM_ROLES:
                self.for_role(role)
        return self

    @staticmethod
    def group_for(role: str) -> str:
        return (
            "verifier"
            if role == "checker" or role.startswith("verifier")
            else (
                "analyzer"
                if role
                in {
                    "analyzer",
                    "extractor",
                    "text_assessor",
                    "text_batch_planner",
                }
                else "generator"
            )
        )

    def for_role(self, role: str) -> LLMConfig:
        group = self.group_for(role)
        data = self.model_dump(exclude={"modules"})
        data["model"] = getattr(self, f"{group}_model") or self.model
        for name in dict.fromkeys((group, role)):
            override = self.modules.get(name)
            if override:
                data.update(override.model_dump(exclude_none=True))
        return LLMConfig.model_validate(data)

    def model_for(self, role: str) -> str:
        return self.for_role(role).model


class ChoiceConfig(ConfigModel):
    target_option_count: int = Field(default=4, ge=2, le=255)


class ScoreConfig(ConfigModel):
    default_levels: int = Field(default=5, ge=2, le=10)


class NoulConfig(ConfigModel):
    labels: list[Literal[0, 1]] = [1, 0]

    @model_validator(mode="after")
    def check_labels(self):
        if not self.labels or len(set(self.labels)) != len(self.labels):
            raise ValueError("noul.labels must contain unique binary labels")
        return self


class VerificationConfig(ConfigModel):
    max_semantic_repairs: int = Field(default=2, ge=0, le=5)


class ExecutionConfig(ConfigModel):
    text_batch_concurrency: int = Field(default=4, ge=1, le=64)
    concurrency: int = Field(default=4, ge=1, le=64)
    llm_concurrency: int = Field(default=8, ge=1, le=128)
    max_transport_attempts: int = Field(default=3, ge=1, le=10)
    max_llm_calls_per_record: int = Field(default=24, ge=1)
    budget_limit_usd: float | None = Field(default=None, ge=0)
    on_budget_exhausted: Literal["checkpoint_and_stop"] = "checkpoint_and_stop"
    cache: bool = True
    resume: bool = True
    seed: int = 42


class OutputConfig(ConfigModel):
    directory: str = "outputs/run_001"
    include_audit: bool = False
    export_requests: bool = False
    jev_model: str = "jev-latest"
    validation_fraction: float = Field(default=0.0, ge=0, lt=1, strict=True, allow_inf_nan=False)
    validation_seed: int = Field(default=42, ge=0, strict=True)


class Config(ConfigModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    input: InputConfig
    task: TaskConfig = Field(default_factory=TaskConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    mineru: MinerUConfig = Field(default_factory=MinerUConfig)
    choice: ChoiceConfig = Field(default_factory=ChoiceConfig)
    score: ScoreConfig = Field(default_factory=ScoreConfig)
    noul: NoulConfig = Field(default_factory=NoulConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @model_validator(mode="before")
    @classmethod
    def reject_removed_search(cls, value):
        if isinstance(value, dict) and (
            "deep_search" in value
            or isinstance(value.get("input"), dict)
            and value["input"].get("type") == "deep_search"
        ):
            raise ValueError(
                "Deep Search has been removed; provide local JSON, TXT or PDF input instead"
            )
        return value

    @model_validator(mode="after")
    def check_budget(self):
        if self.execution.budget_limit_usd is not None:
            for role in LLM_ROLES:
                if self.llm.model_for(role) not in self.llm.pricing:
                    raise ValueError("A USD budget requires explicit pricing for every model")
        return self


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    raw = path.read_text(encoding="utf-8")

    def replace(match):
        key = match.group(1)
        if key not in os.environ or not os.environ[key]:
            raise ValueError(f"Environment variable {key} is required by configuration")
        return os.environ[key]

    data = yaml.safe_load(re.sub(r"\$\{([A-Z_][A-Z0-9_]*)\}", replace, raw))
    config = Config.model_validate(data)
    # All configured paths are relative to the YAML file, not process cwd.
    for obj, key in (
        (config.input, "path"),
        (config.input, "mapping_spec"),
        (config.output, "directory"),
    ):
        value = getattr(obj, key)
        if value:
            setattr(obj, key, str((path.parent / value).resolve()))
    config.input.paths = [str((path.parent / p).resolve()) for p in config.input.paths]
    if config.input.type == "auto":
        from decisionforge.adapters import input_kind

        config.input.type = "pdf" if config.input.paths else input_kind(Path(config.input.path))
    return config
