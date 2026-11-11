from typing import Literal

from pydantic import Field, model_validator

from decisionforge.models import Mode, StrictModel


class TextOrigin(StrictModel):
    """Evidence and reasons for a QA constructed from narrative TXT."""

    evidence: list[str] = Field(min_length=1)
    assessment_reason: str = Field(min_length=1)
    construction_reason: str = Field(min_length=1)


class StandardRecord(StrictModel):
    """Normalized QA; text_origin identifies questions constructed from narrative TXT."""

    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    context: str = ""
    mode: Mode | None = None
    options: list[str] = Field(default_factory=list)
    statement: str | None = None
    noul_label: Literal[0, 1] | None = None
    ranking_items: list[str] = Field(default_factory=list)
    score_index: int | None = Field(default=None, ge=0)
    split: str = "train"
    group_id: str | None = None
    text_origin: TextOrigin | None = None

    @model_validator(mode="after")
    def nonempty(self):
        if not self.question.strip() or not self.answer.strip() or not self.split.strip():
            raise ValueError("question, answer and split must not be blank")
        if self.text_origin is not None and self.mode is None:
            raise ValueError("Constructed text QA requires an assessed mode")
        return self


class ExtractedRecord(StrictModel):
    # Exact spans are enforced in code: this stage cannot synthesize missing parts.
    question: str
    answer: str
    context: str
    mode: Mode | None = None
    options: list[str] = Field(default_factory=list)
    statement: str | None = None
    noul_label: Literal[0, 1] | None = None
    ranking_items: list[str] = Field(default_factory=list)
    score_index: int | None = None


class Extraction(StrictModel):
    records: list[ExtractedRecord]
    reason: str


class TextOpportunity(StrictModel):
    mode: Mode
    focus: str = Field(min_length=1)
    evidence: list[str] = Field(min_length=1)
    reason: str = Field(min_length=1)


class TextAssessment(StrictModel):
    is_narrative: bool
    opportunities: list[TextOpportunity]
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def narrative_only(self):
        if not self.is_narrative and self.opportunities:
            raise ValueError("Only narrative material can supply question opportunities")
        return self


class ConstructedQA(StrictModel):
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)


class QAConstruction(StrictModel):
    qa: ConstructedQA | None
    reason: str = Field(min_length=1)


class BatchQuestionQuota(StrictModel):
    mode: Mode
    count: int = Field(ge=1, le=50, strict=True)


class TextBatchSpec(StrictModel):
    # One-based inclusive unit endpoint. Starts follow the previous endpoint.
    end_unit: int = Field(ge=1, strict=True)
    questions: list[BatchQuestionQuota]
    reason: str = Field(min_length=1)


class TextWindowPlan(StrictModel):
    reason: str = Field(min_length=1)
    batches: list[TextBatchSpec] = Field(min_length=1)


class TextBatchPlan(StrictModel):
    should_split: bool
    reason: str = Field(min_length=1)
    batches: list[TextBatchSpec]

    @model_validator(mode="after")
    def consistent(self):
        if self.should_split and len(self.batches) < 2:
            raise ValueError("A split plan must contain at least two batches")
        if not self.should_split and self.batches:
            raise ValueError("A no-split decision must have no batches")
        return self


class TextBatchContext(StrictModel):
    schema_version: Literal["decisionforge.text-batch.v1"] = "decisionforge.text-batch.v1"
    original_source: str
    original_sha256: str
    group_id: str
    batch_index: int = Field(ge=1)
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    text_sha256: str
    questions: list[BatchQuestionQuota]
    reason: str


class ContextOutput(StrictModel):
    context: str


class ChoiceOutput(StrictModel):
    distractors: list[str]


class NoulOutput(StrictModel):
    statement: str


class ScoreOutput(StrictModel):
    ranking_items: list[str]
    score_index: int


class CheckReport(StrictModel):
    passed: bool
    reason: str
    failed_fields: list[str]
    repair_instructions: list[str]
    modules: list[Literal["context", "choice", "noul", "score", "source"]]

    @model_validator(mode="after")
    def consistent(self):
        if not self.reason.strip():
            raise ValueError("Checker must explain its verdict")
        if self.passed and (self.failed_fields or self.modules or self.repair_instructions):
            raise ValueError("Passing checker report contains unresolved issues")
        if not self.passed and (
            not self.failed_fields or not self.modules or not self.repair_instructions
        ):
            raise ValueError(
                "Failed checker report requires fields, repair instructions and responsible modules"
            )
        return self
