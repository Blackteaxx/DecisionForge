from __future__ import annotations

import math
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_serializer,
    model_validator,
)

from decisionforge.media import MediaRequest
from decisionforge.util import json_text

Mode = Literal["choice", "noul", "score"]
JEVContent = str | dict[str, JsonValue] | list[JsonValue]


def content_texts(value) -> list[str]:
    """Text leaves of a native JEV value, without treating field names as facts."""
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, dict):
        return [text for item in value.values() for text in content_texts(item)]
    if isinstance(value, list):
        return [text for item in value for text in content_texts(item)]
    return []


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class MappingSpec(StrictModel):
    schema_version: Literal["mapping.v1"] = "mapping.v1"
    question: str
    answer: str
    context: str | None = None
    options: str | None = None
    mode: str | None = None
    statement: str | None = None
    noul_label: str | None = None
    ranking_items: str | None = None
    score_index: str | None = None
    original_id: str | None = None
    group_id: str | None = None
    split: str | None = None
    answer_encoding: Literal["text", "key", "index0", "index1"] = "text"
    label_map: dict[str, str | int] = Field(default_factory=dict)


class RawRecord(StrictModel):
    source_record_id: str
    payload: dict[str, Any]
    source: str
    row: int


class JEVQuestion(StrictModel):
    type: Mode
    instructions: JEVContent
    criteria: dict[str, JEVContent | None] | list[JEVContent] | None = None

    @field_validator("instructions", "criteria")
    @classmethod
    def finite_content(cls, value):
        json_text(value)  # Reject NaN/Infinity anywhere inside nested JSON.
        return value

    @model_serializer(mode="wrap")
    def serialize_question(self, handler):
        data = handler(self)
        if self.criteria is None:
            data.pop("criteria", None)
        return data

    @model_validator(mode="after")
    def validate_criteria(self):
        if not content_texts(self.instructions):
            raise ValueError("Instructions require nonempty judgment text")
        if self.type == "choice":
            if not isinstance(self.criteria, dict) or not 2 <= len(self.criteria) <= 255:
                raise ValueError("Choice requires 2–255 options")
            if any(not key.strip() for key in self.criteria):
                raise ValueError("Choice option keys must be nonempty")
            descriptions = [
                json_text(value if value is not None else key).strip().casefold()
                for key, value in self.criteria.items()
            ]
            if len(set(descriptions)) != len(self.criteria):
                raise ValueError("Duplicate Choice content")
            if any(
                value is not None and not content_texts(value) for value in self.criteria.values()
            ):
                raise ValueError("Choice descriptions must contain text or be null")
        elif self.type == "score":
            if not isinstance(self.criteria, list) or not 2 <= len(self.criteria) <= 10:
                raise ValueError("Score requires 2–10 ordered levels")
            if any(not content_texts(level) for level in self.criteria):
                raise ValueError("Score levels require descriptive text")
            if len({json_text(level).casefold() for level in self.criteria}) != len(self.criteria):
                raise ValueError("Duplicate Score levels")
        elif self.criteria is not None and (
            not isinstance(self.criteria, dict) or not set(self.criteria) <= {"true", "false"}
        ):
            raise ValueError("Noul criteria may only describe true and false")
        return self


class JEVRequest(MediaRequest, StrictModel):
    model: str
    state: JEVContent
    questions: dict[str, JEVQuestion] = Field(min_length=1)

    @field_validator("state")
    @classmethod
    def finite_state(cls, value):
        json_text(value)
        return value


class Supervision(StrictModel):
    kind: Literal["choice", "binary", "ordinal_level"]
    value: str | int
    origin: Literal["original_gt", "derived", "synthetic"]
    evidence_ids: list[str] = Field(min_length=1)
    candidate_id: str | None = None
    rubric_id: str | None = None


PRIVATE_KEYS = {
    "answer",
    "label",
    "labels",
    "target",
    "correct_option",
    "correct_answer",
    "reference_answer",
    "private_supervision",
    "supervision",
    "rationale",
    "why_not_correct",
    "verification",
    "audit",
    "metadata",
}


def assert_public(value: Any) -> None:
    if isinstance(value, dict):
        if any(k.casefold() in PRIVATE_KEYS for k in value):
            raise ValueError("Private supervision field in public state")
        for item in value.values():
            assert_public(item)
    elif isinstance(value, list):
        for item in value:
            assert_public(item)


class TrainingRecord(StrictModel):
    schema_version: Literal["decisionforge.training.v1"] = "decisionforge.training.v1"
    record_id: str
    group_id: str
    request: JEVRequest
    supervision: dict[str, Supervision]
    metadata: dict[str, Any]

    @model_validator(mode="after")
    def check_labels(self):
        assert_public(self.request.state)
        if set(self.request.questions) != set(self.supervision):
            raise ValueError("Questions and supervision must match")
        for name, question in self.request.questions.items():
            target = self.supervision[name]
            if question.type == "choice":
                if target.kind != "choice" or target.value not in question.criteria:
                    raise ValueError("Choice label does not refer to an option")
            elif question.type == "noul":
                if (
                    target.kind != "binary"
                    or type(target.value) is not int
                    or target.value not in (0, 1)
                ):
                    raise ValueError("Noul requires a known binary integer label")
            elif (
                target.kind != "ordinal_level"
                or type(target.value) is not int
                or not 0 <= target.value < len(question.criteria)
                or not target.rubric_id
            ):
                raise ValueError("Score level/rubric invalid")
        if not self.metadata.get("split"):
            raise ValueError("Missing split")
        return self


class LegacyTrainingExample(StrictModel):
    """Compact training export; detailed provenance stays in the checkpoint."""

    schema_version: Literal["decisionforge.training.v2"] = "decisionforge.training.v2"
    record_id: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    split: str = Field(min_length=1)
    request: JEVRequest
    supervision: dict[str, str | int]

    @classmethod
    def from_record(cls, record: TrainingRecord) -> LegacyTrainingExample:
        mapping = record.metadata.get("candidate_to_option", {})
        for target in record.supervision.values():
            if target.kind == "choice" and mapping.get(target.candidate_id) != target.value:
                raise ValueError("Candidate-to-option mapping disagrees with supervision")
        return cls(
            record_id=record.record_id,
            group_id=record.group_id,
            split=record.metadata["split"],
            request=record.request,
            supervision={name: target.value for name, target in record.supervision.items()},
        )

    @model_validator(mode="after")
    def check_labels(self):
        assert_public(self.request.state)
        if set(self.request.questions) != set(self.supervision):
            raise ValueError("Questions and supervision must match")
        for name, question in self.request.questions.items():
            value = self.supervision[name]
            if question.type == "choice":
                if not isinstance(value, str) or value not in question.criteria:
                    raise ValueError("Choice label does not refer to an option")
            elif question.type == "noul":
                if type(value) is not int or value not in (0, 1):
                    raise ValueError("Noul requires a known binary integer label")
            elif type(value) is not int or not 0 <= value < len(question.criteria):
                raise ValueError("Score level out of range")
        return self


class JEVInput(MediaRequest, StrictModel):
    state: JEVContent
    questions: dict[str, JEVQuestion] = Field(min_length=1)

    @field_validator("state")
    @classmethod
    def finite_state(cls, value):
        json_text(value)
        return value


Probability = Annotated[float, Field(ge=0, le=1)]


class ChoiceAnswer(StrictModel):
    type: Literal["choice"] = "choice"
    choice: str
    confidence: Probability
    probabilities: dict[str, Probability]


class ScoreAnswer(StrictModel):
    type: Literal["score"] = "score"
    score: float = Field(ge=0)
    confidence: Probability
    legend: dict[str, JEVContent]
    probabilities: dict[str, Probability]


class NoulAnswer(StrictModel):
    type: Literal["noul"] = "noul"
    noul: Probability


class JEVAnswers(StrictModel):
    answers: dict[
        str, Annotated[ChoiceAnswer | ScoreAnswer | NoulAnswer, Field(discriminator="type")]
    ] = Field(min_length=1)


class TrainingExample(StrictModel):
    """Model-independent JEV input and answer-shaped training target."""

    schema_version: Literal["decisionforge.training.v3"] = "decisionforge.training.v3"
    record_id: int = Field(ge=1)
    group_id: int = Field(ge=1)
    split: str = Field(min_length=1)
    request: JEVInput
    supervision: JEVAnswers

    @classmethod
    def from_record(
        cls, record: TrainingRecord, *, record_id: int, group_id: int
    ) -> TrainingExample:
        example = cls.from_legacy(
            LegacyTrainingExample.from_record(record), record_id=record_id, group_id=group_id
        )
        if record.metadata.get("generation_version") == "v2":
            # New generation preserves source wording even when it mentions bookkeeping names.
            for index, question in enumerate(record.request.questions.values(), 1):
                example.request.questions[f"q{index}"].instructions = question.instructions
        if "model_answers" in record.metadata:
            value = example.model_dump()
            value["supervision"] = record.metadata["model_answers"]
            return cls.model_validate(value)
        return example

    @classmethod
    def from_legacy(
        cls, record: LegacyTrainingExample, *, record_id: int, group_id: int
    ) -> TrainingExample:
        from decisionforge.util import clean_instructions

        questions, answers = {}, {}
        for index, (name, question) in enumerate(record.request.questions.items(), 1):
            key, value = f"q{index}", record.supervision[name]
            question = question.model_copy(deep=True)
            question.instructions = clean_instructions(
                question.instructions, [record.record_id, record.group_id, name]
            )
            questions[key] = question
            # These are hard-label training targets, not measured model confidence.
            if question.type == "choice":
                answers[key] = ChoiceAnswer(
                    choice=value,
                    confidence=1.0,
                    probabilities={k: float(k == value) for k in question.criteria},
                )
            elif question.type == "noul":
                answers[key] = NoulAnswer(noul=float(value))
            else:
                answers[key] = ScoreAnswer(
                    score=float(value),
                    confidence=1.0,
                    legend={str(i): level for i, level in enumerate(question.criteria)},
                    probabilities={
                        str(i): float(i == value) for i in range(len(question.criteria))
                    },
                )
        return cls(
            record_id=record_id,
            group_id=group_id,
            split=record.split,
            request=JEVInput(state=record.request.state, questions=questions),
            supervision=JEVAnswers(answers=answers),
        )

    @model_validator(mode="after")
    def check_answers(self):
        assert_public(self.request.state)
        if set(self.request.questions) != set(self.supervision.answers):
            raise ValueError("Questions and answers must match")
        for name, question in self.request.questions.items():
            answer = self.supervision.answers[name]
            if answer.type != question.type:
                raise ValueError("Answer type differs from question type")
            if isinstance(answer, NoulAnswer):
                continue
            expected = (
                set(question.criteria)
                if question.type == "choice"
                else {str(i) for i in range(len(question.criteria))}
            )
            if set(answer.probabilities) != expected or not math.isclose(
                sum(answer.probabilities.values()), 1.0, abs_tol=1e-6
            ):
                raise ValueError("Probabilities must cover the criteria and sum to one")
            if isinstance(answer, ChoiceAnswer):
                if answer.choice not in expected or answer.probabilities[answer.choice] != max(
                    answer.probabilities.values()
                ):
                    raise ValueError("Choice must select an option with maximum probability")
            else:
                if answer.legend != {str(i): level for i, level in enumerate(question.criteria)}:
                    raise ValueError("Score legend must match ordered criteria")
                mean = sum(int(k) * p for k, p in answer.probabilities.items())
                if not math.isclose(answer.score, mean, abs_tol=1e-6):
                    raise ValueError("Score must equal the probability-weighted level")
            from decisionforge.jevmath import choice_confidence, score_confidence

            confidence = (
                choice_confidence(list(answer.probabilities.values()))
                if isinstance(answer, ChoiceAnswer)
                else score_confidence(
                    [answer.probabilities[str(i)] for i in range(len(question.criteria))]
                )
            )
            # Official examples/older exports can round confidence to two decimal places.
            if not math.isclose(answer.confidence, confidence, rel_tol=0, abs_tol=0.01):
                raise ValueError("Confidence must agree with the JEV probability formula")
        return self
