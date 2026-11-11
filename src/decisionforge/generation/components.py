"""Only generate missing/failed components; compile locally to the shared JEV contract."""

from __future__ import annotations

import random
import re

from decisionforge.generation.contracts import ChoiceOutput, ContextOutput, NoulOutput, ScoreOutput
from decisionforge.models import JEVQuestion, JEVRequest, Supervision, TrainingRecord
from decisionforge.util import NeedsReview, digest


def same_text(left, right):
    return re.sub(r"[\W_]+", "", left).casefold() == re.sub(r"[\W_]+", "", right).casefold()


def choice_label(index):
    """Zero-based option index to A, B, ..., Z, AA, AB, ... ."""
    number, label = index + 1, ""
    while number:
        number, remainder = divmod(number - 1, 26)
        label = chr(ord("A") + remainder) + label
    return label


def initial_draft(source, mode, label):
    return {
        "context": source.context,
        "options": list(source.options),
        "statement": source.statement if source.noul_label == label else None,
        "noul_label": label,
        "ranking_items": list(source.ranking_items),
        "score_index": source.score_index,
    }


def missing_modules(source, draft, mode, config):
    missing = []
    if config.context.generate_if_missing and not draft["context"].strip():
        missing.append("context")
    if mode == "choice" and not any(o != source.answer for o in draft["options"]):
        missing.append("choice")
    if mode == "noul" and not draft["statement"]:
        missing.append("noul")
    if mode == "score" and len(draft["ranking_items"]) < 2:
        missing.append("score")
    return missing


async def generate(module, source, draft, config, call, record, stage, feedback, current_jev):
    payload = {
        "source": source.model_dump(),
        "draft": draft,
        "difficulty": config.task.difficulty,
        "relevance": config.context.relevance,
        "domain": config.task.domain_profile,
        "feedback": feedback,
        "current_jev": current_jev,
    }
    if module == "context":
        if not config.context.generate_if_missing:
            raise NeedsReview("Context generation is disabled; source context requires correction")
        result = await call(module, payload, ContextOutput, record, stage)
        draft["context"] = result.context
    elif module == "choice":
        # Existing source distractors survive initial augmentation, but may be replaced on repair.
        existing = [] if feedback else [o for o in draft["options"] if o != source.answer]
        payload.update(
            existing_options=existing + [source.answer],
            count=max(1, config.choice.target_option_count - 1 - len(existing)),
        )
        result = await call(module, payload, ChoiceOutput, record, stage)
        draft["options"] = existing + result.distractors + [source.answer]
    elif module == "noul":
        payload["requested_label"] = draft["noul_label"]
        result = await call(module, payload, NoulOutput, record, stage)
        draft["statement"] = result.statement
    elif module == "score":
        payload["target_count"] = max(config.score.default_levels, len(source.ranking_items))
        result = await call(module, payload, ScoreOutput, record, stage)
        if any(item not in result.ranking_items for item in source.ranking_items):
            raise NeedsReview("Score generator removed source ranking items")
        draft["ranking_items"], draft["score_index"] = result.ranking_items, result.score_index
    else:
        raise NeedsReview("Immutable source question/answer requires correction")


def compile_record(source, draft, mode, source_id, child_id, config):
    context = draft["context"]
    if context.strip() and same_text(context, source.question):
        raise NeedsReview("context must not equal question")
    if config.context.generate_if_missing and not context.strip():
        raise NeedsReview("context is required by configuration")
    instructions = source.question
    mapping = {}
    if mode == "choice":
        options = list(draft["options"])
        if source.answer not in options:
            options.append(source.answer)
        if len({o.strip().casefold() for o in options}) != len(options):
            raise NeedsReview("Duplicate choice options")
        if not 2 <= len(options) <= 255:
            raise NeedsReview("choice requires at least one incorrect distractor")
        random.Random(digest([config.execution.seed, child_id])).shuffle(options)
        criteria = {choice_label(i): value for i, value in enumerate(options)}
        correct = next(key for key, value in criteria.items() if value == source.answer)
        mapping[correct] = correct
        target = Supervision(
            kind="choice",
            value=correct,
            origin="derived",
            evidence_ids=[source_id],
            candidate_id=correct,
        )
    elif mode == "noul":
        if not draft["statement"] or not draft["statement"].strip():
            raise NeedsReview("noul requires a nonempty statement")
        instructions = {"question": source.question, "statement": draft["statement"]}
        criteria = None
        target = Supervision(
            kind="binary", value=draft["noul_label"], origin="derived", evidence_ids=[source_id]
        )
    else:
        criteria = draft["ranking_items"]
        index = draft["score_index"]
        if index is None and criteria.count(source.answer) == 1:
            index = criteria.index(source.answer)
        if (
            type(index) is not int
            or not 0 <= index < len(criteria)
            or criteria[index] != source.answer
        ):
            raise NeedsReview("score_index must point to the unchanged reference answer")
        target = Supervision(
            kind="ordinal_level",
            value=index,
            origin="derived",
            evidence_ids=[source_id],
            rubric_id="source_order",
        )
    question = JEVQuestion(type=mode, instructions=instructions, criteria=criteria)
    return TrainingRecord(
        record_id=child_id,
        group_id=source.group_id or source_id,
        request=JEVRequest(
            model=config.output.jev_model, state=context, questions={"q1": question}
        ),
        supervision={"q1": target},
        metadata={
            "source_record_id": source_id,
            "split": source.split,
            "candidate_to_option": mapping,
            "generation_version": "v2",
        },
    )
