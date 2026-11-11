"""Lossless extraction, with evidence-grounded question construction for narrative TXT."""

from __future__ import annotations

from collections import Counter

from decisionforge.adapters import field
from decisionforge.generation.contracts import (
    Extraction,
    QAConstruction,
    StandardRecord,
    TextAssessment,
    TextOrigin,
)
from decisionforge.util import NeedsReview, json_text, map_bounded

ALIASES = {
    "question": {"question", "query", "prompt", "q", "问题", "题目", "问"},
    "answer": {
        "answer",
        "response",
        "reference_answer",
        "correct_answer",
        "output",
        "a",
        "答案",
        "回答",
        "答",
    },
    "context": {
        "context",
        "passage",
        "background",
        "article",
        "document",
        "上下文",
        "背景",
        "材料",
    },
    "options": {"options", "choices", "candidates", "选项"},
    "ranking_items": {"ranking_items", "levels", "ranking", "ordered_items", "排序项"},
    "statement": {"statement", "claim", "陈述"},
    "noul_label": {"noul_label", "truth", "判断标签"},
    "score_index": {"score_index", "score_level", "排序标签"},
    "mode": {"mode", "type", "题型"},
    "split": {"split"},
    "group_id": {"group_id", "id"},
}


def lookup(payload, name):
    if name in payload:
        return payload[name]
    direct = [value for key, value in payload.items() if key.casefold() in ALIASES[name]]
    if len(direct) == 1:
        return direct[0]
    if len(direct) > 1:
        raise NeedsReview(f"Ambiguous {name} fields; specify mapping_spec")
    matches = []
    # An option key "A" is never a reference answer; context has its own semantic scope.
    opaque = ALIASES["options"] | ALIASES["ranking_items"] | ALIASES["context"]

    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key.casefold() in ALIASES[name]:
                    matches.append(item)
                elif isinstance(item, dict) and key.casefold() not in opaque:
                    walk(item)

    walk(payload)
    if len(matches) > 1:
        raise NeedsReview(f"Ambiguous {name} fields; specify mapping_spec")
    return matches[0] if matches else None


def text(value):
    return value if isinstance(value, str) else json_text(value)


def normalize(payload, config, mapping=None):
    mapping = mapping or {}
    values = {}
    for name in ALIASES:
        if name == "context" and not config.input.extract_context:
            continue
        path = mapping.get(name) or (mapping.get("original_id") if name == "group_id" else None)
        try:
            value = field(payload, path) if path else lookup(payload, name)
        except NeedsReview:
            if not path or name in {"question", "answer"}:
                if name != "context":
                    raise
            value = None
        if (
            name == "mode"
            and not path
            and "mode" not in payload
            and (not isinstance(value, str) or value not in {"choice", "noul", "score"})
        ):
            continue
        if value is not None:
            values[name] = value
    missing = [
        k
        for k in ("question", "answer")
        if k not in values
        or values[k] is None
        or isinstance(values[k], str)
        and not values[k].strip()
    ]
    if missing:
        raise NeedsReview("Missing required source fields: " + ", ".join(missing))
    options = values.get("options", [])
    answer = values["answer"]
    encoding = mapping.get("answer_encoding", "text")
    if str(answer) in mapping.get("label_map", {}):
        answer = mapping["label_map"][str(answer)]
    if encoding == "key" or (
        "answer_encoding" not in mapping and isinstance(options, dict) and str(answer) in options
    ):
        if not isinstance(options, dict) or str(answer) not in options:
            raise NeedsReview("Answer key is absent from options")
        answer = options[str(answer)]
    elif encoding in {"index0", "index1"}:
        if type(answer) is not int or not isinstance(options, list):
            raise NeedsReview("Indexed answer requires integer and list options")
        index = answer - int(encoding == "index1")
        if not 0 <= index < len(options):
            raise NeedsReview("Answer index out of range")
        answer = options[index]
    values["answer"] = text(answer)
    for key in ("options", "ranking_items"):
        value = values.get(key, [])
        if isinstance(value, dict) and key == "options":
            value = list(value.values())
        if not isinstance(value, list) or any(
            not isinstance(v, str) or not v.strip() for v in value
        ):
            raise NeedsReview(f"{key} must contain nonempty strings")
        values[key] = value
    if "context" in values and not isinstance(values["context"], str):
        values["context"] = text(values["context"])
    if "group_id" in values:
        values["group_id"] = text(values["group_id"])
    if not config.input.preserve_source_split or "split" not in values:
        values["split"] = config.input.default_split
    return StandardRecord.model_validate(values)


async def extract(raw, config, call, mapping=None, batch_question_plan=None):
    if config.input.type == "json":
        try:
            return [normalize(raw.payload, config, mapping)]
        except NeedsReview as exc:
            # Ambiguous mappings and invalid supplied fields require explicit correction.
            if mapping or not str(exc).startswith("Missing required source fields"):
                raise
        source = raw.payload
        spans = []

        def leaves(value):
            if isinstance(value, dict):
                for item in value.values():
                    leaves(item)
            elif isinstance(value, list):
                for item in value:
                    leaves(item)
            elif value is not None:
                spans.append(text(value))

        leaves(source)
    else:
        source = raw.payload["text"]
        spans = [source]
    result = await call(
        "extractor",
        {"source": source, "extract_context": config.input.extract_context},
        Extraction,
        raw.source_record_id,
        "extract",
    )
    if not result.records:
        if config.input.type == "txt" and config.input.generate_qa_from_text:
            return await construct_from_text(raw, config, call, batch_question_plan)
        raise NeedsReview("No complete source question/answer extracted: " + result.reason)
    records = []
    for item in result.records:
        extracted = item.model_dump()
        fields = [
            (name, extracted[name]) for name in ("question", "answer", "context", "statement")
        ]
        fields += [
            (name, value) for name in ("options", "ranking_items") for value in extracted[name]
        ]
        fields += [
            (name, str(extracted[name]))
            for name in ("noul_label", "score_index")
            if extracted[name] is not None
        ]
        for name, value in fields:
            if value and not any(value in span for span in spans):
                raise NeedsReview(
                    f"Extractor invented or rewrote {name}; exact source span required"
                )
        if config.input.type == "json":
            payload = {**raw.payload, "question": item.question, "answer": item.answer}
            # Preserve mapped source fields; retain optional extracted components when unmapped.
            for name in ("options", "statement", "noul_label", "ranking_items", "score_index"):
                if lookup(raw.payload, name) is None and extracted[name] is not None:
                    payload[name] = extracted[name]
            records.append(normalize(payload, config, mapping))
        else:
            records.append(
                StandardRecord.model_validate(
                    {
                        **extracted,
                        "context": item.context if config.input.extract_context else "",
                        "split": config.input.default_split,
                    }
                )
            )
    return records


async def construct_from_text(raw, config, call, batch_question_plan=None):
    source = raw.payload["text"]
    requirements = {
        "requested_modes": config.task.modes,
        "domain": config.task.domain_profile,
        "difficulty": config.task.difficulty,
        "target_option_count": config.choice.target_option_count,
        "score_levels": config.score.default_levels,
        "max_questions": config.input.max_generated_questions,
    }
    if batch_question_plan is not None:
        requirements["batch_question_plan"] = batch_question_plan
        requirements["max_questions"] = sum(item["count"] for item in batch_question_plan)
        if not requirements["max_questions"]:
            raise NeedsReview(
                "TXT batch planner found no grounded narrative question opportunities"
            )
    assessment = await call(
        "text_assessor",
        {"source": source, **requirements},
        TextAssessment,
        raw.source_record_id,
        "text_assess",
    )
    if not assessment.opportunities:
        raise NeedsReview("No worthwhile grounded decision question: " + assessment.reason)
    if len(assessment.opportunities) > requirements["max_questions"]:
        raise NeedsReview("Text assessment exceeded max_generated_questions")
    if batch_question_plan is not None:
        ceilings = {item["mode"]: item["count"] for item in batch_question_plan}
        counts = Counter(item.mode for item in assessment.opportunities)
        if any(count > ceilings.get(mode, 0) for mode, count in counts.items()):
            raise NeedsReview("Text assessment exceeded the planned per-mode batch question count")
    # Validate the entire plan before spending calls on construction.
    for opportunity in assessment.opportunities:
        if config.task.modes != ["auto"] and opportunity.mode not in config.task.modes:
            raise NeedsReview("Text assessment selected an unrequested mode")
        if any(not quote.strip() or quote not in source for quote in opportunity.evidence):
            raise NeedsReview("Text assessment evidence must be exact nonempty source spans")

    async def construct(item):
        index, opportunity = item
        return await call(
            "qa_constructor",
            {"source": source, "opportunity": opportunity.model_dump(), **requirements},
            QAConstruction,
            raw.source_record_id,
            f"text_{index + 1}:construct_qa",
        )

    results = await map_bounded(
        construct, list(enumerate(assessment.opportunities)), config.execution.concurrency
    )
    records, reasons = [], []
    for opportunity, result in zip(assessment.opportunities, results, strict=True):
        if result.qa is None:
            reasons.append(result.reason)
            continue
        records.append(
            StandardRecord(
                **result.qa.model_dump(),
                mode=opportunity.mode,
                context="\n\n".join(opportunity.evidence) if config.input.extract_context else "",
                split=config.input.default_split,
                group_id=raw.source_record_id,
                text_origin=TextOrigin(
                    evidence=opportunity.evidence,
                    assessment_reason=opportunity.reason,
                    construction_reason=result.reason,
                ),
            )
        )
    if not records:
        raise NeedsReview("No grounded question could be constructed: " + "; ".join(reasons))
    return records
