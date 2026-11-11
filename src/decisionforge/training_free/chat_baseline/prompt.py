"""Versioned, zero-shot protocol. Changes require a new evaluation run."""

from __future__ import annotations

import json

PROMPT_VERSION = "decisionforge.chat.zero-shot.v1"
SCHEMA_VERSION = "decisionforge.chat.answer.v1"
SYSTEM_PROMPT = """Answer the decision question using only the supplied public request.
The state is the task input. Follow the decision instructions and all criteria.
Return only one JSON object matching the supplied schema, with answers.decision.
Do not include explanations, reasoning, Markdown, or additional fields.
Choice: use an original option key for choice; probabilities must include every
option key exactly once, with finite values in [0,1] summing to one.
Noul: noul is your estimated probability that the proposition is true, in [0,1].
Score: criteria are ordered from grade 0 upwards. score is your estimated numeric
grade within that range; legend copies the exact criterion at each numeric key;
probabilities include every grade key, in [0,1], summing to one.
For Choice and Score also provide confidence in [0,1]. These probabilities and
confidence are your own estimates. Preserve structured JSON criterion content.
"""


def object_schema(properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def answer_schema(question):
    probability = {"type": "number", "minimum": 0, "maximum": 1}
    fields = {"type": {"type": "string", "enum": [question.type]}}
    if question.type == "noul":
        fields["noul"] = probability
    else:
        fields.update(
            confidence=probability,
            probabilities=object_schema({key: probability for key in question.labels}),
        )
        if question.type == "choice":
            fields["choice"] = {"type": "string", "enum": question.labels}
        else:
            fields["score"] = {"type": "number", "minimum": 0, "maximum": len(question.labels) - 1}
            fields["legend"] = object_schema(
                {str(index): {"const": value} for index, value in enumerate(question.criteria)}
            )
    return object_schema({"answers": object_schema({"decision": object_schema(fields)})})


def messages(payload, question):
    schema = answer_schema(question)
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
            + "\nOutput JSON schema:\n"
            + json.dumps(schema, ensure_ascii=False, allow_nan=False),
        },
        {
            "role": "user",
            "content": json.dumps(
                {key: payload[key] for key in ("state", "questions")},
                ensure_ascii=False,
                allow_nan=False,
            ),
        },
    ]
