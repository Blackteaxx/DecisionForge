"""Direct Chat API baseline: one semantic request, no output repair."""

from __future__ import annotations

import json

from decisionforge.eval.adapters.jev import parse_answer, usage_from
from decisionforge.eval.data import parse_json
from decisionforge.eval.metrics import distribution
from decisionforge.training_free.chat_baseline.prompt import answer_schema, messages


def json_content_equal(left, right):
    """JSON numbers may be equivalent; booleans never stand in for numbers."""
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(json_content_equal(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            json_content_equal(a, b) for a, b in zip(left, right)
        )
    numeric = type(left) in (int, float) and type(right) in (int, float)
    return (numeric or type(left) is type(right)) and left == right


class ChatProcessing:
    def prepare_request(self, payload, question):
        from decisionforge.media import reject_media

        reject_media(payload, self.model.adapter)
        body = {
            "model": self.model.model,
            "messages": messages(payload, question),
            "temperature": self.model.temperature,
            "max_tokens": self.model.max_tokens,
            "n": 1,
            "stream": False,
        }
        if self.model.response_format == "json_object":
            body["response_format"] = {"type": "json_object"}
        elif self.model.response_format == "json_schema":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "jev_decision",
                    "strict": True,
                    "schema": answer_schema(question),
                },
            }
        if self.model.serving.enable_thinking is not None:
            body["chat_template_kwargs"] = {
                "enable_thinking": self.model.serving.enable_thinking,
            }
        return body

    def probability_source(self, question):
        return "llm_self_reported"

    def response_usage(self, raw):
        usage = raw.get("usage") if isinstance(raw, dict) else None
        if not isinstance(usage, dict):
            return None
        return usage_from(
            {
                "usage": {
                    "input_tokens": usage.get("prompt_tokens"),
                    "output_tokens": usage.get("completion_tokens"),
                }
            }
        )

    def response_answer(self, raw, question, result):
        result["raw_generated_text"] = None
        choices = raw.get("choices") if isinstance(raw, dict) else None
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise ValueError("Chat response needs exactly one completion")
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict):
            raise ValueError("Missing Chat assistant message")
        content = message.get("content")
        result["raw_generated_text"] = content
        if message.get("role") != "assistant" or message.get("refusal"):
            raise ValueError("Chat refusal or unexpected message role")
        if choice.get("finish_reason") != "stop":
            raise ValueError("Chat completion did not finish normally")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Missing Chat generated text")
        generated = parse_json(content)
        json.dumps(generated, allow_nan=False)
        if (
            not isinstance(generated, dict)
            or set(generated) != {"answers"}
            or not isinstance(generated["answers"], dict)
            or set(generated["answers"]) != {"decision"}
        ):
            raise ValueError("Chat output must contain exactly answers.decision")
        answer = parse_answer(generated, question)
        if question.type == "score":
            expected_legend = {str(i): value for i, value in enumerate(question.criteria)}
            if not json_content_equal(answer["legend"], expected_legend):
                raise ValueError("Chat Score legend differs from original JSON rubric")
        expected = (
            {"type", "noul"}
            if question.type == "noul"
            else {
                "type",
                "confidence",
                "probabilities",
                *(["choice"] if question.type == "choice" else ["score", "legend"]),
            }
        )
        if set(answer) != expected:
            raise ValueError("Chat answer has missing or additional fields")
        if question.type != "noul":
            distribution(answer["probabilities"], question.labels)
        return answer
