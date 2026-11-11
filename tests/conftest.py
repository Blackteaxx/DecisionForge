from __future__ import annotations

import json

import pytest

from decisionforge.config import Config
from decisionforge.util import BudgetExhausted


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "input.json"
    path.write_text(
        json.dumps(
            {
                "id": "one",
                "question": "What is 2+2?",
                "context": "Add two and two.",
                "answer": "4",
                "split": "train",
            }
        ),
        encoding="utf-8",
    )
    return Config(
        input={"path": str(path)},
        task={"modes": ["choice"]},
        # CLI tests may load the user's .env; SDK fixtures use the Responses protocol.
        llm={
            "api": "responses",
            "structured_output": "json",
            "chat_token_limit_param": "max_completion_tokens",
        },
        output={"directory": str(tmp_path / "run")},
        noul={"labels": [1]},
        verification={"max_semantic_repairs": 0},
    )


class FixtureProvider:
    """Offline contract double, not a semantic quality assessment."""

    def __init__(self, mode="choice", fail_role=None, bad_selector=False, false_negative=False):
        self.mode, self.fail_role = mode, fail_role
        self.reject = bad_selector or false_negative
        self.calls = []

    async def close(self):
        pass

    async def call(self, role, payload, schema, record, stage):
        self.calls.append((role, payload, stage))
        if role == self.fail_role:
            raise BudgetExhausted("injected budget interruption")
        if role == "choice":
            result = {"distractors": [str(i) for i in range(5, 5 + payload["count"])]}
        elif role == "noul":
            result = {
                "statement": "Two plus two equals "
                + ("four." if payload["requested_label"] else "five.")
            }
        elif role == "score":
            items = ["incorrect", "partly correct", payload["source"]["answer"]]
            result = {"ranking_items": items, "score_index": 2}
        elif role == "context":
            result = {"context": "Two pairs of objects contain four objects in total."}
        elif role == "checker":
            mode = payload["current_jev"]["request"]["questions"]["q1"]["type"]
            result = {
                "passed": not self.reject,
                "reason": "fixture rejection" if self.reject else "Verified fixture",
                "failed_fields": ["criteria"] if self.reject else [],
                "repair_instructions": ["Replace distractors"] if self.reject else [],
                "modules": [mode] if self.reject else [],
            }
        elif role == "extractor":
            result = {"records": [], "reason": "No existing QA pair"}
        else:
            raise AssertionError(f"Unexpected role: {role}")
        return schema.model_validate(result)
