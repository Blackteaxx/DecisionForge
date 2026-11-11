"""Pin the data profile boundary without making clinical judgments in unit tests."""

import copy
import json

import httpx
import pytest


def scenario():
    return {
        "id": "hard-case-1",
        "family_id": "topic-1",
        "split": "validation",
        "parent_ids": ["message:first", "message:second"],
        "synthetic": True,
        "task": "message_priority",
        "comparison_axis": "timeline",
        "answer": "A",
        "priority_relation": "first_more_urgent",
        "state": {
            "patients": [
                {"message": " ".join(f"first{i}" for i in range(120))},
                {"message": " ".join(f"second{i}" for i in range(120))},
            ]
        },
        "source": {
            "id": "topic-1",
            "url": "https://medlineplus.gov/fixture.html",
            "summary": "Fixture source evidence.",
        },
        "evidence": ["Fixture source evidence."],
        "reason": "Fixture comparison rationale.",
    }


def test_hard_profile_preserves_immutable_source_and_private_lineage(tmp_path):
    from examples.domain.medical_triage.scripts.v1.generate import materialize_inputs
    from examples.domain.medical_triage.scripts.v1.profiles import validate_hard_priority

    original = scenario()
    before = copy.deepcopy(original)
    retained = validate_hard_priority(original)
    assert original == before
    assert retained["split"] == "validation"
    assert retained["source"] == original["source"]
    assert retained["generation_profile"] == "hard_priority_v2"
    output = tmp_path / "constructed.json"
    materialize_inputs([retained], output)
    row = json.loads(output.read_text())[0]
    assert row["_provenance"]["generation_profile"] == "hard_priority_v2"
    assert row["_provenance"]["comparison_axis"] == "timeline"
    assert "hard_priority_v2" not in row["context"]
    assert "comparison_axis" not in row["context"]


@pytest.mark.parametrize("change", ["short", "long", "one_short", "wrong_task", "ehr", "axis"])
def test_hard_profile_rejects_out_of_scope_cases(change):
    from examples.domain.medical_triage.scripts.v1.profiles import validate_hard_priority

    raw = scenario()
    if change == "short":
        raw["state"]["patients"] = [{"message": "few words"}, {"message": "few others"}]
    elif change == "long":
        raw["state"]["patients"][0]["message"] = "word " * 600
    elif change == "one_short":
        raw["state"]["patients"][0]["message"] = "word " * 300
        raw["state"]["patients"][1]["message"] = "word " * 20
    elif change == "wrong_task":
        raw["task"] = "triage_action"
    elif change == "ehr":
        raw["state"]["patients"][0]["ehr"] = "Separate history."
    else:
        raw["comparison_axis"] = "unsupported axis"
    with pytest.raises(ValueError):
        validate_hard_priority(raw)


def test_flat_teacher_format_converts_without_changing_any_patient_fact():
    from examples.domain.medical_triage.scripts.v1.profiles import hard_public_case

    raw = {
        "first_message": "First patient's exact words.",
        "second_message": "Second patient's exact words.",
        "answer": "B",
        "comparison_axis": "timeline",
    }
    before = copy.deepcopy(raw)
    converted = hard_public_case(raw)
    assert raw == before
    assert converted["state"]["patients"] == [
        {"message": raw["first_message"]},
        {"message": raw["second_message"]},
    ]
    assert converted["answer"] == "B"
    with pytest.raises(ValueError):
        hard_public_case({**raw, "state": {"patients": []}})
    with pytest.raises(ValueError):
        hard_public_case({**raw, "second_message": ""})


@pytest.mark.asyncio
async def test_teacher_profile_identity_is_distinct_and_resume_reuses_response(tmp_path):
    from examples.domain.medical_triage.scripts.v1.generate import PROMPT, Teacher

    payloads = []

    def handle(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": f"response-{len(payloads)}",
                "model": "deepseek-flash",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"admitted":false,"scenarios":[]}'},
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    source = {"id": "1", "title": "Fixture", "summary": "Fixture evidence"}
    journal = tmp_path / "teacher.sqlite"
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        classic = Teacher(journal, "http://fixture/v1", "fixture", client)
        left = await classic.ask(source, 21, 4)
        classic.close()
        hard = Teacher(journal, "http://fixture/v1", "fixture", client, prompt="Different profile")
        right = await hard.ask(source, 21, 4)
        hard.close()
        resumed = Teacher(
            journal, "http://fixture/v1", "fixture", client, prompt="Different profile"
        )
        replay = await resumed.ask(source, 21, 4)
        resumed.close()
    assert len(payloads) == 2
    assert payloads[0]["messages"][0]["content"] == PROMPT
    assert payloads[1]["messages"][0]["content"] == "Different profile"
    assert left["_api"]["request_hash"] != right["_api"]["request_hash"]
    assert replay["_api"] == right["_api"]
