"""Prevent benchmark label leakage, accidental subsampling and wrong pair reversal."""

import json

import pytest


def test_care_uses_only_official_public_test_and_visible_messages():
    from examples.domain.medical_triage.scripts.prepare_benchmarks import prepare_care

    row = {
        "prefix_id": "case1_r01",
        "case_id": "case1",
        "split": "public_test_1",
        "gold_label": "1B_ESCALATION_URGENT_CARE",
        "input_messages": [{"role": "user", "content": "I cannot breathe."}],
        "information_state": "SECRET_INFORMATION_STATE",
        "round_role": "SECRET_ROUND",
    }
    samples = prepare_care([row, dict(row, split="development", prefix_id="dev")])
    assert len(samples) == 1
    assert samples[0]["target"] == {"kind": "hard", "value": "D"}
    assert samples[0]["request"]["state"] == row["input_messages"]
    assert len(samples[0]["request"]["questions"]["decision"]["criteria"]) == 4
    public = json.dumps(samples[0]["request"])
    assert "SECRET_" not in public
    assert "gold_label" not in public
    assert samples[0]["group_id"] == "care:case1"


def test_care_rejects_unknown_gold_instead_of_losing_a_test_row():
    from examples.domain.medical_triage.scripts.prepare_benchmarks import prepare_care

    with pytest.raises(ValueError, match="label"):
        prepare_care(
            [
                {
                    "prefix_id": "x",
                    "case_id": "x",
                    "split": "public_test_1",
                    "gold_label": "unknown",
                    "input_messages": [],
                }
            ]
        )


def test_synth_position_moves_ehr_message_and_gold_together():
    from examples.domain.medical_triage.scripts.prepare_benchmarks import prepare_synth

    row = {
        "MSG #1": "first message",
        "EHR #1": "first record",
        "MSG #2": "second message",
        "EHR #2": "second record",
        "Label": "Patient #2",
    }
    positions = set()
    for seed in range(16):
        sample = prepare_synth([row], seed=seed)[0]
        patients = sample["request"]["state"]["patients"]
        assert patients in [
            [
                {"message": "first message", "ehr": "first record"},
                {"message": "second message", "ehr": "second record"},
            ],
            [
                {"message": "second message", "ehr": "second record"},
                {"message": "first message", "ehr": "first record"},
            ],
        ]
        want = "A" if patients[0]["message"] == "second message" else "B"
        assert sample["target"]["value"] == want
        positions.add(want)
        assert prepare_synth([row], seed=seed) == [sample]
    assert positions == {"A", "B"}


def test_reddit_hides_chosen_roles_ratings_and_comments_but_preserves_ties():
    from examples.domain.medical_triage.scripts.prepare_benchmarks import prepare_reddit

    row = {
        "chosen": "priority message",
        "rejected": "other message",
        "chosen_level": 2.0,
        "rejected_level": 2.0,
        "chosen_comments": "SECRET rationale",
        "rejected_comments": "SECRET label",
        "difficulty": "SECRET difficulty",
    }
    samples = prepare_reddit([row, row], seed=3407)
    assert len(samples) == 2
    assert samples[0]["id"] != samples[1]["id"]
    assert samples[0]["group_id"] == samples[1]["group_id"]
    for sample in samples:
        public = json.dumps(sample["request"])
        assert all(text not in public for text in ["chosen", "rejected", "SECRET", "level"])
        patients = sample["request"]["state"]["patients"]
        assert sample["target"]["value"] == (
            "A" if patients[0]["message"] == "priority message" else "B"
        )
        assert sample["metadata"]["equal_levels"] is True


def test_synth_shared_ehr_connects_different_messages_into_one_group():
    from examples.domain.medical_triage.scripts.prepare_benchmarks import prepare_synth

    rows = [
        {
            "MSG #1": "shared patient new symptom",
            "EHR #1": "same patient record",
            "MSG #2": "other one",
            "EHR #2": "record two",
            "Label": "Patient #1",
        },
        {
            "MSG #1": "shared patient older symptom",
            "EHR #1": "same patient record",
            "MSG #2": "other two",
            "EHR #2": "record three",
            "Label": "Patient #2",
        },
    ]
    samples = prepare_synth(rows)
    assert samples[0]["group_id"] == samples[1]["group_id"]


def test_nlm_extracts_only_english_summary_without_linked_bodies(tmp_path):
    from examples.domain.medical_triage.scripts.prepare_sources import load_nlm_topics

    xml = tmp_path / "topics.xml"
    xml.write_text("""<health-topics>
      <health-topic id="1" title="Chest Pain" language="English"
       url="https://medlineplus.gov/chestpain.html">
       <full-summary>&lt;p&gt;Call emergency services for severe pain.&lt;/p&gt;
         &lt;p&gt;Read &lt;a href="https://third-party.example"&gt;more&lt;/a&gt;.&lt;/p&gt;</full-summary>
       <site url="https://third-party.example"><description>THIRD PARTY BODY</description></site>
      </health-topic>
      <health-topic id="2" title="Dolor" language="Spanish" url="https://medlineplus.gov/spanish/x.html">
        <full-summary>SPANISH BODY</full-summary>
      </health-topic></health-topics>""")
    topics = load_nlm_topics(xml)
    assert len(topics) == 1
    assert topics[0]["summary"] == "Call emergency services for severe pain.\nRead more."
    assert "THIRD PARTY" not in json.dumps(topics)
    assert "third-party.example" not in topics[0]["summary"]


def test_source_family_role_is_stable_before_scenario_generation():
    from examples.domain.medical_triage.scripts.prepare_sources import assign_source_roles

    topics = [
        {"id": "1", "family_id": "same-rule"},
        {"id": "2", "family_id": "same-rule"},
        {"id": "3", "family_id": "different-rule"},
    ]
    a = {x["id"]: x["split"] for x in assign_source_roles(topics)}
    b = {x["id"]: x["split"] for x in assign_source_roles(list(reversed(topics)))}
    assert a == b
    assert a["1"] == a["2"]
    assert set(a.values()) <= {"train", "validation", "calibration"}


def _scenario_source(role="train"):
    return {
        "id": "10",
        "family_id": "breathing",
        "split": role,
        "title": "Breathing",
        "url": "https://medlineplus.gov/breathing.html",
        "summary": "Severe breathing trouble needs urgent evaluation.",
    }


def _raw_scenario():
    return {
        "task": "triage_action",
        "synthetic": True,
        "state": [{"role": "user", "content": "I have severe breathing trouble now."}],
        "answer": "D",
        "evidence": ["Severe breathing trouble needs urgent evaluation."],
        "reason": "Severe trouble matches the cited urgent-action rule.",
    }


def test_synthetic_scenario_requires_real_source_evidence_and_public_facts():
    from examples.domain.medical_triage.scripts.v1.generate import validate_scenario

    raw = _raw_scenario()
    valid = validate_scenario(raw, _scenario_source())
    assert valid["split"] == "train"
    assert valid["answer"] == "D"
    with pytest.raises(ValueError, match="evidence"):
        validate_scenario(dict(raw, evidence=["INVENTED GUIDELINE"]), _scenario_source())
    with pytest.raises(ValueError, match="[Pp]rivate"):
        validate_scenario(dict(raw, state={"answer": "D"}), _scenario_source())


def test_pair_tie_is_rejected_instead_of_made_into_binary_gold():
    from examples.domain.medical_triage.scripts.v1.generate import validate_scenario

    raw = dict(
        _raw_scenario(),
        task="message_priority",
        answer="A",
        state={"patients": [{"message": "one"}, {"message": "two"}]},
        priority_relation="tie",
    )
    with pytest.raises(ValueError, match="priority|tie"):
        validate_scenario(raw, _scenario_source())


def test_materialized_input_keeps_rules_and_answer_out_of_public_state(tmp_path):
    from examples.domain.medical_triage.scripts.v1.generate import (
        materialize_inputs,
        validate_scenario,
    )

    scenario = validate_scenario(_raw_scenario(), _scenario_source())
    result = materialize_inputs([scenario], tmp_path / "inputs.json")
    rows = json.loads((tmp_path / "inputs.json").read_text())
    assert result["records"] == 1
    assert rows[0]["answer"] == "Recommend urgent or emergency-level evaluation now."
    assert rows[0]["split"] == "train"
    public = rows[0]["context"]
    assert "Severe breathing trouble needs urgent evaluation." not in public
    assert "_provenance" not in public
    assert rows[0]["_provenance"]["synthetic"] is True


def test_reused_patient_cannot_cross_materialized_split_boundaries(tmp_path):
    from examples.domain.medical_triage.scripts.v1.generate import (
        materialize_inputs,
        validate_scenario,
    )

    first = validate_scenario(_raw_scenario(), _scenario_source())
    second = validate_scenario(
        _raw_scenario(), dict(_scenario_source("validation"), id="11", family_id="different-rule")
    )
    with pytest.raises(ValueError, match="cross|split"):
        materialize_inputs([first, second], tmp_path / "inputs.json")


@pytest.mark.asyncio
async def test_teacher_resume_reuses_committed_response_without_another_request(tmp_path):
    import httpx

    from examples.domain.medical_triage.scripts.v1.generate import Teacher

    def handler(request):
        assert json.loads(request.content)["model"] == "deepseek-flash"
        assert json.loads(request.content)["reasoning_effort"] == "low"
        return httpx.Response(
            200,
            json={
                "model": "deepseek-flash",
                "id": "response-1",
                "usage": {"total_tokens": 123},
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {"admitted": True, "scenarios": [_raw_scenario()]}
                            )
                        },
                    }
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        teacher = Teacher(tmp_path / "journal.sqlite", "http://teacher/v1", "fixture", client)
        first = await teacher.ask(_scenario_source(), round_id=1, count=2)
        teacher.close()

    def offline(request):
        raise AssertionError("Committed API response must not be requested again")

    async with httpx.AsyncClient(transport=httpx.MockTransport(offline)) as client:
        teacher = Teacher(tmp_path / "journal.sqlite", "http://teacher/v1", "fixture", client)
        second = await teacher.ask(_scenario_source(), round_id=1, count=2)
        teacher.close()
    assert first == second
    assert second["scenarios"][0]["answer"] == "D"


@pytest.mark.asyncio
async def test_teacher_transport_disconnect_is_bounded_and_unknown_cost_is_recorded(tmp_path):
    import sqlite3

    import httpx

    from examples.domain.medical_triage.scripts.v1.generate import Teacher

    def disconnected(request):
        raise httpx.RemoteProtocolError("Server disconnected", request=request)

    journal = tmp_path / "journal.sqlite"
    async with httpx.AsyncClient(transport=httpx.MockTransport(disconnected)) as client:
        teacher = Teacher(journal, "http://teacher/v1", "fixture", client)
        with pytest.raises(httpx.RemoteProtocolError):
            await teacher.ask(_scenario_source(), round_id=1, count=2)
        teacher.close()
    db = sqlite3.connect(journal)
    attempts = list(db.execute("SELECT status, response FROM attempts"))
    assert len(attempts) == 3
    assert all(status == 0 and json.loads(raw)["outcome_unknown"] for status, raw in attempts)
    assert db.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 0


def test_cross_role_candidates_are_quarantined_without_reassigning_splits():
    from examples.domain.medical_triage.scripts.v1.generate import (
        quarantine_role_overlaps,
        validate_scenario,
    )

    train = validate_scenario(_raw_scenario(), _scenario_source())
    validation = validate_scenario(
        _raw_scenario(), dict(_scenario_source("validation"), id="11", family_id="different-rule")
    )
    kept, rejected = quarantine_role_overlaps([train, validation])
    assert [s["split"] for s in kept] == ["validation"]
    assert len(rejected) == 1
    assert rejected[0]["reason"] == "cross_role_parent_overlap"
    assert rejected[0]["scenario"]["split"] == "train"
    assert rejected[0]["scenario"]["answer"] == "D"
    assert quarantine_role_overlaps([validation, train]) == (kept, rejected)


@pytest.mark.asyncio
async def test_teacher_recovers_durable_raw_success_when_interrupted_before_cache_commit(tmp_path):
    import sqlite3

    import httpx

    from examples.domain.medical_triage.scripts.v1.generate import Teacher

    calls = []

    def response(request):
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "model": "deepseek-flash",
                "id": "durable-original",
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {"admitted": True, "scenarios": [_raw_scenario()]}
                            )
                        },
                    }
                ],
            },
        )

    journal = tmp_path / "journal.sqlite"
    async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
        teacher = Teacher(journal, "http://teacher/v1", "fixture", client)
        teacher.db.executescript(
            "CREATE TRIGGER interrupt_cache BEFORE INSERT ON responses BEGIN SELECT RAISE(ABORT, 'fixture interruption'); END;"
        )
        with pytest.raises(sqlite3.IntegrityError, match="fixture interruption"):
            await teacher.ask(_scenario_source(), round_id=1, count=2)
        assert teacher.db.execute("SELECT count(*) FROM attempts").fetchone()[0] == 1
        assert teacher.db.execute("SELECT count(*) FROM responses").fetchone()[0] == 0
        teacher.db.execute("DROP TRIGGER interrupt_cache")
        teacher.db.commit()
        teacher.close()

    def offline(request):
        raise AssertionError(
            "A complete durable raw response must be recovered without a second request"
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(offline)) as client:
        teacher = Teacher(journal, "http://teacher/v1", "fixture", client)
        result = await teacher.ask(_scenario_source(), round_id=1, count=2)
        assert result["_api"]["response_id"] == "durable-original"
        assert result["scenarios"][0]["answer"] == "D"
        assert teacher.db.execute("SELECT count(*) FROM attempts").fetchone()[0] == 1
        assert teacher.db.execute("SELECT count(*) FROM responses").fetchone()[0] == 1
        teacher.close()
    assert len(calls) == 1
