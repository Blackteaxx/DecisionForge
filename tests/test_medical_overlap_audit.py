"""Training admission audits patient content without using benchmark answers."""

import json


def row(identity, state, split="train", family="source-one"):
    return {"id": identity, "split": split, "family_id": family, "state": state}


def test_normalized_exact_and_near_matches_are_reported_across_roles():
    from examples.domain.medical_triage.scripts.v1.audit import find_overlaps

    first = (
        "My left ankle has been swollen and painful since I fell on the stairs yesterday morning."
    )
    rows = [
        row("train", {"patients": [{"message": first}]}, family="one"),
        row(
            "validation",
            json.dumps({"patients": [{"message": first.upper()}]}),
            "validation",
            "two",
        ),
        row(
            "calibration",
            {"patients": [{"message": first + " Please help."}]},
            "calibration",
            "three",
        ),
    ]
    overlaps = find_overlaps(rows)
    assert any(
        o["left"] == "train" and o["right"] == "validation" and o["kind"] == "exact"
        for o in overlaps
    )
    assert any(o["kind"] == "near" for o in overlaps)


def test_shared_system_prompt_and_same_role_pairs_are_not_patient_leakage():
    from examples.domain.medical_triage.scripts.v1.audit import find_overlaps

    common = "You are a careful medical assistant and should choose the best current action."
    rows = [
        row(
            "one",
            [
                {"role": "system", "content": common},
                {"role": "user", "content": "My ankle hurts today."},
            ],
        ),
        row(
            "two",
            [
                {"role": "system", "content": common},
                {"role": "user", "content": "My tooth broke yesterday."},
            ],
            "validation",
            "two",
        ),
        row("three", {"patients": [{"message": "My ankle hurts today."}]}),
    ]
    assert find_overlaps(rows) == []


def test_exact_ehr_overlap_is_detected_even_when_messages_differ():
    from examples.domain.medical_triage.scripts.v1.audit import find_overlaps

    rows = [
        row(
            "train",
            {"patients": [{"message": "My arm hurts.", "ehr": "History of type 2 diabetes."}]},
        ),
        row(
            "test",
            {"patients": [{"message": "My ears hurt.", "ehr": "History of type 2 diabetes."}]},
            "test",
            "public",
        ),
    ]
    assert find_overlaps(rows)[0]["document_kind"] == "ehr"


def test_same_family_cannot_cross_roles_even_with_distinct_patient_facts():
    from examples.domain.medical_triage.scripts.v1.audit import find_overlaps

    rows = [
        row("one", "An ankle injury occurred."),
        row("two", "A tooth was broken.", "validation"),
    ]
    assert any(o["kind"] == "source_family" for o in find_overlaps(rows))


def test_context_dependent_short_dialogue_replies_are_not_patient_identities():
    from examples.domain.medical_triage.scripts.v1.audit import find_overlaps

    first = [
        {"role": "user", "content": "My ankle hurts after yesterday walking home."},
        {"role": "assistant", "content": "Any swelling or bruising?"},
        {"role": "user", "content": "No, none of that."},
    ]
    second = [
        {"role": "user", "content": "I have nausea and headaches after starting medication."},
        {"role": "assistant", "content": "Any fever or vision changes?"},
        {"role": "user", "content": "No, none of that."},
    ]
    assert find_overlaps([row("one", first), row("two", second, "validation", "two")]) == []
