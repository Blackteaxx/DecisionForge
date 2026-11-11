"""Preserve lineage, roles and clinical action identity when combining real build outputs."""

import json

import pytest


def build_run(tmp_path, name, identity, split="train", family="one", answer="B"):
    root = tmp_path / name
    root.mkdir()
    options = ["Prioritize the first patient.", "Prioritize the second patient."]
    record = {
        "schema_version": "decisionforge.training.v3",
        "record_id": 1,
        "group_id": 1,
        "split": split,
        "request": {
            "state": "Synthetic patient facts: " + identity,
            "questions": {
                "q1": {
                    "type": "choice",
                    "instructions": "Prioritize.",
                    "criteria": {"A": options[1], "B": options[0]},
                }
            },
        },
        "supervision": {
            "answers": {
                "q1": {
                    "type": "choice",
                    "choice": answer,
                    "confidence": 1.0,
                    "probabilities": {"A": float(answer == "A"), "B": float(answer == "B")},
                }
            }
        },
    }
    source = {
        "id": identity,
        "group_id": "nlm-family:" + family,
        "split": split,
        "context": record["request"]["state"],
        "options": options,
        "answer": options[0],
        "_provenance": {
            "task": "message_priority",
            "synthetic": True,
            "parent_ids": ["message:" + identity],
            "source_id": family,
            "source_url": "https://medlineplus.gov/one.html",
            "evidence": ["A source span"],
            "target_semantics": "synthetic_teacher_hard_label",
        },
    }
    meta = {
        "summary": {"status": "complete", "accepted": 1},
        "records": {"1": {"status": "accepted", "source": {"payload": source}}},
    }
    (root / "accepted.json").write_text(json.dumps([record]))
    (root / "meta.json").write_text(json.dumps(meta))
    return root


def test_two_runs_with_local_id_one_get_unique_ids_and_source_family_groups(tmp_path):
    from examples.domain.medical_triage.scripts.v1.assemble import assemble_training

    first = build_run(tmp_path, "run1", "first-case")
    second = build_run(tmp_path, "run2", "second-case")
    result = assemble_training([first, second], tmp_path / "prepared")
    records = json.loads((tmp_path / "prepared/training.json").read_text())
    assert result["records"] == 2
    assert {r["record_id"] for r in records} == {1, 2}
    assert len({r["group_id"] for r in records}) == 1
    lineage = json.loads((tmp_path / "prepared/lineage.json").read_text())
    assert {r["source_record_id"] for r in lineage.values()} == {1}
    assert {r["source_run"] for r in lineage.values()} == {str(first), str(second)}


def test_combined_cross_round_parent_overlap_protects_heldout_role(tmp_path):
    from examples.domain.medical_triage.scripts.v1.assemble import assemble_training

    train = build_run(tmp_path, "train", "same-case", "train", "train-family")
    validation = build_run(tmp_path, "validation", "other-id", "validation", "val-family")
    p = validation / "meta.json"
    meta = json.loads(p.read_text())
    meta["records"]["1"]["source"]["payload"]["_provenance"]["parent_ids"] = ["message:same-case"]
    p.write_text(json.dumps(meta))
    result = assemble_training([train, validation], tmp_path / "prepared")
    records = json.loads((tmp_path / "prepared/training.json").read_text())
    assert len(records) == 1 and records[0]["split"] == "validation"
    assert result["quarantined"] == 1
    rejected = json.loads((tmp_path / "prepared/quarantined.json").read_text())
    assert rejected[0]["scenario"]["split"] == "train"


def test_changed_catalog_or_gold_cannot_be_accepted_as_same_clinical_task(tmp_path):
    from examples.domain.medical_triage.scripts.v1.assemble import assemble_training

    root = build_run(tmp_path, "bad-gold", "one", answer="A")
    with pytest.raises(ValueError, match="answer|gold"):
        assemble_training([root], tmp_path / "prepared")


def test_assembly_deduplicates_identical_cases_with_both_original_lineages(tmp_path):
    from examples.domain.medical_triage.scripts.v1.assemble import assemble_training

    first = build_run(tmp_path, "run1", "same-case")
    second = build_run(tmp_path, "run2", "same-case")
    result = assemble_training([first, second], tmp_path / "prepared")
    assert result["records"] == 1
    assert result["duplicates"] == 1
    lineage = json.loads((tmp_path / "prepared/lineage.json").read_text())
    assert len(lineage["1"]["origins"]) == 2


def test_assembly_never_mutates_original_accepted_records(tmp_path):
    from examples.domain.medical_triage.scripts.v1.assemble import assemble_training

    root = build_run(tmp_path, "run1", "one")
    original = (root / "accepted.json").read_bytes()
    assemble_training([root], tmp_path / "prepared")
    assert (root / "accepted.json").read_bytes() == original


def test_conflicting_teacher_labels_are_quarantined_instead_of_choosing_one(tmp_path):
    from examples.domain.medical_triage.scripts.v1.assemble import assemble_training

    first = build_run(tmp_path, "run1", "same-case")
    second = build_run(tmp_path, "run2", "same-case", answer="A")
    p = second / "meta.json"
    m = json.loads(p.read_text())
    m["records"]["1"]["source"]["payload"]["answer"] = "Prioritize the second patient."
    p.write_text(json.dumps(m))
    third = build_run(tmp_path, "run3", "different-case")
    result = assemble_training([first, second, third], tmp_path / "prepared")
    assert result["records"] == 1
    rejected = json.loads((tmp_path / "prepared/quarantined.json").read_text())
    assert len(rejected) == 2
    assert {r["reason"] for r in rejected} == {"conflicting_teacher_labels"}


def test_json_object_key_order_does_not_inflate_unique_training_cases(tmp_path):
    from examples.domain.medical_triage.scripts.v1.assemble import assemble_training

    first = build_run(tmp_path, "run1", "identity-one")
    second = build_run(tmp_path, "run2", "identity-two")
    for root, state in [
        (first, '{"message":"a patient","ehr":"history"}'),
        (second, '{"ehr":"history","message":"a patient"}'),
    ]:
        p = root / "accepted.json"
        records = json.loads(p.read_text())
        records[0]["request"]["state"] = state
        p.write_text(json.dumps(records))
        p = root / "meta.json"
        meta = json.loads(p.read_text())
        meta["records"]["1"]["source"]["payload"]["context"] = state
        p.write_text(json.dumps(meta))
    result = assemble_training([first, second], tmp_path / "prepared")
    assert result["records"] == 1
    assert result["duplicates"] == 1


def test_validation_aggregate_contains_only_canonical_validation_rows(tmp_path):
    from examples.domain.medical_triage.scripts.v1.assemble import assemble_training

    train = build_run(tmp_path, "train", "one", family="train")
    val = build_run(tmp_path, "val", "two", split="validation", family="val")
    cal = build_run(tmp_path, "cal", "three", split="calibration", family="cal")
    result = assemble_training([train, val, cal], tmp_path / "prepared")
    rows = [
        json.loads(line)
        for line in (tmp_path / "prepared/validation.jsonl").read_text().splitlines()
    ]
    assert len(rows) == result["independent_evaluation"]["validation:all"]["count"] == 1
    assert {r["split"] for r in rows} == {"validation"}
    assert rows[0]["target"]["value"] == "A"


def test_normalized_ehr_reuse_cannot_cross_roles_with_different_punctuation(tmp_path):
    from examples.domain.medical_triage.scripts.v1.assemble import assemble_training

    train = build_run(tmp_path, "train", "train-case", family="train")
    cal = build_run(tmp_path, "cal", "cal-case", split="calibration", family="cal")
    for root, message, ehr in [
        (train, "A distinct ankle injury.", "Type 2 diabetes, hypertension."),
        (cal, "A distinct abdominal complaint.", "Type 2 diabetes; hypertension."),
    ]:
        state = json.dumps({"patients": [{"message": message, "ehr": ehr}]})
        p = root / "accepted.json"
        records = json.loads(p.read_text())
        records[0]["request"]["state"] = state
        p.write_text(json.dumps(records))
        p = root / "meta.json"
        meta = json.loads(p.read_text())
        meta["records"]["1"]["source"]["payload"]["context"] = state
        p.write_text(json.dumps(meta))
    result = assemble_training([train, cal], tmp_path / "prepared")
    assert result["records"] == 1 and result["splits"] == {"calibration": 1}
    assert result["quarantined"] == 1
