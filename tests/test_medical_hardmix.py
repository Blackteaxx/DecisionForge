"""Data ablation preserves frozen patients, sample budgets and independent roles."""

import copy
import json
from collections import Counter

import pytest

from decisionforge.exporter import read_training_records
from examples.domain.medical_triage.scripts.prepare_benchmarks import (
    PAIR_CRITERIA,
    TRIAGE_CRITERIA,
)


def row(identity, role, family, *, task="message_priority", label="A", reverse=False):
    criteria = dict(PAIR_CRITERIA if task == "message_priority" else TRIAGE_CRITERIA)
    if reverse:
        criteria = {"A": PAIR_CRITERIA["B"], "B": PAIR_CRITERIA["A"]}
    state = (
        {
            "patients": [
                {"message": f"This is the first fictional patient message for {identity}"},
                {"message": f"This is the second fictional patient message for {identity}"},
            ]
        }
        if task == "message_priority"
        else [{"role": "user", "content": f"A fictional patient context for {identity}"}]
    )
    r = {
        "schema_version": "decisionforge.training.v3",
        "record_id": identity,
        "group_id": {"train": 1, "validation": 20, "calibration": 30}[role],
        "split": role,
        "request": {
            "state": json.dumps(state),
            "questions": {
                "q1": {
                    "type": "choice",
                    "instructions": "Choose using current patient facts.",
                    "criteria": criteria,
                }
            },
        },
        "supervision": {
            "answers": {
                "q1": {
                    "type": "choice",
                    "choice": label,
                    "confidence": 1.0,
                    "probabilities": {k: float(k == label) for k in criteria},
                }
            }
        },
    }
    lineage = {
        "task": task,
        "split": role,
        "family_id": "nlm-family:" + family,
        "scenario_id": f"scenario-{identity}",
        "parent_ids": [f"message:{identity}"],
        "source_run": "/formal/run",
        "source_record_id": identity,
        "provenance": {"source_id": family, "generation_profile": "hard_priority_v2"},
    }
    return r, lineage


@pytest.fixture
def data(tmp_path):
    frozen, new = tmp_path / "frozen", tmp_path / "new"
    frozen.mkdir()
    new.mkdir()
    old, old_lineage, added, added_lineage = [], {}, [], {}
    identity = 0
    for label in "ABCD":
        identity += 1
        r, detail = row(identity, "train", "t0", task="triage_action", label=label)
        old.append(r)
        old_lineage[str(identity)] = detail
    for label in "AB":
        for reverse in [False, True]:
            for _ in range(2):
                identity += 1
                r, detail = row(identity, "train", "t0", label=label, reverse=reverse)
                old.append(r)
                old_lineage[str(identity)] = detail
    for role, family in [("validation", "v0"), ("calibration", "c0")]:
        for task in ["triage_action", "message_priority"]:
            identity += 1
            r, detail = row(identity, role, family, task=task)
            old.append(r)
            old_lineage[str(identity)] = detail
    identity = 100
    for role, families in [
        ("train", ["t0", "t1"]),
        ("validation", ["v0", "v1"]),
        ("calibration", ["c0", "c1"]),
    ]:
        for family in families:
            for label in "AB":
                for reverse in [False, True]:
                    identity += 1
                    r, detail = row(identity, role, family, label=label, reverse=reverse)
                    added.append(r)
                    added_lineage[str(identity)] = detail
    for root, records, lineage in [(frozen, old, old_lineage), (new, added, added_lineage)]:
        (root / "training.json").write_text(json.dumps(records))
        (root / "lineage.json").write_text(json.dumps(lineage))
    manifest = tmp_path / "sources.jsonl"
    manifest.write_text(
        "".join(
            json.dumps({"id": f, "split": role}) + "\n"
            for role, fs in [
                ("train", ["t0", "t1"]),
                ("validation", ["v0", "v1"]),
                ("calibration", ["c0", "c1"]),
            ]
            for f in fs
        )
    )
    return frozen, new, manifest


def build(data, output):
    from examples.domain.medical_triage.scripts.v1.hardmix import build_recipes

    return build_recipes(
        *data,
        output,
        replacement_counts=(2, 4),
        validation_count=2,
        calibration_count=2,
        minimum_families=2,
    )


def strata(records):
    result = Counter()
    for r in records:
        q = r["request"]["questions"]["q1"]
        if r["split"] == "train" and len(q["criteria"]) == 2:
            a = r["supervision"]["answers"]["q1"]["choice"]
            result[(a, q["criteria"][a])] += 1
    return result


def test_nested_recipes_keep_all_frozen_action_samples_and_both_label_priors(data, tmp_path):
    before = [(p, p.read_bytes()) for root in data[:2] for p in root.iterdir()]
    result = build(data, tmp_path / "recipes")
    old = json.loads((data[0] / "training.json").read_text())
    old_actions = [
        r
        for r in old
        if r["split"] == "train" and len(r["request"]["questions"]["q1"]["criteria"]) == 4
    ]
    replacements = []
    for name in ["control", "hard25", "hard50"]:
        path = tmp_path / "recipes" / name / "training.json"
        records = json.loads(path.read_text())
        assert len(list(read_training_records(path))) == 20
        assert sum(r["split"] == "train" for r in records) == 12
        assert strata(records) == strata(old)
        assert [
            r
            for r in records
            if r["split"] == "train" and len(r["request"]["questions"]["q1"]["criteria"]) == 4
        ] == old_actions
        role_groups = {
            role: {r["group_id"] for r in records if r["split"] == role}
            for role in ["train", "validation", "calibration"]
        }
        assert not role_groups["train"] & (role_groups["validation"] | role_groups["calibration"])
        assert not role_groups["validation"] & role_groups["calibration"]
        replacements.append(set(result["recipes"][name]["replaced_record_ids"]))
    assert len(replacements[1]) == 2 and len(replacements[2]) == 4
    assert replacements[1] < replacements[2]
    for p, content in before:
        assert p.read_bytes() == content
    for role in ["validation", "calibration"]:
        canonical = [
            (tmp_path / "recipes" / name / (role + ".jsonl")).read_bytes()
            for name in ["control", "hard25", "hard50"]
        ]
        assert canonical[0] == canonical[1] == canonical[2]


def test_new_heldout_collision_is_quarantined_without_deleting_old_train(data, tmp_path):
    old = json.loads((data[0] / "training.json").read_text())
    added = json.loads((data[1] / "training.json").read_text())
    victim = next(r for r in added if r["split"] == "validation")
    victim["request"]["state"] = old[4]["request"]["state"]
    (data[1] / "training.json").write_text(json.dumps(added))
    result = build(data, tmp_path / "recipes")
    assert result["quarantined_new"] == 1
    control = json.loads((tmp_path / "recipes/control/training.json").read_text())
    assert next(r for r in control if r["record_id"] == old[4]["record_id"]) == old[4]


def test_source_role_spoof_or_short_new_holdout_fails_closed(data, tmp_path):
    line = json.loads((data[1] / "lineage.json").read_text())
    key = next(iter(line))
    line[key]["provenance"]["source_id"] = "v0"
    (data[1] / "lineage.json").write_text(json.dumps(line))
    with pytest.raises(ValueError, match="role"):
        build(data, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_mirror_moves_full_patient_and_flips_gold_without_mutating_original():
    from examples.domain.medical_triage.scripts.v1.hardmix import mirror_samples

    s = {
        "schema_version": "decisionforge.eval.v1",
        "id": "validation:1",
        "group_id": "9",
        "split": "validation",
        "subset": "message_priority",
        "request": {
            "state": json.dumps(
                {
                    "patients": [
                        {"message": "one", "ehr": "first history"},
                        {"message": "two", "ehr": "second history"},
                    ]
                }
            ),
            "questions": {
                "decision": {
                    "type": "choice",
                    "instructions": "Compare.",
                    "criteria": PAIR_CRITERIA,
                }
            },
        },
        "target": {"kind": "hard", "value": "A"},
        "metadata": {},
    }
    before = copy.deepcopy(s)
    mirrored = mirror_samples([s])[0]
    assert s == before
    assert json.loads(mirrored["request"]["state"])["patients"] == list(
        reversed(json.loads(s["request"]["state"])["patients"])
    )
    assert mirrored["target"]["value"] == "B" and mirrored["group_id"] == "9"
    assert mirrored["metadata"]["paired_original_id"] == s["id"]
    twice = mirror_samples([mirrored])[0]
    assert twice["target"] == s["target"] and twice["request"]["state"] == s["request"]["state"]


def test_insufficient_new_holdout_does_not_leave_a_partial_recipe(data, tmp_path):
    added = json.loads((data[1] / "training.json").read_text())
    added = [r for r in added if r["split"] != "calibration"]
    (data[1] / "training.json").write_text(json.dumps(added))
    with pytest.raises(ValueError, match="Insufficient"):
        build(data, tmp_path / "incomplete")
    assert not (tmp_path / "incomplete").exists()


def test_protected_topup_at_train_cap_only_requests_deficient_calibration(data):
    from examples.domain.medical_triage.scripts.v1 import hardmix

    old = json.loads((data[0] / "training.json").read_text())
    added = json.loads((data[1] / "training.json").read_text())
    calibration = [r for r in added if r["split"] == "calibration"]
    # Raw calibration has enough rows; protection must remove old-training reuse.
    for record in calibration[:-1]:
        record["request"]["state"] = old[4]["request"]["state"]
    (data[1] / "training.json").write_text(json.dumps(added))
    assert hasattr(hardmix, "protected_data_needs"), "quota planning must use frozen protection"
    result = hardmix.protected_data_needs(
        *data,
        replacement_count=4,
        validation_count=2,
        calibration_count=2,
        minimum_families=2,
        role_rounds={"train": 6, "validation": 6, "calibration": 6},
        role_limits={"train": 6, "validation": 10, "calibration": 10},
    )
    assert result["missing_roles"] == ["calibration"]
    assert result["counts"]["calibration"]["rows"] == 1
    assert result["quarantined_new"] == 7
    with pytest.raises(ValueError, match="calibration"):
        hardmix.protected_data_needs(
            *data,
            replacement_count=4,
            validation_count=2,
            calibration_count=2,
            minimum_families=2,
            role_rounds={"train": 6, "validation": 6, "calibration": 10},
            role_limits={"train": 6, "validation": 10, "calibration": 10},
        )
