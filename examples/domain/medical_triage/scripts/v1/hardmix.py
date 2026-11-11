"""Build fixed-budget data ablations while protecting already-trained frozen patients."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from decisionforge.eval.models import Sample
from decisionforge.exporter import read_training_records
from examples.domain.medical_triage.scripts.prepare_benchmarks import PAIR_CRITERIA
from examples.domain.medical_triage.scripts.v1.audit import normalized_patient_parents
from examples.domain.medical_triage.scripts.v1.experiment import evaluation_samples


def _hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _task(record):
    question = next(iter(record["request"]["questions"].values()))
    return "message_priority" if len(question["criteria"]) == 2 else "triage_action"


def _stratum(record):
    key, question = next(iter(record["request"]["questions"].items()))
    answer = record["supervision"]["answers"][key]["choice"]
    return answer + "|" + question["criteria"][answer]


def _parents(record, lineage):
    return (
        set(lineage["parent_ids"])
        | set(normalized_patient_parents(record["request"]["state"]))
        | {"family:" + lineage["family_id"]}
    )


def _diverse(rows, count, minimum_families, seed):
    grouped = defaultdict(list)
    for row in sorted(rows, key=lambda d: _hash([seed, d["key"]])):
        grouped[row["family"]].append(row)
    if len(grouped) < minimum_families or len(rows) < count:
        raise ValueError("Insufficient new heldout records or source families")
    families = sorted(grouped, key=lambda key: _hash([seed, key]))
    result = []
    for index in range(max(map(len, grouped.values()))):
        for family in families:
            if index < len(grouped[family]):
                result.append(grouped[family][index])
                if len(result) == count:
                    return result
    raise ValueError("Insufficient heldout records")


def mirror_samples(samples):
    result = []
    for original in samples:
        if original["subset"] != "message_priority":
            continue
        sample = copy.deepcopy(original)
        question = next(iter(sample["request"]["questions"].values()))
        if question["criteria"] != PAIR_CRITERIA or sample["target"]["value"] not in {"A", "B"}:
            raise ValueError("Mirror requires canonical binary patient-priority labels")
        state = sample["request"]["state"]
        encoded = isinstance(state, str)
        state = json.loads(state) if encoded else state
        if not isinstance(state, dict) or len(state["patients"]) != 2:
            raise ValueError("Mirror requires two complete patient records")
        state["patients"] = list(reversed(state["patients"]))
        sample["request"]["state"] = json.dumps(state, ensure_ascii=False) if encoded else state
        sample["target"]["value"] = "B" if sample["target"]["value"] == "A" else "A"
        sample["id"] += ":mirror"
        sample.setdefault("metadata", {}).update(paired_original_id=original["id"], view="swapped")
        Sample.model_validate(sample)
        result.append(sample)
    return result


def _protected_entries(frozen_dir, new_assembled_dir, source_manifest):
    sources = {s["id"]: s for s in map(json.loads, source_manifest.read_text().splitlines())}
    all_old, all_new = [], []
    for root, destination, frozen in [
        (frozen_dir, all_old, True),
        (new_assembled_dir, all_new, False),
    ]:
        records = json.loads((root / "training.json").read_text())
        if len(list(read_training_records(root / "training.json"))) != len(records):
            raise ValueError("Native parser changed input record count")
        lineage = json.loads((root / "lineage.json").read_text())
        for record in records:
            detail = lineage[str(record["record_id"])]
            if record["split"] not in {"train", "validation", "calibration"}:
                raise ValueError("Only immutable train/validation/calibration roles are allowed")
            if detail["split"] != record["split"] or detail["task"] != _task(record):
                raise ValueError("Lineage task or role disagrees with native record")
            if not frozen:
                source = sources[detail["provenance"]["source_id"]]
                if source["split"] != record["split"]:
                    raise ValueError("New source role cannot change")
                if detail["family_id"] != "nlm-family:" + source.get("family_id", source["id"]):
                    raise ValueError("New source family cannot change")
                if (
                    detail["provenance"].get("generation_profile") != "hard_priority_v2"
                    or _task(record) != "message_priority"
                ):
                    raise ValueError("Only checked hard-priority records may enter new data")
            destination.append(
                {
                    "record": record,
                    "lineage": detail,
                    "key": detail["scenario_id"],
                    "family": detail["family_id"],
                    "role": record["split"],
                    "parents": _parents(record, detail),
                    "frozen": frozen,
                }
            )
    role_by_parent, old_contexts = {}, set()
    for entry in all_old:
        state = entry["record"]["request"]["state"]
        old_contexts.add(_hash(json.loads(state) if isinstance(state, str) else state))
        for parent in entry["parents"]:
            if parent in role_by_parent and role_by_parent[parent] != entry["role"]:
                raise ValueError("Frozen original data already has a cross-role parent")
            role_by_parent[parent] = entry["role"]
    retained, quarantine = [], []
    for entry in all_new:
        state = entry["record"]["request"]["state"]
        conflict = any(
            p in role_by_parent and role_by_parent[p] != entry["role"] for p in entry["parents"]
        )
        duplicate = _hash(json.loads(state) if isinstance(state, str) else state) in old_contexts
        if conflict or duplicate:
            quarantine.append(
                {
                    "scenario_id": entry["key"],
                    "reason": "frozen_cross_role_overlap"
                    if conflict
                    else "frozen_context_duplicate",
                }
            )
        else:
            retained.append(entry)
    if len({d["key"] for d in retained}) != len(retained):
        raise ValueError("Duplicate new scenario identities")
    return all_old, retained, quarantine


def protected_data_needs(
    frozen_dir,
    new_assembled_dir,
    source_manifest,
    *,
    replacement_count=1187,
    validation_count=208,
    calibration_count=266,
    minimum_families=12,
    seed=3407,
    role_rounds=None,
    role_limits=None,
):
    """Plan only deficient roles using the exact recipe collision protection."""
    old, retained, quarantine = _protected_entries(
        Path(frozen_dir), Path(new_assembled_dir), Path(source_manifest)
    )
    counts, missing = {}, []
    for role, target in [("validation", validation_count), ("calibration", calibration_count)]:
        selected = [d for d in retained if d["role"] == role]
        families = len({d["family"] for d in selected})
        counts[role] = dict(rows=len(selected), families=families, target=target)
        if len(selected) < target or families < minimum_families:
            missing.append(role)
    strata = defaultdict(list)
    for d in sorted(old, key=lambda d: _hash([seed, "old", d["key"]])):
        if d["role"] == "train" and _task(d["record"]) == "message_priority":
            strata[_stratum(d["record"])].append(d)
    ordered = sorted(
        [
            ((i + 0.5) / len(values), key, d)
            for key, values in strata.items()
            for i, d in enumerate(values)
        ],
        key=lambda value: (value[0], value[1]),
    )
    required = Counter(_stratum(d["record"]) for _, _, d in ordered[:replacement_count])
    actual = Counter(_stratum(d["record"]) for d in retained if d["role"] == "train")
    counts["train"] = dict(
        rows=sum(actual.values()), required_strata=dict(required), actual_strata=dict(actual)
    )
    if any(actual[key] < count for key, count in required.items()):
        missing.append("train")
    if role_rounds is not None and role_limits is not None:
        for role in missing:
            if role_rounds[role] >= role_limits[role]:
                raise ValueError(f"Bounded generation budget reached for {role}")
    return dict(missing_roles=sorted(missing), counts=counts, quarantined_new=len(quarantine))


def build_recipes(
    frozen_dir,
    new_assembled_dir,
    source_manifest,
    output,
    *,
    replacement_counts=(594, 1187),
    validation_count=208,
    calibration_count=266,
    minimum_families=12,
    seed=3407,
):
    frozen_dir, new_assembled_dir, source_manifest, output = map(
        Path, [frozen_dir, new_assembled_dir, source_manifest, output]
    )
    if output.exists():
        raise ValueError("Recipe output already exists; retain immutable experiment artifacts")
    all_old, retained, quarantine = _protected_entries(
        frozen_dir, new_assembled_dir, source_manifest
    )
    # Rebuild connected groups, preserving an old group's number whenever no merge
    # is needed. Shared new patients must not split a source/patient component.
    parents, node_roles = {}, defaultdict(set)

    def find(key):
        parents.setdefault(key, key)
        if parents[key] != key:
            parents[key] = find(parents[key])
        return parents[key]

    for d in all_old + retained:
        family = "family:" + d["family"]
        nodes = [family, *d["parents"]]
        if d["frozen"]:
            nodes.append("oldgroup:" + str(d["record"]["group_id"]))
        for node in nodes:
            a, b = find(family), find(node)
            parents[max(a, b)] = min(a, b)
            node_roles[node].add(d["role"])
    group_roles, old_numbers = defaultdict(set), defaultdict(set)
    for node, roles in node_roles.items():
        group_roles[find(node)].update(roles)
        if node.startswith("oldgroup:"):
            old_numbers[find(node)].add(int(node.split(":", 1)[1]))
    if any(len(roles) != 1 for roles in group_roles.values()):
        raise ValueError("New data introduces a cross-role connected group")
    next_group = max(d["record"]["group_id"] for d in all_old)
    groups = {}
    for key in sorted(group_roles):
        if old_numbers[key]:
            groups[key] = min(old_numbers[key])
        else:
            next_group += 1
            groups[key] = next_group
    for d in all_old + retained:
        d["group"] = groups[find("family:" + d["family"])]
    next_id = max(d["record"]["record_id"] for d in all_old)
    for index, d in enumerate(sorted(retained, key=lambda v: v["key"]), next_id + 1):
        d["new_id"] = index
    held = []
    for role, count in [("validation", validation_count), ("calibration", calibration_count)]:
        held.extend(
            _diverse([d for d in retained if d["role"] == role], count, minimum_families, seed)
        )
    old_priority = [
        d for d in all_old if d["role"] == "train" and _task(d["record"]) == "message_priority"
    ]
    if len(replacement_counts) != 2 or not 0 < replacement_counts[0] < replacement_counts[1] < len(
        old_priority
    ):
        raise ValueError("Need two increasing priority replacements below the original budget")
    old_by_stratum, new_by_stratum = defaultdict(list), defaultdict(list)
    for d in sorted(old_priority, key=lambda v: _hash([seed, "old", v["key"]])):
        old_by_stratum[_stratum(d["record"])].append(d)
    for d in sorted(retained, key=lambda v: _hash([seed, "new", v["key"]])):
        if d["role"] == "train":
            new_by_stratum[_stratum(d["record"])].append(d)
    ordered = sorted(
        [
            ((i + 0.5) / len(values), key, d)
            for key, values in old_by_stratum.items()
            for i, d in enumerate(values)
        ],
        key=lambda v: (v[0], v[1]),
    )
    plans = {}
    for name, count in [
        ("control", 0),
        ("hard25", replacement_counts[0]),
        ("hard50", replacement_counts[1]),
    ]:
        removed = [d for _, _, d in ordered[:count]]
        needed = Counter(_stratum(d["record"]) for d in removed)
        if any(len(new_by_stratum[k]) < n for k, n in needed.items()):
            raise ValueError(
                "Insufficient new training cases for fixed label/semantic-position strata"
            )
        positions = Counter()
        replacements = {}
        for d in removed:
            key = _stratum(d["record"])
            replacements[d["record"]["record_id"]] = new_by_stratum[key][positions[key]]
            positions[key] += 1
        records, lineage = [], {}
        for old in all_old:
            rid = old["record"]["record_id"]
            d = replacements.get(rid, old)
            record, detail = copy.deepcopy(d["record"]), copy.deepcopy(d["lineage"])
            record.update(record_id=rid, group_id=d["group"])
            if rid in replacements:
                detail["replaced_frozen_record_id"] = rid
            records.append(record)
            lineage[str(rid)] = detail
        for d in sorted(held, key=lambda v: v["new_id"]):
            record = copy.deepcopy(d["record"])
            record.update(record_id=d["new_id"], group_id=d["group"])
            records.append(record)
            lineage[str(record["record_id"])] = copy.deepcopy(d["lineage"])
        plans[name] = records, lineage, sorted(replacements)
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema_version": "medical.hardmix.v1",
        "seed": seed,
        "quarantined_new": len(quarantine),
        "new_retained": len(retained),
        "recipes": {},
        "inputs": {
            str(root / name): hashlib.sha256((root / name).read_bytes()).hexdigest()
            for root in [frozen_dir, new_assembled_dir]
            for name in ["training.json", "lineage.json"]
        },
        "source_manifest_sha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
    }
    reference = {}
    for name, (records, lineage, removed) in plans.items():
        root = output / name
        root.mkdir()
        path = root / "training.json"
        path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
        if len(list(read_training_records(path))) != len(records):
            raise ValueError("Native recipe parser changed record count")
        (root / "lineage.json").write_text(json.dumps(lineage, ensure_ascii=False, indent=2) + "\n")
        tasks = {int(k): v["task"] for k, v in lineage.items()}
        for role in ["validation", "calibration"]:
            samples = evaluation_samples(records, tasks, role)
            for s in samples:
                detail = lineage[str(s["metadata"]["source_record_id"])]
                s["metadata"]["data_cohort"] = (
                    "hard" if detail["scenario_id"] in {d["key"] for d in held} else "original"
                )
            content = "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in samples)
            if role in reference and reference[role] != content:
                raise ValueError("Ablations must share byte-identical holdouts")
            reference[role] = content
            (root / (role + ".jsonl")).write_text(content)
            if role == "validation":
                (root / "validation-mirror.jsonl").write_text(
                    "".join(
                        json.dumps(s, ensure_ascii=False) + "\n" for s in mirror_samples(samples)
                    )
                )
        result["recipes"][name] = {
            "training_json": str(path),
            "training_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "records": len(records),
            "splits": dict(Counter(r["split"] for r in records)),
            "replaced_record_ids": removed,
            "priority_strata": dict(
                Counter(
                    _stratum(r)
                    for r in records
                    if r["split"] == "train" and _task(r) == "message_priority"
                )
            ),
        }
    (output / "quarantined-new.json").write_text(json.dumps(quarantine, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--new-assembled", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build_recipes(args.frozen, args.new_assembled, args.source_manifest, args.output),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
