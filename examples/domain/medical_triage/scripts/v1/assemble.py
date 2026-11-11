"""Combine completed native build outputs without losing provenance or role boundaries."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from decisionforge.exporter import read_training_records
from examples.domain.medical_triage.scripts.prepare_benchmarks import (
    PAIR_CRITERIA,
    TRIAGE_CRITERIA,
)
from examples.domain.medical_triage.scripts.v1.audit import normalized_patient_parents
from examples.domain.medical_triage.scripts.v1.experiment import evaluation_samples
from examples.domain.medical_triage.scripts.v1.generate import quarantine_role_overlaps


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _state_content(state):
    if isinstance(state, str):
        try:
            value = json.loads(state)
        except ValueError:
            return state
        if isinstance(value, (dict, list)):
            return value
    return state


def assemble_training(runs: list[Path], output: Path) -> dict:
    candidates, inputs = [], []
    for root in runs:
        meta_path, accepted_path = root / "meta.json", root / "accepted.json"
        meta, records = json.loads(meta_path.read_text()), json.loads(accepted_path.read_text())
        if meta["summary"]["status"] != "complete" or meta["summary"]["accepted"] != len(records):
            raise ValueError("Only completed, consistent native build outputs can be assembled")
        inputs.append(
            {
                "run": str(root),
                "accepted_sha256": hashlib.sha256(accepted_path.read_bytes()).hexdigest(),
                "meta_sha256": hashlib.sha256(meta_path.read_bytes()).hexdigest(),
            }
        )
        for record in records:
            detail = meta["records"][str(record["record_id"])]
            if detail["status"] != "accepted":
                raise ValueError("Native accepted record has no accepted checker metadata")
            source = detail["source"]["payload"]
            provenance = source["_provenance"]
            if provenance.get("synthetic") is not True:
                raise ValueError("Independent synthetic provenance is required")
            task = provenance["task"]
            expected = (
                TRIAGE_CRITERIA
                if task == "triage_action"
                else (PAIR_CRITERIA if task == "message_priority" else None)
            )
            if expected is None or len(record["request"]["questions"]) != 1:
                raise ValueError("Unknown or multiple medical decision tasks")
            qid, question = next(iter(record["request"]["questions"].items()))
            if question["type"] != "choice" or set(question["criteria"].values()) != set(
                expected.values()
            ):
                raise ValueError("Accepted task catalog differs from the fixed medical actions")
            selected = record["supervision"]["answers"][qid]["choice"]
            answer = question["criteria"][selected]
            if answer != source["answer"]:
                raise ValueError(
                    "Accepted gold answer content differs from immutable source answer"
                )
            if (
                record["split"] != source["split"]
                or record["request"]["state"] != source["context"]
            ):
                raise ValueError("Native build changed the original role or public patient facts")
            candidates.append(
                {
                    "id": source["id"],
                    "family_id": source["group_id"],
                    "split": source["split"],
                    "parent_ids": sorted(
                        set(provenance["parent_ids"])
                        | set(normalized_patient_parents(record["request"]["state"]))
                    ),
                    "task": task,
                    "answer": answer,
                    "record": record,
                    "provenance": provenance,
                    "context_identity": _digest([task, _state_content(record["request"]["state"])]),
                    "origin": {
                        "source_run": str(root),
                        "source_record_id": record["record_id"],
                        "scenario_id": source["id"],
                    },
                }
            )
    labels = defaultdict(set)
    for candidate in candidates:
        labels[candidate["context_identity"]].add(candidate["answer"])
    conflicts = {key for key, values in labels.items() if len(values) != 1}
    quarantined = [
        {"reason": "conflicting_teacher_labels", "scenario": c}
        for c in candidates
        if c["context_identity"] in conflicts
    ]
    candidates = [c for c in candidates if c["context_identity"] not in conflicts]
    retained, overlaps = quarantine_role_overlaps(candidates)
    quarantined.extend(overlaps)
    unique, duplicates = {}, 0
    for candidate in retained:
        key = candidate["context_identity"]
        if key in unique:
            duplicates += 1
            unique[key]["origins"].append(candidate["origin"])
        else:
            unique[key] = {**candidate, "origins": [candidate["origin"]]}
    parents = {}

    def find(key):
        parents.setdefault(key, key)
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    for candidate in unique.values():
        family = "family:" + candidate["family_id"]
        for parent in candidate["parent_ids"]:
            a, b = find(family), find(parent)
            parents[max(a, b)] = min(a, b)
        find(family)
    group_ids = {
        key: index + 1
        for index, key in enumerate(
            sorted({find("family:" + candidate["family_id"]) for candidate in unique.values()})
        )
    }
    records, lineage = [], {}
    for index, candidate in enumerate(sorted(unique.values(), key=lambda c: c["id"]), 1):
        record = copy.deepcopy(candidate["record"])
        record["record_id"] = index
        record["group_id"] = group_ids[find("family:" + candidate["family_id"])]
        records.append(record)
        lineage[index] = {
            "task": candidate["task"],
            "split": candidate["split"],
            "source_run": candidate["origin"]["source_run"],
            "source_record_id": candidate["origin"]["source_record_id"],
            "scenario_id": candidate["id"],
            "family_id": candidate["family_id"],
            "parent_ids": candidate["parent_ids"],
            "origins": candidate["origins"],
            "provenance": candidate["provenance"],
        }
    if not records:
        raise ValueError("No unambiguous role-isolated records remain after data audit")
    output.mkdir(parents=True, exist_ok=False)
    path = output / "training.json"
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    # The actual public native parser is the final contract gate; no mocked schema.
    validated = list(read_training_records(path))
    if len(validated) != len(records):
        raise ValueError("Native training parser changed assembled record count")
    (output / "lineage.json").write_text(json.dumps(lineage, ensure_ascii=False, indent=2) + "\n")
    (output / "quarantined.json").write_text(
        json.dumps(quarantined, ensure_ascii=False, indent=2) + "\n"
    )
    tasks = {int(key): value["task"] for key, value in lineage.items()}
    evaluation = {}
    for role in ["validation", "calibration"]:
        if any(r["split"] == role for r in records):
            samples = evaluation_samples(records, tasks, role)
            combined = output / (role + ".jsonl")
            content = "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in samples)
            combined.write_text(content)
            evaluation[role + ":all"] = {
                "count": len(samples),
                "sha256": hashlib.sha256(content.encode()).hexdigest(),
                "path": str(combined),
            }
            for task in ["triage_action", "message_priority"]:
                selected = [s for s in samples if s["subset"] == task]
                p = output / (role + "-" + task + ".jsonl")
                content = "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in selected)
                p.write_text(content)
                evaluation[role + ":" + task] = {
                    "count": len(selected),
                    "sha256": hashlib.sha256(content.encode()).hexdigest(),
                    "path": str(p),
                }
    result = {
        "schema_version": "medical.training-assembly.v1",
        "records": len(records),
        "groups": len(group_ids),
        "splits": dict(Counter(r["split"] for r in records)),
        "tasks": dict(Counter(c["task"] for c in unique.values())),
        "semantic_labels": dict(Counter(c["task"] + ":" + c["answer"] for c in unique.values())),
        "duplicates": duplicates,
        "quarantined": len(quarantined),
        "inputs": inputs,
        "training_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "independent_evaluation": evaluation,
        "benchmark_overlap_audit": "pending; required before formal training",
        "target_semantics": "synthetic_teacher_hard_reference_not_clinical_certainty",
    }
    (output / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(assemble_training(args.runs, args.output), indent=2))


if __name__ == "__main__":
    main()
