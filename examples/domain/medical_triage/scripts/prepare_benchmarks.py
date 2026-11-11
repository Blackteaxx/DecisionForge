"""Convert all official medical test rows without showing labels to the model."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from decisionforge.eval.models import Sample

CARE_LABELS = (
    "0A_NO_ESCALATION_INFO_NEEDED",
    "0B_NO_ESCALATION_SELF_CARE_MONITOR",
    "1A_ESCALATION_NONURGENT_CARE",
    "1B_ESCALATION_URGENT_CARE",
)
TRIAGE_CRITERIA = {
    "A": "Ask the smallest necessary clarification; current information is insufficient.",
    "B": "Offer self-care, monitoring, reassurance, or conditional safety-netting.",
    "C": "Recommend current nonurgent professional evaluation or care.",
    "D": "Recommend urgent or emergency-level evaluation now.",
}
TRIAGE_INSTRUCTIONS = (
    "Based only on the information currently available in the conversation, "
    "choose the action the medical assistant should take now."
)
PAIR_CRITERIA = {"A": "Prioritize the first patient.", "B": "Prioritize the second patient."}
PAIR_INSTRUCTIONS = (
    "Which of the two patient messages should the clinical team address first? "
    "Compare clinical urgency using the message and any provided medical history."
)


def _hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _sample(identity, group, subset, state, instructions, criteria, gold, metadata):
    sample = {
        "schema_version": "decisionforge.eval.v1",
        "id": identity,
        "group_id": group,
        "split": "test",
        "subset": subset,
        "request": {
            "state": state,
            "questions": {
                "decision": {"type": "choice", "instructions": instructions, "criteria": criteria}
            },
        },
        "target": {"kind": "hard", "value": gold},
        "metadata": metadata,
    }
    Sample.model_validate(sample)
    return sample


def prepare_care(rows: list[dict]) -> list[dict]:
    result = []
    for row in rows:
        if row["split"] != "public_test_1":
            continue
        label = row["gold_label"]
        if label not in CARE_LABELS:
            raise ValueError(f"Unknown CARE gold label: {label!r}")
        messages = row["input_messages"]
        if not messages or any(not m.get("content") for m in messages):
            raise ValueError("CARE input_messages must be nonempty")
        result.append(
            _sample(
                "care:" + row["prefix_id"],
                "care:" + row["case_id"],
                "triage_action",
                messages,
                TRIAGE_INSTRUCTIONS,
                TRIAGE_CRITERIA,
                "ABCD"[CARE_LABELS.index(label)],
                {
                    "dataset": "CARE-Bench",
                    "target_source": "official_gold_label",
                    "source_id": row["prefix_id"],
                    "source_case_id": row["case_id"],
                    "source_label": label,
                    "task": "triage_action",
                },
            )
        )
    _unique_ids(result)
    return result


def _pair_sample(identity, first, second, winner, seed, metadata):
    swapped = bool(int(_hash([seed, identity])[:8], 16) % 2)
    patients = [second, first] if swapped else [first, second]
    gold = "A" if (winner == 1) != swapped else "B"
    parents = ["message:" + _hash(p["message"]) for p in [first, second]]
    parents += ["ehr:" + _hash(p["ehr"]) for p in [first, second] if p.get("ehr")]
    return _sample(
        identity,
        None,
        "message_priority",
        {"patients": patients},
        PAIR_INSTRUCTIONS,
        PAIR_CRITERIA,
        gold,
        {
            **metadata,
            "task": "message_priority",
            "swapped": swapped,
            "position_seed": seed,
            "parent_ids": parents,
            "target_source": "official_pair_preference",
        },
    )


def _text(row, key):
    value = row[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must contain a nonempty string")
    return value


def _group_connected(samples):
    """All pairs sharing a patient belong to one component, not independent rows."""
    parents = {}

    def find(x):
        parents.setdefault(x, x)
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x

    for sample in samples:
        family = sample["metadata"]["parent_ids"]
        for parent in family[1:]:
            x, y = find(family[0]), find(parent)
            if x != y:
                parents[max(x, y)] = min(x, y)
    for sample in samples:
        sample["group_id"] = find(sample["metadata"]["parent_ids"][0])
    _unique_ids(samples)
    return samples


def prepare_synth(rows: list[dict], seed: int = 3407) -> list[dict]:
    samples = []
    for i, row in enumerate(rows):
        label = row["Label"]
        if label not in {"Patient #1", "Patient #2"}:
            raise ValueError(f"Unknown Synth label: {label!r}")
        samples.append(
            _pair_sample(
                f"pmr-synth:{i:04d}",
                {"message": _text(row, "MSG #1"), "ehr": _text(row, "EHR #1")},
                {"message": _text(row, "MSG #2"), "ehr": _text(row, "EHR #2")},
                1 if label == "Patient #1" else 2,
                seed,
                {"dataset": "PMP-Synth-AllPairs-Annotated", "source_row": i, "source_label": label},
            )
        )
    return _group_connected(samples)


def prepare_reddit(rows: list[dict], seed: int = 3407) -> list[dict]:
    samples = []
    for i, row in enumerate(rows):
        samples.append(
            _pair_sample(
                f"pmr-reddit:{i:04d}",
                {"message": _text(row, "chosen")},
                {"message": _text(row, "rejected")},
                1,
                seed,
                {
                    "dataset": "PMR-Reddit-Test-Pairs",
                    "source_row": i,
                    "chosen_level": row.get("chosen_level"),
                    "rejected_level": row.get("rejected_level"),
                    "equal_levels": row.get("chosen_level") == row.get("rejected_level"),
                    "difficulty": row.get("difficulty"),
                },
            )
        )
    return _group_connected(samples)


def _unique_ids(samples):
    ids = [s["id"] for s in samples]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate official sample ids")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    import pyarrow.parquet as pq

    care_path = args.raw_root / "care/evaluation_prefixes_public.jsonl"
    inputs = {
        "care": [care_path],
        "synth": sorted((args.raw_root / "synth").glob("*.parquet")),
        "reddit": sorted((args.raw_root / "reddit").glob("*.parquet")),
    }
    if len(inputs["synth"]) != 2 or len(inputs["reddit"]) != 1:
        raise ValueError("Expected two Synth parquet files and one Reddit test file")
    collections = {
        "care": prepare_care([json.loads(x) for x in care_path.read_text().splitlines() if x]),
        "synth": prepare_synth(
            [r for p in inputs["synth"] for r in pq.read_table(p).to_pylist()], args.seed
        ),
        "reddit": prepare_reddit(
            [r for p in inputs["reddit"] for r in pq.read_table(p).to_pylist()], args.seed
        ),
    }
    expected = {"care": 135, "synth": 869, "reddit": 1502}
    args.output.mkdir(parents=True, exist_ok=True)
    audit = {
        "schema_version": "medical-triage-benchmarks.v1",
        "position_seed": args.seed,
        "datasets": {},
    }
    for name, samples in collections.items():
        if len(samples) != expected[name]:
            raise ValueError(f"{name}: expected {expected[name]} rows, received {len(samples)}")
        path = args.output / (name + ".jsonl")
        content = "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in samples)
        path.write_text(content)
        audit["datasets"][name] = {
            "count": len(samples),
            "sha256": hashlib.sha256(content.encode()).hexdigest(),
            "path": str(path.resolve()),
            "input_hashes": {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs[name]
            },
            "labels": dict(Counter(s["target"]["value"] for s in samples)),
            "groups": len({s["group_id"] for s in samples}),
            "equal_level_rows": sum(s["metadata"].get("equal_levels", False) for s in samples),
        }
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
