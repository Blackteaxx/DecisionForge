"""Freeze native Typed Decisions group roles before method-specific preparation.

First run prepare_data.sh to create the complete converted train/test source.
This offline step uses the native seed-3407 splitter and never downloads data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from decisionforge.training.config import TrainConfig
from decisionforge.training.data import prepare_data
from decisionforge.util import json_text


def freeze(source, output, seed=3407):
    source, output = Path(source).resolve(), Path(output).resolve()
    audit_path = output.with_suffix(".roles.json")
    if source == output:
        raise ValueError("Frozen-role output must differ from the original source")
    for path in (output, audit_path):
        if path.exists():
            raise FileExistsError(path)
    config = TrainConfig(
        data={"path": str(source)},
        training={"seed": seed},
        calibration={"enabled": True},
    )
    prepared = prepare_data(config)
    roles = {item["record_id"]: item["role"] for item in prepared.assignments}
    text = "".join(
        json_text({**record.model_dump(mode="json"), "split": roles[record.record_id]}) + "\n"
        for record in prepared.records
    )
    audit = {
        "schema_version": "decisionforge.training.frozen_roles.v1",
        "source": os.path.relpath(source, output.parent),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "seed": seed,
        "auto_holdout": config.data.auto_holdout.model_dump(),
        "operation": "native_group_role_assignment_only_public_requests_and_supervision_unchanged",
        "assignments": prepared.assignments,
        "report": prepared.report,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(text)
    with audit_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return audit


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root / "data/local/typed-decisions-training.jsonl")
    parser.add_argument("--output", type=Path, default=root / "data/local/typed-decisions-frozen-roles.v3.jsonl")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    audit = freeze(args.source, args.output, args.seed)
    print(json.dumps({"output": str(args.output), "roles": audit["report"]["roles"]}))


if __name__ == "__main__":
    main()
