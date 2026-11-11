from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from decisionforge.exporter import read_training_records
from decisionforge.models import TrainingExample
from decisionforge.util import atomic_write, json_text

ROLES = ("train", "validation", "calibration", "test")
MIN_CALIBRATION_QUESTIONS = 10  # Official Unsloth decision calibration floor.


def to_unsloth_row(record: TrainingExample) -> dict:
    gold = {}
    for key, question in record.request.questions.items():
        answer = record.supervision.answers[key]
        probabilities = (
            {"false": 1 - answer.noul, "true": answer.noul}
            if question.type == "noul"
            else dict(answer.probabilities)
        )
        gold[key] = {"probabilities": probabilities}
    return {**record.request.model_dump(exclude_none=True), "gold": gold}


def counts(records):
    types = Counter(q.type for r in records for q in r.request.questions.values())
    return {
        "records": len(records),
        "questions": sum(types.values()),
        "groups": len({r.group_id for r in records}),
        "types": dict(types),
    }


@dataclass
class PreparedData:
    records: list[TrainingExample]
    assignments: list[dict]
    report: dict

    @property
    def roles(self):
        by_id = {a["record_id"]: a["role"] for a in self.assignments}
        return {role: [r for r in self.records if by_id[r.record_id] == role] for role in ROLES}


def export_splits(prepared: PreparedData, directory: Path):
    """Expose effective holdouts to eval without changing the original sample snapshot."""
    for role, records in prepared.roles.items():
        atomic_write(
            directory / "data" / f"{role}.jsonl",
            "".join(
                json_text({**record.model_dump(mode="json"), "split": role}) + "\n"
                for record in records
            ),
        )


def prepare_data(config) -> PreparedData:
    from decisionforge.training.backend import capabilities

    minimum = capabilities(config.backend).calibration_min_questions
    path = Path(config.data.path)
    records = list(read_training_records(path))
    from decisionforge.media import has_media, resolve_request

    for record in records:
        if has_media(record.request):
            resolve_request(record.request, path.resolve().parent)
    if not records:
        raise ValueError(f"{path}: Empty training data")
    ids, groups, requests, assignments = set(), {}, {}, []
    for position, r in enumerate(records, 1):
        where = f"{path}: Record {position}, record_id={r.record_id}"
        if r.record_id in ids:
            raise ValueError(f"{where}: Duplicate record ID")
        ids.add(r.record_id)
        if r.split not in config.data.split_map:
            raise ValueError(f"{where}: Unknown split {r.split!r}; configure data.split_map")
        if r.group_id in groups and groups[r.group_id] != r.split:
            raise ValueError(f"{where}: Same source group {r.group_id} crosses splits")
        groups[r.group_id] = r.split
        # A canonical object representation, no hash or content fingerprint.
        public = json_text(r.request.model_dump())
        if public in requests:
            raise ValueError(f"{where}: Duplicate public request of record_id={requests[public]}")
        requests[public] = r.record_id
        assignments.append(
            {
                "record_id": r.record_id,
                "group_id": r.group_id,
                "original_split": r.split,
                "role": config.data.split_map[r.split],
                "source": "existing",
            }
        )
    train_groups = sorted({a["group_id"] for a in assignments if a["role"] == "train"})
    random.Random(config.training.seed).shuffle(train_groups)
    original_groups = len(train_groups)
    for role, fraction, required in (
        ("validation", config.data.auto_holdout.validation_fraction, True),
        ("calibration", config.data.auto_holdout.calibration_fraction, config.calibration.enabled),
    ):
        if required and not any(a["role"] == role for a in assignments):
            n = int(original_groups * fraction)
            if n < 1 or n >= len(train_groups):
                raise ValueError(
                    f"{path}: Insufficient train groups for nonempty {role} holdout; "
                    "provide explicit splits or increase data/fraction"
                )
            held, train_groups = set(train_groups[:n]), train_groups[n:]
            for a in assignments:
                if a["role"] == "train" and a["group_id"] in held:
                    a.update(role=role, source="auto_holdout")
    prepared = PreparedData(records, assignments, {})
    roles = prepared.roles
    if not roles["train"]:
        raise ValueError(f"{path}: Empty train split")
    if config.calibration.enabled and counts(roles["calibration"])["questions"] < minimum:
        raise ValueError(
            f"{path}: Calibration needs at least {minimum} usable "
            "questions; provide more calibration samples or disable calibration"
        )
    prepared.report = {
        "total": counts(records),
        "roles": {k: counts(v) for k, v in roles.items()},
        "original_splits": dict(Counter(r.split for r in records)),
    }
    return prepared
