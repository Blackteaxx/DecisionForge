from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from decisionforge.config import OutputConfig
from decisionforge.models import LegacyTrainingExample, TrainingExample, TrainingRecord
from decisionforge.storage import Store
from decisionforge.util import atomic_write, digest, json_text, write_json


def numbered_id(mapping: dict, key: str) -> int:
    return mapping.setdefault(key, len(mapping) + 1)


def numbered_group(ids: dict, key: str, *, validation=False) -> int:
    """Allocate original and held-out groups in one collision-free integer namespace."""
    mapping = ids.setdefault("validation_groups" if validation else "groups", {})
    if key not in mapping:
        mapping[key] = 1 + max(
            [0, *ids["groups"].values(), *ids.get("validation_groups", {}).values()]
        )
    return mapping[key]


def assign_validation(accepted: list[dict], metadata: dict, ids: dict, output: OutputConfig):
    """Opt-in question holdout at export; leave checkpoint outcomes and source splits intact."""
    if not output.validation_fraction:
        return None
    candidates = [row for row in accepted if row["split"] == "train"]
    candidates.sort(
        key=lambda row: (
            digest([output.validation_seed, metadata[str(row["record_id"])]["source_record_id"]]),
            metadata[str(row["record_id"])]["source_record_id"],
        )
    )
    total = len(candidates)
    count = min(total - 1, max(1, int(total * output.validation_fraction))) if total >= 2 else 0
    held = {row["record_id"] for row in candidates[:count]}
    for row in accepted:
        original_group, original_split = row["group_id"], row["split"]
        if row["record_id"] in held:
            row["split"] = "validation"
            row["group_id"] = numbered_group(ids, str(original_group), validation=True)
        metadata[str(row["record_id"])]["split_assignment"] = {
            "original_group_id": original_group,
            "original_split": original_split,
            "group_id": row["group_id"],
            "split": row["split"],
        }
    return {
        "strategy": "accepted_train_records",
        "validation_fraction": output.validation_fraction,
        "seed": output.validation_seed,
        "eligible_records": total,
        "validation_records": count,
        "train_records": total - count,
        "effective_fraction": count / total if total else 0.0,
        "status": "assigned" if total >= 2 else "insufficient_records",
    }


def failure_case(value: dict) -> dict:
    """Keep the original case and its locator, without duplicating internal evidence objects."""
    source = value.get("source") or {}
    locator = {key: source[key] for key in ("source", "row", "page") if key in source}
    case = source.get("raw", source.get("payload"))
    if case is None:
        case = {
            key: source[key]
            for key in ("question", "context", "answer", "options", "quotes", "assumptions")
            if key in source
        }
    result = {
        "record_id": value["record_id"],
        "status": value["status"],
        "reason": value["reason"],
    }
    if locator:
        result["source"] = locator
    if case:
        result["case"] = case
    return result


def export_run(
    store: Store, directory: Path, export_requests=False, include_audit=False, export_api=False
) -> dict:
    accepted, failed, metadata = [], [], {}
    counts, modes = Counter(), Counter()
    config = store.get("effective_config") or {}
    output = OutputConfig.model_validate(config.get("output") or {})
    ids = store.get("export_ids") or {"records": {}, "groups": {}}
    for identifier, status, value in store.outcomes():
        if status == "complete":
            continue
        number = numbered_id(ids["records"], identifier)
        counts[status] += 1
        record_data = value if status == "accepted" else value.get("jev")
        row = None
        if record_data:
            record = TrainingRecord.model_validate(record_data)
            example = TrainingExample.from_record(
                record, record_id=number, group_id=numbered_group(ids, record.group_id)
            )
            row = example.model_dump(mode="json")
            (accepted if status == "accepted" else failed).append(row)
            if status == "accepted":
                for question in example.request.questions.values():
                    modes[question.type] += 1
        detail = store.get("generation_meta:" + identifier) or value
        metadata[str(number)] = {
            "source_record_id": identifier,
            "status": status,
            "output": "accepted.json" if status == "accepted" else "fail.json" if row else None,
            **{k: v for k, v in detail.items() if k not in {"jev", "request", "supervision"}},
        }
    split_report = assign_validation(accepted, metadata, ids, output)
    store.put("export_ids", ids)
    accepted.sort(key=lambda r: r["record_id"])
    failed.sort(key=lambda r: r["record_id"])
    summary = {
        "accepted": len(accepted),
        "failed": len(failed),
        "input_failures": sum(v["output"] is None for v in metadata.values()),
        "review": counts["review"],
        "rejected": counts["rejected"],
        "status": store.get("run_status"),
        "modes": dict(modes),
        "splits": dict(Counter(row["split"] for row in accepted)),
        "performance": {
            **(store.get("run_metrics") or {}),
            "llm_api_attempts": store.db.execute("SELECT COUNT(*) FROM charges").fetchone()[0],
        },
        "recorded_or_reserved_cost_usd": None
        if store.get("pricing_complete") is False
        else store.spent(),
    }
    if split_report is not None:
        summary["validation_split"] = split_report
    write_json(directory / "accepted.json", accepted)
    write_json(directory / "fail.json", failed)
    events = [
        {"event_id": event, "source_record_id": record, "stage": stage, **json.loads(value)}
        for event, record, stage, value in store.db.execute("SELECT * FROM events ORDER BY id")
    ]
    write_json(
        directory / "meta.json",
        {
            "schema_version": "decisionforge.meta.v2",
            "summary": summary,
            "config": store.get("effective_config"),
            "records": metadata,
            "events": events,
            "text_batching": store.get("text_batching"),
            "active_text_batch": store.get("active_text_batch"),
            "active_text_batches": store.get("active_text_batches") or [],
        },
    )
    rows = {}
    if export_requests:
        rows["requests.jsonl"] = [r["request"] for r in accepted]
        rows["targets.jsonl"] = [
            {k: r[k] for k in ("record_id", "group_id", "split", "supervision")} for r in accepted
        ]
    if export_api:
        config = store.get("effective_config") or {}
        model = config.get("output", {}).get("jev_model", "jev-latest")
        rows["jev_requests.jsonl"] = [{"model": model, **r["request"]} for r in accepted]
    if include_audit:
        rows["audit.jsonl"] = events
    for name, items in rows.items():
        atomic_write(directory / name, "".join(json_text(row) + "\n" for row in items))
    return summary


def read_training_records(path: Path):
    """Read the default JSON array or legacy/explicit JSONL without changing its line format."""
    record_ids, group_ids = {}, {}
    with path.open(encoding="utf-8") as stream:
        if path.suffix.lower() == ".json":
            try:
                values = json.load(stream)
            except ValueError as exc:
                raise ValueError(f"{path}: Invalid JSON: {exc}") from exc
            if not isinstance(values, list):
                raise ValueError(f"{path}: Training JSON must be an array of records")
            values = enumerate(values, 1)
        else:

            def lines():
                for line_number, line in enumerate(stream, 1):
                    if line.strip():
                        try:
                            yield line_number, json.loads(line)
                        except ValueError as exc:
                            raise ValueError(
                                f"{path}: line {line_number}: Invalid JSON: {exc}"
                            ) from exc

            values = lines()
        for number, value in values:
            try:
                if not isinstance(value, dict):
                    raise ValueError("Training record must be an object")
                version = value.get("schema_version", "decisionforge.training.v1")
                if version in {"decisionforge.training.v1", "decisionforge.training.v2"}:
                    legacy = (
                        LegacyTrainingExample.from_record(TrainingRecord.model_validate(value))
                        if version == "decisionforge.training.v1"
                        else LegacyTrainingExample.model_validate(value)
                    )
                    yield TrainingExample.from_legacy(
                        legacy,
                        record_id=numbered_id(record_ids, legacy.record_id),
                        group_id=numbered_id(group_ids, legacy.group_id),
                    )
                else:
                    yield TrainingExample.model_validate(value)
            except ValueError as exc:
                identity = (
                    value.get("record_id", "unknown") if isinstance(value, dict) else "unknown"
                )
                questions = (
                    value.get("request", {}).get("questions", {})
                    if isinstance(value, dict) and isinstance(value.get("request"), dict)
                    else {}
                )
                raise ValueError(
                    f"{path}: {'array position' if path.suffix.lower() == '.json' else 'line'} {number}: "
                    f"Record {number}, record_id={identity}, "
                    f"questions={list(questions) if isinstance(questions, dict) else '<invalid>'}: {exc}"
                ) from exc


def validate_file(path: Path) -> int:
    groups, seen, fingerprints = {}, set(), {}
    count = 0
    for number, record in enumerate(read_training_records(path), 1):
        try:
            if record.record_id in seen:
                raise ValueError("Duplicate record ID")
            seen.add(record.record_id)
            split = record.split
            if record.group_id in groups and groups[record.group_id] != split:
                raise ValueError("Same source group crosses splits")
            groups[record.group_id] = split
            fingerprint = digest(record.request.model_dump())
            if fingerprint in fingerprints:
                raise ValueError("Duplicate public request")
            fingerprints[fingerprint] = split
        except (ValueError, KeyError) as exc:
            raise ValueError(f"Record {number}: {exc}") from exc
        count += 1
    return count


def export_chat_sft(directory: Path) -> Path:
    source = directory / "accepted.json"
    if not source.exists():
        source = directory / "accepted.jsonl"
    validate_file(source)
    split_rows = {"train": []}
    for record in read_training_records(source):
        split_rows.setdefault(record.split, []).append(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "Evaluate each question independently. Return a JSON object with answers keyed by question name. Choice answers contain type, choice, confidence and probabilities; Score answers contain type, score, confidence, legend and probabilities; Noul answers contain type and noul.",
                    },
                    {
                        "role": "user",
                        "content": json_text(
                            {
                                "state": record.request.state,
                                "questions": {
                                    k: q.model_dump() for k, q in record.request.questions.items()
                                },
                            }
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": json_text(record.supervision.model_dump()),
                    },
                ]
            }
        )
    path = directory / "chat_sft.jsonl"
    split_files = {}
    for split, rows in split_rows.items():
        destination = (
            path if split == "train" else directory / f"chat_sft.split_{digest(split)[:12]}.jsonl"
        )
        atomic_write(destination, "".join(json_text(row) + "\n" for row in rows))
        split_files[split] = {"file": destination.name, "records": len(rows)}
    write_json(directory / "chat_sft.splits.json", split_files)
    return path


def export_canonical_jsonl(directory: Path) -> Path:
    source = directory / "accepted.json"
    validate_file(source)
    destination = directory / "accepted.jsonl"
    atomic_write(
        destination,
        "".join(
            json_text(row.model_dump(mode="json")) + "\n" for row in read_training_records(source)
        ),
    )
    return destination
