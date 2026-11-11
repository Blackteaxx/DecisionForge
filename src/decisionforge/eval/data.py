from __future__ import annotations

import json
from pathlib import Path

from decisionforge.eval.models import DataConfig, Sample
from decisionforge.util import atomic_write


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def parse_json(text):
    return json.loads(text, object_pairs_hook=unique_object)


def training_samples(data):
    """Interface adapter only; metrics and native model execution remain unchanged."""
    from decisionforge.models import TrainingExample

    record = TrainingExample.model_validate(data)
    for name, question in record.request.questions.items():
        answer = record.supervision.answers[name]
        probabilities = (
            {"false": 1.0 - answer.noul, "true": answer.noul}
            if question.type == "noul"
            else answer.probabilities
        )
        hard = next((key for key, value in probabilities.items() if value == 1.0), None)
        target = (
            {
                "kind": "hard",
                "value": (
                    hard == "true"
                    if question.type == "noul"
                    else int(hard)
                    if question.type == "score"
                    else hard
                ),
            }
            if hard is not None
            else {"kind": "soft", "probabilities": probabilities}
        )
        yield Sample.model_validate(
            {
                "schema_version": "decisionforge.eval.v1",
                "id": f"{record.record_id}:{name}",
                "group_id": str(record.group_id),
                "split": record.split,
                "subset": question.type,
                "request": {
                    **{k: v for k, v in record.request.model_dump(mode="json").items() if k != "questions"},
                    "questions": {"decision": question.model_dump(mode="json")},
                },
                "target": target,
                "metadata": {"target_source": "decisionforge.training.v3"},
            }
        )


def load_samples(path: str | Path, *, resolve_media=True) -> list[Sample]:
    path = Path(path)
    result, seen = [], set()
    with path.open(encoding="utf-8") as stream:
        if path.suffix.lower() == ".json":
            values = parse_json(stream.read())
            if not isinstance(values, list):
                raise ValueError(f"{path}: JSON dataset must be an array")
            rows = enumerate(values, 1)
        else:
            rows = ((number, line) for number, line in enumerate(stream, 1) if line.strip())
        for line_number, raw in rows:
            identity = "unknown"
            try:
                data = parse_json(raw) if isinstance(raw, str) else raw
                identity = data.get("id", "unknown") if isinstance(data, dict) else "unknown"
                values = (
                    training_samples(data)
                    if isinstance(data, dict)
                    and data.get("schema_version") == "decisionforge.training.v3"
                    else [Sample.model_validate(data)]
                )
                for value in values:
                    from decisionforge.media import has_media, resolve_request

                    if resolve_media and has_media(value.request):
                        resolve_request(value.request, path.resolve().parent)
                    if value.id in seen:
                        raise ValueError(f"duplicate sample id {value.id!r}")
                    seen.add(value.id)
                    result.append(value)
            except ValueError as exc:
                raise ValueError(f"{path}: line {line_number}, id={identity}: {exc}") from exc
    if not result:
        raise ValueError(f"{path}: empty evaluation dataset")
    return result


def select_samples(samples: list[Sample], config: DataConfig) -> list[Sample]:
    selected = [
        s
        for s in samples
        if (config.splits is None or s.split in config.splits)
        and (config.subsets is None or s.subset in config.subsets)
    ]
    if config.limit is not None:
        selected = selected[: config.limit]
    if not selected:
        raise ValueError("No evaluation samples match the selection")
    return selected


def write_jsonl(path: Path, values) -> None:
    atomic_write(
        path,
        "".join(json.dumps(v, ensure_ascii=False, allow_nan=False) + "\n" for v in values),
    )
