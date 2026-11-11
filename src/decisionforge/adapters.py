"""Generation input IO; PDF is prepared as TXT before record ingestion."""

from __future__ import annotations

import json
from pathlib import Path

from decisionforge.models import RawRecord
from decisionforge.util import NeedsReview, digest, json_text


def input_kind(path: Path) -> str:
    if path.is_dir():
        return "pdf"
    kinds = {".json": "json", ".txt": "txt", ".pdf": "pdf"}
    if path.suffix.lower() not in kinds:
        raise ValueError("Generation supports .json, .txt, .pdf or a PDF directory")
    return kinds[path.suffix.lower()]


def field(payload, path):
    if path is None:
        return None
    if isinstance(payload, dict) and path in payload:
        return payload[path]
    value = payload
    try:
        for part in path.split("."):
            value = value[int(part)] if isinstance(value, list) else value[part]
    except (KeyError, IndexError, ValueError, TypeError) as exc:
        raise NeedsReview(f"Mapping path missing or incompatible: {path}") from exc
    return value


def json_records(value, records_path=None):
    if records_path:
        value = field(value, records_path)
        if not isinstance(value, list):
            raise ValueError("input.records_path must point to an array")
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        if any(k.casefold() in {"question", "query", "prompt", "q", "问题", "题目"} for k in value):
            return [value]
        containers = [
            value[k]
            for k in ("data", "records", "rows", "items", "examples", "dataset")
            if isinstance(value.get(k), (list, dict))
        ]
        if len(containers) > 1:
            raise ValueError("Ambiguous JSON containers; set input.records_path")
        return json_records(containers[0]) if containers else [value]
    raise ValueError("JSON must contain an object or an array of objects")


def read_records(path, kind="auto", records_path=None):
    kind = input_kind(path) if kind == "auto" else kind
    if kind == "pdf":
        raise ValueError("PDF must be parsed to TXT first; use build or parse-pdf")
    if kind != input_kind(path):
        raise ValueError("Input type and file extension disagree")
    text = path.read_bytes().decode("utf-8-sig")
    rows = json_records(json.loads(text), records_path) if kind == "json" else [{"text": text}]
    for row, item in enumerate(rows, 1):
        # Malformed rows are routed as individual intake failures by the pipeline.
        payload = item if isinstance(item, dict) else {"invalid_input": item}
        yield RawRecord(
            source_record_id=digest([str(path.resolve()), row, item])[:24],
            payload=payload,
            source=str(path.resolve()),
            row=row,
        )


def shape(value):
    if isinstance(value, dict):
        return {key: shape(val) for key, val in sorted(value.items())}
    if isinstance(value, list):
        return sorted({json_text(shape(item)) for item in value})
    return type(value).__name__
