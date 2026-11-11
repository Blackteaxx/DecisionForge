from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


async def map_bounded(function, items, concurrency, *, stop=None):
    """Keep input order while bounding workers and joining them on failure/cancellation."""
    indexed = iter(enumerate(items))
    results = [None] * len(items)

    async def worker():
        while stop is None or not stop.is_set():
            entry = next(indexed, None)
            if entry is None:
                return
            index, item = entry
            results[index] = await function(item)

    tasks = [asyncio.create_task(worker()) for _ in range(min(len(items), concurrency))]
    try:
        await asyncio.gather(*tasks)
        if stop is not None and stop.is_set():
            raise BudgetExhausted("Budget exhausted")
        return results
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def clean_instructions(text, identifiers=()):
    """Replace bookkeeping references with plain wording, leaving task facts intact."""
    if isinstance(text, dict):
        return {key: clean_instructions(value, identifiers) for key, value in text.items()}
    if isinstance(text, list):
        return [clean_instructions(value, identifiers) for value in text]
    if not isinstance(text, str):
        return text
    fields = (
        r"\b(?:source_record_id|record_id|group_id|evidence_id|candidate_id|question_id|rubric_id)"
        r"s?\b"
    )
    text = re.sub(
        fields + r"\s*[:=]\s*[`\"']?[\w:./-]+[`\"']?",
        "the supplied information",
        text,
    )
    text = re.sub(fields, "the supplied information", text)
    for identifier in sorted(set(identifiers), key=len, reverse=True):
        if len(identifier) >= 4 and not identifier.isdecimal():
            text = re.sub(
                rf"(?<!\w){re.escape(identifier)}(?!\w)", "the supplied information", text
            )
    text = re.sub(
        r"\b(?:[\w.-]+:)*(?=[a-fA-F0-9]*[a-fA-F])[a-fA-F0-9]{16,}(?::[\w.-]+)*\b"
        r"|\b[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b",
        "the supplied information",
        text,
    )
    return re.sub(r"[ \t]+", " ", text).strip()


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(json_text(value).encode("utf-8")).hexdigest()


def now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def write_json(path: Path, value: Any) -> str:
    """Write human-readable JSON without changing JSONL, cache keys or prompt serialization."""
    text = json.dumps(value, ensure_ascii=False, indent=4, allow_nan=False) + "\n"
    atomic_write(path, text)
    return text


class PipelineError(Exception):
    code = "pipeline_error"


class NeedsReview(PipelineError):
    code = "needs_review"


class Unsupported(PipelineError):
    code = "unsupported"


class BudgetExhausted(PipelineError):
    code = "budget_exhausted"


class CallLimit(NeedsReview):
    code = "call_limit"


class ProviderError(NeedsReview):
    code = "provider_error"
