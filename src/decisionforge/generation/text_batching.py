"""Model-planned batching with bounded context, capped slices and resumable decisions."""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

import yaml
from filelock import FileLock

from decisionforge.generation.contracts import TextBatchContext, TextBatchPlan, TextWindowPlan
from decisionforge.util import NeedsReview, atomic_write, digest, json_text

logger = logging.getLogger(__name__)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_text(path: Path) -> str:
    # Preserve CRLF and every decoded character; an optional UTF-8 BOM is encoding metadata.
    return path.read_bytes().decode("utf-8-sig")


def context_path(path: Path) -> Path:
    return path.with_suffix(".batch.json")


def question_limit(config) -> int:
    if not config.input.generate_qa_from_text:
        return 0
    # Extraction and assessment consume two calls on the same raw input record.
    return min(
        config.input.max_generated_questions, max(0, config.execution.max_llm_calls_per_record - 2)
    )


def validate_quotas(questions, config):
    modes = [item.mode for item in questions]
    if len(modes) != len(set(modes)):
        raise NeedsReview("Batch question plan contains duplicate modes")
    if config.task.modes != ["auto"] and any(m not in config.task.modes for m in modes):
        raise NeedsReview("Batch question plan selected an unrequested mode")
    if sum(item.count for item in questions) > question_limit(config):
        raise NeedsReview("Batch question plan exceeds question/call limit")


def load_batch_context(config) -> TextBatchContext | None:
    path = Path(config.input.path)
    if path.suffix.lower() != ".txt" or not context_path(path).exists():
        return None
    context = TextBatchContext.model_validate_json(context_path(path).read_bytes())
    text = read_text(path)
    if context.text_sha256 != sha256(text) or context.char_end - context.char_start != len(text):
        raise NeedsReview("Generated TXT batch was modified; rebuild batches from the source")
    validate_quotas(context.questions, config)
    return context


def batch_char_limit(config) -> int:
    # Leave room for source, evidence, current JEV, the envelope and JSON string quotes.
    available = (config.llm.max_input_chars - 4096) // 3 - 2
    if available < 100:
        raise NeedsReview(
            "TXT batching requires llm.max_input_chars >= 4402 for planning and evidence"
        )
    return min(config.input.text_batching.threshold_chars, available)


def source_units(text: str, max_chars: int) -> list[dict]:
    """Sentence boundaries when present; bounded character spans for unpunctuated text."""
    units = []
    for sentence in re.finditer(r".+?(?:[。！？!?；;\n]|\.(?=\s)|$)", text, re.S):
        for start in range(sentence.start(), sentence.end(), max_chars):
            end = min(start + max_chars, sentence.end())
            units.append(
                {"id": len(units) + 1, "start": start, "end": end, "text": text[start:end]}
            )
    assert "".join(u["text"] for u in units) == text
    return units


def planning_payload(text, units, config):
    payload = {
        "source_characters": len(text),
        "requested_modes": config.task.modes,
        "domain": config.task.domain_profile,
        "difficulty": config.task.difficulty,
        "target_option_count": config.choice.target_option_count,
        "score_levels": config.score.default_levels,
        "max_batches": config.input.text_batching.max_batches,
        "max_batch_chars": batch_char_limit(config),
        "max_questions_per_batch": question_limit(config),
        "view": "full",
        "units": units,
    }

    def fits():
        return (
            len(json_text({"output_language": config.project.language, **payload}))
            <= config.llm.max_input_chars
        )

    if fits():
        return payload
    # Every unit stays represented; omitted middle text is explicit, never mistaken for full input.
    for width in (128, 64, 32, 16):
        payload["view"] = "head_tail_previews"
        payload["units"] = [
            {
                "id": u["id"],
                "start": u["start"],
                "end": u["end"],
                "head": u["text"][:width],
                "tail": u["text"][max(width, len(u["text"]) - width) :],
                "omitted_middle_characters": max(0, len(u["text"]) - 2 * width),
            }
            for u in units
        ]
        if fits():
            return payload
    raise NeedsReview(
        "Even the complete TXT boundary index exceeds max_input_chars; "
        "use a larger supported context limit or prepare smaller source documents"
    )


def validate_plan(plan, text, units, config) -> list[tuple[int, int]]:
    if not plan.should_split:
        if 3 * len(json_text(text)) + 4096 > config.llm.max_input_chars:
            raise NeedsReview(
                "Planner declined splitting but full TXT exceeds the safe generation envelope"
            )
        return []
    return validate_batches(plan.batches, text, units, config)


def validate_batches(batches, text, units, config) -> list[tuple[int, int]]:
    if len(batches) > config.input.text_batching.max_batches:
        raise NeedsReview("TXT planner exceeded max_batches")
    spans, previous = [], 0
    limit = batch_char_limit(config)
    for batch in batches:
        if not previous < batch.end_unit <= len(units):
            raise NeedsReview("TXT batch endpoints must increase and reference existing units")
        start, planned_end = units[previous]["start"], units[batch.end_unit - 1]["end"]
        # Truncate only this batch's tail. The next batch retains its model-planned start.
        end = min(planned_end, start + limit)
        fragment = text[start:end]
        if not fragment.strip():
            raise NeedsReview("TXT batch is empty")
        if 3 * len(json_text(fragment)) + 4096 > config.llm.max_input_chars:
            raise NeedsReview("TXT batch exceeds safe serialized input envelope")
        validate_quotas(batch.questions, config)
        spans.append((start, end))
        previous = batch.end_unit
    if previous != len(units):
        raise NeedsReview("TXT batch plan does not cover the entire source")
    return spans


def planning_windows(text, units, config):
    """Bound planner requests in memory; oversized model batches get a code fallback."""
    window_chars = min(batch_char_limit(config) * 8, config.llm.max_input_chars // 3)
    first = 0
    while first < len(units):
        start = units[first]["start"]
        end = first + 1
        while end < len(units) and units[end]["end"] - start <= window_chars:
            end += 1
        # Prefer a paragraph boundary so rules and observations travel together.
        paragraphs = [
            i
            for i in range(first + 1, end + 1)
            if text[max(0, units[i - 1]["end"] - 2) : units[i - 1]["end"]] == "\n\n"
            and units[i - 1]["end"] - start >= window_chars // 2
        ]
        if end < len(units) and paragraphs:
            end = paragraphs[-1]
        while True:
            fragment = text[start : units[end - 1]["end"]]
            local = [
                {**u, "id": i + 1, "start": u["start"] - start, "end": u["end"] - start}
                for i, u in enumerate(units[first:end])
            ]
            try:
                # Reserve envelope room for the window marker and its explanatory fields.
                window_config = config.model_copy(deep=True)
                window_config.llm.max_input_chars -= 256
                payload = planning_payload(fragment, local, window_config)
                payload.update(
                    planning_window=True,
                    max_batches=min(32, config.input.text_batching.max_batches),
                )
                # Use the actual configured batch limit, not the envelope-reserved one.
                payload["max_batch_chars"] = batch_char_limit(config)
                break
            except NeedsReview:
                if end - first <= 1:
                    raise
                end = first + (end - first) // 2
        yield first, fragment, local, payload
        first = end


async def plan_windows(text, units, config, store, call):
    batches = []
    count = 0
    source_hash = sha256(text)
    for count, (first, fragment, local, payload) in enumerate(
        planning_windows(text, units, config), 1
    ):
        identifier = f"text-plan-window:{source_hash[:24]}:{count}"
        saved = store.get(identifier)
        if saved is None:
            plan = await call(
                "text_batch_planner", payload, TextWindowPlan, identifier, "text_batch_plan"
            )
            saved = {
                "text_sha256": sha256(fragment),
                "first_unit": first,
                "plan": plan.model_dump(),
            }
            store.put(identifier, saved)
            store.event(identifier, "text_batch_plan", saved)
        if saved["text_sha256"] != sha256(fragment) or saved["first_unit"] != first:
            raise NeedsReview("TXT planning window changed since checkpoint")
        plan = TextWindowPlan.model_validate(saved["plan"])
        if len(plan.batches) > payload["max_batches"]:
            raise NeedsReview("TXT window planner exceeded max_batches")
        validate_batches(plan.batches, fragment, local, config)
        batches.extend(b.model_copy(update={"end_unit": b.end_unit + first}) for b in plan.batches)
        if len(batches) > config.input.text_batching.max_batches:
            raise NeedsReview("TXT planner exceeded total max_batches")
    return TextBatchPlan(
        should_split=True,
        reason=f"Model planned {count} bounded context windows; oversized batch tails are truncated by code.",
        batches=batches,
    )


def materialize(config, text, plan, spans, view):
    source = Path(config.input.path).resolve()
    directory = source.parent / "batches"
    if directory.resolve().parent != source.parent or directory.is_symlink():
        raise NeedsReview(
            "The source batches directory must not redirect outside its source directory"
        )
    directory.mkdir(exist_ok=True)
    source_hash = sha256(text)
    identity = [str(source), source_hash, plan.model_dump()]
    planned_ends = [start for start, _ in spans[1:]] + [len(text)]
    if any(end != planned_end for (_, end), planned_end in zip(spans, planned_ends, strict=True)):
        # Different truncation limits must never reuse a different batch's files or checkpoint IDs.
        identity.append(spans)
    plan_id = digest(identity)[:16]
    base = config.model_dump(mode="json")
    # Freeze path meaning before moving the YAML next to its batch, not its original config.
    for section, name in (("input", "path"), ("input", "mapping_spec"), ("output", "directory")):
        if base[section][name]:
            base[section][name] = str(Path(base[section][name]).resolve())
    config_id = digest(base)[:12]
    prefix = f"{source.stem[:60]}.{plan_id}"
    entries, files = [], {}
    for index, ((start, end), spec) in enumerate(zip(spans, plan.batches, strict=True), 1):
        path = directory / f"{prefix}.{index:03d}.txt"
        yaml_path = path.with_suffix(f".{config_id}.yaml")
        fragment = text[start:end]
        planned_end = planned_ends[index - 1]
        dropped = planned_end - end
        reason = spec.reason
        if dropped:
            reason += (
                f" [Code truncated batch from {planned_end - start} to {end - start} "
                f"characters; discarded {dropped} trailing characters.]"
            )
            logger.warning(
                "TXT batch %s truncated from %s to %s characters; discarded %s characters",
                index,
                planned_end - start,
                end - start,
                dropped,
            )
        context = TextBatchContext(
            original_source=str(source),
            original_sha256=source_hash,
            group_id=digest([str(source), source_hash]),
            batch_index=index,
            char_start=start,
            char_end=end,
            text_sha256=sha256(fragment),
            questions=spec.questions,
            reason=reason,
        )
        data = {**base, "input": {**base["input"], "path": str(path)}}
        files[path] = fragment
        files[yaml_path] = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
        files[context_path(path)] = json_text(context.model_dump()) + "\n"
        entries.append(
            {"input_path": str(path), "config_path": str(yaml_path), **context.model_dump()}
        )
    manifest = {
        "schema_version": "decisionforge.text-batches.v1",
        "original_source": str(source),
        "source_sha256": source_hash,
        "should_split": True,
        "reason": plan.reason,
        "planning_view": view,
        "batch_count": len(entries),
        "batches": entries,
    }
    manifest_path = directory / f"{prefix}.{config_id}.manifest.json"
    files[manifest_path] = json_text(manifest) + "\n"
    # Only own deterministic files are touched. Existing unrelated TXT files are never swept in.
    with FileLock(str(directory / ".dforge-batches.lock"), timeout=0):
        for path, content in files.items():
            if path.is_symlink() or (
                path.exists() and path.read_bytes() != content.encode("utf-8")
            ):
                raise NeedsReview(f"Refusing to overwrite modified batch artifact: {path}")
        for path, content in files.items():
            if not path.exists():
                atomic_write(path, content)
    return {**manifest, "manifest_path": str(manifest_path)}


async def prepare_text_batches(config, store, call) -> dict | None:
    """Plan once (or reuse a checkpoint), then validate and publish exact source slices."""
    if config.input.type != "txt" or not config.input.text_batching.enabled:
        return None
    text = read_text(Path(config.input.path))
    if len(text) <= config.input.text_batching.threshold_chars:
        return None
    units = source_units(text, max(25, min(1000, batch_char_limit(config) // 4)))
    source_hash = sha256(text)
    saved = store.get("text_batch_decision")
    if saved is None:
        # Bound both input and expected plan output. Ordinary sources keep the existing one-call path.
        payload = None
        if len(text) <= batch_char_limit(config) * 32:
            try:
                payload = planning_payload(text, units, config)
            except NeedsReview:
                pass
        if payload is None:
            if len(text) > batch_char_limit(config) * config.input.text_batching.max_batches:
                raise NeedsReview(
                    "TXT cannot fit within max_batches * max_batch_chars; increase max_batches"
                )
            plan = await plan_windows(text, units, config, store, call)
            view = "windowed"
        else:
            plan = await call(
                "text_batch_planner",
                payload,
                TextBatchPlan,
                "text-plan:" + source_hash[:24],
                "text_batch_plan",
            )
            view = payload["view"]
        saved = {
            "source_sha256": source_hash,
            "planning_view": view,
            "plan": plan.model_dump(),
        }
        # Save even an invalid plan for diagnosis; do not spend repeated semantic calls on retries.
        store.put("text_batch_decision", saved)
        store.event("text-plan:" + source_hash[:24], "text_batch_plan", saved)
    if saved["source_sha256"] != source_hash:
        raise NeedsReview("TXT source changed since its batch planning checkpoint")
    plan = TextBatchPlan.model_validate(saved["plan"])
    spans = validate_plan(plan, text, units, config)
    result = (
        materialize(config, text, plan, spans, saved["planning_view"])
        if plan.should_split
        else {
            "should_split": False,
            "reason": plan.reason,
            "batches": [],
            "source_sha256": source_hash,
            "planning_view": saved["planning_view"],
        }
    )
    store.put("text_batching", result)
    return result
