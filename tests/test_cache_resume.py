"""Recovery should do work proportional to pending batches, preserving durable checkpoints."""

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_pipeline import run
from test_text_batching import NORTH, SOUTH, BatchProvider, setup_text
from test_text_concurrency import ConcurrentProvider

from decisionforge import pipeline
from decisionforge.pipeline import Pipeline
from decisionforge.storage import Store


@pytest.mark.parametrize("invalid", [None, "assessment_count"])
async def test_completed_batches_skip_workers_writes_and_duplicate_events(
    config, monkeypatch, invalid
):
    setup_text(config)
    _, before, _ = await run(config, BatchProvider(invalid=invalid))
    directory = Path(config.output.directory)
    store = Store(directory)
    exports = []
    original_export = pipeline.export_run

    def export(*args, **kwargs):
        exports.append(1)
        return original_export(*args, **kwargs)

    def unexpected(*args, **kwargs):
        pytest.fail("A completed batch must not instantiate a child pipeline")

    monkeypatch.setattr(pipeline, "export_run", export)
    monkeypatch.setattr(pipeline, "load_config", unexpected)
    statements = []
    store.db.set_trace_callback(statements.append)
    try:
        provider = BatchProvider()
        result = await Pipeline(config, store, provider).run()
        assert result["accepted"] == len(before["accepted"])
        assert provider.calls == []
        assert len(exports) == 1
        after = json.loads((directory / "meta.json").read_text())
        assert after["events"] == before["meta"]["events"]
        assert after["records"] == before["meta"]["records"]
        assert json.loads((directory / "accepted.json").read_text()) == before["accepted"]
        assert after["active_text_batches"] == []
        writes = [s for s in statements if s.startswith("INSERT")]
        assert not any("'normalized_records" in s for s in writes)
        assert not any("INTO events" in s for s in writes)
    finally:
        store.close()


async def test_partial_resume_skips_finished_batch_and_exports_once(config, monkeypatch):
    setup_text(config)
    await run(config, BatchProvider(interrupt_south=True))
    store = Store(Path(config.output.directory))
    original_export = pipeline.export_run
    exports = []

    def export(*args, **kwargs):
        exports.append(1)
        return original_export(*args, **kwargs)

    monkeypatch.setattr(pipeline, "export_run", export)
    monkeypatch.setattr(pipeline, "BATCH_EXPORT_INTERVAL_SECONDS", float("inf"))
    try:
        provider = BatchProvider()
        result = await Pipeline(config, store, provider).run()
        assert result["status"] == "complete" and result["accepted"] == 2
        assert len(exports) == 1
        assert [role for role, _, _ in provider.calls].count("extractor") == 1
        assert store.db.execute(
            "SELECT record,COUNT(*) FROM events WHERE stage='text_batch_complete' GROUP BY record"
        ).fetchall() == [("text-batch:1", 1), ("text-batch:2", 1)]
    finally:
        store.close()


async def test_public_export_is_throttled_but_checkpoints_stay_current(config, monkeypatch):
    setup_text(
        config, NORTH + SOUTH + NORTH.replace("North", "East") + NORTH.replace("North", "West")
    )
    store = Store(Path(config.output.directory))
    clock = [0.0]
    original_event, original_export = store.event, pipeline.export_run
    exports = []

    def event(record, stage, value):
        original_event(record, stage, value)
        if stage == "text_batch_complete":
            clock[0] += 20.0

    def export(*args, **kwargs):
        exports.append(clock[0])
        return original_export(*args, **kwargs)

    monkeypatch.setattr(store, "event", event)
    monkeypatch.setattr(pipeline, "export_run", export)
    monkeypatch.setattr(
        pipeline,
        "time",
        SimpleNamespace(monotonic=lambda: clock[0], perf_counter=pipeline.time.perf_counter),
    )
    try:
        result = await Pipeline(config, store, ConcurrentProvider(store, config)).run()
        assert result["accepted"] == 12
        assert exports == [60.0, 80.0]
        assert store.get("source_total") == 12
        assert store.get("active_text_batches") == []
    finally:
        store.close()


def test_done_index_and_failure_count_follow_commits_and_reopen(tmp_path):
    store = Store(tmp_path)
    try:
        store.finish("old", "complete", {})
        statements = []
        store.db.set_trace_callback(statements.append)
        assert store.done("old") and not store.done("new")
        assert store.input_failure_count() == 0
        store.finish("new", "review", {"jev": None})
        assert store.done("new") and store.input_failure_count() == 1
        store.finish("new", "review", {"jev": {"request": {}}})
        assert store.input_failure_count() == 0
        store.finish("new", "review", {"jev": None})
        assert store.input_failure_count() == 1
        assert statements.count("SELECT record FROM outcomes") == 1
    finally:
        store.close()
    store = Store(tmp_path)
    try:
        assert store.done("old") and store.done("new")
        assert store.input_failure_count() == 1
    finally:
        store.close()


def test_put_many_publishes_once_skips_noops_and_rolls_back(tmp_path, monkeypatch):
    monkeypatch.setenv("DFORGE_SQLITE_MODE", "local")
    store = Store(tmp_path)
    original = store._publish_snapshot
    snapshots = []

    def publish():
        snapshots.append(1)
        original()

    monkeypatch.setattr(store, "_publish_snapshot", publish)
    try:
        store.put_many({"a": 1, "b": 2})
        before = store.snapshot.read_bytes()
        store.put_many({"b": 2, "a": 1})
        store.put("a", 1)
        assert len(snapshots) == 1 and store.snapshot.read_bytes() == before
        store.db.execute(
            "CREATE TRIGGER reject_fail BEFORE INSERT ON kv WHEN NEW.key='fail' "
            "BEGIN SELECT RAISE(ABORT, 'injected'); END"
        )
        with pytest.raises(sqlite3.IntegrityError, match="injected"):
            store.put_many({"a": 3, "fail": 4})
        assert store.get("a") == 1 and store.get("fail") is None
        assert len(snapshots) == 1 and store.snapshot.read_bytes() == before
    finally:
        store.close()
