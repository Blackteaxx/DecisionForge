import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import FixtureProvider

from decisionforge import storage
from decisionforge.cli import main
from decisionforge.storage import Store, state_directory
from decisionforge.util import BudgetExhausted, write_json


@pytest.mark.parametrize("statement", ["PRAGMA journal_mode=DELETE", "COMMIT"])
def test_shared_io_failure_uses_local_database_and_closes_failed_connection(
    tmp_path, monkeypatch, statement
):
    original = sqlite3.connect
    connections = []
    remote = tmp_path / ".dforge/checkpoint.sqlite"

    class BrokenSharedConnection(sqlite3.Connection):
        def execute(self, sql, *args):
            if sql == statement:
                error = sqlite3.OperationalError("disk I/O error")
                error.sqlite_errorcode = sqlite3.SQLITE_IOERR_FSYNC
                error.sqlite_errorname = "SQLITE_IOERR_FSYNC"
                raise error
            return super().execute(sql, *args)

    def connect(path, *args, **kwargs):
        if Path(path) == remote:
            kwargs["factory"] = BrokenSharedConnection
        db = original(path, *args, **kwargs)
        connections.append(db)
        return db

    monkeypatch.setattr(storage.sqlite3, "connect", connect)
    store = Store(tmp_path)
    local_path = store._local_path
    try:
        assert local_path.parent != remote.parent
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connections[0].execute("SELECT 1")
        store.put("saved", 42)
        assert store.snapshot.exists()
        snapshot = original(store.snapshot)
        try:
            assert (
                json.loads(snapshot.execute("SELECT value FROM kv WHERE key='saved'").fetchone()[0])
                == 42
            )
        finally:
            snapshot.close()
    finally:
        store.close()
    assert not local_path.exists()
    # The shared snapshot is authoritative, even without the environment override.
    store = Store(tmp_path)
    try:
        assert store.get("saved") == 42
    finally:
        store.close()


@pytest.mark.parametrize(
    "message",
    ["database is locked", "database or disk is full", "attempt to write a readonly database"],
)
def test_non_io_errors_do_not_silently_create_another_database(tmp_path, monkeypatch, message):
    def fail(_):
        raise sqlite3.OperationalError(message)

    monkeypatch.setattr(storage, "connect_database", fail)
    with pytest.raises(sqlite3.OperationalError, match=message):
        Store(tmp_path)
    assert not (state_directory(tmp_path) / "checkpoint.snapshot.sqlite").exists()


def test_local_snapshot_persists_every_commit_and_rollback(tmp_path, monkeypatch):
    monkeypatch.setenv("DFORGE_SQLITE_MODE", "local")
    store = Store(tmp_path)
    try:
        store.put("cache", {"value": 1})
        store.save_step("r", "generate", {"output": "saved"})
        store.event("r", "generate", {"ok": True})
        store.register_group("g", "train")
        store.duplicate("hash", "r", "train")
        store.finish("r", "accepted", {"answer": "saved"})
        charge = store.reserve("r", "generate", 0.6, 1.0, 3)
        before = store.snapshot.read_bytes()
        with pytest.raises(BudgetExhausted):
            store.reserve("r", "next", 0.6, 1.0, 3)
        assert store.snapshot.read_bytes() == before
        store.settle(charge, 0.2)
        # Inspect a copied snapshot while the store is still open: close() is not the save point.
        copy = tmp_path / "inspection.sqlite"
        copy.write_bytes(store.snapshot.read_bytes())
        db = sqlite3.connect(copy)
        try:
            for table in ["kv", "steps", "events", "groups", "duplicates", "outcomes", "charges"]:
                assert db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 1
            assert db.execute("SELECT usd,settled FROM charges").fetchone() == (0.2, 1)
            assert db.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        finally:
            db.close()
    finally:
        store.close()


@pytest.mark.parametrize("failure", ["write", "fsync", "replace"])
def test_snapshot_failure_preserves_last_checkpoint_and_stops_writes(
    tmp_path, monkeypatch, failure
):
    monkeypatch.setenv("DFORGE_SQLITE_MODE", "local")
    store = Store(tmp_path)
    store.put("value", "old")
    before = store.snapshot.read_bytes()
    try:
        with monkeypatch.context() as patch:

            def fail(*args, **kwargs):
                raise OSError("snapshot storage unavailable")

            target, name = {
                "write": (storage.shutil, "copyfileobj"),
                "fsync": (storage.os, "fsync"),
                "replace": (Path, "replace"),
            }[failure]
            patch.setattr(target, name, fail)
            with pytest.raises(sqlite3.OperationalError, match="Cannot save checkpoint"):
                store.put("value", "new")
        assert store.snapshot.read_bytes() == before
        with pytest.raises(sqlite3.OperationalError, match="Cannot save checkpoint"):
            store.reserve("r", "no-api-request", 1, None, 10)
    finally:
        store.close()
    store = Store(tmp_path)
    try:
        assert store.get("value") == "old"
        assert store.spent() == 0
    finally:
        store.close()


@pytest.mark.parametrize("journal", ["WAL", "DELETE"])
def test_migration_recovers_old_journals_without_modifying_originals(
    tmp_path, monkeypatch, journal
):
    state = state_directory(tmp_path)
    state.mkdir()
    original = state / "checkpoint.sqlite"
    # Abrupt child exit leaves a WAL or hot rollback journal, without another live writer.
    code = """
import json, os, sqlite3, sys
db = sqlite3.connect(sys.argv[1])
db.execute('PRAGMA journal_mode=' + sys.argv[2])
db.execute('PRAGMA cache_size=2')
db.execute('CREATE TABLE kv (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
db.execute('INSERT INTO kv VALUES (?, ?)', ('saved', json.dumps('committed')))
db.commit()
if sys.argv[2] == 'DELETE':
    db.execute('UPDATE kv SET value=?', (json.dumps('not committed'),))
    for i in range(200):
        db.execute('INSERT INTO kv VALUES (?,?)', (str(i), 'x' * 4096))
os._exit(0)
"""
    subprocess.run([sys.executable, "-c", code, str(original), journal], check=True)
    side = Path(f"{original}-{'wal' if journal == 'WAL' else 'journal'}")
    assert side.exists() and side.stat().st_size > 0
    before = {p: p.read_bytes() for p in state.iterdir()}
    monkeypatch.setenv("DFORGE_SQLITE_MODE", "local")
    store = Store(tmp_path)
    try:
        assert store.get("saved") == "committed"
        assert store.get("0") is None
        assert store.db.execute("PRAGMA journal_mode").fetchone() == ("delete",)
        store.put("next", "new")
    finally:
        store.close()
    assert all(path.read_bytes() == data for path, data in before.items())
    monkeypatch.delenv("DFORGE_SQLITE_MODE")
    store = Store(tmp_path)
    try:
        assert store.get("next") == "new"
    finally:
        store.close()


def test_corrupt_snapshot_never_falls_back_to_stale_database(tmp_path, monkeypatch):
    monkeypatch.setenv("DFORGE_SQLITE_MODE", "local")
    state = state_directory(tmp_path)
    state.mkdir()
    snapshot = state / "checkpoint.snapshot.sqlite"
    snapshot.write_bytes(b"corrupt snapshot")
    with pytest.raises(sqlite3.DatabaseError):
        Store(tmp_path)
    assert snapshot.read_bytes() == b"corrupt snapshot"


def test_local_mode_cli_resume_export_and_legacy_root(config, monkeypatch):
    monkeypatch.setenv("DFORGE_SQLITE_MODE", "local")
    monkeypatch.setattr("decisionforge.cli.OpenAIProvider", lambda *_: FixtureProvider())
    directory = Path(config.output.directory)
    # A legacy run stores its configuration and checkpoint at the root.
    write_json(directory / "config.resolved.json", config.model_dump(mode="json"))
    assert main(["data", "resume", "--run-dir", str(directory)]) == 0
    actual = Path(config.output.directory)
    assert (actual / ".dforge" / "checkpoint.snapshot.sqlite").exists() or (actual / "checkpoint.snapshot.sqlite").exists()
    accepted = (actual / "accepted.json").read_bytes()
    monkeypatch.delenv("DFORGE_SQLITE_MODE")
    assert main(["data", "export", "--run-dir", str(directory), "--format", "chat-sft"]) == 0
    assert actual.joinpath("accepted.json").read_bytes() == accepted
    assert main(["data", "resume", "--run-dir", str(directory)]) == 0
    assert actual.joinpath("accepted.json").read_bytes() == accepted


def test_cli_reports_sqlite_failure_without_traceback(tmp_path, monkeypatch, caplog):
    state = state_directory(tmp_path)
    state.mkdir()
    (state / "checkpoint.snapshot.sqlite").write_bytes(b"invalid sqlite")
    assert main(["data", "export", "--run-dir", str(tmp_path)]) == 1
    assert "Checkpoint database error" in caplog.text
