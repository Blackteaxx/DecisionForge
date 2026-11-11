from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from decisionforge.util import BudgetExhausted, CallLimit, json_text, now

logger = logging.getLogger(__name__)


def state_directory(directory: Path) -> Path:
    """Keep new run internals together; still read checkpoints created by older versions."""
    hidden = directory / ".dforge"
    if not hidden.exists() and any(
        (directory / name).exists()
        for name in (
            "checkpoint.sqlite",
            "checkpoint.snapshot.sqlite",
            "manifest.json",
            "config.resolved.json",
        )
    ):
        return directory
    return hidden


def _is_disk_io_error(exc: BaseException) -> bool:
    code = getattr(exc, "sqlite_errorcode", 0)
    return code & 0xFF == sqlite3.SQLITE_IOERR or "disk i/o error" in str(exc).lower()


def _use_rollback_journal(db: sqlite3.Connection) -> None:
    db.execute("PRAGMA journal_mode=DELETE")
    db.execute("PRAGMA synchronous=FULL")
    # An empty transaction does not test journal writes or fsync on all filesystems.
    version = db.execute("PRAGMA user_version").fetchone()[0]
    db.execute("BEGIN IMMEDIATE")
    db.execute(f"PRAGMA user_version={version}")
    db.execute("COMMIT")


def connect_database(path: Path) -> sqlite3.Connection:
    """Open and test a real write transaction, closing the connection on failure."""
    db = sqlite3.connect(path)
    try:
        _use_rollback_journal(db)
    except BaseException:
        db.close()
        raise
    return db


class Store:
    """Single-process store; the caller must hold the run lock for its whole lifetime."""

    def __init__(self, directory: Path):
        self.directory = state_directory(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.snapshot = self.directory / "checkpoint.snapshot.sqlite"
        self._temporary = None
        self._sync_error = None
        self._done_records = None
        self._input_failures = None
        mode = os.getenv("DFORGE_SQLITE_MODE", "auto")
        if mode not in {"auto", "local"}:
            raise ValueError("DFORGE_SQLITE_MODE must be auto or local")
        path = self.directory / "checkpoint.sqlite"
        try:
            if self.snapshot.exists() or mode == "local":
                self.db = self._open_local(self.snapshot if self.snapshot.exists() else path)
            else:
                try:
                    self.db = connect_database(path)
                except sqlite3.OperationalError as exc:
                    if not _is_disk_io_error(exc):
                        raise
                    logger.warning(
                        "SQLite I/O failed on %s (%s); using local SQLite with a shared snapshot",
                        path,
                        getattr(exc, "sqlite_errorname", str(exc)),
                    )
                    self.db = self._open_local(path)
            self._initialize()
            self._publish_snapshot()
        except BaseException:
            if hasattr(self, "db"):
                self.db.close()
            if self._temporary is not None:
                self._temporary.cleanup()
            raise

    def _open_local(self, source: Path) -> sqlite3.Connection:
        # On clusters TMPDIR may itself point to GPFS; prefer node-local /tmp on Linux.
        root = os.getenv("DFORGE_SQLITE_TMPDIR") or (
            "/tmp" if sys.platform.startswith("linux") else None
        )
        self._temporary = tempfile.TemporaryDirectory(prefix="dforge-sqlite-", dir=root)
        self._local_path = Path(self._temporary.name) / "checkpoint.sqlite"
        if source.exists():
            shutil.copyfile(source, self._local_path)
            # Keep recovery journals; -shm is a disposable index SQLite rebuilds locally.
            for suffix in ("-wal", "-journal"):
                side = Path(f"{source}{suffix}")
                if side.exists():
                    shutil.copyfile(side, Path(f"{self._local_path}{suffix}"))
        return connect_database(self._local_path)

    def _initialize(self):
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS steps (
                record TEXT, stage TEXT, value TEXT, PRIMARY KEY(record, stage));
            CREATE TABLE IF NOT EXISTS outcomes (
                record TEXT PRIMARY KEY, status TEXT, value TEXT);
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY, record TEXT, stage TEXT, value TEXT);
            CREATE TABLE IF NOT EXISTS charges (
                id INTEGER PRIMARY KEY, record TEXT, stage TEXT, usd REAL, settled INTEGER);
            CREATE TABLE IF NOT EXISTS groups (id TEXT PRIMARY KEY, split TEXT);
            CREATE TABLE IF NOT EXISTS duplicates (hash TEXT PRIMARY KEY, record TEXT, split TEXT);
        """)
        self.db.commit()

    def _publish_snapshot(self):
        if self._temporary is None:
            return
        temporary = self.snapshot.with_suffix(".sqlite.tmp")
        try:
            # Local commits use DELETE/FULL, so the main file contains every committed page.
            with self._local_path.open("rb") as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target)
                target.flush()
                os.fsync(target.fileno())
            temporary.replace(self.snapshot)
        except OSError as exc:
            self._sync_error = (
                f"Cannot save checkpoint snapshot to {self.snapshot}: {exc}. "
                "Run stopped; check shared storage space, quota and write permissions."
            )
            raise sqlite3.OperationalError(self._sync_error) from exc
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    @contextmanager
    def _transaction(self):
        if self._sync_error:
            raise sqlite3.OperationalError(self._sync_error)
        with self.db:
            yield
        # Persist each stage/reservation before returning, not merely on normal shutdown.
        self._publish_snapshot()

    def close(self):
        try:
            self.db.close()
        finally:
            if self._temporary is not None:
                self._temporary.cleanup()

    def get(self, key: str) -> Any:
        row = self.db.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: Any):
        self.put_many({key: value})

    def put_many(self, values: dict[str, Any]):
        """Publish related state together, without committing unchanged values."""
        if self._sync_error:
            raise sqlite3.OperationalError(self._sync_error)
        changed = []
        for key, value in values.items():
            serialized = json_text(value)
            if not self.db.execute(
                "SELECT 1 FROM kv WHERE key=? AND value=?", (key, serialized)
            ).fetchone():
                changed.append((key, serialized))
        if not changed:
            return
        with self._transaction():
            self.db.executemany("INSERT OR REPLACE INTO kv VALUES (?,?)", changed)

    def step(self, record: str, stage: str) -> Any:
        row = self.db.execute(
            "SELECT value FROM steps WHERE record=? AND stage=?", (record, stage)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def save_step(self, record: str, stage: str, value: Any):
        with self._transaction():
            self.db.execute(
                "INSERT OR REPLACE INTO steps VALUES (?,?,?)", (record, stage, json_text(value))
            )

    def event(self, record: str, stage: str, value: Any):
        with self._transaction():
            self.db.execute(
                "INSERT INTO events(record,stage,value) VALUES (?,?,?)",
                (record, stage, json_text({"at": now(), **value})),
            )

    def done(self, record: str) -> bool:
        # The run lock gives this Store exclusive ownership. Load once, then update on finish.
        if self._done_records is None:
            self._done_records = {row[0] for row in self.db.execute("SELECT record FROM outcomes")}
        return record in self._done_records

    def input_failure_count(self) -> int:
        if self._input_failures is None:
            self._input_failures = sum(
                not json.loads(value).get("jev")
                for (value,) in self.db.execute("SELECT value FROM outcomes WHERE status='review'")
            )
        return self._input_failures

    def finish(self, record: str, status: str, value: Any):
        previous_failure = False
        if self._input_failures is not None and self.done(record):
            previous = self.db.execute(
                "SELECT status,value FROM outcomes WHERE record=?", (record,)
            ).fetchone()
            previous_failure = previous[0] == "review" and not json.loads(previous[1]).get("jev")
        with self._transaction():
            self.db.execute(
                "INSERT OR REPLACE INTO outcomes VALUES (?,?,?)", (record, status, json_text(value))
            )
        if self._done_records is not None:
            self._done_records.add(record)
        if self._input_failures is not None:
            self._input_failures += int(status == "review" and not value.get("jev")) - int(
                previous_failure
            )

    def outcomes(self):
        for record, status, value in self.db.execute("SELECT * FROM outcomes ORDER BY record"):
            yield record, status, json.loads(value)

    def spent(self) -> float:
        return float(self.db.execute("SELECT COALESCE(SUM(usd),0) FROM charges").fetchone()[0])

    def reserve(
        self, record: str, stage: str, usd: float, limit: float | None, max_calls: int
    ) -> int:
        with self._transaction():
            calls = self.db.execute(
                "SELECT COUNT(*) FROM charges WHERE record=?", (record,)
            ).fetchone()[0]
            if calls >= max_calls:
                raise CallLimit(f"LLM call limit {max_calls} reached for source {record}")
            if limit is not None and self.spent() + usd > limit:
                raise BudgetExhausted(
                    "Insufficient budget for the next bounded request; checkpoint saved"
                )
            cursor = self.db.execute(
                "INSERT INTO charges(record,stage,usd,settled) VALUES (?,?,?,0)",
                (record, stage, usd),
            )
            return cursor.lastrowid

    def settle(self, charge: int, usd: float):
        with self._transaction():
            self.db.execute("UPDATE charges SET usd=?,settled=1 WHERE id=?", (usd, charge))

    def register_group(self, group: str, split: str):
        row = self.db.execute("SELECT split FROM groups WHERE id=?", (group,)).fetchone()
        if row and row[0] != split:
            raise ValueError(f"Source group {group} crosses splits: {row[0]} / {split}")
        with self._transaction():
            self.db.execute("INSERT OR IGNORE INTO groups VALUES (?,?)", (group, split))

    def duplicate(self, fingerprint: str, record: str, split: str) -> str | None:
        row = self.db.execute(
            "SELECT record,split FROM duplicates WHERE hash=?", (fingerprint,)
        ).fetchone()
        if row and row[0] != record:
            if row[1] != split:
                raise ValueError("Duplicate public task crosses source splits")
            return row[0]
        with self._transaction():
            self.db.execute(
                "INSERT OR IGNORE INTO duplicates VALUES (?,?,?)", (fingerprint, record, split)
            )
        return None
