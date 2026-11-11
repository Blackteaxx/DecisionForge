from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

from decisionforge.eval.data import parse_json, write_jsonl
from decisionforge.util import digest, now


class EvalStore:
    """Fsynced JSONL journal, used under the runner's single-writer lock.

    One complete line commits all its updates. A torn final line is uncommitted;
    online reopen truncates it, while offline scoring leaves the file untouched.
    """

    def __init__(self, directory, *, read_only=False):
        self.directory = Path(directory)
        state = self.directory / ".eval"
        self.path = state / "events.jsonl"
        self.read_only = read_only
        self._tables = {table: {} for table in ("sessions", "attempts", "predictions")}
        self._sequence = 0
        self._failed = False
        self._stream = None
        if read_only and not self.path.exists():
            raise ValueError("Evaluation journal does not exist")
        if self.path.exists():
            self._replay()
        if not read_only:
            state.mkdir(parents=True, exist_ok=True)
            self._stream = self.path.open("ab", buffering=0)
            os.fsync(self._stream.fileno())
            # Persist the journal name and .eval directory before any HTTP send.
            for directory in (state, self.directory):
                if os.name != "nt":
                    fd = os.open(directory, os.O_RDONLY)
                    try:
                        os.fsync(fd)
                    finally:
                        os.close(fd)

    def _validate_event(self, event):
        if not isinstance(event, dict) or set(event) != {
            "version",
            "sequence",
            "updates",
            "sha256",
        }:
            raise ValueError("Invalid event envelope")
        if type(event["version"]) is not int or event["version"] != 1:
            raise ValueError("Unsupported journal version")
        if type(event["sequence"]) is not int or event["sequence"] != self._sequence + 1:
            raise ValueError("Missing or reordered journal event")
        content = {k: event[k] for k in ("version", "sequence", "updates")}
        if event["sha256"] != digest(content):
            raise ValueError("Journal checksum mismatch")
        if not isinstance(event["updates"], list) or not event["updates"]:
            raise ValueError("Empty journal event")
        for update in event["updates"]:
            if not isinstance(update, dict) or set(update) != {"table", "record"}:
                raise ValueError("Invalid journal update")
            table, record = update["table"], update["record"]
            if not isinstance(table, str) or table not in self._tables:
                raise ValueError("Invalid journal table")
            if (
                not isinstance(record, dict)
                or not isinstance(record.get("id"), str)
                or not record["id"]
            ):
                raise ValueError("Invalid journal record id")

    def _apply(self, event):
        for update in event["updates"]:
            self._tables[update["table"]][update["record"]["id"]] = update["record"]
        self._sequence = event["sequence"]

    def _replay(self):
        committed_bytes, torn_tail = 0, False
        with self.path.open("rb") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.endswith(b"\n"):
                    torn_tail = True
                    break
                try:
                    event = parse_json(line)
                    self._validate_event(event)
                except (ValueError, TypeError) as exc:
                    raise ValueError(
                        f"Evaluation journal {self.path}: line {line_number}: {exc}"
                    ) from exc
                self._apply(event)
                committed_bytes = stream.tell()
        if torn_tail and not self.read_only:
            with self.path.open("r+b") as stream:
                stream.truncate(committed_bytes)
                stream.flush()
                os.fsync(stream.fileno())

    def _write_event(self, data):
        remaining = memoryview(data)
        while remaining:
            written = self._stream.write(remaining)
            if not written:
                raise OSError("Evaluation journal write made no progress")
            remaining = remaining[written:]
        os.fsync(self._stream.fileno())

    def _commit(self, updates):
        if self.read_only:
            raise ValueError("Evaluation journal is read-only")
        if self._failed:
            raise ValueError("Cannot append to a failed journal; close and reopen for recovery")
        if self._stream is None:
            raise ValueError("Evaluation journal is closed")
        content = {
            "version": 1,
            "sequence": self._sequence + 1,
            "updates": [{"table": table, "record": record} for table, record in updates],
        }
        data = (
            json.dumps({**content, "sha256": digest(content)}, ensure_ascii=False, allow_nan=False)
            + "\n"
        ).encode("utf-8")
        event = parse_json(data)
        self._validate_event(event)
        try:
            self._write_event(data)
            self._apply(event)
        except BaseException:
            self._failed = True
            raise

    def records(self, table):
        return deepcopy(list(self._tables[table].values()))

    def put(self, table, record):
        self._commit([(table, record)])

    def get(self, table, identity):
        if identity not in self._tables[table]:
            raise ValueError(f"Missing {table} record: {identity}")
        return deepcopy(self._tables[table][identity])

    def report_attempts(self):
        """A held run lock makes orphaned in-flight sends unknown for offline reports."""
        records = self.records("attempts")
        for record in records:
            if record["status"] == "in_flight":
                record.update(
                    status="outcome_unknown", usage=None, attempt_latency_ms=None, http_status=None
                )
        return records

    def start_session(self, settings):
        record = {"id": str(uuid4()), "started_at": now(), "status": "running", **settings}
        self.put("sessions", record)
        return record["id"]

    def finish_session(self, identity, status, window_seconds):
        record = self.get("sessions", identity)
        record.update(status=status, finished_at=now(), window_seconds=window_seconds)
        self.put("sessions", record)

    def start_attempt(self, sample_id, session_id, payload):
        record = {
            "id": str(uuid4()),
            "sample_id": sample_id,
            "session_id": session_id,
            "status": "in_flight",
            "started_at": now(),
            "request": payload,
        }
        self.put("attempts", record)
        return record["id"]

    def finish_attempt(self, identity, outcome, prediction=None):
        record = self.get("attempts", identity)
        record.update(outcome, status="finished", finished_at=now())
        updates = [("attempts", record)]
        if prediction is not None:
            updates.append(("predictions", prediction))
        self._commit(updates)

    def recover(self):
        updates = []
        for record in self.records("attempts"):
            if record["status"] == "in_flight":
                record.update(
                    status="outcome_unknown",
                    recovered_at=now(),
                    usage=None,
                    attempt_latency_ms=None,
                    http_status=None,
                )
                updates.append(("attempts", record))
        for record in self.records("sessions"):
            if record["status"] == "running":
                record.update(status="interrupted", window_seconds=None)
                updates.append(("sessions", record))
        if updates:
            self._commit(updates)

    def export(self):
        write_jsonl(self.directory / "attempts.jsonl", self.records("attempts"))
        write_jsonl(self.directory / "predictions.jsonl", self.records("predictions"))

    def close(self):
        if self._stream is not None:
            self._stream.close()
            self._stream = None
