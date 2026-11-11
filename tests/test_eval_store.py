import json

import pytest

from decisionforge.eval.store import EvalStore


def test_terminal_attempt_and_prediction_are_one_durable_jsonl_event(tmp_path):
    store = EvalStore(tmp_path)
    session = store.start_session({})
    attempt = store.start_attempt("one", session, {"state": {"text": "hello"}})
    store.finish_attempt(attempt, {"http_status": 200}, {"id": "one", "answer": {"noul": 0.9}})
    store.finish_session(session, "completed", 1)
    store.close()

    path = tmp_path / ".eval/events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert [u["table"] for u in events[2]["updates"]] == ["attempts", "predictions"]
    assert not list(tmp_path.rglob("*.sqlite"))
    reopened = EvalStore(tmp_path)
    assert reopened.get("attempts", attempt)["status"] == "finished"
    assert reopened.get("predictions", "one")["answer"] == {"noul": 0.9}
    detached = reopened.get("predictions", "one")
    detached["answer"]["noul"] = 0
    assert reopened.records("predictions")[0]["answer"]["noul"] == 0.9
    reopened.close()


@pytest.mark.parametrize("tail", [b'{"updates":', b"\xff", b'{"complete_json": true}'])
def test_incomplete_tail_is_read_only_until_online_recovery(tmp_path, tail):
    store = EvalStore(tmp_path)
    session = store.start_session({})
    attempt = store.start_attempt("one", session, {"state": "hello"})
    store.close()
    path = tmp_path / ".eval/events.jsonl"
    committed = path.read_bytes()
    path.write_bytes(committed + tail)

    offline = EvalStore(tmp_path, read_only=True)
    assert offline.report_attempts()[0]["status"] == "outcome_unknown"
    with pytest.raises(ValueError, match="read-only"):
        offline.start_session({})
    offline.close()
    assert path.read_bytes() == committed + tail

    recovered = EvalStore(tmp_path)
    assert path.read_bytes() == committed
    recovered.recover()
    assert recovered.get("attempts", attempt)["status"] == "outcome_unknown"
    assert recovered.get("sessions", session)["status"] == "interrupted"
    once = path.read_bytes()
    recovered.recover()
    assert path.read_bytes() == once
    recovered.start_session({})
    recovered.close()
    reopened = EvalStore(tmp_path)
    assert len(reopened.records("sessions")) == 2
    reopened.close()


@pytest.mark.parametrize("corruption", ["invalid_json", "tampered_record", "missing_event"])
def test_committed_corruption_is_rejected_without_rewriting(tmp_path, corruption):
    store = EvalStore(tmp_path)
    session = store.start_session({})
    store.start_attempt("one", session, {"state": "hello"})
    store.close()
    path = tmp_path / ".eval/events.jsonl"
    lines = path.read_bytes().splitlines(keepends=True)
    if corruption == "invalid_json":
        lines[-1] = b'{"broken":\n'
    elif corruption == "tampered_record":
        event = json.loads(lines[-1])
        event["updates"][0]["record"]["sample_id"] = "different"
        lines[-1] = json.dumps(event).encode() + b"\n"
    else:
        lines = lines[1:]
    corrupted = b"".join(lines)
    path.write_bytes(corrupted)
    with pytest.raises(ValueError, match="journal.*line"):
        EvalStore(tmp_path)
    assert path.read_bytes() == corrupted


@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt])
def test_partial_write_never_exposes_half_of_terminal_commit(tmp_path, monkeypatch, failure):
    store = EvalStore(tmp_path)
    session = store.start_session({})
    attempt = store.start_attempt("one", session, {"state": "hello"})
    path = tmp_path / ".eval/events.jsonl"

    def partial_write(data):
        with path.open("ab") as stream:
            stream.write(data[: len(data) // 2])
        raise failure("interrupted disk write")

    monkeypatch.setattr(store, "_write_event", partial_write)
    with pytest.raises(failure, match="disk write"):
        store.finish_attempt(attempt, {"http_status": 200}, {"id": "one"})
    assert store.records("predictions") == []
    assert store.get("attempts", attempt)["status"] == "in_flight"
    with pytest.raises(ValueError, match="failed journal"):
        store.start_session({})
    store.close()
    recovered = EvalStore(tmp_path)
    recovered.recover()
    assert recovered.records("predictions") == []
    assert recovered.get("attempts", attempt)["status"] == "outcome_unknown"
    recovered.close()


def test_missing_read_only_journal_is_not_created(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        EvalStore(tmp_path, read_only=True)
    assert not list(tmp_path.rglob("events.jsonl"))
