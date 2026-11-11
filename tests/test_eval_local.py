"""Local evaluation contracts and orchestration without Torch, CUDA or model downloads."""

import asyncio
import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from decisionforge.eval.data import write_jsonl
from decisionforge.eval.models import EvalConfig, load_config
from tests.eval_fixtures import sample


def checkpoint_at(path, merged=False, base=None):
    path.mkdir(parents=True)
    (path / "joint_head_config.json").write_text('{"hidden_size":16,"width":16}')
    (path / "unsloth_decision_config.json").write_text(
        '{"layout":"clef","max_len":2048,"temperature":[1,1,1]}'
    )
    (path / "tokenizer.json").write_text("{}")
    (path / "tokenizer_config.json").write_text('{"tokenizer_class":"Qwen2Tokenizer"}')
    names = [
        "joint_head.safetensors",
        "model.safetensors" if merged else "adapter_model.safetensors",
    ]
    for name in names:
        header = json.dumps(
            {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
        ).encode()
        (path / name).write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<f", 1.0))
    (path / ("config.json" if merged else "adapter_config.json")).write_text(
        json.dumps(
            {"model_type": "qwen3_5"}
            if merged
            else {"peft_type": "LORA", "base_model_name_or_path": str(base or "fixture/base")}
        )
    )
    return path


def config_at(tmp_path, items=None, **execution):
    cp = checkpoint_at(tmp_path / "model")
    data = tmp_path / "samples.jsonl"
    write_jsonl(data, items or [sample(kind, "soft", kind) for kind in ("choice", "noul", "score")])
    return EvalConfig(
        data={"path": str(data)},
        model={"adapter": "unsloth_local", "path": str(cp)},
        output=str(tmp_path / "run"),
        execution=execution,
    )


def fixture_worker(connection, options):
    """Deterministic process test double implementing the worker's public-only protocol."""
    from decisionforge.eval.local_worker import answer_from_probabilities

    connection.send(
        {
            "kind": "ready",
            "metadata": {
                "model": options["path"],
                "model_identity_source": "loaded_checkpoint",
                "backbone": {"name": "fixture/base", "revision": "fixture-v1"},
                "calibration": {"temperature": [1, 1, 1]},
                "runtime": {"gpu": "fixture", "model_load_seconds": 0.0},
            },
        }
    )
    while True:
        command = connection.recv()
        if command["kind"] == "close":
            break
        public = command["request"]
        assert set(public) == {"state", "questions"}
        q = public["questions"]["decision"]
        p = (
            {"A": 0.206572, "B": 0.793428}
            if q["type"] == "choice"
            else {"false": 0.206572, "true": 0.793428}
            if q["type"] == "noul"
            else {"0": 0.106572, "1": 0.2, "2": 0.693428}
        )
        answer = answer_from_probabilities(q, p)
        connection.send(
            {
                "kind": "prediction",
                "response": {
                    "model": options["path"],
                    "answers": {"decision": answer},
                    "usage": {"input_tokens": 16, "output_tokens": 0},
                },
                "input_tokens": 16,
            }
        )
    connection.close()


def hanging_worker(connection, options):
    import time

    connection.send(
        {"kind": "ready", "metadata": {"model": options["path"], "backbone": {}, "runtime": {}}}
    )
    connection.recv()
    time.sleep(60)


def failing_worker(connection, options):
    connection.send(
        {
            "kind": "error",
            "error": {"kind": "ImportError", "message": "Missing Unsloth test fixture"},
        }
    )
    connection.close()


def test_local_config_and_http_backwards_compatibility(tmp_path):
    legacy = EvalConfig(data={"path": "test.jsonl"}, model={"api_key_env": None})
    assert legacy.model.adapter == "jev"
    c = config_at(tmp_path)
    assert c.model.adapter == "unsloth_local" and c.model.model == str(tmp_path / "model")
    for change in (
        {"base_url": "http://wrong"},
        {"api_key_env": "KEY"},
        {"model": "wrong"},
        {"load_in_4bit": 1},
    ):
        values = c.model_dump()
        values["model"].update(change)
        with pytest.raises(ValidationError):
            EvalConfig.model_validate(values)
    for change in ({"concurrency": 2}, {"max_attempts": 2}):
        values = c.model_dump()
        values["execution"].update(change)
        with pytest.raises(ValidationError):
            EvalConfig.model_validate(values)


def partial_frame_worker(connection, options):
    import os
    import time

    connection.send({"kind": "ready", "metadata": {"model": options["path"]}})
    connection.recv()
    # poll() becomes readable before a complete multiprocessing frame arrives.
    os.write(connection.fileno(), struct.pack("!i", 1000) + b"{")
    time.sleep(30)


def unreading_worker(connection, options):
    import time

    connection.send({"kind": "ready", "metadata": {"model": options["path"]}})
    time.sleep(30)


@pytest.mark.asyncio
@pytest.mark.parametrize("partial_frame", [True, False])
@pytest.mark.parametrize("cancelled", [True, False])
async def test_timeout_bounds_entire_ipc_exchange(tmp_path, partial_frame, cancelled):
    if partial_frame and sys.platform == "win32":
        pytest.skip(
            "Partial POSIX pipe frame injection requires a file descriptor, not a Windows handle"
        )
    from decisionforge.eval.local_adapter import LocalAdapter
    from decisionforge.eval.models import Sample

    c = config_at(tmp_path, timeout=60 if cancelled else 0.1)
    adapter = await LocalAdapter(
        c.model,
        c.execution,
        worker_target=partial_frame_worker if partial_frame else unreading_worker,
    ).start()
    process = adapter.process
    value = Sample.model_validate(sample())
    public = value.request.model_dump(exclude_none=True)
    if not partial_frame:
        # Worker never reads requests; fill the pipe so send itself must block.
        public["state"] = "x" * (16 * 1024 * 1024)
    try:
        task = asyncio.create_task(adapter.send(public, value.question))
        if cancelled:
            await asyncio.sleep(0.1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=2)
        else:
            outcome = await asyncio.wait_for(task, timeout=2)
            assert outcome["fatal"] and outcome["error"]["kind"] == "local_timeout"
        assert not process.is_alive()
    finally:
        await adapter.close(force=True)


def test_local_path_relative_to_yaml(tmp_path):
    p = tmp_path / "eval.yaml"
    p.write_text(
        "data: {path: data.jsonl}\nmodel: {adapter: unsloth_local, path: ./model}\noutput: run\n"
    )
    c = load_config(p)
    assert c.model.path == str(tmp_path / "model")


def test_checkpoint_layout_and_content_identity(tmp_path):
    from decisionforge.eval.checkpoint import checkpoint_identity, validate_checkpoint

    base = checkpoint_at(tmp_path / "base", merged=True)
    cp = checkpoint_at(tmp_path / "model", base=base)
    validate_checkpoint(cp)
    a = checkpoint_identity(cp)
    path = base / "model.safetensors"
    original = path.read_bytes()
    path.write_bytes(original[:-4] + struct.pack("<f", 2.0))
    assert checkpoint_identity(cp) != a
    (cp / "joint_head.safetensors").unlink()
    with pytest.raises(ValueError, match="joint_head"):
        validate_checkpoint(cp)


def test_plain_language_model_rejected_before_loading(tmp_path):
    from decisionforge.eval.checkpoint import validate_checkpoint

    cp = tmp_path / "model"
    cp.mkdir()
    (cp / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="decision|决策"):
        validate_checkpoint(cp)


def test_nested_checkpoint_shards_participate_in_identity(tmp_path):
    from decisionforge.eval.checkpoint import checkpoint_identity

    cp = checkpoint_at(tmp_path / "model", merged=True)
    (cp / "weights").mkdir()
    shard = cp / "weights/model.safetensors"
    (cp / "model.safetensors").rename(shard)
    (cp / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "weights/model.safetensors"}})
    )
    before = checkpoint_identity(cp)
    content = shard.read_bytes()
    shard.write_bytes(content[:-4] + struct.pack("<f", 2.0))
    assert checkpoint_identity(cp) != before


@pytest.mark.parametrize("name", ["../outside.safetensors", "/tmp/outside.safetensors"])
def test_checkpoint_index_rejects_paths_outside_directory(tmp_path, name):
    from decisionforge.eval.checkpoint import validate_checkpoint

    cp = checkpoint_at(tmp_path / "model", merged=True)
    (cp / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": name}}))
    with pytest.raises(ValueError, match="relative paths"):
        validate_checkpoint(cp)


@pytest.mark.asyncio
async def test_worker_spawn_failure_closes_pipe(tmp_path, monkeypatch):
    from multiprocessing.context import SpawnProcess

    from decisionforge.eval.local_adapter import LocalAdapter

    c = config_at(tmp_path)
    adapter = LocalAdapter(c.model, c.execution, worker_target=fixture_worker)

    def fail_start(process):
        raise OSError("process quota exhausted")

    monkeypatch.setattr(SpawnProcess, "start", fail_start)
    try:
        with pytest.raises(ValueError, match="process quota"):
            await adapter.start()
        assert adapter.connection is None and adapter.process is None
    finally:
        if adapter.connection is not None:
            adapter.connection.close()


@pytest.mark.asyncio
async def test_cancelling_shutdown_still_releases_worker(tmp_path):
    from decisionforge.eval.local_adapter import LocalAdapter

    c = config_at(tmp_path)
    adapter = await LocalAdapter(c.model, c.execution, worker_target=unreading_worker).start()
    process, connection = adapter.process, adapter.connection
    task = asyncio.create_task(adapter.close())
    await asyncio.sleep(0.03)
    task.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not process.is_alive()
        assert connection.closed
        await adapter.close()
    finally:
        if process.is_alive():
            process.terminate()
            await asyncio.to_thread(process.join, 2)
        connection.close()


@pytest.mark.asyncio
async def test_export_failure_still_closes_worker_and_journal(tmp_path, monkeypatch):
    from decisionforge.eval.local_adapter import LocalAdapter
    from decisionforge.eval.runner import run_evaluation
    from decisionforge.eval.store import EvalStore

    c = config_at(tmp_path)
    adapters, stores = [], []
    original_close = EvalStore.close

    def close(store):
        stores.append(store)
        original_close(store)

    def fail_export(store):
        raise OSError("disk full")

    async def factory(model, execution, transport=None):
        adapter = await LocalAdapter(model, execution, worker_target=fixture_worker).start()
        adapters.append((adapter, adapter.process))
        return adapter

    monkeypatch.setattr(EvalStore, "export", fail_export)
    monkeypatch.setattr(EvalStore, "close", close)
    try:
        with pytest.raises(OSError, match="disk full"):
            await run_evaluation(c, adapter_factory=factory)
        assert adapters[0][0].process is None and not adapters[0][1].is_alive()
        assert len(stores) == 1 and stores[0]._stream is None
    finally:
        # Clean up even when exercising the pre-fix failure.
        for adapter, _ in adapters:
            await adapter.close(force=True)


def test_full_precision_native_mapping():
    from decisionforge.eval.local_worker import answer_from_probabilities

    q = sample("score")["request"]["questions"]["decision"]
    p = {"0": 0.106572, "1": 0.2, "2": 0.693428}
    a = answer_from_probabilities(q, p)
    assert a["probabilities"] == p and a["score"] == pytest.approx(1.586856)
    assert a["legend"] == {"0": "low", "1": "middle", "2": "high"}
    q = sample("choice")["request"]["questions"]["decision"]
    assert answer_from_probabilities(q, {"A": 0.206572, "B": 0.793428})["choice"] == "B"
    q = sample("noul")["request"]["questions"]["decision"]
    assert answer_from_probabilities(q, {"false": 0.206572, "true": 0.793428})["noul"] == 0.793428
    with pytest.raises(ValueError):
        answer_from_probabilities(q, {"false": float("nan"), "true": 1})


@pytest.mark.asyncio
async def test_local_worker_integration_resume_score_only_and_reference_isolation(tmp_path):
    from decisionforge.eval.local_adapter import LocalAdapter
    from decisionforge.eval.runner import run_evaluation

    c = config_at(tmp_path)

    async def factory(model, execution, transport=None):
        adapter = LocalAdapter(model, execution, worker_target=fixture_worker)
        await adapter.start()
        return adapter

    result = await run_evaluation(c, adapter_factory=factory)
    report = json.loads((Path(c.output) / result["report"] / "metrics.json").read_text())
    assert report["counts"]["strict_valid"] == 3
    assert report["counts"]["scoring_invalid"] == 0
    assert report["prediction_provenance"]["unresolved_model_predictions"] == 0
    assert report["performance"]["cost_basis"] == "local_compute_unpriced"
    assert report["metrics"]["soft_brier"]["used"] == 3
    predictions = (Path(c.output) / "predictions.jsonl").read_bytes()

    async def reject(*args, **kwargs):
        pytest.fail("A completed resume loaded a model")

    await run_evaluation(c, resume=True, adapter_factory=reject)
    assert predictions == (Path(c.output) / "predictions.jsonl").read_bytes()
    for p in Path(c.model.path).iterdir():
        p.unlink()
    Path(c.model.path).rmdir()
    Path(c.data.path).unlink()
    scored = await run_evaluation(c, score_only=True, adapter_factory=reject)
    assert scored["status"] == "scored"
    assert predictions == (Path(c.output) / "predictions.jsonl").read_bytes()


@pytest.mark.asyncio
async def test_local_timeout_and_cancellation_terminate_worker(tmp_path):
    from decisionforge.eval.local_adapter import LocalAdapter

    c = config_at(tmp_path, timeout=0.1)
    adapter = LocalAdapter(c.model, c.execution, worker_target=hanging_worker)
    await adapter.start()
    process = adapter.process
    s = sample()  # question constructed independently of evaluation target.
    from decisionforge.eval.models import Sample

    value = Sample.model_validate(s)
    result = await adapter.send(
        {"model": c.model.model, **value.request.model_dump(exclude_none=True)}, value.question
    )
    assert result["fatal"] and result["error"]["kind"] == "local_timeout"
    assert not process.is_alive()
    await adapter.close()
    adapter = LocalAdapter(c.model, c.execution, worker_target=hanging_worker)
    await adapter.start()
    process = adapter.process
    task = asyncio.create_task(
        adapter.send(
            {"model": c.model.model, **value.request.model_dump(exclude_none=True)}, value.question
        )
    )
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not process.is_alive()
    await adapter.close()


@pytest.mark.asyncio
async def test_model_failure_stops_scheduling_and_reports_partial(tmp_path):
    from decisionforge.eval.local_adapter import LocalAdapter
    from decisionforge.eval.runner import run_evaluation

    c = config_at(tmp_path, timeout=0.1)

    async def factory(model, execution, transport=None):
        a = LocalAdapter(model, execution, worker_target=hanging_worker)
        await a.start()
        return a

    r = await run_evaluation(c, adapter_factory=factory)
    report = json.loads((Path(c.output) / r["report"] / "metrics.json").read_text())
    assert r["status"] == "model_failed"
    assert report["counts"]["attempted"] == 1 and report["counts"]["unattempted"] == 2
    assert report["partial"]


def test_local_offline_cli_imports_no_sdk_or_network(tmp_path):
    c = config_at(tmp_path)
    p = tmp_path / "eval.json"
    p.write_text(json.dumps(c.model_dump()))
    code = """import sys
class Guard:
 def find_spec(self, name, *args):
  if name.split('.')[0] in {'torch','unsloth','transformers'}: raise AssertionError(name)
sys.meta_path.insert(0, Guard())
from decisionforge.cli import main
raise SystemExit(main(sys.argv[1:]))"""
    r = subprocess.run(
        [sys.executable, "-c", code, "eval", "run", "--config", str(p), "--validate-only"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    assert not Path(c.output).exists()


def graceful_worker(connection, options):
    connection.send(
        {"kind": "ready", "metadata": {"model": options["path"], "backbone": {}, "runtime": {}}}
    )
    assert connection.recv()["kind"] == "close"
    Path(options["path"], "worker-closed.txt").write_text("graceful")
    connection.close()


@pytest.mark.asyncio
async def test_normal_close_requests_worker_shutdown(tmp_path):
    from decisionforge.eval.local_adapter import LocalAdapter

    c = config_at(tmp_path)
    adapter = LocalAdapter(c.model, c.execution, worker_target=graceful_worker)
    await adapter.start()
    await adapter.close()
    assert (Path(c.model.path) / "worker-closed.txt").read_text() == "graceful"


@pytest.mark.asyncio
async def test_resume_after_worker_start_failure_and_changed_weights(tmp_path):
    from decisionforge.eval.local_adapter import LocalAdapter
    from decisionforge.eval.runner import run_evaluation

    c = config_at(tmp_path)

    async def failure(model, execution, transport=None):
        a = LocalAdapter(model, execution, worker_target=failing_worker)
        return await a.start()

    with pytest.raises(ValueError, match="Missing Unsloth"):
        await run_evaluation(c, adapter_factory=failure)

    async def working(model, execution, transport=None):
        return await LocalAdapter(model, execution, worker_target=fixture_worker).start()

    assert (await run_evaluation(c, resume=True, adapter_factory=working))["status"] == "completed"
    weights = Path(c.model.path) / "adapter_model.safetensors"
    weights.write_bytes(weights.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Resume incompatible"):
        await run_evaluation(c, resume=True, adapter_factory=working)


@pytest.mark.asyncio
async def test_partial_journal_resume_does_not_repeat_predictions(tmp_path):
    from decisionforge.eval.local_adapter import LocalAdapter
    from decisionforge.eval.runner import run_evaluation

    c = config_at(tmp_path)

    # Use a request counter solely in this test dependency, not in production adapter behavior.
    class Interrupted(LocalAdapter):
        calls = 0

        async def send(self, payload, question):
            self.calls += 1
            if self.calls == 2:
                raise asyncio.CancelledError()
            return await super().send(payload, question)

    async def interrupted(model, execution, transport=None):
        return await Interrupted(model, execution, worker_target=fixture_worker).start()

    with pytest.raises(asyncio.CancelledError):
        await run_evaluation(c, adapter_factory=interrupted)
    first = json.loads(Path(c.output, "predictions.jsonl").read_text().splitlines()[0])

    async def working(model, execution, transport=None):
        return await LocalAdapter(model, execution, worker_target=fixture_worker).start()

    await run_evaluation(c, resume=True, adapter_factory=working)
    predictions = [
        json.loads(line) for line in Path(c.output, "predictions.jsonl").read_text().splitlines()
    ]
    assert len(predictions) == 3 and predictions[0] == first
    assert len({p["id"] for p in predictions}) == 3
    assert first["attempt_ids"] == predictions[0]["attempt_ids"]
