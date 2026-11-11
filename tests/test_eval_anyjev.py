"""AnyJev boundaries: public requests, unmodified L0 readout and local lifecycle."""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from decisionforge.eval.data import write_jsonl
from decisionforge.eval.models import EvalConfig, Request, load_config
from tests.eval_fixtures import sample

REVISION = "35311c58955716cb8955ce1368a665988bc63735"


def config_at(tmp_path, items=None, **execution):
    data = tmp_path / "samples.jsonl"
    write_jsonl(data, items or [sample(k, "soft", k) for k in ("choice", "noul", "score")])
    return EvalConfig(
        data={"path": str(data)},
        model={
            "adapter": "anyjev_local",
            "model": "fixture/base",
            "revision": "a" * 40,
        },
        output=str(tmp_path / "run"),
        execution=execution,
    )


def test_typed_config_fixed_method_and_local_execution(tmp_path):
    c = config_at(tmp_path)
    assert c.model.adapter == "anyjev_local"
    for field, value in [
        ("level", "raw"),
        ("prior", "batch"),
        ("adaptive_shifts", True),
        ("combine", "mean"),
        ("max_permutations", 1),
        ("training_free", False),
        ("model", "nokia/Tacit-7B"),
        ("revision", "main"),
    ]:
        values = c.model_dump()
        values["model"][field] = value
        with pytest.raises(ValidationError):
            EvalConfig.model_validate(values)
    for change in ({"concurrency": 2}, {"max_attempts": 2}):
        values = c.model_dump()
        values["execution"].update(change)
        with pytest.raises(ValidationError):
            EvalConfig.model_validate(values)


def test_relative_backbone_and_hf_id(tmp_path):
    path = tmp_path / "eval.yaml"
    path.write_text("data: {path: data.jsonl}\nmodel: {adapter: anyjev_local, model: ./base}\n")
    assert load_config(path).model.model == str(tmp_path / "base")
    path.write_text("data: {path: data.jsonl}\nmodel: {adapter: anyjev_local, model: org/base}\n")
    with pytest.raises(ValueError, match="revision"):
        load_config(path)


@pytest.mark.parametrize("size", [27, 77, 151])
def test_full_catalog_is_validated_without_loss(tmp_path, size):
    from decisionforge.eval.runner import preflight
    from decisionforge.training_free.anyjev.processing import prepare_request

    item = sample()
    criteria = {str(i): {"description": f"intent {i}"} for i in range(size)}
    item["request"]["questions"]["decision"]["criteria"] = criteria
    item["target"] = None
    config = config_at(tmp_path, [item])
    preflight(config)
    converted = prepare_request(item["request"])
    assert len(converted["options"]) == size
    assert [json.loads(option)["key"] for option in converted["options"]] == list(criteria)
    assert not Path(config.output).exists()


def test_lossless_mapping_structured_state_and_score_order():
    from decisionforge.training_free.anyjev.processing import prepare_request

    public = {
        "state": [{"role": "user", "content": "hello", "extra": {"z": 1, "a": 2}}],
        "questions": {
            "decision": {
                "type": "choice",
                "instructions": {"z": "last", "a": "first"},
                "criteria": {"复杂 key": None, "other": None},
            }
        },
    }
    converted = prepare_request(Request.model_validate(public))
    assert converted["state"] == '[{"content":"hello","extra":{"a":2,"z":1},"role":"user"}]'
    assert [json.loads(v) for v in converted["options"]] == [
        {"key": "复杂 key", "description": None},
        {"key": "other", "description": None},
    ]
    assert converted["labels"] == ["复杂 key", "other"]
    assert json.loads(converted["text"])["instructions"] == {"a": "first", "z": "last"}
    public["questions"]["decision"] = {
        "type": "score",
        "criteria": ["high", {"x": ["low"]}, "high"],
    }
    converted = prepare_request(Request.model_validate(public))
    assert converted["options"] == ['"high"', '{"x":["low"]}', '"high"']
    public["questions"]["decision"] = {
        "type": "noul",
        "criteria": {"false": ["no"], "true": {"yes": "y"}},
    }
    assert json.loads(prepare_request(Request.model_validate(public))["text"])["criteria"] == {
        "false": ["no"],
        "true": {"yes": "y"},
    }


def decision_for(kind, probs):
    return SimpleNamespace(
        probs=probs,
        level="L0",
        confidence=max(probs, default=0),
        argmax='{"description":"two","key":"B"}',
        p_true=probs[0] if probs else None,
        value=sum(i * p for i, p in enumerate(probs)),
        diagnostics={"permutations": 1},
    )


@pytest.mark.parametrize(
    "kind,probs,field,want",
    [
        ("choice", [0.2, 0.8], "choice", "B"),
        ("noul", [0.8, 0.2], "noul", 0.8),
        ("score", [0.2, 0.3, 0.5], "score", 1.3),
    ],
)
def test_native_readout(kind, probs, field, want):
    from decisionforge.training_free.anyjev.processing import response_from_decision

    q = Request.model_validate(sample(kind)["request"]).questions["decision"]
    response = response_from_decision(q, decision_for(kind, probs))
    answer = response["answers"]["decision"]
    assert answer[field] == want
    assert answer["confidence"] == max(probs)
    assert response["anyjev"]["confidence_basis"] == "maximum_candidate_probability"
    assert response["anyjev"]["probs"] == probs
    assert response.get("usage") is None
    if kind == "noul":
        assert answer["probabilities"] == {"false": 0.2, "true": 0.8}
    if kind == "score":
        assert answer["legend"] == {"0": "low", "1": "middle", "2": "high"}


@pytest.mark.parametrize(
    "probs",
    [[], [1.0], [0.2, 0.3, 0.5], [float("nan"), 1.0], [-0.1, 1.1], [0.4, 0.4], [float("inf"), 0.0]],
)
def test_invalid_readout_preserves_evidence_without_repair(probs):
    from decisionforge.training_free.anyjev.processing import response_from_decision

    q = Request.model_validate(sample("noul")["request"]).questions["decision"]
    raw = response_from_decision(q, decision_for("noul", probs))
    assert "answers" not in raw
    assert raw["anyjev"]["probs"] == [
        "NaN" if p != p else "Infinity" if p == float("inf") else p for p in probs
    ]
    assert raw["anyjev"]["invalid_output"]
    json.dumps(raw, allow_nan=False)


@pytest.mark.parametrize(
    "field,value", [("confidence", float("nan")), ("argmax", "bad"), ("value", float("inf"))]
)
def test_invalid_result_snapshots_independently_readable_fields(field, value):
    from decisionforge.training_free.anyjev.processing import response_from_decision

    kind = "score" if field == "value" else "choice"
    probs = [0.2, 0.3, 0.5] if kind == "score" else [0.2, 0.8]
    decision = decision_for(kind, probs)
    setattr(decision, field, value)
    decision.to_dict = lambda: {"offending": value, "original": probs}
    q = Request.model_validate(sample(kind)["request"]).questions["decision"]
    response = response_from_decision(q, decision)
    assert "answers" not in response
    raw = response["anyjev"]
    expected = "NaN" if field == "confidence" else "Infinity" if field == "value" else value
    assert raw[field] == expected
    assert raw["result"] == {"offending": expected, "original": probs}
    assert raw["probs"] == probs
    json.dumps(response, allow_nan=False)


def test_unreadable_result_field_does_not_drop_other_raw_evidence():
    from decisionforge.training_free.anyjev.processing import response_from_decision

    class Decision:
        probs, level, diagnostics = [0.2, 0.8], "L0", {"probe": 3}
        argmax = "bad"

        @property
        def confidence(self):
            raise ValueError("unreadable confidence")

        def to_dict(self):
            return {"payload": [float("nan")]}

    q = Request.model_validate(sample("choice")["request"]).questions["decision"]
    response = response_from_decision(q, Decision())
    assert "answers" not in response
    assert response["anyjev"]["argmax"] == "bad"
    assert response["anyjev"]["result"] == {"payload": ["NaN"]}
    json.dumps(response, allow_nan=False)


def fixture_worker(connection, options):
    from decisionforge.training_free.anyjev.processing import response_from_decision

    connection.send(
        {
            "kind": "ready",
            "metadata": {
                "model": options["model"],
                "backbone": {"name": options["model"], "revision": options["revision"]},
                "runtime": {"precision": "torch.bfloat16"},
                "method": {"revision": REVISION},
            },
        }
    )
    while True:
        command = connection.recv()
        if command["kind"] == "close":
            break
        assert set(command["request"]) == {"state", "questions"}
        assert "target" not in json.dumps(command["request"])
        question = Request.model_validate(command["request"]).questions["decision"]
        probs = [0.2, 0.3, 0.5] if question.type == "score" else [0.2, 0.8]
        response = response_from_decision(question, decision_for(question.type, probs))
        connection.send({"kind": "prediction", "response": response})
    connection.close()


@pytest.mark.asyncio
async def test_local_journal_resume_identity_and_offline_scoring(tmp_path):
    from decisionforge.eval.adapters.anyjev import AnyJevAdapter
    from decisionforge.eval.runner import run_evaluation

    c = config_at(tmp_path)
    workers = []

    async def factory(model, execution, transport=None):
        adapter = await AnyJevAdapter(model, execution, worker_target=fixture_worker).start()
        workers.append(adapter.process)
        return adapter

    result = await run_evaluation(c, adapter_factory=factory)
    assert not workers[0].is_alive()
    report = json.loads((Path(c.output) / result["report"] / "metrics.json").read_text())
    assert report["counts"]["strict_valid"] == 3
    assert report["prediction_provenance"]["unresolved_model_predictions"] == 0
    assert report["prediction_provenance"]["method"] == "anyjev_l0"
    assert report["prediction_provenance"]["confidence_basis"] == "maximum_candidate_probability"
    assert report["prediction_provenance"]["declared"]["upstream_revision"] == REVISION
    assert report["performance"]["usage"]["input_tokens"] is None
    assert report["performance"]["cost_basis"] == "local_compute_unpriced"

    async def reject(*args):
        pytest.fail("completed resume or score-only loaded model")

    await run_evaluation(c, resume=True, adapter_factory=reject)
    c.model.precision = "float32"
    with pytest.raises(ValueError, match="Resume incompatible"):
        await run_evaluation(c, resume=True, adapter_factory=reject)
    Path(c.data.path).unlink()
    assert (await run_evaluation(c, score_only=True, adapter_factory=reject))["status"] == "scored"


def test_cpu_preflight_imports_no_optional_sdk(tmp_path):
    c = config_at(tmp_path)
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(c.model_dump()))
    code = """import sys
class Guard:
 def find_spec(self, name, *args):
  if name.split('.')[0] in {'torch', 'unsloth', 'transformers', 'numpy', 'anyjev'}: raise AssertionError(name)
sys.meta_path.insert(0, Guard())
from decisionforge.cli import main
raise SystemExit(main(sys.argv[1:]))"""
    result = subprocess.run(
        [sys.executable, "-c", code, "eval", "run", "--config", str(path), "--validate-only"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert not Path(c.output).exists()


@pytest.mark.parametrize("missing", ["probs", "confidence", "level", "diagnostics"])
def test_missing_upstream_fields_are_invalid_evidence(missing):
    from decisionforge.training_free.anyjev.processing import response_from_decision

    d = decision_for("noul", [0.8, 0.2])
    delattr(d, missing)
    q = Request.model_validate(sample("noul")["request"]).questions["decision"]
    response = response_from_decision(q, d)
    assert "answers" not in response
    assert response["anyjev"]["invalid_output"]
    json.dumps(response, allow_nan=False)


def test_backbone_content_identity_tracks_weights_and_method_json_roundtrip(tmp_path):
    from decisionforge.training_free.anyjev import backbone_identity, method_identity

    config = config_at(tmp_path)
    config.model.model = str(tmp_path / "base")
    root = Path(config.model.model)
    root.mkdir()
    weights = root / "model.safetensors"
    weights.write_bytes(b"original")
    before = backbone_identity(config.model)
    weights.write_bytes(b"changed")
    assert backbone_identity(config.model) != before
    assert json.loads(json.dumps(method_identity())) == method_identity()


def test_bounded_backend_checks_flat_and_shared_complete_prompts():
    from decisionforge.eval.adapters.anyjev_worker import BoundedBackend

    class Backend:
        tokenizer = SimpleNamespace(encode=lambda text, **kwargs: list(text))
        name = "fixture"

        def next_token_logprobs(self, prompts, token_ids):
            return ["unchanged"]

        def score_shared(self, groups, token_ids):
            return [["unchanged"]]

    backend = BoundedBackend(Backend(), 4)
    assert backend.next_token_logprobs(["1234"], [[1]]) == ["unchanged"]
    assert backend.score_shared([("12", ["34"])], [[1]]) == [["unchanged"]]
    with pytest.raises(ValueError, match="refusing to truncate"):
        backend.next_token_logprobs(["12345"], [[1]])
    with pytest.raises(ValueError, match="refusing to truncate"):
        backend.score_shared([("123", ["45"])], [[1]])


def nonfinite_worker(connection, options):
    connection.send({"kind": "ready", "metadata": {"model": options["model"]}})
    connection.recv()
    connection.send(
        {
            "kind": "prediction",
            "response": {
                "answers": {"decision": {"type": "noul", "noul": float("nan")}},
                "diagnostics": {"bad": float("inf")},
            },
        }
    )
    assert connection.recv()["kind"] == "close"
    connection.close()


@pytest.mark.asyncio
async def test_nonfinite_worker_result_preserves_raw_evidence(tmp_path):
    from decisionforge.eval.adapters.anyjev import AnyJevAdapter

    c = config_at(tmp_path)
    adapter = await AnyJevAdapter(c.model, c.execution, worker_target=nonfinite_worker).start()
    request = Request.model_validate(sample("noul")["request"])
    try:
        outcome = await adapter.send(request.model_dump(), request.questions["decision"])
        assert not outcome["protocol_ok"]
        assert outcome["raw_response"]["answers"]["decision"]["noul"] == "NaN"
        assert outcome["raw_response"]["diagnostics"]["bad"] == "Infinity"
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_partial_resume_preserves_predictions_and_closes_workers(tmp_path):
    from decisionforge.eval.adapters.anyjev import AnyJevAdapter
    from decisionforge.eval.runner import run_evaluation

    c = config_at(tmp_path)
    processes = []

    class Interrupted(AnyJevAdapter):
        calls = 0

        async def send(self, payload, question):
            self.calls += 1
            if self.calls == 2:
                raise asyncio.CancelledError()
            return await super().send(payload, question)

    async def interrupted(model, execution, transport=None):
        adapter = await Interrupted(model, execution, worker_target=fixture_worker).start()
        processes.append(adapter.process)
        return adapter

    with pytest.raises(asyncio.CancelledError):
        await run_evaluation(c, adapter_factory=interrupted)
    first = Path(c.output, "predictions.jsonl").read_text()

    async def working(model, execution, transport=None):
        return await AnyJevAdapter(model, execution, worker_target=fixture_worker).start()

    await run_evaluation(c, resume=True, adapter_factory=working)
    predictions = Path(c.output, "predictions.jsonl").read_text()
    assert predictions.startswith(first)
    assert len(predictions.splitlines()) == 3
    assert not processes[0].is_alive()


@pytest.mark.parametrize("source_commit", [REVISION, "b" * 40, None])
def test_installed_upstream_source_must_match_pin(monkeypatch, source_commit):
    import importlib.metadata

    from decisionforge.training_free.anyjev import validate_installation

    source = {"vcs_info": {"commit_id": source_commit}} if source_commit else {}
    fake = SimpleNamespace(version="0.3.0", read_text=lambda name: json.dumps(source))
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: fake)
    if source_commit == REVISION:
        assert validate_installation()["commit_id"] == REVISION
    else:
        with pytest.raises(ValueError, match="pinned"):
            validate_installation()


def test_preflight_rejects_local_trained_decision_artifact(tmp_path):
    from decisionforge.eval.runner import preflight

    c = config_at(tmp_path)
    root = tmp_path / "backbone"
    root.mkdir()
    (root / "adapter_config.json").write_text("{}")
    c.model.model = str(root)
    with pytest.raises(ValueError, match="ordinary backbone"):
        preflight(c)


def waiting_worker(connection, options):
    import time

    connection.send({"kind": "ready", "metadata": {"model": options["model"]}})
    connection.recv()
    time.sleep(30)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_anyjev_deadline_and_cancellation_release_owned_worker(tmp_path, cancelled):
    from decisionforge.eval.adapters.anyjev import AnyJevAdapter

    c = config_at(tmp_path, timeout=60 if cancelled else 0.1)
    adapter = await AnyJevAdapter(c.model, c.execution, worker_target=waiting_worker).start()
    process = adapter.process
    public = Request.model_validate(sample()["request"])
    try:
        task = asyncio.create_task(adapter.send(public.model_dump(), public.questions["decision"]))
        if cancelled:
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 3)
        else:
            assert (await asyncio.wait_for(task, 3))["fatal"]
        assert not process.is_alive()
    finally:
        await adapter.close()


def test_original_anyjev_shared_prefix_optimization_can_be_disabled_explicitly(tmp_path):
    from decisionforge.training_free.anyjev import method_identity

    values = config_at(tmp_path).model_dump()
    values["model"]["shared_prefix"] = False
    config = EvalConfig.model_validate(values)
    assert config.model.shared_prefix is False
    from decisionforge.eval.provenance import method_metadata

    assert method_metadata(config.model)["declared"]["settings"]["shared_prefix"] is False
    assert (
        method_metadata(config.model.model_dump())["declared"]["settings"]["shared_prefix"] is False
    )
    assert method_identity(shared_prefix=False)["settings"]["shared_prefix"] is False
    assert method_identity()["settings"]["shared_prefix"] == "auto"
    assert method_identity(shared_prefix=False)["settings"]["max_permutations"] is None
    assert method_identity(shared_prefix=False)["settings"]["prior"] == "content_free"
    values["model"]["shared_prefix"] = "invented"
    with pytest.raises(ValidationError):
        EvalConfig.model_validate(values)
