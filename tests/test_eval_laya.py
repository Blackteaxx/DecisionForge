import copy
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from test_training_laya import artifact_at

from decisionforge.eval.models import EvalConfig


def test_laya_eval_config_distinct_and_single_worker():
    c = EvalConfig(data={"path": "unused"}, model={"adapter": "laya_local", "path": "model"})
    assert c.model.model == "model"
    for options in ({"load_in_4bit": True}, {"precision": "bf16"}):
        with pytest.raises(ValidationError):
            EvalConfig(
                data={"path": "unused"}, model={"adapter": "laya_local", "path": "model", **options}
            )
    with pytest.raises(ValidationError, match="concurrency"):
        EvalConfig(
            data={"path": "unused"},
            model={"adapter": "laya_local", "path": "model"},
            execution={"concurrency": 2},
        )


@pytest.mark.parametrize("kind", ["choice", "score", "noul"])
def test_laya_public_predict_keeps_native_response_and_never_injects_targets(kind):
    from decisionforge.eval.adapters.laya_worker import predict_public

    criteria = (
        {"B": "bee", "A": "aye", "C": "see"}
        if kind == "choice"
        else ["low", "middle", "high"]
        if kind == "score"
        else None
    )
    public = {
        "state": {"text": "complete input"},
        "questions": {
            "decision": {
                "type": kind,
                "instructions": "judge",
                **({"criteria": criteria} if criteria else {}),
            }
        },
    }
    answer = (
        {"type": kind, "noul": 0.3333}
        if kind == "noul"
        else {
            "type": kind,
            "probabilities": dict(
                zip(
                    ["B", "A", "C"] if kind == "choice" else ["0", "1", "2"],
                    [0.3333, 0.3333, 0.3333],
                )
            ),
            "confidence": 0.0001,
        }
    )
    response = {
        "model": "laya-rl-agent",
        "answers": {"decision": answer},
        "usage": {"truncated": False, "input_tokens": 3},
    }

    class Agent:
        def system_one(self, state, questions):
            assert state == public["state"] and questions == public["questions"]
            return response

    before = copy.deepcopy(response)
    assert predict_public(Agent(), public) == before
    assert response == before
    with pytest.raises(ValueError, match="public|extra|Extra"):
        predict_public(Agent(), {**public, "gold": {}})


@pytest.mark.parametrize(
    "usage", [{"truncated": True}, {"options": {"decision": {"distinct": 1, "total": 3}}}]
)
def test_laya_refuses_input_loss(usage):
    from decisionforge.eval.adapters.laya_worker import predict_public

    public = {"state": "x", "questions": {"decision": {"type": "noul", "instructions": "yes?"}}}
    agent = SimpleNamespace(
        system_one=lambda state, questions: {
            "answers": {"decision": {"type": "noul", "noul": 0.2}},
            "usage": usage,
        }
    )
    with pytest.raises(ValueError, match="truncat|collaps"):
        predict_public(agent, public)


def test_laya_scoring_view_preserves_structured_rubric_without_mutating_raw():
    from decisionforge.eval.adapters.laya import LayaAdapter
    from decisionforge.eval.models import ExecutionConfig, Question

    c = EvalConfig(data={"path": "unused"}, model={"adapter": "laya_local", "path": "model"})
    adapter = LayaAdapter(c.model, ExecutionConfig())
    q = Question(type="score", instructions="judge", criteria=[{"zh": "低"}, ["high"]])
    raw = {
        "answers": {
            "decision": {
                "type": "score",
                "score": 0.8,
                "confidence": 0.3,
                "probabilities": {"0": 0.2, "1": 0.8},
                "legend": {"0": '{"zh": "低"}', "1": '["high"]'},
            }
        }
    }
    before = copy.deepcopy(raw)
    answer = adapter.response_answer(raw, q)
    assert answer["legend"] == {"0": {"zh": "低"}, "1": ["high"]}
    assert raw == before and answer["probabilities"] == {"0": 0.2, "1": 0.8}
    raw["answers"]["decision"]["legend"]["0"] = "wrong"
    with pytest.raises(ValueError, match="legend"):
        adapter.response_answer(raw, q)


def test_laya_scoring_view_allows_only_native_rounding_bound():
    from decisionforge.eval.adapters.laya import LayaAdapter
    from decisionforge.eval.models import ExecutionConfig, Question

    c = EvalConfig(data={"path": "unused"}, model={"adapter": "laya_local", "path": "model"})
    adapter = LayaAdapter(c.model, ExecutionConfig())
    q = Question(type="choice", criteria={"A": "a", "B": "b", "C": "c"})
    raw = {
        "answers": {
            "decision": {
                "type": "choice",
                "choice": "A",
                "confidence": 0.1,
                "probabilities": {"A": 0.3333, "B": 0.3333, "C": 0.3333},
            }
        }
    }
    assert adapter.response_answer(raw, q)["probabilities"]["A"] == 0.3333
    raw["answers"]["decision"]["probabilities"]["A"] = 0.2
    with pytest.raises(ValueError, match="mass|sum"):
        adapter.response_answer(raw, q)


@pytest.mark.parametrize("raw", [None, [], {"answers": []}, {"answers": {"decision": []}}])
def test_malformed_laya_score_response_is_protocol_error(raw):
    from decisionforge.eval.adapters.laya import LayaAdapter
    from decisionforge.eval.models import Question

    c = EvalConfig(data={"path": "unused"}, model={"adapter": "laya_local", "path": "model"})
    with pytest.raises(ValueError):
        LayaAdapter(c.model, c.execution).response_answer(
            raw, Question(type="score", criteria=["low", "high"])
        )


@pytest.mark.parametrize("kind", ["choice", "noul", "score"])
def test_absent_instructions_are_losslessly_encoded_as_null(kind):
    from decisionforge.eval.adapters.laya_worker import predict_public

    criteria = (
        {"A": {"text": "one"}, "B": ["two"]}
        if kind == "choice"
        else ["low", "high"]
        if kind == "score"
        else {"false": "absent", "true": {"text": "present"}}
    )
    public = {
        "state": [{"role": "user", "content": "text", "extra": {"keep": "all"}}],
        "questions": {"decision": {"type": kind, "criteria": criteria}},
    }
    before = copy.deepcopy(public)

    class Agent:
        def system_one(self, state, questions):
            assert state == before["state"]
            assert questions["decision"]["instructions"] == "null"
            assert questions["decision"]["criteria"] == criteria
            return {"usage": {"truncated": False}, "answers": {}}

    predict_public(Agent(), public)
    assert public == before


@pytest.mark.parametrize("value", ["", " ", {}, []])
def test_empty_canonical_instructions_remain_lossless(value):
    from decisionforge.eval.adapters.laya_worker import render_question

    q = {"type": "noul", "instructions": value, "criteria": {"true": "present"}}
    rendered = render_question(q)
    assert json.loads(rendered["instructions"]) == value
    assert q["instructions"] == value


def test_official_cpu_fallback_is_recorded_without_changing_raw(monkeypatch):
    from decisionforge.eval.adapters import laya_worker

    response = {"answers": {"decision": {"type": "noul", "noul": 0.2}}, "usage": {}}

    class Agent:
        cpu_fallback_count, device, dtype = 0, "cuda:0", "torch.float16"
        last_fallback_reason = "memory"

        def system_one(self, state, questions):
            self.cpu_fallback_count += 1
            return response

    monkeypatch.setattr(laya_worker, "agent_input_loss", lambda *args: None)
    runtime = laya_worker.LayaRuntime.__new__(laya_worker.LayaRuntime)
    runtime.agent = Agent()
    result = runtime.predict(
        {"state": "x", "questions": {"decision": {"type": "noul", "instructions": "judge"}}}
    )
    assert result is response
    assert runtime.last_runtime["device"] == "cpu"
    assert runtime.last_runtime["cpu_fallbacks"] == 1
    assert runtime.last_runtime["precision"] == "torch.float32"


def test_laya_preflight_and_identity_are_gpu_free(tmp_path):
    from decisionforge.eval.data import write_jsonl
    from decisionforge.eval.runner import preflight
    from tests.eval_fixtures import sample

    root = artifact_at(tmp_path / "model")
    data = tmp_path / "samples.jsonl"
    write_jsonl(data, [sample("choice", "soft", "one")])
    c = EvalConfig(data={"path": str(data)}, model={"adapter": "laya_local", "path": str(root)})
    assert preflight(c)[0].model.adapter == "laya_local"
    (root / "rl_agent_config.json").unlink()
    with pytest.raises(ValueError, match="Laya|laya"):
        preflight(c)


@pytest.mark.asyncio
async def test_laya_eval_resume_tracks_weight_identity_and_score_only_survives_removal(tmp_path):
    from test_eval_local import fixture_worker

    from decisionforge.eval.adapters.laya import LayaAdapter
    from decisionforge.eval.data import write_jsonl
    from decisionforge.eval.runner import run_evaluation
    from tests.eval_fixtures import sample

    root = artifact_at(tmp_path / "model")
    data = tmp_path / "samples.jsonl"
    write_jsonl(data, [sample("choice", "soft", "one")])
    c = EvalConfig(
        data={"path": str(data)},
        model={"adapter": "laya_local", "path": str(root)},
        output=str(tmp_path / "eval"),
    )

    async def factory(model, execution, transport):
        return await LayaAdapter(model, execution, worker_target=fixture_worker).start()

    result = await run_evaluation(c, adapter_factory=factory)
    assert result["status"] == "completed"
    manifest = json.loads((tmp_path / "eval/manifest.json").read_text())
    assert manifest["inference"]["laya"]["artifact_format"] == "laya"
    (root / "model.safetensors").write_bytes(b"modified")
    with pytest.raises(ValueError, match="Resume incompatible"):
        await run_evaluation(c, resume=True, adapter_factory=factory)
    (root / "model.safetensors").unlink()
    assert (await run_evaluation(c, score_only=True))["status"] == "scored"
