import json

import httpx
import pytest

from decisionforge.eval.models import EvalConfig
from decisionforge.eval.runner import run_evaluation
from tests.eval_fixtures import sample
from tests.test_eval_chat import chat_config, completion, records
from tests.test_eval_runner import config_at, response

LLM2JEV = {
    "method": "llm2jev",
    "upstream_repo": "https://github.com/Yinsongxu/LLM2Jev",
    "upstream_revision": "f2bdad648e7a34bfce64401fb0f7c44bc13209ae",
    "backbone_model": "Qwen/Qwen3.5-4B",
    "backbone_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "serving_backend": "sglang",
    "precision": "bf16",
    "dependencies": {"sglang": "0.5.20"},
    "settings": {"enable_thinking": False, "submission": "staged"},
}


@pytest.mark.parametrize(
    "kind,criteria,count",
    [
        ("choice", {str(i): {"intent": str(i)} for i in range(77)}, 77),
        ("choice", {str(i): str(i) for i in range(151)}, 151),
        ("score", ["low", "middle", "high"], 3),
        ("noul", None, 1),
        ("noul", {"true": "applies", "false": "does not apply"}, 2),
    ],
)
@pytest.mark.asyncio
async def test_llm2jev_full_catalog_provenance_and_binary_accounting(
    tmp_path, kind, criteria, count
):
    item = sample(kind=kind, target=None)
    if criteria is not None:
        item["request"]["questions"]["decision"]["criteria"] = criteria
    config = config_at(tmp_path, [item])
    config.model = EvalConfig(
        data=config.data,
        model={
            **config.model.model_dump(),
            "provenance": LLM2JEV,
        },
    ).model

    def handle(req):
        body = json.loads(req.content)
        assert body == {"model": "test", **item["request"]}
        if kind == "choice":
            answer = {
                "type": "choice",
                "choice": "0",
                "confidence": 0.0,
                "probabilities": {str(i): 1 / count for i in range(count)},
            }
            return httpx.Response(
                200, json={"model": "service-alias", "answers": {"decision": answer}}
            )
        return response(req)

    result = await run_evaluation(config, transport=httpx.MockTransport(handle))
    pred = records(tmp_path, "predictions.jsonl")[0]
    assert pred["probability_source"] == "llm2jev_yes_no_logits"
    assert pred["candidate_task_count"] == count
    manifest = json.loads((tmp_path / "run/manifest.json").read_text())
    assert manifest["inference"]["model"]["provenance"] == LLM2JEV
    report = json.loads((tmp_path / "run" / result["report"] / "metrics.json").read_text())
    assert report["prediction_provenance"]["method"] == "llm2jev"
    assert report["prediction_provenance"]["declared"]["backbone_model"] == "Qwen/Qwen3.5-4B"
    assert report["performance"]["candidate_tasks"]["known_total"] == count
    assert report["performance"]["sessions"][0]["attempts"] == 1
    assert report["performance"]["reported_cost_usd"] is None


@pytest.mark.asyncio
async def test_chat_prompt_and_serving_settings_enter_resume_identity(tmp_path, monkeypatch):
    from decisionforge.eval import chat_prompt

    config = chat_config(tmp_path, serving={"enable_thinking": False})

    def handle(req):
        assert json.loads(req.content)["chat_template_kwargs"] == {"enable_thinking": False}
        return httpx.Response(200, json=completion())

    await run_evaluation(config, transport=httpx.MockTransport(handle))
    monkeypatch.setattr(chat_prompt, "PROMPT_VERSION", "decisionforge.chat.zero-shot.v2")
    with pytest.raises(ValueError, match="incompatible"):
        await run_evaluation(config, resume=True)


@pytest.mark.asyncio
async def test_llm2jev_settings_change_refuses_resume(tmp_path):
    config = config_at(tmp_path)
    config.model = EvalConfig(
        data=config.data,
        model={
            **config.model.model_dump(),
            "provenance": LLM2JEV,
        },
    ).model
    await run_evaluation(config, transport=httpx.MockTransport(response))
    config.model.provenance.settings["submission"] = "all"
    with pytest.raises(ValueError, match="incompatible"):
        await run_evaluation(config, resume=True)


@pytest.mark.parametrize(
    "provenance",
    [
        {"method": "llm2jev"},
        {**LLM2JEV, "upstream_repo": "https://github.com/tic-top/llm2jev"},
        {**LLM2JEV, "upstream_revision": "main"},
        {**LLM2JEV, "backbone_revision": "main"},
    ],
)
def test_llm2jev_requires_explicit_pinned_provenance(provenance):
    with pytest.raises(ValueError):
        EvalConfig(data={"path": "samples.jsonl"}, model={"provenance": provenance})


@pytest.mark.asyncio
@pytest.mark.parametrize("dependency", ["eval/metrics.py", "eval/data.py"])
async def test_chat_probability_validator_source_change_refuses_resume(
    tmp_path, monkeypatch, dependency
):
    import decisionforge.eval.runner as runner

    config = chat_config(tmp_path)
    await run_evaluation(
        config, transport=httpx.MockTransport(lambda req: httpx.Response(200, json=completion()))
    )
    original = runner.source_versions()
    monkeypatch.setattr(runner, "source_versions", lambda: {**original, dependency: "a" * 64})
    with pytest.raises(ValueError, match="incompatible"):
        await run_evaluation(config, resume=True)
