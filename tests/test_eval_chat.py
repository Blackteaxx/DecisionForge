import json
from pathlib import Path

import httpx
import pytest

from decisionforge.eval.data import write_jsonl
from decisionforge.eval.models import EvalConfig
from decisionforge.eval.runner import run_evaluation
from tests.eval_fixtures import prediction, sample


def chat_config(tmp_path, items=None, **model):
    path = tmp_path / "samples.jsonl"
    write_jsonl(path, items or [sample()])
    return EvalConfig(
        data={"path": str(path)},
        output=str(tmp_path / "run"),
        model={
            "adapter": "chat_api",
            "base_url": "http://chat.test",
            "model": "baseline",
            "api_key_env": None,
            "max_tokens": 4096,
            **model,
        },
    )


def completion(kind="choice", *, content=None, finish="stop", refusal=None, usage=True):
    raw = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "actual-baseline",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish,
                "message": {
                    "role": "assistant",
                    "content": content
                    if content is not None
                    else json.dumps({"answers": {"decision": prediction(kind=kind)["answer"]}}),
                    "refusal": refusal,
                },
            }
        ],
    }
    if usage:
        raw["usage"] = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    return raw


def records(tmp_path, name):
    return [json.loads(line) for line in (tmp_path / "run" / name).read_text().splitlines()]


@pytest.mark.parametrize("kind", ["choice", "noul", "score"])
@pytest.mark.asyncio
async def test_chat_public_request_native_answer_and_exact_wire_journal(tmp_path, kind):
    item = sample(kind=kind)
    item["metadata"]["private_secret"] = "PRIVATE_SENTINEL"
    config = chat_config(tmp_path, [item], response_format="json_schema")
    seen = []

    def handle(req):
        assert str(req.url) == "http://chat.test/v1/chat/completions"
        body = json.loads(req.content)
        seen.append(body)
        event = json.loads((tmp_path / "run/.eval/events.jsonl").read_text().splitlines()[-1])
        assert event["updates"][0]["record"]["request"] == body
        assert body["model"] == "baseline"
        assert body["temperature"] == 0
        assert body["max_tokens"] == 4096
        assert body["stream"] is False
        assert body["n"] == 1
        assert len(body["messages"]) == 2
        public = json.loads(body["messages"][1]["content"])
        assert public == item["request"]
        assert "PRIVATE_SENTINEL" not in req.content.decode()
        assert "target" not in public and "metadata" not in public
        assert body["response_format"]["json_schema"]["strict"] is True
        return httpx.Response(200, json=completion(kind))

    result = await run_evaluation(config, transport=httpx.MockTransport(handle))
    pred = records(tmp_path, "predictions.jsonl")[0]
    assert pred["protocol_ok"] is True
    assert pred["answer"] == prediction(kind=kind)["answer"]
    assert pred["probability_source"] == "llm_self_reported"
    assert pred["usage"] == {"input_tokens": 10, "output_tokens": 20}
    assert pred["raw_response"]["id"] == "chatcmpl-test"
    assert json.loads(pred["raw_generated_text"])["answers"]["decision"] == pred["answer"]
    assert pred["model"] == "actual-baseline"
    assert pred["cost_usd"] is None
    await run_evaluation(config, resume=True, transport=httpx.MockTransport(handle))
    assert len(seen) == 1
    report = json.loads((tmp_path / "run" / result["report"] / "metrics.json").read_text())
    assert report["counts"]["scoring_valid"] == 1


BAD_CONTENT = [
    "not JSON",
    "```json\n{}\n```",
    '{"answers":{},"answers":{}}',
    json.dumps({"answers": {"decision": {"type": "noul", "noul": 0.5}}}),
]
for field, value in [
    ("choice", "UNKNOWN"),
    ("confidence", True),
    ("probabilities", {"A": 1}),
    ("probabilities", {"A": -0.1, "B": 1.1}),
    ("probabilities", {"A": float("nan"), "B": 0.5}),
]:
    answer = prediction()["answer"]
    answer[field] = value
    BAD_CONTENT.append(json.dumps({"answers": {"decision": answer}}))


@pytest.mark.parametrize("probabilities", [{"A": 0.76, "B": 0.99}, {"A": 0.0, "B": 0.0}])
@pytest.mark.asyncio
async def test_chat_nonunit_probability_total_is_accepted_and_scored_without_repair(
    tmp_path, probabilities
):
    config = chat_config(tmp_path)
    answer = prediction(probabilities=probabilities)["answer"]
    raw = completion(content=json.dumps({"answers": {"decision": answer}}))
    result = await run_evaluation(
        config, transport=httpx.MockTransport(lambda req: httpx.Response(200, json=raw))
    )
    saved = records(tmp_path, "predictions.jsonl")[0]
    report = json.loads((tmp_path / "run" / result["report"] / "metrics.json").read_text())
    assert saved["protocol_ok"] is True
    assert saved["answer"]["probabilities"] == probabilities
    assert saved["raw_response"] == raw
    assert report["counts"]["scoring_valid"] == 1
    assert report["counts"]["renormalized"] == 0


@pytest.mark.parametrize("content", BAD_CONTENT)
@pytest.mark.asyncio
async def test_chat_invalid_output_is_terminal_without_repair(tmp_path, content):
    config = chat_config(tmp_path)
    config.execution.max_attempts = 3
    calls = []
    raw = completion(content=content)
    await run_evaluation(
        config,
        transport=httpx.MockTransport(
            lambda req: calls.append(req) or httpx.Response(200, json=raw)
        ),
    )
    pred = records(tmp_path, "predictions.jsonl")[0]
    assert pred["transport_ok"] is True
    assert pred["protocol_ok"] is False
    assert pred["answer"] is None
    assert pred["raw_generated_text"] == content
    assert pred["error"]["kind"] == "protocol"
    assert len(calls) == 1


@pytest.mark.parametrize(
    "finish,refusal", [("length", None), ("content_filter", None), ("stop", "I cannot answer")]
)
@pytest.mark.asyncio
async def test_chat_refusal_and_truncation_are_failures(tmp_path, finish, refusal):
    config = chat_config(tmp_path)
    await run_evaluation(
        config,
        transport=httpx.MockTransport(
            lambda req: httpx.Response(200, json=completion(finish=finish, refusal=refusal))
        ),
    )
    assert records(tmp_path, "predictions.jsonl")[0]["protocol_ok"] is False


@pytest.mark.asyncio
async def test_chat_missing_usage_retry_auth_and_redaction(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_CHAT_KEY", "secret-baseline-key")
    config = chat_config(
        tmp_path, [sample(identity="one"), sample(identity="two")], api_key_env="TEST_CHAT_KEY"
    )
    config.execution.max_attempts = 2
    config.execution.backoff = 0
    calls = []

    def handle(req):
        assert req.headers["Authorization"] == "Bearer secret-baseline-key"
        calls.append(req)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, json={"error": "limit"})
        raw = completion(usage=False)
        raw["echo"] = "secret-baseline-key"
        if len(calls) == 3:
            return httpx.Response(403, json={"error": "secret-baseline-key"})
        return httpx.Response(200, json=raw)

    result = await run_evaluation(config, transport=httpx.MockTransport(handle))
    assert result["status"] == "access_denied"
    assert len(calls) == 3
    assert records(tmp_path, "predictions.jsonl")[0]["usage"] is None
    assert records(tmp_path, "predictions.jsonl")[0]["protocol_ok"] is True
    for file in (tmp_path / "run").rglob("*"):
        if file.is_file():
            assert b"secret-baseline-key" not in file.read_bytes()


@pytest.mark.asyncio
async def test_chat_offline_preflight_and_score_only_need_no_secret_or_network(tmp_path):
    config = chat_config(tmp_path, api_key_env="ABSENT_CHAT_KEY")
    transport = httpx.MockTransport(lambda req: pytest.fail("offline mode made a request"))
    assert (await run_evaluation(config, validate_only=True, transport=transport))[
        "status"
    ] == "validated"
    config.model.api_key_env = None
    await run_evaluation(
        config, transport=httpx.MockTransport(lambda req: httpx.Response(200, json=completion()))
    )
    config.model.api_key_env = "ABSENT_CHAT_KEY"
    Path(config.data.path).unlink()
    assert (await run_evaluation(config, score_only=True, transport=transport))[
        "status"
    ] == "scored"


@pytest.mark.parametrize(
    "settings",
    [
        {"temperature": -1},
        {"max_tokens": 0},
        {"base_url": "https://u:p@host"},
        {"response_format": "auto"},
        {"serving": {"messages": []}},
        {"endpoint_path": "/v1/systemone"},
    ],
)
def test_chat_rejects_invalid_config(tmp_path, settings):
    with pytest.raises(ValueError):
        chat_config(tmp_path, **settings)


@pytest.mark.asyncio
async def test_chat_score_legend_distinguishes_nested_json_boolean_from_number(tmp_path):
    item = sample("score")
    item["request"]["questions"]["decision"]["criteria"] = [{"pass": False}, {"pass": True}]
    item["target"]["value"] = 1
    config = chat_config(tmp_path, [item])
    answer = {
        "type": "score",
        "score": 1,
        "confidence": 0.8,
        "probabilities": {"0": 0.2, "1": 0.8},
        "legend": {"0": {"pass": 0}, "1": {"pass": 1}},
    }
    await run_evaluation(
        config,
        transport=httpx.MockTransport(
            lambda req: httpx.Response(
                200, json=completion(content=json.dumps({"answers": {"decision": answer}}))
            )
        ),
    )
    assert records(tmp_path, "predictions.jsonl")[0]["protocol_ok"] is False


@pytest.mark.asyncio
async def test_chat_score_legend_preserves_json_values_with_equivalent_numbers(tmp_path):
    item = sample("score")
    item["request"]["questions"]["decision"]["criteria"] = [
        {"count": 0.0, "pass": False},
        {"count": 1.0, "pass": True},
    ]
    item["target"]["value"] = 1
    config = chat_config(tmp_path, [item])
    answer = {
        "type": "score",
        "score": 1,
        "confidence": 0.8,
        "probabilities": {"0": 0.2, "1": 0.8},
        "legend": {"1": {"pass": True, "count": 1}, "0": {"pass": False, "count": 0}},
    }
    await run_evaluation(
        config,
        transport=httpx.MockTransport(
            lambda req: httpx.Response(
                200, json=completion(content=json.dumps({"answers": {"decision": answer}}))
            )
        ),
    )
    result = records(tmp_path, "predictions.jsonl")[0]
    assert result["protocol_ok"] is True
    assert result["answer"] == answer
