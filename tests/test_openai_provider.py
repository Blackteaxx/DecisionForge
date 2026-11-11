import json

import httpx
import pytest
from openai import AsyncOpenAI

from decisionforge.generation.contracts import ChoiceOutput, QAConstruction, TextAssessment
from decisionforge.llm import OpenAIProvider
from decisionforge.storage import Store
from decisionforge.util import ProviderError


@pytest.mark.parametrize("api", ["responses", "chat_completions"])
@pytest.mark.parametrize("mode", ["strict", "json", "prompt"])
@pytest.mark.parametrize("token_param", ["max_completion_tokens", "max_tokens"])
async def test_protocol_and_output_modes(config, tmp_path, api, mode, token_param):
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        assert body["model"] == config.llm.model
        output = '{"distractors":["d1"]}'
        if api == "responses":
            assert request.url.path == "/v1/responses"
            assert body["max_output_tokens"] == config.llm.max_output_tokens
            assert "messages" not in body and "response_format" not in body
            assert "max_tokens" not in body and "max_completion_tokens" not in body
            fmt = body.get("text", {}).get("format")
            messages = body["input"]
            response = responses_body(output)
        else:
            assert request.url.path == "/v1/chat/completions"
            assert body[token_param] == config.llm.max_output_tokens
            other = (
                "max_tokens" if token_param == "max_completion_tokens" else "max_completion_tokens"
            )
            assert other not in body
            assert "input" not in body and "text" not in body and "store" not in body
            assert "max_output_tokens" not in body
            fmt = body.get("response_format")
            messages = body["messages"]
            response = {
                "id": "chat_test",
                "object": "chat.completion",
                "created": 1,
                "model": "fixture",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": output},
                    }
                ],
            }
        if mode == "prompt":
            assert fmt is None
        else:
            assert fmt["type"] == ("json_schema" if mode == "strict" else "json_object")
        if mode != "strict":
            assert "JSON schema:" in messages[0]["content"]
        return httpx.Response(200, json=response)

    config.llm.api = api
    config.llm.structured_output = mode
    config.llm.chat_token_limit_param = token_param
    store = Store(tmp_path / "db")
    client = AsyncOpenAI(
        api_key="test",
        base_url="https://test.invalid/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    provider = OpenAIProvider(config, store, client)
    try:
        result = await provider.call("choice", {"count": 1}, ChoiceOutput, "r", "choice")
        assert result.distractors == ["d1"]
        assert len(requests) == 1
    finally:
        await provider.close()
        store.close()


@pytest.mark.parametrize("mode", ["json", "prompt"])
async def test_bad_request_diagnostics_are_redacted_and_not_retried(config, tmp_path, mode):
    calls = []
    secret = "PRIVATE_TEST_CREDENTIAL"

    def handler(request):
        calls.append(request)
        return httpx.Response(
            400,
            headers={"x-request-id": "req-test"},
            json={
                "error": {
                    "message": f"Unsupported response_format; credential={secret}; Bearer other-secret",
                    "param": "response_format",
                    "code": "unsupported_parameter",
                    "type": "invalid_request_error",
                    "debug_headers": {"authorization": secret},
                }
            },
        )

    config.llm.api = "chat_completions"
    config.llm.structured_output = mode
    store = Store(tmp_path / "db")
    client = AsyncOpenAI(
        api_key=secret,
        base_url="https://test.invalid/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    provider = OpenAIProvider(config, store, client)
    try:
        with pytest.raises(ProviderError) as caught:
            await provider.call("choice", {"count": 1}, ChoiceOutput, "r", "choice")
        message = str(caught.value)
        audit = str(store.db.execute("SELECT value FROM events").fetchall())
        assert "HTTP 400" in message
        assert "response_format" in message and "unsupported_parameter" in message
        assert "req-test" in message and "req-test" in audit
        assert "[REDACTED]" in message
        assert secret not in message + audit and "other-secret" not in message + audit
        assert "debug_headers" not in message + audit
        assert caught.value.__suppress_context__
        assert len(calls) == 1 and store.step("r", "choice") is None
    finally:
        await provider.close()
        store.close()


@pytest.mark.parametrize("api", ["responses", "chat_completions"])
async def test_prompt_mode_still_rejects_invalid_schema(config, tmp_path, api):
    config.llm.api = api
    config.llm.structured_output = "prompt"
    body = (
        responses_body('{"wrong_field":true}')
        if api == "responses"
        else {
            "id": "chat_test",
            "object": "chat.completion",
            "created": 1,
            "model": "fixture",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": '{"wrong_field":true}'},
                }
            ],
        }
    )
    store = Store(tmp_path / "db")
    client = AsyncOpenAI(
        api_key="test",
        base_url="https://test.invalid/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body))
        ),
    )
    provider = OpenAIProvider(config, store, client)
    try:
        with pytest.raises(ProviderError, match="schema validation"):
            await provider.call("choice", {}, ChoiceOutput, "r", "choice")
        assert store.step("r", "choice") is None
    finally:
        await provider.close()
        store.close()


def test_environment_protocol_and_yaml_precedence(monkeypatch, tmp_path):
    from pathlib import Path

    from decisionforge.config import LLMConfig, load_config

    monkeypatch.setenv("OPENAI_MODEL", "fixture-model")
    monkeypatch.setenv("OPENAI_API", "chat_completions")
    monkeypatch.setenv("OPENAI_STRUCTURED_OUTPUT", "prompt")
    monkeypatch.setenv("OPENAI_CHAT_TOKEN_LIMIT_PARAM", "max_tokens")
    root = Path(__file__).resolve().parents[1]
    inherited = LLMConfig()
    assert inherited.api == "chat_completions"
    assert inherited.structured_output == "prompt"
    assert inherited.chat_token_limit_param == "max_tokens"
    import yaml

    source = root / "configs"
    value = yaml.safe_load((source / "default.yaml").read_text())
    value["input"]["path"] = str(root / "examples/standard.json")
    value["llm"].update(api="chat_completions", structured_output="json", chat_token_limit_param="max_completion_tokens")
    fixture = tmp_path / "generation.yaml"
    fixture.write_text(yaml.safe_dump(value))
    config = load_config(fixture)
    assert config.llm.api == "chat_completions"
    assert config.llm.structured_output == "json"
    assert config.llm.chat_token_limit_param == "max_completion_tokens"
    assert Path(config.input.path).is_file()
    explicit = LLMConfig(
        api="responses", structured_output="strict", chat_token_limit_param="max_completion_tokens"
    )
    assert explicit.api == "responses" and explicit.structured_output == "strict"
    assert explicit.chat_token_limit_param == "max_completion_tokens"
    monkeypatch.setenv("OPENAI_API", "invalid-api")
    with pytest.raises(ValueError):
        LLMConfig()


def responses_body(text, status="completed"):
    return {
        "id": "resp_test",
        "object": "response",
        "created_at": 1,
        "status": status,
        "model": "fixture-model",
        "output": [
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 100, "output_tokens": 25, "total_tokens": 125},
    }


@pytest.mark.parametrize("api", ["responses", "chat_completions"])
@pytest.mark.parametrize(
    "role,schema,output",
    [
        (
            "text_assessor",
            TextAssessment,
            {
                "is_narrative": True,
                "opportunities": [
                    {
                        "mode": "choice",
                        "focus": "dispatch",
                        "evidence": ["Two units each supply 2 MW."],
                        "reason": "Explicit capacity",
                    }
                ],
                "reason": "Worthwhile decision",
            },
        ),
        (
            "qa_constructor",
            QAConstruction,
            {"qa": {"question": "Which total capacity?", "answer": "4 MW"}, "reason": "2 + 2 = 4"},
        ),
    ],
)
async def test_narrative_schemas_through_sdk_and_checkpoint(
    config, tmp_path, api, role, schema, output
):
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        if api == "responses":
            assert body["text"]["format"]["schema"]["additionalProperties"] is False
            return httpx.Response(200, json=responses_body(json.dumps(output)))
        assert body["response_format"]["json_schema"]["schema"]["additionalProperties"] is False
        return httpx.Response(
            200,
            json={
                "id": "chat_test",
                "object": "chat.completion",
                "created": 1,
                "model": "fixture",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(output),
                            "refusal": None,
                        },
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 25, "total_tokens": 125},
            },
        )

    config.llm.api = api
    config.llm.structured_output = "strict"
    store = Store(tmp_path / "sdk")
    client = AsyncOpenAI(
        api_key="fixture-key",
        base_url="https://test.invalid/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    provider = OpenAIProvider(config, store, client)
    try:
        first = await provider.call(
            role, {"source": "Two units each supply 2 MW."}, schema, "r", role
        )
        again = await provider.call(
            role, {"source": "Two units each supply 2 MW."}, schema, "r", role
        )
        assert first.model_dump() == again.model_dump() == output
        assert len(requests) == 1
        assert store.step("r", role)["output"] == output
    finally:
        await provider.close()
        store.close()


@pytest.mark.parametrize("api", ["responses", "chat_completions"])
async def test_real_sdk_request_format_and_cache(config, tmp_path, api):
    captured = []

    def handler(request):
        body = json.loads(request.content)
        captured.append(body)
        output = '{"distractors":["d1","d2","d3"]}'
        if api == "responses":
            assert request.url.path == "/v1/responses"
            assert body["text"]["format"]["type"] == "json_schema"
            assert body["text"]["format"]["strict"] is True
            return httpx.Response(200, json=responses_body(output))
        assert request.url.path == "/v1/chat/completions"
        assert body["response_format"]["json_schema"]["strict"] is True
        return httpx.Response(
            200,
            json={
                "id": "chat_test",
                "object": "chat.completion",
                "created": 1,
                "model": "fixture-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": output, "refusal": None},
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 25, "total_tokens": 125},
            },
        )

    config.llm.api = api
    config.llm.structured_output = "strict"
    store = Store(tmp_path / "db")
    client = AsyncOpenAI(
        api_key="TEST_KEY_NEVER_LOG",
        base_url="https://test.invalid/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    provider = OpenAIProvider(config, store, client)
    selection = await provider.call("choice", {"count": 3}, ChoiceOutput, "record", "select")
    assert selection.distractors == ["d1", "d2", "d3"]
    await provider.call("choice", {"count": 3}, ChoiceOutput, "record", "select")
    await provider.call("choice", {"count": 3}, ChoiceOutput, "another-record", "select")
    assert len(captured) == 1
    assert "temperature" not in captured[0] and "reasoning" not in captured[0]
    audit = str(store.db.execute("SELECT value FROM events").fetchall())
    assert "TEST_KEY_NEVER_LOG" not in audit
    await provider.close()
    store.close()


@pytest.mark.parametrize(
    "failure", ["rate_limit", "timeout", "invalid_json", "incomplete", "refusal"]
)
async def test_provider_failure_and_retry_contract(config, tmp_path, failure):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if failure == "rate_limit":
            return httpx.Response(
                429,
                json={
                    "error": {"message": "rate limited", "type": "rate_limit", "code": "rate_limit"}
                },
            )
        if failure == "timeout":
            raise httpx.ReadTimeout("timeout", request=request)
        body = responses_body(
            "not valid JSON" if failure == "invalid_json" else '{"distractors":[],"reason":"x"}'
        )
        if failure == "incomplete":
            body["status"] = "incomplete"
            body["incomplete_details"] = {"reason": "max_output_tokens"}
        if failure == "refusal":
            body["output"][0]["content"] = [{"type": "refusal", "refusal": "Cannot comply"}]
        return httpx.Response(200, json=body)

    config.execution.max_transport_attempts = 2
    store = Store(tmp_path / "db")
    client = AsyncOpenAI(
        api_key="test",
        base_url="https://test.invalid/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    provider = OpenAIProvider(config, store, client)
    with pytest.raises(ProviderError):
        await provider.call("choice", {"count": 3}, ChoiceOutput, "record", "select")
    assert calls == (2 if failure in {"timeout", "rate_limit"} else 1)
    assert store.step("record", "select") is None
    await provider.close()
    store.close()


async def test_stage_checkpoint_survives_new_provider_instance(config, tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=responses_body('{"distractors":["d1"]}'))

    config.execution.cache = False
    for _ in range(2):
        store = Store(tmp_path / "persistent")
        client = AsyncOpenAI(
            api_key="test",
            base_url="https://test.invalid/v1",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        provider = OpenAIProvider(config, store, client)
        await provider.call("choice", {"count": 1}, ChoiceOutput, "record", "select")
        await provider.close()
        store.close()
    assert len(calls) == 1


@pytest.mark.parametrize("structured_output", ["strict", "json"])
@pytest.mark.parametrize("reasoning_content", ["null", "omitted"])
@pytest.mark.parametrize("outcome", ["success", "refusal", "incomplete", "reasoning_only", "empty"])
async def test_responses_reasoning_items_allow_absent_content(
    config, tmp_path, structured_output, reasoning_content, outcome
):
    """Use the actual SDK: reasoning.content is optional, even for valid responses."""
    body = responses_body('{"distractors":["d1","d2","d3"]}')
    reasoning = {"id": "rs_test", "type": "reasoning", "summary": []}
    if reasoning_content == "null":
        reasoning["content"] = None
    body["output"].insert(0, reasoning)
    if outcome == "refusal":
        # A refusal must still block acceptance even when valid text is also present.
        body["output"][1]["content"].append({"type": "refusal", "refusal": "Cannot comply"})
    elif outcome == "incomplete":
        body["status"] = "incomplete"
        body["incomplete_details"] = {"reason": "max_output_tokens"}
    elif outcome == "reasoning_only":
        body["output"] = [reasoning]
    elif outcome == "empty":
        body["output"] = []

    config.llm.structured_output = structured_output
    store = Store(tmp_path / "db")
    client = AsyncOpenAI(
        api_key="test",
        base_url="https://test.invalid/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body))
        ),
    )
    provider = OpenAIProvider(config, store, client)
    try:
        if outcome == "success":
            result = await provider.call("choice", {"count": 3}, ChoiceOutput, "record", "select")
            assert result.distractors == ["d1", "d2", "d3"]
            assert store.step("record", "select") is not None
        else:
            message = (
                "no text output"
                if outcome in {"reasoning_only", "empty"}
                else "refusal or incomplete"
            )
            with pytest.raises(ProviderError, match=message):
                await provider.call("choice", {"count": 3}, ChoiceOutput, "record", "select")
            assert store.step("record", "select") is None
    finally:
        await provider.close()
        store.close()
