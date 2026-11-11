"""Per-module endpoint/protocol routing, inheritance, credentials and persistence."""

import json

import httpx
import pytest
from openai import AsyncOpenAI
from test_openai_provider import responses_body

from decisionforge.config import Config, LLMConfig
from decisionforge.generation.contracts import ChoiceOutput
from decisionforge.llm import OpenAIProvider
from decisionforge.storage import Store


def test_fieldwise_inheritance_and_legacy_groups():
    config = LLMConfig(
        model="global",
        generator_model="legacy",
        base_url="https://global.invalid/v1",
        modules={
            "generator": {"api_key_env": "GEN_KEY"},
            "choice": {"model": "choice-expert"},
            "checker": {"base_url": "https://verify.invalid/v1", "api_key_env": "VERIFY_KEY"},
        },
    )
    choice = config.for_role("choice")
    assert (choice.model, choice.api_key_env, choice.base_url) == (
        "choice-expert",
        "GEN_KEY",
        "https://global.invalid/v1",
    )
    assert config.model_for("score") == "legacy"
    checker = config.for_role("checker")
    assert checker.model == "global" and checker.api_key_env == "VERIFY_KEY"
    assert config.model_for("analyzer") == "global"
    assert config.modules["choice"].base_url is None


@pytest.mark.parametrize(
    "modules",
    [
        {"typo": {"model": "x"}},
        {"choice": {"base_url": "https://user:password@api.invalid"}},
        {"choice": {"temperature": 0.5}},
        {"checker": {"api_key": "never-persist-this"}},
    ],
)
def test_invalid_overrides_rejected(modules):
    with pytest.raises(ValueError):
        LLMConfig(modules=modules)


def test_budget_requires_prices_for_overridden_roles():
    with pytest.raises(ValueError, match="pricing"):
        Config(
            input={"path": "a.txt"},
            execution={"budget_limit_usd": 1},
            llm={
                "model": "common",
                "modules": {"choice": {"model": "special"}},
                "pricing": {"common": {"input_per_million": 1, "output_per_million": 1}},
            },
        )


async def test_real_sdk_routes_concurrent_roles_and_closes_clients(config, tmp_path, monkeypatch):
    calls, clients = [], []
    monkeypatch.setenv("OPENAI_API_KEY", "global-test-key")
    monkeypatch.setenv("SPECIAL_KEY", "special-test-key")
    config.llm.base_url = "https://global.invalid/v1"
    config.llm = LLMConfig.model_validate(
        {
            **config.llm.model_dump(),
            "modules": {
                "choice": {
                    "base_url": "https://special.invalid/v2",
                    "api_key_env": "SPECIAL_KEY",
                    "model": "choice-expert",
                    "api": "chat_completions",
                    "chat_token_limit_param": "max_tokens",
                },
            },
        }
    )

    def handler(request):
        body = json.loads(request.content)
        calls.append((request.url.host, request.headers["authorization"], body))
        output = '{"distractors":["wrong"]}'
        if request.url.host == "special.invalid":
            assert request.url.path == "/v2/chat/completions"
            assert body["model"] == "choice-expert" and "max_tokens" in body
            response = {
                "id": "chat",
                "object": "chat.completion",
                "created": 1,
                "model": "choice-expert",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": output},
                    }
                ],
            }
        else:
            assert request.url.path == "/v1/responses"
            response = responses_body(output)
        return httpx.Response(200, json=response)

    def factory(**kwargs):
        client = AsyncOpenAI(
            **kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        )
        clients.append(client)
        return client

    monkeypatch.setattr("decisionforge.llm.AsyncOpenAI", factory)
    store = Store(tmp_path / "routing")
    provider = OpenAIProvider(config, store)
    try:
        import asyncio

        await asyncio.gather(
            provider.call("choice", {}, ChoiceOutput, "r1", "choice"),
            provider.call("context", {}, ChoiceOutput, "r2", "context"),
        )
        assert {c[:2] for c in calls} == {
            ("special.invalid", "Bearer special-test-key"),
            ("global.invalid", "Bearer global-test-key"),
        }
        # Same model/payload, changed endpoint cannot reuse a saved checkpoint.
        config.llm.modules["choice"].base_url = "https://changed.invalid/v2"
        with pytest.raises(Exception, match="identity mismatch"):
            await provider.call("choice", {}, ChoiceOutput, "r1", "choice")
        assert len(clients) == 2
        assert "special-test-key" not in config.model_dump_json()
    finally:
        await provider.close()
        store.close()
    assert all(client.is_closed() for client in clients)
