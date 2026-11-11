from __future__ import annotations

import asyncio
import os
import re
import time
from typing import Protocol, TypeVar

from openai import (
    APIConnectionError,
    APIStatusError,
    AsyncOpenAI,
    ContentFilterFinishReasonError,
    LengthFinishReasonError,
    RateLimitError,
)
from pydantic import BaseModel, ValidationError

from decisionforge.config import LLM_ROLES, Config
from decisionforge.prompts import VERSION, prompt
from decisionforge.storage import Store
from decisionforge.util import ProviderError, digest, json_text

T = TypeVar("T", bound=BaseModel)


class Provider(Protocol):
    async def call(
        self, role: str, payload: dict, schema: type[T], record: str, stage: str
    ) -> T: ...
    async def close(self) -> None: ...


class OpenAIProvider:
    """All production semantic work uses the OpenAI Python SDK, with explicit capabilities."""

    def __init__(self, config: Config, store: Store, client=None):
        self.config, self.store = config, store
        store.put(
            "pricing_complete",
            all(config.llm.model_for(role) in config.llm.pricing for role in LLM_ROLES),
        )
        self._client = client
        self._clients = {}

    @property
    def client(self):
        return self.client_for(self.config.llm)

    def client_for(self, llm):
        # Injected test clients retain the original provider contract.
        if self._client is not None:
            return self._client
        key = os.getenv(llm.api_key_env)
        if not key:
            raise ProviderError(f"Set {llm.api_key_env} in environment or .env")
        endpoint = llm.base_url or os.getenv("OPENAI_BASE_URL")
        identity = (endpoint, key, llm.timeout_seconds)
        if identity not in self._clients:
            self._clients[identity] = AsyncOpenAI(
                api_key=key,
                base_url=endpoint,
                timeout=llm.timeout_seconds,
                max_retries=0,
            )
        return self._clients[identity]

    async def close(self):
        if self._client is not None:
            await self._client.close()
        for client in self._clients.values():
            await client.close()

    def _error_details(self, exc):
        # Only retain useful scalar fields, never a full body, request or headers.
        secrets = (
            os.getenv(self.config.llm.api_key_env),
            getattr(self._client, "api_key", None),
            *(os.getenv(self.config.llm.for_role(role).api_key_env) for role in LLM_ROLES),
            *(getattr(client, "api_key", None) for client in self._clients.values()),
        )

        def clean(value):
            if not isinstance(value, (str, int)) or isinstance(value, bool):
                return None
            text = str(value)
            for secret in secrets:
                if isinstance(secret, str) and secret:
                    text = text.replace(secret, "[REDACTED]")
            text = re.sub(r"(?i)Bearer\s+\S+", "Bearer [REDACTED]", text)
            text = re.sub(r"\bsk-[\w-]+", "[REDACTED]", text)
            text = re.sub(r"https?://\S+", "[URL REDACTED]", text)
            return " ".join(text.split())[:1000]

        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            body = body.get("error", body)
        details = {}
        if isinstance(body, dict):
            for field in ("message", "param", "code", "type"):
                value = clean(body.get(field))
                if value:
                    details["api_error_" + field] = value
        request_id = clean(getattr(exc, "request_id", None))
        if request_id:
            details["request_id"] = request_id
        return details

    async def call(self, role: str, payload: dict, schema: type[T], record: str, stage: str) -> T:
        llm, execution = self.config.llm.for_role(role), self.config.execution
        system = prompt(role)
        payload = {"output_language": self.config.project.language, **payload}
        serialized = json_text(payload)
        if len(serialized) > llm.max_input_chars:
            raise ProviderError("Input context limit exceeded; refusing to silently truncate")
        model = llm.model
        identity = {
            "role": role,
            "model": model,
            "prompt_version": VERSION,
            "prompt": system,
            "schema": schema.model_json_schema(),
            "payload": payload,
            "config": llm.model_dump(mode="json"),
        }
        key = digest(identity)
        previous = self.store.step(record, stage)
        if previous is not None:
            if previous["key"] != key:
                raise ProviderError(f"Checkpoint identity mismatch at {stage}")
            return schema.model_validate(previous["output"])
        cached = self.store.get("llm:" + key) if execution.cache else None
        if cached is not None:
            result = schema.model_validate(cached)
            self.store.save_step(record, stage, {"key": key, "output": cached})
            self.store.event(record, stage, {"role": role, "cache_hit": True, "input_hash": key})
            return result

        messages = [{"role": "system", "content": system}, {"role": "user", "content": serialized}]
        if llm.structured_output != "strict":
            messages[0]["content"] += "\nJSON schema: " + json_text(schema.model_json_schema())
        params = {"model": model}
        if llm.temperature is not None:
            params["temperature"] = llm.temperature
        if llm.reasoning_effort is not None:
            if llm.api == "responses":
                params["reasoning"] = {"effort": llm.reasoning_effort}
            else:
                params["reasoning_effort"] = llm.reasoning_effort
        rate = llm.pricing.get(model)
        # UTF-8 byte count + schema/envelope margin is a conservative local upper estimate.
        input_bound = (
            len(json_text(messages).encode())
            + len(json_text(schema.model_json_schema()).encode())
            + 4096
        )
        reserve = (
            (
                (
                    input_bound * rate.input_per_million
                    + llm.max_output_tokens * rate.output_per_million
                )
                / 1e6
            )
            if rate
            else 0
        )
        for attempt in range(1, execution.max_transport_attempts + 1):
            # Validate credentials before recording any reservation or attempted API charge.
            client = self.client_for(llm)
            charge = self.store.reserve(
                record,
                stage,
                reserve,
                execution.budget_limit_usd,
                execution.max_llm_calls_per_record,
            )
            started = time.monotonic()
            base_audit = {
                "role": role,
                "model": model,
                "prompt_version": VERSION,
                "schema_name": schema.__name__,
                "schema_hash": digest(schema.model_json_schema()),
                "input_hash": key,
                "attempt": attempt,
                "sampling": params,
                "api": llm.api,
                "structured_output": llm.structured_output,
                "max_output_tokens": llm.max_output_tokens,
                "evidence_ids": [e.get("evidence_id") for e in payload.get("evidence", [])],
            }
            try:
                if llm.api == "responses":
                    if llm.structured_output == "strict":
                        response = await client.responses.parse(
                            **params,
                            input=messages,
                            text_format=schema,
                            max_output_tokens=llm.max_output_tokens,
                            store=False,
                        )
                    else:
                        response = await client.responses.create(
                            **params,
                            input=messages,
                            **(
                                {"text": {"format": {"type": "json_object"}}}
                                if llm.structured_output == "json"
                                else {}
                            ),
                            max_output_tokens=llm.max_output_tokens,
                            store=False,
                        )
                    usage = response.usage
                    input_tokens = usage.input_tokens if usage else 0
                    output_tokens = usage.output_tokens if usage else 0
                    raw = response.output_text
                    refused = any(
                        getattr(content, "type", None) == "refusal"
                        for item in response.output
                        # Reasoning/tool items are not messages and may have content=None.
                        if getattr(item, "type", None) == "message"
                        for content in (getattr(item, "content", None) or [])
                    )
                    valid_finish = response.status == "completed" and not refused
                else:
                    token_limit = {llm.chat_token_limit_param: llm.max_output_tokens}
                    if llm.structured_output == "strict":
                        response = await client.chat.completions.parse(
                            **params,
                            messages=messages,
                            response_format=schema,
                            **token_limit,
                        )
                    else:
                        response = await client.chat.completions.create(
                            **params,
                            messages=messages,
                            **(
                                {"response_format": {"type": "json_object"}}
                                if llm.structured_output == "json"
                                else {}
                            ),
                            **token_limit,
                        )
                    usage = response.usage
                    input_tokens = usage.prompt_tokens if usage else 0
                    output_tokens = usage.completion_tokens if usage else 0
                    raw = response.choices[0].message.content or ""
                    valid_finish = (
                        response.choices[0].finish_reason == "stop"
                        and not response.choices[0].message.refusal
                    )
                actual = (
                    (
                        input_tokens * rate.input_per_million
                        + output_tokens * rate.output_per_million
                    )
                    / 1e6
                    if rate
                    else 0
                )
                if usage is not None:
                    self.store.settle(charge, actual)
                self.store.event(
                    record,
                    stage,
                    {
                        **base_audit,
                        "raw_output": raw,
                        "elapsed_seconds": time.monotonic() - started,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "cost_usd": actual if rate else None,
                        "response_id": response.id,
                        "completed": valid_finish,
                    },
                )
                if not valid_finish:
                    raise ProviderError(
                        "Model refusal or incomplete output; no training label accepted"
                    )
                if not raw.strip():
                    self.store.event(
                        record, stage, {"schema_valid": False, "error": "empty_output"}
                    )
                    raise ProviderError(
                        "Model returned no text output; expected a JSON object for "
                        + schema.__name__
                    )
                result = schema.model_validate_json(raw)
                output = result.model_dump(mode="json")
                self.store.save_step(record, stage, {"key": key, "output": output})
                if execution.cache:
                    self.store.put("llm:" + key, output)
                self.store.event(record, stage, {"schema_valid": True})
                return result
            except (APIConnectionError, APIStatusError) as exc:
                retriable = isinstance(exc, (APIConnectionError, RateLimitError)) or (
                    isinstance(exc, APIStatusError)
                    and (exc.status_code >= 500 or exc.status_code in {408, 409})
                )
                details = self._error_details(exc)
                self.store.event(
                    record,
                    stage,
                    {
                        **base_audit,
                        "error": type(exc).__name__,
                        "status_code": getattr(exc, "status_code", None),
                        **details,
                    },
                )
                if not retriable or attempt == execution.max_transport_attempts:
                    description = "; ".join(f"{k}={v}" for k, v in details.items())
                    status = getattr(exc, "status_code", None)
                    raise ProviderError(
                        f"OpenAI transport failed: {type(exc).__name__} "
                        f"(HTTP {status}, api={llm.api}, "
                        f"structured_output={llm.structured_output})"
                        + (f": {description}" if description else "")
                    ) from None
                await asyncio.sleep(min(2 ** (attempt - 1), 8))
            except ValidationError as exc:
                self.store.event(
                    record,
                    stage,
                    {
                        **base_audit,
                        "schema_valid": False,
                        "validation_errors": exc.errors(
                            include_input=False, include_url=False, include_context=False
                        ),
                    },
                )
                raise ProviderError("Model response failed strict schema validation") from exc
            except (LengthFinishReasonError, ContentFilterFinishReasonError) as exc:
                self.store.event(
                    record,
                    stage,
                    {**base_audit, "schema_valid": False, "error": type(exc).__name__},
                )
                raise ProviderError("Model output was truncated or filtered") from exc
        raise ProviderError("Transport retry loop exhausted")
