from __future__ import annotations

import math
import os
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx

from decisionforge.eval.data import parse_json
from decisionforge.eval.numbers import number
from decisionforge.eval.provenance import candidate_task_count, method_metadata


def parse_answer(raw, question):
    if not isinstance(raw, dict) or not isinstance(raw.get("answers"), dict):
        raise ValueError("Missing answers object")
    answer = raw["answers"].get("decision")
    if not isinstance(answer, dict) or answer.get("type") != question.type:
        raise ValueError("Missing or mismatched decision answer")
    if question.type == "noul":
        if not number(answer.get("noul"), 0, 1):
            raise ValueError("Invalid native Noul probability")
    else:
        if not number(answer.get("confidence"), 0, 1):
            raise ValueError("Invalid native confidence")
        if not isinstance(answer.get("probabilities"), dict):
            raise ValueError("Missing native probabilities")
        if question.type == "choice":
            if answer.get("choice") not in question.labels:
                raise ValueError("Invalid native Choice label")
        else:
            if not number(answer.get("score"), 0, len(question.labels) - 1):
                raise ValueError("Invalid native Score value")
            expected = {str(i): v for i, v in enumerate(question.criteria)}
            if answer.get("legend") != expected:
                raise ValueError("Native Score legend differs from rubric")
    return answer


def usage_from(raw):
    usage = raw.get("usage") if isinstance(raw, dict) else None
    if not isinstance(usage, dict):
        return None
    if any(
        type(usage.get(k)) is not int or usage[k] < 0 for k in ("input_tokens", "output_tokens")
    ):
        return None
    return {k: usage[k] for k in ("input_tokens", "output_tokens")}


def retry_after(value):
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=UTC)
            seconds = (date - datetime.now(UTC)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    return max(0, seconds) if math.isfinite(seconds) else None


class JevAdapter:
    """Native transport; accepts public requests, never evaluation references."""

    def __init__(self, model, execution, transport=None):
        self.model, self.execution = model, execution
        self.secret = os.getenv(model.api_key_env) if model.api_key_env else None
        if model.api_key_env and not self.secret:
            raise ValueError(f"Missing API key environment variable: {model.api_key_env}")
        headers = {"Authorization": f"Bearer {self.secret}"} if self.secret else {}
        self.client = httpx.AsyncClient(
            timeout=execution.timeout,
            headers=headers,
            transport=transport,
            follow_redirects=False,
        )

    def redact(self, value):
        if isinstance(value, str):
            return value.replace(self.secret, "[REDACTED]") if self.secret else value
        if isinstance(value, dict):
            return {self.redact(k): self.redact(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.redact(v) for v in value]
        return value

    def prepare_request(self, payload, question):
        from decisionforge.media import reject_media

        reject_media(payload, self.model.adapter)
        return payload

    def response_usage(self, raw):
        return usage_from(raw)

    def response_answer(self, raw, question, result):
        return parse_answer(raw, question)

    def probability_source(self, question):
        if method_metadata(self.model)["method"] == "llm2jev":
            return "llm2jev_yes_no_logits"
        return "derived_from_native_noul" if question.type == "noul" else "native"

    async def send(self, payload, question):
        from decisionforge.media import reject_media

        reject_media(payload, self.model.adapter)
        start = time.monotonic()
        result = {
            "transport_ok": False,
            "protocol_ok": False,
            "answer": None,
            "raw_response": None,
            "http_status": None,
            "usage": None,
            "model": self.model.model,
            "model_identity_source": "configured_unresolved",
            "probability_source": self.probability_source(question),
            "error": None,
            "retryable": False,
            "retry_after_seconds": None,
            "cost_usd": None,
            "cost_basis": "unknown",
        }
        if method_metadata(self.model)["method"] == "llm2jev":
            result["candidate_task_count"] = candidate_task_count(question)
            result["candidate_task_count_basis"] = "pinned_upstream_compiler"
        try:
            response = await self.client.post(
                f"{self.model.base_url}{self.model.endpoint_path}",
                json=payload,
            )
            result["attempt_latency_ms"] = (time.monotonic() - start) * 1000
            result["http_status"] = response.status_code
            result["retry_after_seconds"] = retry_after(response.headers.get("Retry-After"))
            try:
                raw = parse_json(response.text)
                # JSON nonfinite values must never reach checkpoint serialization.
                import json

                json.dumps(raw, allow_nan=False)
            except ValueError:
                raw = None
            result["raw_response"] = raw if raw is not None else response.text
            result["usage"] = self.response_usage(raw)
            raw_usage = raw.get("usage") if isinstance(raw, dict) else None
            cost = raw_usage.get("cost") if isinstance(raw_usage, dict) else None
            if number(cost, 0, float("inf")):
                result.update(cost_usd=cost, cost_basis="reported_usage_cost")
            if isinstance(raw, dict) and isinstance(raw.get("model"), str):
                result.update(model=raw["model"], model_identity_source="response")
            if 200 <= response.status_code < 300:
                result["transport_ok"] = True
                try:
                    result["answer"] = self.response_answer(raw, question, result)
                    result["protocol_ok"] = True
                except (ValueError, TypeError) as exc:
                    result["error"] = {"kind": "protocol", "message": str(exc)}
            else:
                result["retryable"] = (
                    response.status_code in {429, 529} or 500 <= response.status_code < 600
                )
                result["error"] = {"kind": "http", "message": f"HTTP {response.status_code}"}
        except httpx.TransportError as exc:
            result["attempt_latency_ms"] = (time.monotonic() - start) * 1000
            result["retryable"] = isinstance(exc, (httpx.NetworkError, httpx.TimeoutException))
            result["error"] = {"kind": type(exc).__name__, "message": str(exc)}
        return self.redact(result)

    async def close(self):
        await self.client.aclose()


async def create_adapter(model, execution, transport=None):
    if model.adapter in {"swift_local", "swift_chat_local"}:
        from decisionforge.eval.adapters.swift import SwiftAdapter

        return await SwiftAdapter(model, execution).start()
    if model.adapter == "chat_api":
        from decisionforge.eval.adapters.chat import ChatAdapter

        return ChatAdapter(model, execution, transport)
    if model.adapter == "jev":
        return JevAdapter(model, execution, transport)
    if transport is not None:
        raise ValueError("HTTP transport is not supported for local model evaluation")
    if model.adapter == "anyjev_local":
        from decisionforge.eval.adapters.anyjev import AnyJevAdapter

        return await AnyJevAdapter(model, execution).start()
    if model.adapter == "openjev_local":
        from decisionforge.eval.adapters.openjev import OpenJevAdapter

        return await OpenJevAdapter(model, execution).start()
    if model.adapter == "laya_local":
        from decisionforge.eval.adapters.laya import LayaAdapter

        return await LayaAdapter(model, execution).start()
    if model.adapter != "unsloth_local":
        raise ValueError(f"Unsupported local adapter: {model.adapter}")
    from decisionforge.eval.adapters.local import LocalAdapter

    return await LocalAdapter(model, execution).start()
