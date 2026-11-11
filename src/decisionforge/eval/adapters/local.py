"""Persistent single-GPU worker; parent-side journaling never holds model/GPU dependencies."""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import time

from decisionforge.eval.adapter import parse_answer, usage_from
from decisionforge.eval.adapters.local_worker import worker_main


class LocalAdapter:
    def __init__(self, model, execution, *, worker_target=worker_main):
        self.model, self.execution = model, execution
        self.worker_target = worker_target
        self.process = self.connection = None
        self._closing = None
        self.metadata = {}
        self.lock = asyncio.Lock()

    def redact(self, value):
        return value

    def response_answer(self, raw, question):
        return parse_answer(raw, question)

    async def _receive(self, timeout):
        # Read an entire frame off the event loop: poll() only promises that some
        # bytes arrived, and recv() can still block on a partial frame.
        return await asyncio.wait_for(asyncio.to_thread(self.connection.recv), timeout)

    async def start(self):
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(
            target=self.worker_target, args=(child, self.model.model_dump())
        )
        try:
            try:
                self.process.start()
            finally:
                child.close()
            message = await self._receive(self.model.load_timeout)
            if message.get("kind") != "ready":
                raise ValueError(f"Local model initialization failed: {message.get('error')}")
            self.metadata = message["metadata"]
        except BaseException as exc:
            await self.close(force=True)
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise ValueError(f"Local model could not start: {exc}") from exc
        return self

    async def send(self, payload, question):
        if not getattr(self, "supports_media", False):
            from decisionforge.media import reject_media

            reject_media(payload, self.model.adapter)
        result = {
            "transport_ok": False,
            "protocol_ok": False,
            "answer": None,
            "raw_response": None,
            "http_status": None,
            "usage": None,
            "model": self.metadata.get("model", self.model.model),
            "model_identity_source": getattr(self, "model_identity_source", "loaded_checkpoint"),
            "probability_source": getattr(
                self,
                "result_probability_source",
                "derived_from_local_noul"
                if question.type == "noul"
                else "local_calibrated_softmax",
            ),
            "error": None,
            "retryable": False,
            "retry_after_seconds": None,
            "cost_usd": None,
            "cost_basis": "local_compute_unpriced",
            "fatal": False,
        }
        started = time.monotonic()
        async with self.lock:
            try:
                # The deadline includes writing a request into a full pipe as
                # well as waiting for the complete response. Killing the worker
                # on timeout/cancellation releases any blocked pipe thread.
                async with asyncio.timeout(self.execution.timeout):
                    await asyncio.to_thread(
                        self.connection.send,
                        {
                            "kind": "predict",
                            "request": {k: payload[k] for k in ("state", "questions", "images", "videos") if k in payload},
                        },
                    )
                    message = await asyncio.to_thread(self.connection.recv)
                if message.get("kind") == "error":
                    result.update(error=message["error"], fatal=message.get("fatal", True))
                elif message.get("kind") == "prediction":
                    raw = message["response"]
                    # Keep illegal result evidence even when strict JSON validation fails.
                    from decisionforge.training_free.anyjev.processing import json_safe

                    result["raw_response"] = json_safe(raw)
                    json.dumps(raw, allow_nan=False)
                    result.update(transport_ok=True, raw_response=raw, usage=usage_from(raw))
                    if message.get("runtime") is not None:
                        result["local_runtime"] = message["runtime"]
                    result["answer"] = self.response_answer(raw, question)
                    result["protocol_ok"] = True
                else:
                    raise ValueError("Unexpected local worker response")
            except asyncio.CancelledError:
                await self.close(force=True)
                raise
            except TimeoutError:
                result.update(
                    error={
                        "kind": "local_timeout",
                        "message": f"Local model request exceeded {self.execution.timeout}s",
                    },
                    fatal=True,
                )
            except (EOFError, OSError, BrokenPipeError) as exc:
                result.update(
                    error={"kind": "local_worker_failed", "message": str(exc)}, fatal=True
                )
            except (ValueError, TypeError, KeyError) as exc:
                result.update(error={"kind": "protocol", "message": str(exc)})
            finally:
                result["attempt_latency_ms"] = (time.monotonic() - started) * 1000
            if result["fatal"]:
                await self.close(force=True)
        return result

    async def close(self, *, force=False):
        process = self.process
        if self._closing is None:
            self.process = None
            connection, self.connection = self.connection, None
            self._closing = asyncio.create_task(self._close_resources(process, connection, force))
        cancelled = False
        while not self._closing.done():
            try:
                await asyncio.shield(self._closing)
            except asyncio.CancelledError:
                cancelled = True
                if process is not None and process.pid is not None and process.is_alive():
                    process.terminate()
        self._closing.result()
        if cancelled:
            raise asyncio.CancelledError()

    @staticmethod
    async def _close_resources(process, connection, force):
        try:
            if process is not None and process.pid is not None:
                if process.is_alive():
                    if force:
                        process.terminate()
                    elif connection is not None:
                        try:
                            await asyncio.wait_for(
                                asyncio.to_thread(connection.send, {"kind": "close"}), 2
                            )
                        except (OSError, EOFError, TimeoutError):
                            if process.is_alive():
                                process.terminate()
                await asyncio.to_thread(process.join, 5)
                if process.is_alive():
                    process.terminate()
                    await asyncio.to_thread(process.join, 5)
                if process.is_alive():
                    process.kill()
                    await asyncio.to_thread(process.join, 5)
        finally:
            if connection is not None:
                connection.close()
