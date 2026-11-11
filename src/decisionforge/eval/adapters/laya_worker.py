"""Official Laya loader/readout in an isolated process; CPU-safe module import."""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import sys
import tempfile
import time
from importlib.metadata import distribution
from pathlib import Path

from decisionforge.eval.adapters.local_worker import LocalInputError, safe_error
from decisionforge.eval.models import Request
from decisionforge.training.backends.laya import (
    UPSTREAM_REVISION,
    artifact_identity,
    dependency_versions,
    input_loss,
    stage_source,
    validate_artifact,
)

RENDERING = {
    "version": "decisionforge.laya.rendering.v1",
    "absent_instructions": "JSON_literal_text_null",
    "empty_instructions": "JSON_literal_text_of_empty_value",
    "structured_fields": "upstream_lossless_JSON_serialization",
}


def render_question(question):
    # Upstream rejects absent/empty instructions. Encoding the absent JSON value
    # as its literal text preserves absence without inventing a judgment task.
    instructions = question.get("instructions")
    if not isinstance(instructions, str) or not instructions.strip():
        instructions = json.dumps(instructions, ensure_ascii=False)
    return {**question, "instructions": instructions}


def predict_public(agent, public, *, input_check=None):
    request = Request.model_validate(public)
    try:
        if input_check is not None:
            detail = input_check(request)
            if detail:
                raise LocalInputError(f"Laya public input would be truncated: {detail}")
        response = agent.system_one(
            request.state,
            {
                qid: render_question(q.model_dump(exclude_none=True))
                for qid, q in request.questions.items()
            },
        )
    except (ValueError, TypeError, KeyError) as exc:
        raise LocalInputError(str(exc)) from exc
    usage = response.get("usage", {})
    if usage.get("truncated"):
        raise LocalInputError(
            "Laya public state was truncated; increase checkpoint sequence budgets"
        )
    if usage.get("options"):
        raise LocalInputError("Laya public options collapsed inside the checkpoint head budget")
    # Preserve native rounded probabilities/confidence. No normalization or custom formula.
    return response


def agent_input_loss(agent, request):
    api = importlib.import_module("laya.train")
    common = importlib.import_module("laya.common")
    return input_loss(
        agent.tok,
        request.state,
        "decision",
        render_question(request.questions["decision"].model_dump(exclude_none=True)),
        api,
        common,
        agent.cfg.get("max_len", 512),
        agent.cfg.get("head_max_len", 192),
    )


class LayaRuntime:
    def __init__(self, options):
        source = validate_artifact(options["path"])
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError(
                "Laya local evaluation requires a single owned worker without torchrun"
            )
        try:
            self.torch = importlib.import_module("torch")
            laya = importlib.import_module("laya")
        except ImportError as exc:
            raise ValueError("Missing Laya dependencies; run uv sync --extra train-laya") from exc
        direct = json.loads(distribution("laya").read_text("direct_url.json") or "{}")
        if direct.get("vcs_info", {}).get("commit_id") != UPSTREAM_REVISION:
            raise ValueError("Install uv sync --extra train-laya for the pinned upstream Laya loader")
        if not self.torch.cuda.is_available():
            raise ValueError("Laya local inference requires CUDA; score-only works on CPU")
        device = self.torch.device(options["device"])
        self.torch.cuda.set_device(device)
        if options["precision"] == "fp16":
            os.environ["LAYA_CUDA_AMP"] = "fp16"
        self.staging = tempfile.TemporaryDirectory(prefix="dforge-laya-eval-")
        staged = stage_source(source, Path(self.staging.name) / "source")
        started = time.monotonic()
        try:
            self.agent = laya.load(
                str(staged), device=str(device), backend="eager", fast=False, compile=False
            )
        except BaseException:
            self.staging.cleanup()
            raise
        self.metadata = {
            "model": str(source),
            "model_identity_source": "loaded_checkpoint",
            "backbone": artifact_identity(source),
            "dependencies": dependency_versions(),
            "method": {
                "backend": "laya",
                "upstream_revision": UPSTREAM_REVISION,
                "loader": "laya.load",
                "readout": "Agent.system_one",
                "rounding_decimals": 4,
                "rendering": RENDERING,
            },
            "calibration": {
                "temperature": self.agent.temperature,
                "temperature_by_options": self.agent.temperature_by_options,
            },
            "runtime": {
                "precision": str(self.agent.dtype),
                "amp_enabled": self.agent.amp_enabled,
                "device": str(self.agent.device),
                "gpu": self.torch.cuda.get_device_name(device),
                "cuda": self.torch.version.cuda,
                "backend": self.agent.backend,
                "model_load_seconds": time.monotonic() - started,
            },
        }

    def predict(self, public):
        before = self.agent.cpu_fallback_count
        response = predict_public(
            self.agent, public, input_check=lambda request: agent_input_loss(self.agent, request)
        )
        fallbacks = self.agent.cpu_fallback_count - before
        self.last_runtime = {
            "device": "cpu" if fallbacks else str(self.agent.device),
            "precision": "torch.float32" if fallbacks else str(self.agent.dtype),
            "cpu_fallbacks": fallbacks,
            "fallback_reason": self.agent.last_fallback_reason if fallbacks else None,
            "post_request_device": str(self.agent.device),
        }
        return response

    def close(self):
        if hasattr(self, "agent"):
            self.agent.__exit__(None, None, None)
        self.staging.cleanup()


def worker_main(connection, options):
    runtime = None
    try:
        with contextlib.redirect_stdout(sys.stderr):
            runtime = LayaRuntime(options)
        connection.send({"kind": "ready", "metadata": runtime.metadata})
        while True:
            command = connection.recv()
            if command["kind"] == "close":
                break
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    response = runtime.predict(command["request"])
                connection.send(
                    {"kind": "prediction", "response": response, "runtime": runtime.last_runtime}
                )
            except (LocalInputError, ValueError, TypeError) as exc:
                connection.send({"kind": "error", "error": safe_error(exc), "fatal": False})
            except Exception as exc:
                connection.send({"kind": "error", "error": safe_error(exc), "fatal": True})
                break
    except EOFError:
        pass
    except BaseException as exc:
        try:
            connection.send({"kind": "error", "error": safe_error(exc), "fatal": True})
        except (OSError, EOFError):
            pass
    finally:
        if runtime is not None:
            runtime.close()
        connection.close()
