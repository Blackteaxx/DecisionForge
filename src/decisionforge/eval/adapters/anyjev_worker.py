"""Isolated AnyJev HF runtime; module imports never load NumPy/Torch/Transformers."""

from __future__ import annotations

import contextlib
import importlib
import os
import sys
import time
from pathlib import Path

from decisionforge.eval.adapters.local_worker import LocalInputError, safe_error
from decisionforge.eval.models import AnyJevModelConfig
from decisionforge.training_free.anyjev import (
    SETTINGS,
    dependency_versions,
    method_identity,
    validate_installation,
)
from decisionforge.training_free.anyjev.bridge import make_decider, make_question
from decisionforge.training_free.anyjev.processing import (
    json_safe,
    prepare_request,
    response_from_decision,
    validate_request,
)
from decisionforge.training_free.anyjev.tokenization import cached_tokenization


class BoundedBackend:
    """Check complete upstream prompts, including probes and every shared suffix."""

    def __init__(self, backend, limit):
        self.backend, self.limit = backend, limit
        self.tokenizer, self.name = backend.tokenizer, backend.name

    def check(self, prompts):
        if any(
            len(self.tokenizer.encode(p, add_special_tokens=False)) > self.limit for p in prompts
        ):
            raise LocalInputError(
                "Public request exceeds max_seq_length; refusing to truncate AnyJev prompt"
            )

    def next_token_logprobs(self, prompts, token_ids):
        self.check(prompts)
        return self.backend.next_token_logprobs(prompts, token_ids)

    def score_shared(self, groups, token_ids):
        self.check([prefix + suffix for prefix, suffixes in groups for suffix in suffixes])
        return self.backend.score_shared(groups, token_ids)


class AnyJevRuntime:
    def __init__(self, options):
        config = AnyJevModelConfig.model_validate(options)
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError("AnyJev local evaluation supports one CUDA device without torchrun")
        try:
            source = validate_installation()
            torch = importlib.import_module("torch")
            anyjev = importlib.import_module("anyjev")
            hf = importlib.import_module("anyjev.backends.hf")
        except ImportError as exc:
            raise ValueError(
                "Missing AnyJev dependencies; run uv sync --extra eval-anyjev in a dedicated environment"
            ) from exc
        if not torch.cuda.is_available():
            raise ValueError(
                "AnyJev inference requires CUDA; static validation and score-only work on CPU"
            )
        device = torch.device(config.device)
        index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_device(index)
        if config.precision == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise ValueError("Requested bfloat16 is unsupported by this CUDA device")
        path = Path(config.model)
        if path.is_dir() and any(
            (path / n).exists() for n in ("adapter_config.json", "unsloth_decision_config.json")
        ):
            raise ValueError(
                "AnyJev training-free requires an ordinary backbone, not trained decision artifacts"
            )
        started = time.monotonic()
        self.backend = hf.HFBackend(
            config.model,
            device=config.device,
            dtype=config.precision,
            batch_size=config.batch_size,
            revision=config.revision,
            trust_remote_code=False,
        )
        devices = {str(p.device) for p in self.backend.model.parameters()}
        if devices != {f"cuda:{index}"}:
            raise ValueError(
                f"AnyJev requires single-device placement; loaded devices: {sorted(devices)}"
            )
        actual_dtype = str(self.backend.model.dtype)
        if actual_dtype != f"torch.{config.precision}":
            raise ValueError(f"AnyJev loaded precision differs from request: {actual_dtype}")
        limit = min(
            config.max_seq_length,
            getattr(self.backend.model.config, "max_position_embeddings", config.max_seq_length),
        )
        self.decider = make_decider(
            BoundedBackend(self.backend, limit),
            **{**SETTINGS, "shared_prefix": config.shared_prefix},
        )
        self.question_type = anyjev.Question
        self.options = config.model_dump()
        model_config = self.backend.model.config
        self.metadata = {
            "model": config.model,
            "model_identity_source": "loaded_backbone",
            "backbone": {
                "name": getattr(model_config, "_name_or_path", config.model),
                "revision": getattr(model_config, "_commit_hash", None) or config.revision,
            },
            "tokenizer": {
                "name": getattr(self.backend.tokenizer, "name_or_path", config.model),
                "revision": self.backend.tokenizer.init_kwargs.get("_commit_hash", config.revision),
            },
            "method": method_identity(shared_prefix=config.shared_prefix),
            "dependencies": dependency_versions(),
            "upstream_installation": source,
            "runtime": {
                "precision": actual_dtype,
                "device": f"cuda:{index}",
                "gpu": torch.cuda.get_device_name(index),
                "cuda": torch.version.cuda,
                "model_load_seconds": time.monotonic() - started,
                "max_seq_length": limit,
            },
        }
        if config.revision and self.metadata["backbone"]["revision"] != config.revision:
            raise ValueError("Loaded AnyJev backbone revision differs from pinned revision")

    def predict(self, public):
        try:
            request = validate_request(public)
            converted = prepare_request(request)
        except (ValueError, TypeError) as exc:
            raise LocalInputError(str(exc)) from exc
        kind = converted["kind"]
        kwargs = {"text": converted["text"], "name": "decision"}
        if kind == "choice":
            kwargs["options"] = converted["options"]
        elif kind == "score":
            kwargs["levels"] = converted["options"]
        question = make_question(self.question_type, kind, **kwargs)
        before = dict(self.decider.stats)
        fallback_before = self.backend.shared_fallbacks
        with cached_tokenization(
            self.backend, self.decider.backend, enabled=kind == "choice" and question.k > 26
        ):
            decision = self.decider.decide(converted["state"], [question], require="L0")["decision"]
        response = response_from_decision(request.questions["decision"], decision)
        response.update(model=self.metadata["model"])
        response["anyjev"].update(
            method=json_safe(method_identity(shared_prefix=self.decider.shared_prefix)),
            mapping=converted,
            workload={
                **{key: value - before.get(key, 0) for key, value in self.decider.stats.items()},
                "shared_fallbacks": self.backend.shared_fallbacks - fallback_before,
                "basis": "upstream_decider_counters_not_gpu_forwards",
            },
        )
        protocol = (
            getattr(self.decider, "label_protocols", {}).get(question.k)
            if kind == "choice"
            else None
        )
        if protocol is not None:
            response["anyjev"]["label_protocol"] = protocol
        return response


def worker_main(connection, options):
    try:
        with contextlib.redirect_stdout(sys.stderr):
            runtime = AnyJevRuntime(options)
        connection.send({"kind": "ready", "metadata": runtime.metadata})
        while True:
            command = connection.recv()
            if command["kind"] == "close":
                break
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    response = runtime.predict(command["request"])
                connection.send({"kind": "prediction", "response": response})
            except LocalInputError as exc:
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
        connection.close()
