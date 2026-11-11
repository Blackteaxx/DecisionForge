"""Isolated CUDA inference. Importing this module is safe in a CPU-only installation."""

from __future__ import annotations

import contextlib
import importlib
import os
import platform
import sys
import time
from pathlib import Path

from decisionforge.eval.metrics import distribution


class LocalInputError(ValueError):
    """A single request cannot be encoded without dropping public input."""


def answer_from_probabilities(question, probabilities):
    kind = question["type"]
    labels = (
        list(question["criteria"])
        if kind == "choice"
        else ["false", "true"]
        if kind == "noul"
        else [str(i) for i in range(len(question["criteria"]))]
    )
    p = distribution(probabilities, labels)
    if kind == "noul":
        return {"type": kind, "noul": p["true"], "probabilities": p}
    answer = {"type": kind, "confidence": max(p.values()), "probabilities": p}
    if kind == "choice":
        answer["choice"] = max(labels, key=p.__getitem__)
    else:
        answer.update(
            score=sum(i * p[str(i)] for i in range(len(labels))),
            legend={str(i): description for i, description in enumerate(question["criteria"])},
        )
    return answer


def safe_error(exc):
    message = str(exc)
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        if os.environ.get(key):
            message = message.replace(os.environ[key], "[REDACTED]")
    if "out of memory" in message.lower():
        message += "; enable 4-bit loading or reduce max_seq_length"
    return {"kind": type(exc).__name__, "message": message}


class LocalRuntime:
    def __init__(self, options):
        if (
            platform.system() not in {"Linux", "Windows"}
            or int(os.environ.get("WORLD_SIZE", "1")) != 1
        ):
            raise ValueError(
                "Local decision evaluation requires Windows or Linux single-GPU CUDA without torchrun"
            )
        if platform.system() == "Windows":
            os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
            os.environ.setdefault("UNSLOTH_DECISION_COMPILE", "0")
        try:
            unsloth = importlib.import_module("unsloth")
            self.torch = importlib.import_module("torch")
            self.decision = importlib.import_module("unsloth.models.decision")
            self.clef = importlib.import_module("unsloth.models.clef")
        except ImportError as exc:
            raise ValueError(
                "Missing local model dependencies; run uv sync --extra train-unsloth"
            ) from exc
        torch = self.torch
        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA is unavailable; local inference needs an NVIDIA GPU (offline modes work on CPU)"
            )
        required = ("_clef_amp_dtype", "_served_temperatures", "_clef_truncated")
        if (
            not hasattr(unsloth, "FastDecisionModel")
            or any(not callable(getattr(self.decision, n, None)) for n in required)
            or not callable(getattr(self.clef, "encode_record", None))
        ):
            raise ValueError(
                "Installed Unsloth lacks required decision inference interfaces; run uv sync --extra train-unsloth"
            )
        precision = options["precision"]
        bf16 = torch.cuda.is_bf16_supported()
        if precision == "bf16" and not bf16:
            raise ValueError("Requested bf16 is unsupported by this CUDA device")
        dtype = (
            torch.bfloat16 if precision == "bf16" or precision == "auto" and bf16 else torch.float16
        )
        started = time.monotonic()
        self.model, self.tokenizer = unsloth.FastDecisionModel.from_pretrained(
            options["path"],
            max_seq_length=options["max_seq_length"],
            dtype=dtype,
            load_in_4bit=options["load_in_4bit"],
            local_files_only=options["local_files_only"],
            use_gradient_checkpointing=False,
        )
        if not getattr(self.model, "is_clef", False):
            raise ValueError(
                "Local evaluation requires a saved Clef decision head; no random head is created"
            )
        self.model.eval()
        self.tokenizer = getattr(self.tokenizer, "tokenizer", self.tokenizer)
        self.device = next(self.model.parameters()).device
        self.amp = self.decision._clef_amp_dtype(self.model, self.device)
        if precision != "auto" and self.amp != dtype:
            raise ValueError(
                "Requested inference precision is incompatible with this model/Unsloth combination; use auto"
            )
        self.limit = options["max_seq_length"]
        config = self.model.encoder.config
        self.metadata = {
            "model": str(Path(options["path"]).resolve()),
            "model_identity_source": "loaded_checkpoint",
            "backbone": {
                "name": getattr(config, "_name_or_path", None),
                "revision": getattr(config, "_commit_hash", None),
            },
            "calibration": self.model.decision_config,
            "runtime": {
                "gpu": torch.cuda.get_device_name(0),
                "cuda": torch.version.cuda,
                "precision": str(self.amp),
                "model_load_seconds": time.monotonic() - started,
                "logits_dtype": "float32",
                "probability_dtype": "float64",
            },
        }

    def predict(self, public):
        if set(public) != {"state", "questions"} or set(public["questions"]) != {"decision"}:
            raise LocalInputError(
                "Local inference accepts exactly public state and one decision question"
            )
        question = public["questions"]["decision"]
        try:
            encoded = self.clef.encode_record(self.tokenizer, public, max_length=self.limit)
            if self.decision._clef_truncated(
                self.tokenizer, public["state"], public["questions"], encoded, self.limit
            ):
                raise LocalInputError(
                    "Public request exceeds max_seq_length; refusing to truncate evaluation input"
                )
        except (ValueError, TypeError) as exc:
            raise LocalInputError(str(exc)) from exc
        torch = self.torch
        ids = torch.tensor([encoded.input_ids], device=self.device)
        # Official Clef encoder/head forward, with no target, gold, metadata or training labels.
        with (
            torch.inference_mode(),
            torch.autocast(self.device.type, dtype=self.amp, enabled=self.amp is not None),
        ):
            logits, _ = self.model(
                input_ids=ids, attention_mask=torch.ones_like(ids), records=[encoded]
            )
        record = encoded.questions[0]
        row = logits[0, : len(record.option_ids)].float().cpu()
        scale = self.decision._served_temperatures(
            self.model.decision_config,
            [row],
            [{"qtype": self.decision.QUESTION_TYPES.index(question["type"])}],
        )[0]
        # Preserve calibrated probability mass before the official API's four-decimal formatter.
        # Float64 softmax of float32 logits avoids accumulation loss; no distribution repair.
        p = dict(zip(record.option_ids, (row.double() / scale).softmax(-1).tolist()))
        answer = answer_from_probabilities(question, p)
        return {
            "model": self.metadata["model"],
            "answers": {"decision": answer},
            "usage": {"input_tokens": len(encoded.input_ids), "output_tokens": 0},
        }


def worker_main(connection, options):
    try:
        with contextlib.redirect_stdout(sys.stderr):
            runtime = LocalRuntime(options)
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
