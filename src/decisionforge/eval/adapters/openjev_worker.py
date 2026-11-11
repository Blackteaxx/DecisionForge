"""Isolated official single-rationale/logit readout; imports remain CPU-safe."""

from __future__ import annotations

import contextlib
import importlib
import os
import sys
import time

from decisionforge.eval.adapters.local_worker import LocalInputError, safe_error
from decisionforge.training.backends.openjev import (
    artifact_identity,
    dependency_versions,
    read_object,
    validate_artifact,
    validate_installation,
)
from decisionforge.training.backends.openjev_task import (
    LABELS,
    PUBLIC_PROTOCOL,
    READOUT_POLICY,
    SAMPLING,
    TASK_NAME,
    answer_from_distribution,
    check_prompt_budget,
    inference_label_protocol,
    public_item,
    register_task,
)
from decisionforge.util import digest


def predict_public(
    model, rat, public, *, focus="decision", max_new_tokens=128, input_check=None, labels=None
):
    labels = LABELS if labels is None else labels
    item = public_item(public, focus, labels=labels)
    if input_check is not None:
        input_check(item)
    groups = rat.sample(model, [item], m=1, max_new_tokens=max_new_tokens)
    if len(groups) != 1 or len(groups[0]) != 1:
        raise ValueError("OpenJev official sampler did not return exactly one rationale")
    group = groups[0]
    u, _ = rat.forward(model, group, grad=False, k=len(item["option_keys"]))
    full = u.detach().cpu().tolist()
    p, rationale, forced = group[0]
    raw = {
        "answers": {},
        "openjev": {
            "public_request": item["public_request"],
            "focus": focus,
            "label_mapping": item["label_mapping"],
            "prompt_ids": p,
            "rationale_ids": rationale,
            "rationale": rat.tok.decode(rationale),
            "forced_marker": forced,
            "appended_marker_ids": list(rat.suffix) if forced else [],
            "raw_label_distributions": full,
            "sampling": {**SAMPLING, "max_new_tokens": max_new_tokens},
            "public_protocol": PUBLIC_PROTOCOL,
        },
        "usage": {"input_tokens": len(p), "output_tokens": len(rationale)},
    }
    try:
        if (
            len(full) != 1
            or len(full[0]) != len(labels)
            or any(value != 0 for value in full[0][len(item["option_keys"]) :])
        ):
            raise ValueError("Invalid OpenJev official masked label distribution")
        raw["answers"][focus] = answer_from_distribution(
            public["questions"][focus], full[0][: len(item["option_keys"])]
        )
    except (ValueError, TypeError) as exc:
        # Preserve illegal raw numerical output for the parent's protocol journal.
        raw["openjev"]["validation_error"] = str(exc)
    return raw


class OpenJevRuntime:
    def __init__(self, options):
        root = validate_artifact(options["path"])
        from decisionforge.eval.models import OpenJevModelConfig

        config = OpenJevModelConfig.model_validate({"adapter": "openjev_local", **options})
        manifest = read_object(root / "openjev_artifact.json")
        source = validate_installation()
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError("OpenJev inference requires a single CUDA device without DDP")
        self.torch = importlib.import_module("torch")
        torch = self.torch
        if not torch.cuda.is_available():
            raise ValueError(
                "OpenJev inference requires a BF16 CUDA GPU; static validation/score-only are CPU-safe"
            )
        device = torch.device(config.device)
        torch.cuda.set_device(device)
        if not torch.cuda.is_bf16_supported():
            raise ValueError("OpenJev requires BF16-capable CUDA; FP16 substitution is unsupported")
        # Upstream resolves DEVICE once at import; select it explicitly in this isolated worker.
        os.environ["OPENJEV_DEVICE"] = str(device)
        upstream = importlib.import_module("openjev_rlcd.train")
        upstream.DEVICE = str(device)
        transformers = importlib.import_module("transformers")
        started = time.monotonic()
        tok = transformers.AutoTokenizer.from_pretrained(
            str(root / "backbone"), local_files_only=True
        )
        self.manifest = manifest
        self.trained_protocol = inference_label_protocol(tok, manifest, len(manifest["labels"]))
        register_task(upstream.TASKS, {}, labels=self.trained_protocol["labels"])
        self.rat = upstream.Rationales(tok, TASK_NAME)
        self.readers = {len(self.trained_protocol["labels"]): self.rat}
        cfg = transformers.AutoConfig.from_pretrained(str(root / "backbone"), local_files_only=True)
        self.model = transformers.AutoModelForCausalLM.from_config(
            cfg, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        if not callable(getattr(self.model, "model", None)) or not callable(
            getattr(self.model, "lm_head", None)
        ):
            raise ValueError("OpenJev requires model.model hidden states and model.lm_head")
        state = torch.load(
            root / "stages/reinforce/weights.pt", map_location="cpu", weights_only=True
        )
        self.model.load_state_dict(state, strict=True)
        self.model.generation_config = transformers.GenerationConfig.from_pretrained(
            str(root / "backbone"), local_files_only=True
        )
        del state
        self.model.to(device).eval()
        self.max_new_tokens = config.max_new_tokens or manifest["sampling"]["max_new_tokens"]
        self.limit = min(
            config.max_seq_length or manifest["max_seq_length"],
            manifest["max_seq_length"],
            cfg.max_position_embeddings,
        )
        self.seed = config.seed
        self.upstream = upstream
        self.metadata = {
            "model": str(root),
            "model_identity_source": "loaded_openjev_stage2_weights",
            **source,
            **artifact_identity(root),
            "sampling": {
                **SAMPLING,
                "max_new_tokens": self.max_new_tokens,
                "seed": self.seed,
                "seed_basis": "sha256_public_request_focus_plus_config_seed",
            },
            "dependencies": dependency_versions(),
            "generation_config": self.model.generation_config.to_dict(),
            "public_protocol": PUBLIC_PROTOCOL,
            "label_protocol": self.trained_protocol,
            "trained_pool": {
                "labels": manifest["labels"],
                "token_ids": manifest["label_token_ids"],
                "training_max_options": manifest.get("training_max_options"),
                "artifact_schema_version": manifest["schema_version"],
            },
            "readout_policy": READOUT_POLICY,
            "runtime": {
                "device": str(device),
                "gpu": torch.cuda.get_device_name(device),
                "precision": "bfloat16_parameters_and_autocast",
                "logits_precision": "float32",
                "cuda": torch.version.cuda,
                "load_seconds": time.monotonic() - started,
                "effective_max_seq_length": self.limit,
                "attention_implementation": "sdpa",
            },
        }

    def predict(self, public, *, focus="decision"):
        from decisionforge.training.backends.openjev_task import option_keys

        # Validate every sibling before deriving the focused reader. All public input is retained.
        public_item(public, focus)
        k = len(option_keys(public["questions"][focus]))
        protocol = inference_label_protocol(self.rat.tok, self.manifest, k)
        count = len(protocol["labels"])
        if count not in self.readers:
            register_task(self.upstream.TASKS, {}, labels=protocol["labels"])
            self.readers[count] = self.upstream.Rationales(self.rat.tok, TASK_NAME)
        reader = self.readers[count]
        extended = count > len(self.trained_protocol["labels"])

        def check(item):
            check_prompt_budget(
                reader.tok,
                self.upstream.prompt(reader.tok, item, TASK_NAME),
                self.limit,
                self.max_new_tokens,
                reader.suffix,
            )

        # Stable request-local seeds make journal resume independent of earlier draws.
        seed_input = {"public": public, "focus": focus, "seed": self.seed}
        if extended:
            seed_input["readout_protocol_hash"] = protocol["protocol_hash"]
        seed = int(digest(seed_input)[:8], 16)
        self.torch.manual_seed(seed)
        with self.torch.inference_mode():
            raw = predict_public(
                self.model,
                reader,
                public,
                focus=focus,
                max_new_tokens=self.max_new_tokens,
                input_check=check,
                labels=protocol["labels"],
            )
        raw.update(model=self.metadata["model"])
        raw["openjev"]["effective_request_seed"] = seed
        raw["openjev"].update(
            label_protocol=protocol,
            trained_pool_size=len(self.trained_protocol["labels"]),
            training_max_options=self.manifest.get("training_max_options"),
            inference_only_label_extension=extended,
            readout_policy=READOUT_POLICY,
        )
        return raw

    def close(self):
        self.model = None
        self.torch.cuda.empty_cache()


def worker_main(connection, options):
    runtime = None
    try:
        with contextlib.redirect_stdout(sys.stderr):
            runtime = OpenJevRuntime(options)
        connection.send({"kind": "ready", "metadata": runtime.metadata})
        while True:
            command = connection.recv()
            if command["kind"] == "close":
                break
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    response = runtime.predict(command["request"])
                connection.send({"kind": "prediction", "response": response})
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
