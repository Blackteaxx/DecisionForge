"""Native Transformers inference for registered decision routes."""

from __future__ import annotations

import contextlib
import importlib.metadata
import json
import os
import time

from decisionforge.training.backends.swift import validate_artifact
from decisionforge.training.backends.swift_runtime import Runtime, installation
from decisionforge.training.backends.swift_task import map_request

from . import get_route
from .artifacts import file_hash, source_roots, sources
from .sdk import head_environment, preprocess, register_templates, temperature_buffers


class RouteRuntime(Runtime):
    def __init__(self, config, adapter, *, device="cuda:0", chat=False):
        if chat:
            raise ValueError("The existing Swift Chat baseline is OmniJev/Qwen3.5-4B only")
        self.config, self.chat = config, False
        self.route = get_route(config.swift.model_type)
        validated = validate_artifact(adapter, model_type=self.route.name)
        source = installation()
        import torch
        from swift.infer_engine import TransformersEngine
        from swift.model.decision_head import get_scoring_head

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ValueError("Swift inference requires an assigned BF16 CUDA device")
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError("Swift routes require single-device execution")
        self.torch = torch
        torch.cuda.set_device(device)
        register_templates()
        trained = validated.is_dir() and (validated / "dforge_swift_artifact.json").is_file()
        manifest = (
            json.loads((validated / "dforge_swift_artifact.json").read_text()) if trained else None
        )
        if trained and manifest["finetuning"] == "full":
            model_root, native_adapter = validated, None
        else:
            model_root, native_adapter = source_roots(self.route)
            if trained:
                native_adapter = validated
        head_environment(
            self.route, model_root, native_adapter, artifact=validated if trained else None
        )
        started = time.monotonic()
        self.engine = TransformersEngine(
            str(model_root),
            model_type=self.route.name,
            task_type="decision",
            adapters=[str(native_adapter)] if native_adapter else [],
            template_type=self.route.template,
            torch_dtype=torch.bfloat16,
            attn_impl="eager",
            device_map=device,
            use_hf=True,
        )
        self.engine.template.max_length = config.model.max_seq_length
        self.engine.template.max_pixels = config.swift.max_pixels
        self.engine.template.truncation_strategy = "raise"
        self.engine.template.remove_unused_columns = False
        self.engine.model.eval()
        if manifest:
            checkpoint = manifest["files"]
        else:
            checkpoint = {}
            for owner, relative, exported, _ in self.route.assets:
                root = native_adapter if owner == "adapter" else model_root
                checkpoint[exported] = file_hash(root / relative)
            if native_adapter and (native_adapter / "adapter_model.safetensors").is_file():
                checkpoint["adapter_model.safetensors"] = file_hash(
                    native_adapter / "adapter_model.safetensors"
                )
        self.metadata = {
            "model": str(adapter),
            "model_type": self.route.name,
            "model_identity_source": "loaded_swift_checkpoint",
            "backbone": {"name": self.route.model, "revision": self.route.revision},
            "checkpoint": checkpoint,
            "lineage": manifest["lineage"] if manifest else sources(self.route),
            "calibration": get_scoring_head(self.engine.model).extra_state_for_save(),
            "generation": None,
            "model_revision": self.route.revision,
            "adapter_revision": self.route.adapter_revision,
            "upstream": source,
            "probability_contract": self.route.probability_contract,
            "route_options": {"allow_none": config.swift.allow_none, "usage": config.swift.usage},
            "runtime": {
                "device": device,
                "gpu": torch.cuda.get_device_name(device),
                "cuda": torch.version.cuda,
                "precision": "bf16_native_decision_head_dtype",
                "load_seconds": time.monotonic() - started,
                "effective_max_seq_length": config.model.max_seq_length,
                "attention_implementation": "eager",
            },
            "dependencies": {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "peft", "datasets", "ms-swift")
            },
        }

    @contextlib.contextmanager
    def temperature(self, calibrated=True):
        if calibrated:
            yield
            return
        from swift.model.decision_head import get_scoring_head

        head = get_scoring_head(self.engine.model)
        before = {name: value.clone() for name, value in temperature_buffers(head).items()}
        try:
            for name in before:
                getattr(head, name).fill_(1)
            yield
        finally:
            for name, value in before.items():
                getattr(head, name).copy_(value)

    def predict(self, public, *, calibrated=True):
        from swift.infer_engine import RequestConfig

        rows, mappings = map_request(public, self.config.swift)
        answers, columns, native_records = {}, [], []
        started = time.monotonic()
        self.torch.cuda.reset_peak_memory_stats()
        with self.temperature(calibrated):
            for index, row in enumerate(rows):
                group = sorted(
                    (m for m in mappings if m["row_index"] == index),
                    key=lambda m: m["question_index"],
                )
                response = self.engine.infer(
                    [preprocess(self.route.name, row)], RequestConfig(), use_tqdm=False
                )[0]
                predictions = response.choices[0].message.content
                predictions = (
                    json.loads(predictions) if isinstance(predictions, str) else predictions
                )
                if not isinstance(predictions, list) or len(predictions) != len(group):
                    raise ValueError("Native Swift question cardinality changed")
                native_records.append(predictions)
                for mapping, prediction in zip(group, predictions):
                    entry = {**mapping, "native": [prediction], "extra_choice_probability": None}
                    columns.append(entry)
                    try:
                        answer, extra = self.route.readout(
                            public["questions"][mapping["question_id"]],
                            prediction["probs"],
                            mapping,
                        )
                        answers[mapping["question_id"]] = answer
                        entry["extra_choice_probability"] = extra
                    except (ValueError, TypeError) as exc:
                        entry["validation_error"] = str(exc)
        self.torch.cuda.synchronize()
        return {
            "answers": answers,
            "swift": {
                "model_type": self.route.name,
                "columns": columns,
                "native_records": native_records,
                "probability_contract": self.route.probability_contract,
                "calibrated": calibrated,
                "seconds": time.monotonic() - started,
                "peak_memory_bytes": self.torch.cuda.max_memory_allocated(),
            },
        }
