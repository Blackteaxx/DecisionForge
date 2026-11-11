"""Official Unsloth decision APIs. No training dependency is imported at module import time."""

from __future__ import annotations

import gc
import hashlib
import importlib
import inspect
import json
import math
import os
import platform
import random
import time
from collections import Counter
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from decisionforge.eval.adapter import parse_answer
from decisionforge.eval.data import parse_json
from decisionforge.eval.models import Question
from decisionforge.training.data import MIN_CALIBRATION_QUESTIONS, to_unsloth_row
from decisionforge.training.state import optimizer_summary, state_fingerprint
from decisionforge.util import json_text, now, write_json

TYPES = ("choice", "score", "noul")
CLEF_TYPES = ("noul", "choice", "score")
DECISION_REVISION = "94a00e95a75d380da6b8477f1d42710d2d54e58f"
DECISION_SOURCE_SHA256 = "4b538ffda29f7b2af74a13a21b7bb520f112c035efe69d8e9b7b20a6011ec5ee"
PACKAGES = (
    "unsloth",
    "unsloth_zoo",
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "bitsandbytes",
    "safetensors",
    "numpy",
    "datasets",
    "pyarrow",
    "xformers",
    "torchvision",
)


def local_model_metadata(name):
    """Record local model file metadata, without hashing model contents or training data."""
    root = Path(name).expanduser()
    if not root.is_dir():
        return None
    root = root.resolve()
    files = {
        path
        for path in root.rglob("*")
        if path.is_file()
        and not any(part.startswith(".") for part in path.relative_to(root).parts)
        and path.suffix in {".json", ".safetensors", ".bin", ".model", ".txt", ".jinja", ".py"}
    }
    index = root / "model.safetensors.index.json"
    if index.is_file():
        files.update(root / name for name in json.loads(index.read_text())["weight_map"].values())
    metadata = {}
    for path in sorted(files):
        stat = path.stat()
        metadata[str(path.relative_to(root))] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return {"path": str(root), "files": metadata}


def encoding_report(records, items, raw, max_length):
    kept = {i: [] for i in range(len(records))}
    types, truncated = Counter(), []
    for item in items:
        row = item["row"]
        if not 0 <= row < len(records):
            raise ValueError("Unsupported Unsloth dataset row index")
        if "source" in item:
            keys = list(item["source"]["questions"])
        else:
            candidates = [
                k
                for k, q in records[row].request.questions.items()
                if q.type == TYPES[item["qtype"]] and k not in kept[row]
            ]
            keys = candidates[:1]
        kept[row].extend(keys)
        types.update(records[row].request.questions[k].type for k in keys)
        if len(item["input_ids"]) >= max_length:
            truncated.append({"record_id": records[row].record_id, "questions": keys})
    skipped = [
        {
            "record_id": r.record_id,
            "questions": [k for k in r.request.questions if k not in kept[i]],
        }
        for i, r in enumerate(records)
        if set(kept[i]) != set(r.request.questions)
    ]
    return {
        "items": len(items),
        "questions": sum(types.values()),
        "types": dict(types),
        "upstream": raw,
        "skipped": skipped,
        "truncated": truncated,
    }


def check_encoding(config, role, report):
    if not report["items"]:
        raise ValueError(f"All {role} records were skipped; cannot continue")
    for field in ("skipped", "truncated"):
        if report.get(field) and getattr(config.data, "on_" + field) == "error":
            raise ValueError(
                f"{role}: Unsloth {field} data: {report[field]}; "
                f"see data_report.json; explicitly set data.on_{field}=allow to continue"
            )
    if (
        role == "calibration"
        and config.calibration.enabled
        and report["questions"] < MIN_CALIBRATION_QUESTIONS
    ):
        raise ValueError(
            f"Calibration needs at least {MIN_CALIBRATION_QUESTIONS} usable encoded "
            "questions; provide more calibration samples or disable calibration"
        )


def type_items(items, kind):
    """Keep original encoded context while selecting a type's head outputs for diagnostics."""
    selected = []
    for item in items:
        if "qtypes" not in item:
            if item["qtype"] == TYPES.index(kind):
                selected.append(item)
            continue
        indices = [i for i, t in enumerate(item["qtypes"]) if t == CLEF_TYPES.index(kind)]
        if not indices:
            continue
        record = replace(
            item["record"], questions=tuple(item["record"].questions[i] for i in indices)
        )
        subset = {**item, "record": record}
        for key in ("qtypes", "targets", "labels"):
            subset[key] = [item[key][i] for i in indices]
        selected.append(subset)
    return selected


def native_prediction(answers, questions):
    if set(answers) != set(questions):
        raise ValueError("Reloaded model changed original question keys")
    for key, raw in answers.items():
        q = Question.model_validate(questions[key])
        answer = parse_answer({"answers": {"decision": raw}}, q)
        probabilities = answer.get("probabilities")
        # Official Decision API rounds probabilities to 4 decimal places; retain, never repair.
        if probabilities is not None and (
            set(probabilities) != set(q.labels)
            or any(
                type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
                for v in probabilities.values()
            )
            or abs(sum(probabilities.values()) - 1) > len(q.labels) * 0.00005 + 1e-6
        ):
            raise ValueError(f"Reloaded model returned invalid probabilities for {key}")
    return {"answers": answers}


def retain_unfitted_calibration(before, after, fitted, clamp):
    missing = set(range(3)) - set(fitted)
    if not missing:
        return {"shared_head_temperature_retained": False, "clamped_fitted_types": []}
    old_head, new_head = before.get("head_temperature", 1.0), after.get("head_temperature", 1.0)
    original = before.get("temperature", [1.0] * 3)
    temperatures = list(after.get("temperature", [1.0] * 3))
    clamped = []
    # A shared head scale would change absent types too. Keep its previous scale and
    # represent fitted types as relative temperatures within the official serving bounds.
    for index in range(3):
        if index in missing:
            temperatures[index] = original[index]
        else:
            effective_relative = new_head * clamp(temperatures[index]) / old_head
            temperatures[index] = clamp(effective_relative)
            if not math.isclose(temperatures[index], effective_relative):
                clamped.append(TYPES[index])
    after["head_temperature"] = old_head
    after["temperature"] = temperatures
    buckets = {
        k: v
        for k, v in before.get("temperature_by_options", {}).items()
        if k.split(":")[0] in {TYPES[i] for i in missing}
    }
    if buckets:
        after["temperature_by_options"] = buckets
    return {
        "shared_head_temperature_retained": True,
        "clamped_fitted_types": clamped,
        "meaning": "preserves effective calibration of unfitted types; fitted relative values "
        "may be limited by official serving bounds",
    }


def checkpoint_configs(checkpoint, finetuning="lora"):
    def read(name):
        try:
            value = parse_json((checkpoint / name).read_text())
            if not isinstance(value, dict):
                raise ValueError("expected JSON object")
            return value
        except (ValueError, OSError) as exc:
            raise ValueError(f"Corrupt checkpoint {name}: {exc}") from exc

    adapter = read("adapter_config.json") if finetuning == "lora" else read("config.json")
    if finetuning == "lora" and (
        adapter.get("peft_type") != "LORA"
        or not isinstance(adapter.get("base_model_name_or_path"), str)
        or not adapter["base_model_name_or_path"]
        or type(adapter.get("r")) is not int
        or adapter["r"] < 1
    ):
        raise ValueError("Corrupt checkpoint adapter_config.json: invalid LoRA configuration")
    head = read("joint_head_config.json")
    for key in ("hidden_size", "width", "heads", "feedforward", "routing_layers", "layers"):
        if type(head.get(key)) is not int or head[key] < (
            0 if key in {"routing_layers", "layers"} else 1
        ):
            raise ValueError(f"Corrupt checkpoint joint_head_config.json: invalid {key}")
    decision = read("unsloth_decision_config.json")
    temperatures = decision.get("temperature", [1.0] * 3)
    if (
        decision.get("layout") != "clef"
        or type(decision.get("max_len")) is not int
        or decision["max_len"] < 1
        or not isinstance(temperatures, list)
        or len(temperatures) != 3
        or any(type(t) not in (int, float) or not math.isfinite(t) or t <= 0 for t in temperatures)
    ):
        raise ValueError(
            "Corrupt checkpoint unsloth_decision_config.json: invalid decision configuration"
        )
    return adapter, head, decision


def verify_calibration_reload(original, reloaded):
    for key, default in (("temperature", [1.0] * 3), ("temperature_by_options", {})):
        if original.get(key, default) != reloaded.get(key, default):
            raise ValueError(f"Saved calibration {key} changed during reload")
    expected = original.get("folded_temperature", 1.0) * original.get("head_temperature", 1.0)
    actual = reloaded.get("folded_temperature", 1.0) * reloaded.get("head_temperature", 1.0)
    if not math.isclose(expected, actual, rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError("Saved calibration head temperature changed during reload")


class UnslothBackend:
    def __init__(self, config, directory):
        self.config, self.directory = config, Path(directory)
        self.model = self.tokenizer = self.trainer = None
        self.items, self.encoding = {}, {}
        self.data = None
        self.resume_state, self._checkpoint_audits = [], {}
        self._resume_checkpoint = None

    @property
    def is_main(self):
        return int(os.environ.get("RANK", "0")) == 0

    def barrier(self):
        if self.config.distributed.world_size > 1:
            self.torch.distributed.barrier()

    def finish_distributed(self):
        if self.config.distributed.world_size > 1:
            self.barrier()
            self.torch.distributed.destroy_process_group()

    def environment(self):
        if platform.system() not in {"Linux", "Windows"}:
            raise ValueError("Unsloth decision training requires Windows or Linux + NVIDIA CUDA")
        world = self.config.distributed.world_size
        if platform.system() == "Windows":
            if self.config.finetuning != "lora" or self.config.distributed.strategy != "single":
                raise ValueError("Windows currently supports single-GPU Unsloth LoRA training")
            os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
            os.environ.setdefault("UNSLOTH_DECISION_COMPILE", "0")
        if int(os.environ.get("WORLD_SIZE", "1")) != world:
            raise ValueError("Worker WORLD_SIZE differs from configured distributed world_size")
        try:
            # Unsloth patches must load before Transformers/PEFT, including through torch.
            unsloth = importlib.import_module("unsloth")
            torch = importlib.import_module("torch")
            from transformers import TrainerCallback, TrainingArguments, set_seed
        except ImportError as exc:
            raise ValueError(
                "Missing training dependencies. Install .[train-unsloth] in the "
                "documented Windows or Linux CUDA environment"
            ) from exc
        api, trainer = (
            getattr(unsloth, "FastDecisionModel", None),
            getattr(unsloth, "DecisionTrainer", None),
        )
        required = (
            "from_pretrained",
            "get_peft_model",
            "build_dataset",
            "evaluate",
            "calibrate",
            "predict",
        )
        if (
            api is None
            or trainer is None
            or any(not callable(getattr(api, n, None)) for n in required)
        ):
            raise ValueError(
                "Installed Unsloth lacks official FastDecisionModel / DecisionTrainer APIs; "
                "install the documented version"
            )
        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA is unavailable; train on a Linux NVIDIA GPU (validate-only works on CPU)"
            )
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.device_count() < world or not 0 <= local_rank < torch.cuda.device_count():
            raise ValueError(
                "Insufficient visible CUDA devices for configured single-host world_size"
            )
        torch.cuda.set_device(local_rank)
        source = Path(inspect.getfile(api)).resolve()
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        if self.config.finetuning == "full":
            if "full_finetuning" not in inspect.signature(api.from_pretrained).parameters:
                raise ValueError("Installed Unsloth lacks the pinned full_finetuning API")
            if source_hash != DECISION_SOURCE_SHA256:
                raise ValueError(
                    "Full finetuning requires the pinned Unsloth decision source; "
                    "run uv sync --extra train-unsloth-full"
                )
        bf16 = torch.cuda.is_bf16_supported()
        precision = self.config.training.precision
        if precision == "bf16" and not bf16:
            raise ValueError("Requested bf16 is unsupported by this CUDA device; choose auto/fp16")
        self.dtype = (
            torch.bfloat16 if precision == "bf16" or precision == "auto" and bf16 else torch.float16
        )
        self.torch, self.api, self.Trainer = torch, api, trainer
        self.TrainingArguments, self.Callback, self.set_seed = (
            TrainingArguments,
            TrainerCallback,
            set_seed,
        )
        packages = {}
        for name in PACKAGES:
            try:
                packages[name] = version(name)
            except PackageNotFoundError:
                packages[name] = None
        return {
            "packages": packages,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(local_rank),
            "precision": str(self.dtype),
            "visible_devices": torch.cuda.device_count(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "world_size": world,
            "devices": [torch.cuda.get_device_name(i) for i in range(world)],
            "upstream": {
                "repository": "https://github.com/unslothai/unsloth",
                "revision": DECISION_REVISION if source_hash == DECISION_SOURCE_SHA256 else None,
                "expected_revision": DECISION_REVISION,
                "decision_source_sha256": source_hash,
                "pinned_source_verified": source_hash == DECISION_SOURCE_SHA256,
            },
        }

    def prepare(self, data):
        self.data = data
        c = self.config
        source_metadata = local_model_metadata(c.model.name)
        self.set_seed(c.training.seed)
        self.model, self.tokenizer = self.api.from_pretrained(
            model_name=c.model.name,
            revision=c.model.revision,
            max_seq_length=c.model.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=c.model.load_in_4bit,
            full_finetuning=c.finetuning == "full",
            use_gradient_checkpointing="unsloth" if c.training.gradient_checkpointing else False,
            random_state=c.training.seed,
        )
        if c.finetuning == "lora":
            self.model = self.api.get_peft_model(
                self.model,
                r=c.lora.rank,
                lora_alpha=c.lora.alpha,
                lora_dropout=c.lora.dropout,
                target_modules="all-linear",
                use_gradient_checkpointing="unsloth"
                if c.training.gradient_checkpointing
                else False,
                random_state=c.training.seed,
            )
        if not getattr(self.model, "is_clef", False):
            raise ValueError(
                "This backend requires a Clef decision model or a plain language model "
                "converted by FastDecisionModel (e.g. Qwen3.5-4B)"
            )
        if not callable(getattr(self.model, "save_pretrained", None)) or not callable(
            getattr(self.model, "save_pretrained_merged", None)
        ):
            raise ValueError("Unsloth model lacks official adapter/merged saving APIs")
        self.before, self.trainable_parameters = {}, {"groups": {}, "total": 0}
        scopes = (
            (("backbone", "encoder."), ("head", "head."))
            if c.finetuning == "full"
            else (("lora", "lora_"), ("head", "head."))
        )
        for group, prefix in scopes:
            params = [
                (n, p)
                for n, p in self.model.named_parameters()
                if p.requires_grad and (prefix in n if group == "lora" else n.startswith(prefix))
            ]
            if not params:
                raise ValueError(
                    f"No trainable {group} parameters; refusing incomplete {c.finetuning} training"
                )
            count = sum(p.numel() for _, p in params)
            self.trainable_parameters["groups"][group] = {
                "count": count,
                "names": [n for n, _ in params],
            }
            self.trainable_parameters["total"] += count
            # Small deterministic audit samples: avoids a second copy of the backbone.
            self.before[group] = {
                n: p.detach().flatten()[:4096].float().cpu().clone() for n, p in params
            }
        for role, records in data.roles.items():
            if not records or role == "calibration" and not c.calibration.enabled:
                self.items[role] = []
                continue
            rows = [to_unsloth_row(r) for r in records]
            items, raw = self.api.build_dataset(rows, self.tokenizer, self.model)
            report = encoding_report(records, items, raw, c.model.max_seq_length)
            # Obtain each affected row's own upstream reason instead of attributing the batch's
            # most frequent reason to every skipped record.
            affected = {r["record_id"] for r in report["skipped"]}
            for r, row in zip(records, rows):
                if r.record_id in affected:
                    _, detail = self.api.build_dataset([row], self.tokenizer, self.model)
                    for entry in report["skipped"]:
                        if entry["record_id"] == r.record_id:
                            entry["upstream_reason"] = detail.get("reason")
            self.items[role], self.encoding[role] = items, report
            if self.is_main:
                write_json(
                    self.directory / "data_report.json",
                    {**data.report, "encoding": {"roles": self.encoding}},
                )
            check_encoding(c, role, report)
        encoder_config = getattr(self.model.encoder, "config", None)
        loaded_model = {
            "configured": c.model.model_dump(),
            "actual_name": getattr(encoder_config, "_name_or_path", c.model.name),
            "actual_revision": getattr(encoder_config, "_commit_hash", None),
            "decision_config": self.model.decision_config,
            "source_metadata": source_metadata,
            "backbone_metadata": local_model_metadata(
                getattr(encoder_config, "_name_or_path", c.model.name)
            ),
        }
        self._create_trainer()
        return {
            "model": loaded_model,
            "roles": self.encoding,
            "trainable_parameters": self.trainable_parameters,
        }

    def _create_trainer(self):
        c, backend = self.config.training, self

        class Monitor(self.Callback):
            def on_train_begin(self, args, state, control, **kwargs):
                backend._record_restored_state(state)

            def on_log(self, args, state, control, logs=None, **kwargs):
                if not backend.is_main:
                    return
                values = {"at": now(), "step": state.global_step, **(logs or {})}
                if any(isinstance(v, float) and not math.isfinite(v) for v in values.values()):
                    raise ValueError("Nonfinite training loss/diagnostics")
                values["gpu_memory_allocated"] = backend.torch.cuda.memory_allocated()
                values["gpu_memory_peak"] = backend.torch.cuda.max_memory_allocated()
                if (
                    backend.config.distributed.world_size == 1
                    and "loss" in values
                    and backend.items.get("validation")
                ):
                    values["validation"] = backend.evaluate("validation", "training_monitor")
                backend._log(values)

            def on_save(self, args, state, control, **kwargs):
                backend._mark_checkpoint(Path(args.output_dir) / f"checkpoint-{state.global_step}")

            def on_step_begin(self, args, state, control, **kwargs):
                # A terminal same-step save may follow the automatic save. Invalidate
                # its commit marker before any later save rewrites files.
                if backend.is_main:
                    target = Path(args.output_dir) / f"checkpoint-{state.global_step + 1}"
                    (target / "dforge_checkpoint.json").unlink(missing_ok=True)

        args = self.TrainingArguments(
            output_dir=str(
                Path(self.config.output.checkpoint_directory or self.directory / "checkpoints")
            ),
            per_device_train_batch_size=c.batch_size,
            per_device_eval_batch_size=c.batch_size,
            gradient_accumulation_steps=c.gradient_accumulation_steps,
            learning_rate=c.learning_rate,
            warmup_steps=c.warmup_steps,
            weight_decay=c.weight_decay,
            lr_scheduler_type=c.lr_scheduler_type,
            bf16=self.dtype == self.torch.bfloat16,
            fp16=self.dtype == self.torch.float16,
            gradient_checkpointing=c.gradient_checkpointing,
            logging_steps=c.logging_steps,
            logging_first_step=True,
            save_steps=c.save_steps,
            save_strategy="steps",
            max_steps=c.max_steps if c.max_steps is not None else -1,
            num_train_epochs=c.num_train_epochs if c.num_train_epochs is not None else 1.0,
            seed=c.seed,
            data_seed=c.seed,
            report_to="none",
            dataloader_num_workers=0,
            optim="adamw_torch",
            save_only_model=False,
            load_best_model_at_end=False,
            remove_unused_columns=False,
            logging_nan_inf_filter=False,
            ddp_backend="nccl" if self.config.distributed.world_size > 1 else None,
            ddp_timeout=self.config.distributed.timeout_seconds,
            ddp_find_unused_parameters=True if self.config.distributed.world_size > 1 else None,
        )
        self.trainer = self.Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=args,
            train_dataset=self.items["train"],
            head_learning_rate=c.head_learning_rate,
            callbacks=[Monitor()],
        )
        self._install_rng_audit()
        expected_bf16 = self.dtype == self.torch.bfloat16
        expected_fp16 = self.dtype == self.torch.float16
        if self.trainer.args.bf16 != expected_bf16 or self.trainer.args.fp16 != expected_fp16:
            raise ValueError(
                "Selected precision/quantization is unsupported by this model and Unsloth version; "
                "use a supported precision/device instead of silently changing training precision"
            )

    def _record_restored_state(self, state):
        if self._resume_checkpoint is None:
            return
        expected = self._checkpoint_audits[str(self._resume_checkpoint)]
        actual = {
            "global_step": state.global_step,
            "optimizer": optimizer_summary(self.trainer.optimizer.state_dict(), self.torch),
            "scheduler_sha256": state_fingerprint(
                self.trainer.lr_scheduler.state_dict(), self.torch
            ),
        }
        event = {
            "event": "optimizer_scheduler_restored",
            "rank": int(os.environ.get("RANK", "0")),
            "checkpoint": str(self._resume_checkpoint),
            "timing": "on_train_begin after upstream optimizer/scheduler load",
            "expected": expected,
            "actual": actual,
            "matches_checkpoint": actual == expected,
            "meaning": "optimizer tensor samples and all step counters; complete scheduler state",
        }
        self.resume_state.append(event)
        if not event["matches_checkpoint"]:
            raise ValueError(
                "Actual restored optimizer/scheduler/global-step differs from checkpoint"
            )

    def _rng_state(self):
        numpy = importlib.import_module("numpy")
        return {
            "python": random.getstate(),
            "numpy": numpy.random.get_state(),
            "cpu": self.torch.random.get_rng_state(),
            "cuda": self.torch.cuda.random.get_rng_state_all()
            if self.config.distributed.world_size > 1
            else self.torch.cuda.random.get_rng_state(),
        }

    def _install_rng_audit(self):
        upstream = getattr(self.trainer, "_load_rng_state", None)
        if not callable(upstream):
            return

        def load_and_observe(checkpoint):
            upstream(checkpoint)
            if checkpoint is None:
                return
            rank = int(os.environ.get("RANK", "0"))
            filename = (
                f"rng_state_{rank}.pth"
                if self.config.distributed.world_size > 1
                else "rng_state.pth"
            )
            expected = self.torch.load(
                Path(checkpoint) / filename, map_location="cpu", weights_only=False
            )
            actual_hash = state_fingerprint(self._rng_state(), self.torch)
            expected_hash = state_fingerprint(expected, self.torch)
            event = {
                "event": "rng_restored",
                "rank": rank,
                "checkpoint": str(checkpoint),
                "timing": "immediately after official Trainer._load_rng_state, before further random use",
                "expected_sha256": expected_hash,
                "actual_sha256": actual_hash,
                "matches_checkpoint": actual_hash == expected_hash,
                "meaning": "complete Python/NumPy/CPU/CUDA RNG states; observation consumes no draws",
            }
            self.resume_state.append(event)
            if not event["matches_checkpoint"]:
                raise ValueError("Actual restored per-rank RNG differs from checkpoint")

        self.trainer._load_rng_state = load_and_observe

    def _log(self, value):
        if not self.is_main:
            return
        logs = self.directory / "logs"
        logs.mkdir(exist_ok=True)
        with (logs / "training.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json_text(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def train(self, checkpoint=None):
        self._resume_checkpoint = Path(checkpoint) if checkpoint is not None else None
        if checkpoint is not None:
            self.validate_checkpoint(checkpoint)
        if checkpoint is not None:
            self.trainer._load_from_checkpoint(str(checkpoint))
            params = dict(self.model.named_parameters())
            for group, before in self.before.items():
                self.before[group] = {
                    n: params[n].detach().flatten()[:4096].float().cpu().clone() for n in before
                }
        started = time.monotonic()
        result = self.trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
        if not math.isfinite(result.training_loss):
            raise ValueError("Nonfinite final training loss")
        changed = {}
        params = dict(self.model.named_parameters())
        for group, before in self.before.items():
            changed[group] = any(
                not self.torch.equal(sample, params[n].detach().flatten()[:4096].float().cpu())
                for n, sample in before.items()
            )
        rank = int(os.environ.get("RANK", "0"))
        observation = {
            "rank": rank,
            "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
            "current_device": self.torch.cuda.current_device(),
            "device": self.torch.cuda.get_device_name(),
            "precision": str(self.dtype),
            "resume_state": self.resume_state,
        }
        observations = [observation]
        if self.config.distributed.world_size > 1:
            observations = [None] * self.config.distributed.world_size
            self.torch.distributed.all_gather_object(observations, observation)
        rank_devices = [
            {key: value for key, value in entry.items() if key != "resume_state"}
            for entry in observations
        ]
        resume_events = [event for entry in observations for event in entry["resume_state"]]
        if self.is_main:
            write_json(
                self.directory / "logs" / "resume_state.json",
                {
                    "checkpoint": str(checkpoint) if checkpoint else None,
                    "timing": "after official train returns, before final checkpoint save",
                    "rank_devices": rank_devices,
                    "events": resume_events,
                },
            )
        return {
            "raw": result.metrics,
            "training_loss": result.training_loss,
            "global_step": self.trainer.state.global_step,
            "elapsed_seconds": time.monotonic() - started,
            "parameter_update_probe": changed,
            "trainable_parameters": self.trainable_parameters,
            "parameter_update_probe_meaning": "deterministic first-4096-value samples per trainable tensor",
            "parameter_update_probe_reference": str(checkpoint)
            if checkpoint
            else "initial loaded model",
            "rank_devices": rank_devices,
            "resume_state": resume_events,
            "selected_checkpoint": "final",
        }

    def _mark_checkpoint(self, checkpoint):
        self.barrier()
        if self.is_main:
            self.validate_checkpoint(checkpoint, require_marker=False)
            write_json(
                checkpoint / "dforge_checkpoint.json",
                {
                    "global_step": self.trainer.state.global_step,
                    "finetuning": self.config.finetuning,
                    "world_size": self.config.distributed.world_size,
                    "run_identity": self._run_identity(),
                    "created_at": now(),
                    "files": {
                        p.name: p.stat().st_size
                        for p in checkpoint.iterdir()
                        if p.is_file() and p.name != "dforge_checkpoint.json"
                    },
                },
            )
        self.barrier()

    def save_final_checkpoint(self):
        # Use the official full Trainer saver: includes optimizer/scheduler/RNG and head.
        checkpoint = (
            Path(self.config.output.checkpoint_directory or self.directory / "checkpoints")
            / f"checkpoint-{self.trainer.state.global_step}"
        )
        if self.is_main:
            (checkpoint / "dforge_checkpoint.json").unlink(missing_ok=True)
        self.barrier()
        self.trainer._save_checkpoint(self.trainer.model, trial=None)
        self._mark_checkpoint(checkpoint)
        return checkpoint

    def _run_identity(self):
        manifest = self.directory / "manifest.json"
        return json.loads(manifest.read_text()).get("identity") if manifest.is_file() else None

    def validate_checkpoint(self, checkpoint, require_marker=True):
        checkpoint = Path(checkpoint)
        required = [
            "trainer_state.json",
            "optimizer.pt",
            "scheduler.pt",
            "joint_head.safetensors",
            "joint_head_config.json",
            "unsloth_decision_config.json",
        ]
        world = self.config.distributed.world_size
        rng_files = (
            ["rng_state.pth"] if world == 1 else [f"rng_state_{rank}.pth" for rank in range(world)]
        )
        required.extend(rng_files)
        if self.config.finetuning == "lora":
            weights = ["adapter_model.safetensors"]
            required.append("adapter_config.json")
        else:
            required.append("config.json")
            if (checkpoint / "adapter_config.json").exists():
                raise ValueError("Full checkpoint cannot contain LoRA adapter configuration")
            index_path = checkpoint / "model.safetensors.index.json"
            if index_path.is_file():
                index = parse_json(index_path.read_text())
                mapping = index.get("weight_map") if isinstance(index, dict) else None
                if not isinstance(mapping, dict) or not mapping:
                    raise ValueError("Corrupt full checkpoint shard index")
                if any(
                    not isinstance(key, str)
                    or not isinstance(name, str)
                    or Path(name).name != name
                    or not name.endswith(".safetensors")
                    for key, name in mapping.items()
                ):
                    raise ValueError("Corrupt full checkpoint shard path")
                weights = sorted(set(mapping.values()))
                required.append(index_path.name)
            else:
                weights = ["model.safetensors"]
            if {p.name for p in checkpoint.glob("model*.safetensors")} != set(weights):
                raise ValueError("Corrupt full checkpoint: missing or unindexed backbone shard")
        required.extend(weights)
        if self.dtype == self.torch.float16:
            required.append("scaler.pt")
        for name in required:
            if (
                (checkpoint / name).resolve().parent != checkpoint.resolve()
                or not (checkpoint / name).is_file()
                or (checkpoint / name).stat().st_size == 0
            ):
                raise ValueError(f"Corrupt checkpoint {checkpoint}: missing/empty {name}")
        checkpoint_configs(checkpoint, self.config.finetuning)
        state = parse_json((checkpoint / "trainer_state.json").read_text())
        step = state["global_step"]
        if (
            type(step) is not int
            or step < 1
            or type(state.get("max_steps")) is not int
            or not step <= state["max_steps"]
            or checkpoint.name != f"checkpoint-{step}"
        ):
            raise ValueError(f"Corrupt checkpoint step: {checkpoint}")
        history = state.get("log_history", [])
        if not isinstance(history, list) or not any(
            isinstance(entry, dict)
            and type(entry.get("loss")) in (int, float)
            and math.isfinite(entry["loss"])
            for entry in history
        ):
            raise ValueError(
                f"Corrupt checkpoint: missing finite training loss history: {checkpoint}"
            )
        optimizer = self.torch.load(
            checkpoint / "optimizer.pt", map_location="cpu", weights_only=True
        )
        scheduler = self.torch.load(
            checkpoint / "scheduler.pt", map_location="cpu", weights_only=True
        )
        if (
            not optimizer.get("state")
            or not optimizer.get("param_groups")
            or scheduler.get("last_epoch") != step
        ):
            raise ValueError(f"Corrupt optimizer/scheduler state: {checkpoint}")
        self._checkpoint_audits[str(checkpoint)] = {
            "global_step": step,
            "optimizer": optimizer_summary(optimizer, self.torch),
            "scheduler_sha256": state_fingerprint(scheduler, self.torch),
        }
        if self.dtype == self.torch.float16:
            scaler = self.torch.load(
                checkpoint / "scaler.pt", map_location="cpu", weights_only=True
            )
            scale = scaler.get("scale") if isinstance(scaler, dict) else None
            if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
                raise ValueError(f"Corrupt scaler state: {checkpoint}")
        # RNG checkpoint contains NumPy/Python state objects in the official Trainer format.
        for name in rng_files:
            rng = self.torch.load(checkpoint / name, map_location="cpu", weights_only=False)
            if not {"python", "numpy", "cpu", "cuda"} <= set(rng):
                raise ValueError(f"Corrupt RNG state: {checkpoint / name}")
        from safetensors import safe_open

        for name in (*weights, "joint_head.safetensors"):
            with safe_open(checkpoint / name, framework="pt", device="cpu") as tensors:
                if not tensors.keys():
                    raise ValueError(f"Empty {name}")
                for key in tensors.keys():
                    tensors.get_tensor(key)
                if (
                    self.config.finetuning == "full"
                    and name in weights
                    and (checkpoint / "model.safetensors.index.json").is_file()
                ):
                    expected = {key for key, shard in mapping.items() if shard == name}
                    if not expected <= set(tensors.keys()):
                        raise ValueError(f"Corrupt shard tensor mapping: {name}")
        if require_marker:
            marker = json.loads((checkpoint / "dforge_checkpoint.json").read_text())
            if (self.config.finetuning == "full" or world > 1) and not {
                "finetuning",
                "world_size",
                "run_identity",
                "files",
                "global_step",
            } <= set(marker):
                raise ValueError("Corrupt checkpoint identity metadata")
            if (
                marker.get("finetuning", "lora") != self.config.finetuning
                or marker.get("world_size", 1) != world
            ):
                raise ValueError("Checkpoint finetuning/world_size differs from configured run")
            if marker.get("run_identity") != self._run_identity() and "run_identity" in marker:
                raise ValueError("Checkpoint run/data identity differs from configured run")
            if any(
                not isinstance(name, str) or Path(name).name != name for name in marker["files"]
            ):
                raise ValueError("Corrupt checkpoint marker file paths")
            if (
                not set(required) <= set(marker["files"])
                or marker["global_step"] != step
                or any(
                    not (checkpoint / n).is_file() or (checkpoint / n).stat().st_size != size
                    for n, size in marker["files"].items()
                )
            ):
                raise ValueError(f"Corrupt committed checkpoint: {checkpoint}")
        return step

    def load_final_checkpoint(self, checkpoint):
        self.validate_checkpoint(checkpoint)
        self.trainer._load_from_checkpoint(str(checkpoint))
        from transformers.trainer_callback import TrainerState

        self.trainer.state = TrainerState.load_from_json(
            str(Path(checkpoint) / "trainer_state.json")
        )
        # Exact checkpoints save the unfurled temperature state separately from the head.
        self.model.decision_config = json.loads(
            (Path(checkpoint) / "unsloth_decision_config.json").read_text()
        )

    def evaluate(self, role, phase):
        def evaluate(items):
            return self.api.evaluate(
                self.model, self.tokenizer, items, batch_size=self.config.training.batch_size
            )

        raw = evaluate(self.items[role])
        by_type = {}
        for kind, questions in self.encoding[role]["types"].items():
            subset = type_items(self.items[role], kind)
            by_type[kind] = {"questions": questions, "items": len(subset), "raw": evaluate(subset)}
        return {
            "role": role,
            "phase": phase,
            "source": "Unsloth FastDecisionModel.evaluate",
            "types": self.encoding[role]["types"],
            "questions": self.encoding[role]["questions"],
            "items": len(self.items[role]),
            "raw": raw,
            "by_type": by_type,
            "meaning": "teacher target agreement; not human correctness",
        }

    def calibrate(self):
        before = json.loads(json.dumps(self.model.decision_config))
        raw = self.api.calibrate(
            self.model,
            self.tokenizer,
            self.items["calibration"],
            batch_size=self.config.training.batch_size,
        )
        from unsloth.models.decision import _laya

        adjustment = retain_unfitted_calibration(
            before,
            self.model.decision_config,
            raw.get("fitted_types", []),
            _laya().common.clamp_temperature,
        )
        fitted = [TYPES[i] for i in raw.get("fitted_types", [])]
        return {
            "raw": raw,
            "relative_temperature_fitted_types": fitted,
            "relative_temperature_fallback_types": [t for t in TYPES if t not in fitted],
            "head_temperature_fitted": not adjustment["shared_head_temperature_retained"],
            "preservation_adjustment": adjustment,
            "usable_questions": self.encoding["calibration"]["questions"],
            "upstream_target": "Clef calibration uses modal teacher labels",
            "meaning": "fitting diagnostics (upstream may report cross-fitted metrics), not independent test",
        }

    def calibration_state(self):
        return self.model.decision_config

    def restore_calibration(self, state):
        self.model.decision_config = state

    def save(self):
        self.model.save_pretrained(str(self.directory / "model"), self.tokenizer)
        artifacts = {
            "model": "model",
            "final_checkpoint": os.path.relpath(
                Path(self.config.output.checkpoint_directory or self.directory / "checkpoints")
                / f"checkpoint-{self.trainer.state.global_step}",
                self.directory,
            ),
        }
        if self.config.output.save_merged:
            self.model.save_pretrained_merged(
                str(self.directory / "merged"), self.tokenizer, save_method="merged_16bit"
            )
            artifacts["merged"] = "merged"
        return artifacts

    def reload_probe(self, data):
        requests, covered = [], set()
        for r in data.records:
            types = {q.type for q in r.request.questions.values()}
            if types - covered:
                requests.append(r.request.model_dump(exclude_none=True))
                covered.update(types)
            if covered == set(TYPES):
                break
        result = {}
        calibration = json.loads(json.dumps(self.model.decision_config))
        # Release training allocations before loading saved models on a single GPU.
        self.trainer = self.model = self.tokenizer = None
        gc.collect()
        self.torch.cuda.empty_cache()
        for folder in ("model", "merged") if self.config.output.save_merged else ("model",):
            model, tokenizer = self.api.from_pretrained(
                str(self.directory / folder),
                dtype=self.dtype,
                load_in_4bit=self.config.model.load_in_4bit,
                local_files_only=True,
                use_gradient_checkpointing=False,
            )
            verify_calibration_reload(calibration, model.decision_config)
            predictions = [
                native_prediction(
                    self.api.predict(model, tokenizer, **request), request["questions"]
                )
                for request in requests
            ]
            result[folder] = {
                "types": sorted(covered),
                "predictions": predictions,
                "decision_config": model.decision_config,
                "protocol_valid": True,
                "calibration_restored": True,
            }
            del model, tokenizer
            gc.collect()
            self.torch.cuda.empty_cache()
        write_json(self.directory / "reload_predictions.json", result)
        return result

    def close(self):
        self.trainer = self.model = self.tokenizer = None
        self.items.clear()
        gc.collect()
        if hasattr(self, "torch"):
            self.torch.cuda.empty_cache()
