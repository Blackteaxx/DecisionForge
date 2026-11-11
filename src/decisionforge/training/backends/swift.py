"""Experimental ms-swift OmniJev backend.

The parent runner owns snapshots, holdouts, reports, and file locks. This module
only translates the canonical v3 records and invokes the pinned upstream CLI.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from decisionforge.training.backends.swift_task import (
    ADAPTER_REVISION,
    BASE_REVISION,
    SWIFT_REVISION,
    diagnostics,
    keys_for,
    map_request,
)
from decisionforge.util import write_json


def validate_artifact(path, *, model_type=None):
    root = Path(path).expanduser().resolve()
    marker = root / "dforge_swift_artifact.json"
    if (
        marker.is_file()
        and json.loads(marker.read_text()).get("schema_version") == "decisionforge.swift.decision.v1"
    ) or model_type not in (None, "omnijev"):
        from decisionforge.training.backends.swift_routes.artifacts import validate

        return validate(path, model_type)
    required = ("adapter_config.json", "head.pt", "ord.pt", "head_meta.json", "new_tok_emb.pt")
    if not root.is_dir() or any(not (root / f).is_file() for f in required):
        raise ValueError(f"Not an OmniJev/ms-swift adapter artifact: {root}")
    marker = root / "dforge_swift_artifact.json"
    if marker.is_file():
        manifest = json.loads(marker.read_text())
        if manifest.get("schema_version") != "decisionforge.swift.omnijev.v1":
            raise ValueError("Unsupported ms-swift artifact version")
        required_trained = {
            "adapter_model.safetensors",
            "scoring_head.safetensors",
            "scoring_head.json",
        }
        if not required_trained <= set(manifest.get("files", {})):
            raise ValueError("Incomplete trained adapter/scoring-head artifact")
        lineage = manifest.get("lineage", {})
        if lineage.get("base_model") != {"name": "Qwen/Qwen3.5-4B", "revision": BASE_REVISION}:
            raise ValueError("ms-swift artifact base lineage is incompatible")
        if (
            lineage.get("adapter") != {"name": "tinnel123/OmniJev", "revision": ADAPTER_REVISION}
            or lineage.get("ms_swift") != SWIFT_REVISION
        ):
            raise ValueError(
                "ms-swift artifact adapter/upstream lineage is incomplete or incompatible"
            )
        for name, expected in manifest["files"].items():
            candidate = (root / name).resolve()
            if (
                candidate.parent != root
                or not candidate.is_file()
                or hashlib.sha256(candidate.read_bytes()).hexdigest() != expected
            ):
                raise ValueError(f"ms-swift artifact hash mismatch: {name}")
    elif not (root / "release_manifest.json").is_file():
        raise ValueError("Original OmniJev requires its published release manifest")
    else:
        # The released adapter-only checkpoint's core weight digests are anchored
        # independently of any caller-supplied release manifest.
        originals = {
            "adapter_model.safetensors": "96247914d7e2e68a7124c35c49393267c2beb98769d096f5172ed0043851f0f0",
            "head.pt": "7b801a1901040e70559deb74eadb11709540ff65ef926213ecf881e4f4070320",
            "ord.pt": "38c8e1bdb23f8ccd915dba75e05596d36fe2896f95c984166fc3713a027f5b68",
            "new_tok_emb.pt": "cd78e3440c03c7ace689ddb3b05f860c5cc59b710a8202cbc8f8252de1cd18de",
            "head_meta.json": "d184edd664ad85bbbb782157a8965db33c36d4a144d8b781d19f5c11cd2778ee",
        }
        for name, expected in originals.items():
            if (
                not (root / name).is_file()
                or hashlib.sha256((root / name).read_bytes()).hexdigest() != expected
            ):
                raise ValueError(f"Original OmniJev pinned weight hash mismatch: {name}")
    return root


def artifact_identity(path, *, model_type=None):
    root = validate_artifact(path, model_type=model_type)
    if not root.is_dir():
        from decisionforge.training.backends.swift_routes import get_route
        from decisionforge.training.backends.swift_routes.artifacts import sources

        route = get_route(model_type)
        return {"model_type": route.name, **sources(route)}
    marker = root / "dforge_swift_artifact.json"
    if marker.is_file():
        manifest = json.loads(marker.read_text())
        if manifest.get("schema_version") == "decisionforge.swift.decision.v1":
            # validate_artifact already streamed every file. Do not allocate a
            # multi-GB backbone shard merely to compute the run fingerprint.
            return {"path": str(root), "manifest": manifest}
    return {
        "path": str(root),
        "files": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.iterdir()
            if p.is_file()
            and p.suffix in {".json", ".safetensors", ".pt"}
            and p.name not in {"optimizer.pt", "scheduler.pt"}
        },
    }


def validate_data(config, data):
    from decisionforge.training.backends.swift_routes import get_route

    route = get_route(config.swift.model_type)
    if not data.roles["train"]:
        raise ValueError("swift requires nonempty train split")
    for role, records in data.roles.items():
        # map_request performs all media, option, marker and score restrictions.
        for record in records:
            map_request(
                record.request,
                config.swift,
                root=Path(config.data.path).parent,
                answers=record.supervision.answers,
                record_id=record.record_id,
            )
    return {
        "model_type": route.name,
        "records": len(data.records),
        "swift_revision": SWIFT_REVISION,
    }


class SwiftBackend:
    def __new__(cls, config, directory):
        if cls is SwiftBackend and config.swift.model_type != "omnijev":
            from decisionforge.training.backends.swift_routes.backend import RouteBackend

            return object.__new__(RouteBackend)
        return object.__new__(cls)

    def __init__(self, config, directory):
        self.config, self.directory = config, Path(directory)
        self.settings = config.swift
        self.data_dir = self.directory / "swift_data"
        self.output_dir = self.directory / "swift_output"
        self.adapter = None
        self.training_report = {}
        self.runtime = None
        self.calibrated = False

    def environment(self):
        from decisionforge.training.backends.swift_runtime import installation

        try:
            deps = installation()
        except Exception as exc:
            deps = {"validation_error": str(exc), "expected_revision": SWIFT_REVISION}
        return {
            "backend": "swift",
            "model_type": self.settings.model_type,
            "upstream": deps,
            "base_model": {"name": self.config.model.name, "revision": BASE_REVISION},
            "adapter": {"name": self.settings.adapter, "revision": self.settings.adapter_revision},
        }

    def _write_rows(self, role, records):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        rows, mappings = [], []
        for record in records:
            items, mapping = map_request(
                record.request,
                self.settings,
                root=Path(self.config.data.path).parent,
                answers=record.supervision.answers,
                record_id=record.record_id,
            )
            rows.extend(items)
            offset = len(mappings)
            mappings.extend(
                {**item, "item_index": offset + i, "group_id": record.group_id, "role": role}
                for i, item in enumerate(mapping)
            )
        path = self.data_dir / f"{role}.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        write_json(
            self.data_dir / f"{role}.mapping.json",
            {"version": "decisionforge.swift.mapping.v1", "rows": mappings},
        )
        return path, mappings

    def prepare(self, data):
        validate_data(self.config, data)
        from decisionforge.training.backends.swift_runtime import encode_check

        # Check every split before allocating a GPU model. Native dataset loading is
        # strict as well; no overlength row can be silently removed downstream.
        all_rows = [
            row for record in data.records for row in map_request(record.request, self.settings)[0]
        ]
        encoding = encode_check(self.config, all_rows)
        encoded = {}
        for role, records in data.roles.items():
            path, mappings = self._write_rows(role, records)
            encoded[role] = {
                "path": str(path),
                "rows": len(mappings),
                "mapping": str(self.data_dir / f"{role}.mapping.json"),
            }
        plugin = str(Path(__file__).with_name("swift_plugin.py"))
        self.data = data
        self.adapter = (
            validate_artifact(self.settings.init_checkpoint)
            if self.settings.init_checkpoint
            else None
        )
        from decisionforge.training.backends.swift_runtime import adapter_dir

        if self.adapter is None:
            self.adapter = adapter_dir(self.settings)
        self.plugin = plugin
        write_json(
            self.directory / "swift_lineage.json",
            {
                "base_model": {"name": self.config.model.name, "revision": BASE_REVISION},
                "adapter": {
                    "name": self.settings.adapter,
                    "revision": self.settings.adapter_revision,
                },
                "ms_swift": SWIFT_REVISION,
                "parent_artifact": artifact_identity(self.settings.init_checkpoint)
                if self.settings.init_checkpoint
                else None,
                "plugin": plugin,
                "probability_contract": "decisionforge.omnijev.probabilities.v1",
            },
        )
        return {
            "model": {
                "actual_name": self.config.model.name,
                "actual_revision": BASE_REVISION,
                "adapter": str(self.adapter),
                "adapter_revision": self.settings.adapter_revision,
            },
            "rows": encoded,
            "plugin": plugin,
            "encoding": encoding,
        }

    def _cmd(self):
        t = self.config.training
        args = {
            "external_plugins": self.plugin,
            "callbacks": "dforge_update_evidence",
            "optim": "adamw_torch",
            "model": self.config.model.name,
            "model_revision": self.config.model.revision,
            "model_type": "omnijev",
            "task_type": "decision",
            "adapters": str(self.adapter),
            "dataset": "dforge_train",
            "loss_type": "ce_rps",
            "torch_dtype": "bfloat16",
            "attn_impl": "eager",
            "freeze_vit": str(self.settings.freeze_vit).lower(),
            "gradient_checkpointing": "false",
            # Swift globally overrides checkpoint(..., use_reentrant=False)
            # even with layer checkpointing off. Its default reentrant path
            # disconnects OmniJev's token-ID-only branch from the LoRA graph.
            "gradient_checkpointing_kwargs": json.dumps({"use_reentrant": False}),
            "max_length": self.config.model.max_seq_length,
            "decision_subsample_options": 0,
            "decision_drop_instructions": 0,
            "decision_shuffle_fields": "false",
            "decision_state_as_text": 0,
            "output_dir": str(self.output_dir),
            "split_dataset_ratio": 0,
            "add_version": "false",
            "dataloader_num_workers": 0,
            "learning_rate": t.learning_rate,
            "per_device_train_batch_size": t.batch_size,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": t.gradient_accumulation_steps,
            "logging_steps": t.logging_steps,
            "save_strategy": "steps" if t.save_steps else "epoch",
            "warmup_steps": t.warmup_steps,
            "weight_decay": t.weight_decay,
            "lr_scheduler_type": t.lr_scheduler_type,
            "seed": t.seed,
            "data_seed": t.seed,
            "eval_strategy": "no",
            "report_to": "none",
            "create_checkpoint_symlink": "false",
            "remove_unused_columns": "false",
            # Sources are immutable revisions; querying whether a ModelScope
            # checkout is "latest" is unrelated and breaks offline execution.
            "check_model": "false",
            "strict": "true",
            "load_best_model_at_end": "false",
        }
        if t.save_steps:
            args["save_steps"] = t.save_steps
        args["max_steps" if t.max_steps else "num_train_epochs"] = t.max_steps or t.num_train_epochs
        if self.config.calibration.enabled:
            args["val_dataset"] = "dforge_calibration"
        return [
            sys.executable,
            "-m",
            "decisionforge.training.backends.swift_train",
            *[str(v) for key, value in args.items() for v in ("--" + key, value)],
        ]

    def train(self, checkpoint=None):
        if checkpoint is not None:
            raise ValueError("swift complete state resume is unsupported; initialize a new run")
        from decisionforge.training.backends.swift_runtime import head_environment

        head_environment(self.adapter)
        env = {
            **os.environ,
            "DFORGE_SWIFT_DATA": str(self.data_dir),
            "DFORGE_SWIFT_CALIBRATE": "1" if self.config.calibration.enabled else "0",
            "SWIFT_OUTPUT_DIR": str(self.output_dir),
            "DFORGE_SWIFT_LINEAGE": str(self.directory / "swift_lineage.json"),
        }
        command = self._cmd()
        write_json(self.directory / "swift_command.json", command)
        with (
            (self.directory / "swift.stdout.log").open("w") as out,
            (self.directory / "swift.stderr.log").open("w") as err,
        ):
            result = subprocess.run(command, env=env, stdout=out, stderr=err, check=False)
        if result.returncode:
            raise ValueError(f"ms-swift exited {result.returncode}; see swift.stderr.log")
        receipt_path = self.output_dir / "dforge_training.json"
        receipt = json.loads(receipt_path.read_text())
        receipt.update(json.loads((self.output_dir / "dforge_update_evidence.json").read_text()))
        self.checkpoint = Path(receipt["last_model_checkpoint"])
        if not self.checkpoint.is_absolute():
            self.checkpoint = self.output_dir / self.checkpoint
        for file in (
            "adapter_model.safetensors",
            "adapter_config.json",
            "scoring_head.safetensors",
            "scoring_head.json",
        ):
            if not (self.checkpoint / file).is_file():
                raise ValueError(f"Missing trained ms-swift checkpoint artifact: {file}")
        # The base loader needs original files before applying the trained head overlay.
        for file in ("head.pt", "ord.pt", "head_meta.json", "new_tok_emb.pt"):
            shutil.copy2(self.adapter / file, self.checkpoint / file)
        write_json(
            self.checkpoint / "swift_lineage.json",
            json.loads((self.directory / "swift_lineage.json").read_text()),
        )
        self.training_report = receipt
        return receipt

    def _runtime(self):
        if self.runtime is None:
            from decisionforge.training.backends.swift_runtime import Runtime

            self.runtime = Runtime(self.config, self.checkpoint)
        return self.runtime

    def evaluate(self, role, phase):
        runtime = self._runtime()
        predictions, details = [], []
        for record in self.data.roles[role]:
            public = record.request.model_dump(mode="json")
            raw = runtime.predict(public, calibrated=self.calibrated)
            for key, q in record.request.questions.items():
                a = raw["answers"][key]
                labels = keys_for(q.model_dump())
                p = (
                    [a["noul"], 1 - a["noul"]]
                    if q.type == "noul"
                    else [a["probabilities"][k] for k in labels]
                )
                target = record.supervision.answers[key]
                y = (
                    [target.noul, 1 - target.noul]
                    if q.type == "noul"
                    else [target.probabilities[k] for k in labels]
                )
                predictions.append((p, y))
            details.append({"record_id": record.record_id, "prediction": raw})
        write_json(self.directory / f"{role}.{phase}.predictions.json", details)
        return {"role": role, "phase": phase, **diagnostics(predictions)}

    def calibrate(self):
        path = self.output_dir / "dforge_calibration.json"
        if not path.is_file():
            raise ValueError("ms-swift did not complete independent calibration")
        self.calibrated = True
        return json.loads(path.read_text())

    def calibration_state(self):
        return {
            "owner": "ms-swift",
            "enabled": self.calibrated,
            "scoring_head": json.loads((self.checkpoint / "scoring_head.json").read_text()),
        }

    def restore_calibration(self, state):
        raise ValueError("swift complete state resume is unsupported")

    def save(self):
        from huggingface_hub import snapshot_download

        target = self.directory / "model"
        shutil.copytree(
            self.checkpoint,
            target,
            ignore=shutil.ignore_patterns(
                "optimizer.pt", "scheduler.pt", "rng_state*.pth", "training_args.bin"
            ),
        )
        # The adapter, head assets and saved processor are derived from the two
        # pinned releases. Carry their notices with the exported artifact.
        base = Path(
            snapshot_download(
                self.config.model.name, revision=self.config.model.revision, local_files_only=True
            )
        )
        shutil.copy2(self.adapter / "LICENSE", target / "LICENSE")
        shutil.copy2(base / "LICENSE", target / "QWEN_LICENSE")
        (target / "NOTICE").write_text(
            "Continued-training derivative of tinnel123/OmniJev on Qwen/Qwen3.5-4B.\n"
            "Adapter weights, scoring_head.* and the processor/tokenizer were saved by ms-swift; "
            "the adapter and decision head were updated, including independent temperature calibration when enabled.\n"
            "Original head.pt, ord.pt, head_meta.json and new_tok_emb.pt are retained as loader assets.\n"
            "See LICENSE, QWEN_LICENSE and swift_lineage.json for notices and pinned sources.\n",
            encoding="utf-8",
        )
        # The SDK has already stored the sole fitted temperature in scoring_head.json.
        write_json(
            target / "dforge_swift_artifact.json",
            {
                "schema_version": "decisionforge.swift.omnijev.v1",
                "lineage": json.loads((target / "swift_lineage.json").read_text()),
                "files": {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in target.iterdir()
                    if p.is_file()
                    and p.name not in {"optimizer.pt", "rng_state.pth", "scheduler.pt"}
                },
            },
        )
        self.exported = target
        return {
            "format": "swift_omnijev",
            "model": str(target),
            "lineage": str(target / "swift_lineage.json"),
        }

    def reload_probe(self, data):
        from decisionforge.training.backends.swift_runtime import Runtime

        validate_artifact(self.exported)
        samples = data.roles["validation"]
        if not samples:
            raise ValueError("swift reload probe requires an independent validation record")
        public = samples[0].request.model_dump(mode="json")
        before = self._runtime().predict(public, calibrated=self.calibrated)
        self.runtime.close()
        self.runtime = None
        reloaded = Runtime(self.config, self.exported)
        try:
            after = reloaded.predict(public, calibrated=self.calibrated)
            delta = 0.0
            for key, q in samples[0].request.questions.items():
                a, b = before["answers"][key], after["answers"][key]
                values = (
                    [abs(a["noul"] - b["noul"])]
                    if q.type == "noul"
                    else [
                        abs(a["probabilities"][k] - b["probabilities"][k])
                        for k in a["probabilities"]
                    ]
                )
                delta = max(delta, max(values))
            before_meta = json.loads((self.checkpoint / "scoring_head.json").read_text())
            after_meta = json.loads((self.exported / "scoring_head.json").read_text())
            if delta > 1e-5 or before_meta != after_meta:
                raise ValueError("swift exported/reloaded predictions or calibration differ")
            report = {
                "status": "passed",
                "max_abs_probability_delta": delta,
                "calibration_identical": True,
                "record_id": samples[0].record_id,
                "before": before,
                "after": after,
            }
            write_json(self.directory / "reload_probe.json", report)
            return report
        finally:
            reloaded.close()

    def close(self):
        if self.runtime is not None:
            self.runtime.close()
            self.runtime = None
