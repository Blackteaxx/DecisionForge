"""Shared native trainer orchestration for explicitly registered new decision routes."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from decisionforge.training.backends.swift import (
    SwiftBackend,
    artifact_identity,
    validate_artifact,
    validate_data,
)
from decisionforge.training.backends.swift_runtime import encode_check, installation
from decisionforge.training.backends.swift_task import map_request
from decisionforge.util import write_json

from . import get_route
from .artifacts import source_roots, sources, write_manifest
from .sdk import PROCESSOR_FILES, head_environment


class RouteBackend(SwiftBackend):
    @property
    def route(self):
        return get_route(self.settings.model_type)

    def environment(self):
        try:
            sdk = installation()
        except Exception as exc:
            sdk = {"validation_error": str(exc)}
        return {
            "backend": "swift",
            "model_type": self.route.name,
            "upstream": sdk,
            **sources(self.route),
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
            row_offset, question_offset = len(rows), len(mappings)
            rows.extend(items)
            mappings.extend(
                {
                    **item,
                    "item_index": question_offset + i,
                    "row_index": row_offset + item["row_index"],
                    "group_id": record.group_id,
                    "role": role,
                }
                for i, item in enumerate(mapping)
            )
        path = self.data_dir / f"{role}.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows)
        )
        write_json(
            self.data_dir / f"{role}.mapping.json",
            {"version": "decisionforge.swift.mapping.v2", "rows": mappings},
        )
        return path, mappings

    def prepare(self, data):
        validate_data(self.config, data)
        self.data = data
        rows = [
            row for record in data.records for row in map_request(record.request, self.settings)[0]
        ]
        encoding = encode_check(self.config, rows)
        encoded = {}
        for role, records in data.roles.items():
            path, mapping = self._write_rows(role, records)
            encoded[role] = {
                "path": str(path),
                "rows": sum(bool(line.strip()) for line in path.open()),
                "questions": len(mapping),
                "mapping": str(self.data_dir / f"{role}.mapping.json"),
            }
        import torch

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ValueError(
                "Real Swift training requires an assigned BF16 CUDA device; use validate-only on CPU"
            )
        self.model_root, self.source_adapter = source_roots(self.route)
        parent = self.settings.init_checkpoint
        if parent:
            validate_artifact(parent, model_type=self.route.name)
            info = json.loads((Path(parent) / "dforge_swift_artifact.json").read_text())
            if info["finetuning"] != self.config.finetuning:
                raise ValueError(
                    "Swift init artifact finetuning mode differs; no implicit conversion"
                )
        self.adapter = (
            Path(parent) if parent and self.config.finetuning == "lora" else self.source_adapter
        )
        self.start_model = (
            Path(parent) if parent and self.config.finetuning == "full" else self.model_root
        )
        self.plugin = str(Path(__file__).parents[1] / "swift_plugin.py")
        self.lineage = {
            **sources(self.route),
            "model_type": self.route.name,
            "parent_artifact": artifact_identity(parent, model_type=self.route.name)
            if parent
            else None,
            "probability_contract": self.route.probability_contract["version"],
            "usage": self.settings.usage,
            "plugin": self.plugin,
        }
        write_json(self.directory / "swift_lineage.json", self.lineage)
        return {
            "model": {
                "actual_name": str(self.start_model),
                "actual_revision": self.route.revision,
                "model_type": self.route.name,
                "source": sources(self.route),
                "adapter": str(self.adapter) if self.adapter else None,
            },
            "rows": encoded,
            "encoding": encoding,
            "plugin": self.plugin,
        }

    def _cmd(self):
        original = super()._cmd()
        args = dict(zip(original[3::2], original[4::2]))
        args.update(
            {
                "--model_type": self.route.name,
                "--template": self.route.template,
                "--model": str(getattr(self, "start_model", self.config.model.name)),
                "--loss_type": self.route.loss,
                "--tuner_type": self.config.finetuning,
                "--gradient_checkpointing": str(
                    self.config.training.gradient_checkpointing
                ).lower(),
                "--packing": "false",
                "--padding_free": "false",
                "--max_pixels": str(self.settings.max_pixels),
            }
        )
        if self.adapter:
            args["--adapters"] = str(self.adapter)
        else:
            args.pop("--adapters", None)
            if self.config.finetuning == "lora":
                args.update(
                    {
                        "--target_modules": "all-linear",
                        "--lora_rank": str(self.config.lora.rank),
                        "--lora_alpha": str(self.config.lora.alpha),
                        "--lora_dropout": str(self.config.lora.dropout),
                    }
                )
        return original[:3] + [value for pair in args.items() for value in pair]

    def train(self, checkpoint=None):
        if checkpoint is not None:
            raise ValueError("Swift complete state resume is unsupported; initialize a new run")
        head_environment(
            self.route, self.model_root, self.source_adapter, artifact=self.settings.init_checkpoint
        )
        env = {
            **os.environ,
            "DFORGE_SWIFT_MODEL_TYPE": self.route.name,
            "DFORGE_SWIFT_FINETUNING": self.config.finetuning,
            "DFORGE_SWIFT_USAGE": self.settings.usage or "",
            "DFORGE_SWIFT_DATA": str(self.data_dir),
            "DFORGE_SWIFT_CALIBRATE": "1" if self.config.calibration.enabled else "0",
            "DFORGE_SWIFT_LINEAGE": str(self.directory / "swift_lineage.json"),
            "SWIFT_OUTPUT_DIR": str(self.output_dir),
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
        receipt = json.loads((self.output_dir / "dforge_training.json").read_text())
        receipt.update(json.loads((self.output_dir / "dforge_update_evidence.json").read_text()))
        self.checkpoint = Path(receipt["last_model_checkpoint"])
        if not self.checkpoint.is_absolute():
            self.checkpoint = self.output_dir / self.checkpoint
        for owner, relative, exported, _ in self.route.assets:
            source = self.source_adapter if owner == "adapter" else self.model_root
            shutil.copy2(source / relative, self.checkpoint / exported)
        for name in PROCESSOR_FILES:
            source = self.model_root / name
            if source.is_file() and not (self.checkpoint / name).exists():
                shutil.copy2(source, self.checkpoint / name)
        if self.source_adapter:
            for name, dest in (("README.md", "ADAPTER_README.md"), ("LICENSE", "ADAPTER_LICENSE")):
                source = self.source_adapter / name
                if not source.exists():
                    source = self.source_adapter.parent / name
                if source.is_file():
                    shutil.copy2(source, self.checkpoint / dest)
        write_manifest(self.checkpoint, self.route, self.config.finetuning, self.lineage)
        validate_artifact(self.checkpoint, model_type=self.route.name)
        self.training_report = receipt
        return receipt

    def save(self):
        target = self.directory / "model"
        shutil.copytree(
            self.checkpoint,
            target,
            ignore=shutil.ignore_patterns(
                "optimizer.pt", "scheduler.pt", "rng_state*.pth", "training_args.bin"
            ),
        )
        prior = (target / "NOTICE").read_text() if (target / "NOTICE").exists() else ""
        (target / "NOTICE").write_text(
            f"DecisionForge continued-training derivative: Swift {self.route.name}.\n"
            f"Published source: {self.route.model}@{self.route.revision}.\n"
            "Backbone/adapter and decision-head parameters were continued with the pinned native trainer.\n"
            "Original loader assets, source licenses and adapter cards are retained; see swift_lineage.json.\n"
            + (
                "StartLux weights: CC-BY-NC-4.0; noncommercial research use only.\n"
                if self.route.noncommercial
                else ""
            )
            + prior
        )
        write_manifest(target, self.route, self.config.finetuning, self.lineage)
        validate_artifact(target, model_type=self.route.name)
        self.exported = target
        return {
            "format": "swift_decision",
            "model_type": self.route.name,
            "model": str(target),
            "lineage": str(target / "swift_lineage.json"),
        }
