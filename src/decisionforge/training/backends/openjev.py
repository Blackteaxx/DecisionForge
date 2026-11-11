"""Pinned OpenJev CLI orchestration and CPU-safe two-stage artifact checks."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import sys
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path

from decisionforge.eval.checkpoint import file_identity, read_object
from decisionforge.label_tokens import label_protocol_hash
from decisionforge.training.backends.openjev_task import (
    LABELS,
    MAPPING_VERSION,
    MAPPING_VERSION_V2,
    PUBLIC_PROTOCOL,
    SAMPLING,
    TASK_NAME,
    check_prompt_budget,
    map_roles,
    option_keys,
    validate_labels,
)
from decisionforge.util import digest, write_json

UPSTREAM_REPO = "https://github.com/ZimmyGao/openjev-rlcd"
UPSTREAM_REVISION = "22db40ad3333f5cf0bc5139097da70e0f8c05018"
TRAIN_SHA256 = "d71fb77dab4b8276d241d5175dfe40881e8d355d48e20d5548c3243f0c02cba6"
SOURCE_SHA256 = {
    "train.py": TRAIN_SHA256,
    "estimators.py": "8213f1b0b585f268889ddbc92a15402e9dc91d78e89955b3e10667f7563d3b42",
    "tasks/__init__.py": "7adc650eec095716c24bd20f6715ce999eb68545045ca9dbba9a82b44ded1ab5",
}
ARTIFACT_VERSION = "decisionforge.openjev.artifact.v1"
ARTIFACT_VERSION_V2 = "decisionforge.openjev.artifact.v2"


def dependency_versions():
    result = {}
    for name in (
        "openjev-rlcd",
        "torch",
        "transformers",
        "datasets",
        "numpy",
        "pyarrow",
        "huggingface_hub",
        "tokenizers",
        "safetensors",
    ):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def validate_installation():
    try:
        package = distribution("openjev-rlcd")
        direct = json.loads(package.read_text("direct_url.json") or "{}")
    except PackageNotFoundError as exc:
        raise ValueError(
            "Missing OpenJev dependencies; run uv sync --extra train-openjev in a dedicated environment"
        ) from exc
    if direct.get("vcs_info", {}).get("commit_id") != UPSTREAM_REVISION:
        raise ValueError(
            "OpenJev installation must match the uv sync --extra train-openjev pinned revision"
        )
    for name, expected in SOURCE_SHA256.items():
        file = Path(package.locate_file(f"openjev_rlcd/{name}"))
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise ValueError(f"OpenJev {name} differs from the pinned unchanged upstream source")
    return {
        "upstream_repo": UPSTREAM_REPO,
        "upstream_revision": UPSTREAM_REVISION,
        "train_sha256": TRAIN_SHA256,
        "source_sha256": dict(SOURCE_SHA256),
        "source_verified": True,
    }


def inventory(root, paths=None):
    root = Path(root).resolve()
    result = {}
    for path in sorted(paths if paths is not None else root.rglob("*")):
        path = Path(path)
        if not path.is_file() or path.name == "openjev_artifact.json":
            continue
        before = path.stat()
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                hasher.update(block)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ino,
        ):
            raise ValueError(f"OpenJev file changed during identity computation: {path}")
        result[path.relative_to(root).as_posix()] = {
            "size": after.st_size,
            "sha256": hasher.hexdigest(),
        }
    return result


def validate_artifact(path):
    root = Path(path).expanduser().resolve()
    marker = root / "openjev_artifact.json"
    if not marker.is_file() or any(
        (root / name).exists()
        for name in ("adapter_config.json", "unsloth_decision_config.json", "rl_agent_config.json")
    ):
        raise ValueError(f"Not an unambiguous OpenJev artifact: {root}")
    manifest = read_object(marker)
    labels = manifest.get("labels")
    ids = manifest.get("label_token_ids")
    legacy = manifest.get("schema_version") == ARTIFACT_VERSION
    if legacy:
        valid_labels = (
            labels == LABELS
            and manifest.get("mapping_version") == MAPPING_VERSION
            and "label_protocol" not in manifest
        )
    else:
        protocol = manifest.get("label_protocol")
        valid_labels = (
            manifest.get("schema_version") == ARTIFACT_VERSION_V2
            and manifest.get("mapping_version") == MAPPING_VERSION_V2
            and isinstance(labels, list)
            and len(labels) >= 10
            and all(isinstance(label, str) and label for label in labels)
            and len(set(labels)) == len(labels)
            and labels[:10] == LABELS
            and isinstance(protocol, dict)
            and protocol.get("schema_version") == 1
            and protocol.get("encoding_style") == "space_prefixed"
            and protocol.get("legacy_count") == 10
            and protocol.get("labels") == labels
            and protocol.get("token_ids") == ids
            and protocol.get("encodings") == [" " + label for label in labels]
            and protocol.get("protocol_hash") == label_protocol_hash(protocol)
            and type(manifest.get("training_max_options")) is int
            and 2 <= manifest["training_max_options"] <= len(labels)
        )
    if (
        not valid_labels
        or manifest.get("artifact_format") != "openjev_rlcd"
        or manifest.get("upstream_revision") != UPSTREAM_REVISION
        or not isinstance(ids, list)
        or len(ids) != (10 if legacy else len(labels or []))
        or any(type(token) is not int or token < 0 for token in ids)
        or len(set(ids)) != len(ids)
        or manifest.get("public_protocol") != PUBLIC_PROTOCOL
        or manifest.get("full_resume") is not False
        or type(manifest.get("max_seq_length")) is not int
        or manifest["max_seq_length"] < 32
        or not isinstance(manifest.get("sampling"), dict)
        or any(manifest["sampling"].get(k) != v for k, v in SAMPLING.items())
        or type(manifest["sampling"].get("max_new_tokens")) is not int
        or manifest["sampling"]["max_new_tokens"] < 1
    ):
        raise ValueError("Incompatible OpenJev artifact version, method or sampling settings")
    files = manifest.get("files")
    required = {
        "backbone/config.json",
        "backbone/generation_config.json",
        "backbone/tokenizer_config.json",
        "backbone/tokenizer.json",
        "stages/calibrate/weights.pt",
        "stages/reinforce/weights.pt",
    }
    if not isinstance(files, dict) or not required <= set(files):
        raise ValueError("Incomplete OpenJev artifact inventory")
    for name, identity in files.items():
        if (
            not isinstance(name, str)
            or Path(name).is_absolute()
            or ".." in Path(name).parts
            or not (root / name).is_file()
        ):
            raise ValueError("Invalid OpenJev artifact file path")
        if not isinstance(identity, dict) or identity.get("size", 0) <= 0:
            raise ValueError("Invalid OpenJev artifact file identity")
    actual = inventory(root)
    if actual != files:
        raise ValueError("OpenJev artifact digest/identity changed or files are incomplete")
    cfg = read_object(root / "backbone/config.json")
    read_object(root / "backbone/tokenizer_config.json")
    read_object(root / "backbone/generation_config.json")
    if cfg.get("model_type") != "qwen3":
        raise ValueError("OpenJev currently supports the ordinary Qwen3 causal architecture")
    stages = manifest.get("stages")
    if (
        not isinstance(stages, list)
        or len(stages) != 2
        or any(not isinstance(stage, dict) for stage in stages)
        or [s.get("name") for s in stages] != ["calibrate", "reinforce"]
    ):
        raise ValueError("OpenJev stage ordering is invalid")
    for stage in stages:
        weight = f"stages/{stage['name']}/weights.pt"
        if (
            stage.get("weights") != weight
            or stage.get("sha256") != files[weight]["sha256"]
            or stage.get("size") != files[weight]["size"]
        ):
            raise ValueError("OpenJev stage weight identity is invalid")
        if (
            type(stage.get("actual_optimizer_steps")) is not int
            or stage["actual_optimizer_steps"] < 1
        ):
            raise ValueError("OpenJev stage lacks actual optimizer-step evidence")
        delta = stage.get("parameter_updates", {}).get("max_abs_delta")
        if type(delta) not in (int, float) or not math.isfinite(delta) or delta <= 0:
            raise ValueError("OpenJev stage lacks finite nonzero parameter-update evidence")
    if stages[1].get("initialization") != {"kind": "previous_stage", "sha256": stages[0]["sha256"]}:
        raise ValueError("OpenJev stage-2 lineage must reference this artifact's stage-1 weights")
    return root


def artifact_identity(path):
    root = validate_artifact(path)
    marker = root / "openjev_artifact.json"
    return {
        "artifact_format": "openjev_rlcd",
        "checkpoint": {
            "path": str(root),
            "files": {
                **read_object(marker)["files"],
                "openjev_artifact.json": {
                    "size": marker.stat().st_size,
                    "sha256": hashlib.sha256(marker.read_bytes()).hexdigest(),
                },
            },
        },
    }


def stage_arguments(config, stage, backbone, directory, initialization=None):
    if stage not in {"calibrate", "reinforce"} or stage == "reinforce" and initialization is None:
        raise ValueError("OpenJev reinforce stage requires this run's previous-stage checkpoint")
    settings, t = config.openjev, config.training
    steps = t.max_steps if stage == "calibrate" else settings.reinforce_steps or t.max_steps
    seed = (
        t.seed
        if stage == "calibrate"
        else settings.reinforce_seed
        if settings.reinforce_seed is not None
        else (t.seed + 1000) % 2**32
    )
    directory = Path(directory)
    args = [
        "--task",
        TASK_NAME,
        "--objective",
        "rlcd",
        "--readout-score",
        "brier",
        "--model",
        str(backbone),
        "--seed",
        str(seed),
        "--steps",
        str(steps),
        "--items-per-step",
        str(t.batch_size),
        "--m",
        str(settings.m),
        "--m-eval",
        str(settings.m_eval),
        "--max-new-tokens",
        str(settings.max_new_tokens),
        "--lr",
        str(t.learning_rate),
        "--eval-every",
        str(settings.eval_every or steps),
        "--dev-cap",
        str(settings.dev_cap),
        "--test-cap",
        str(settings.test_cap),
        "--kl-beta",
        "0.04",
        "--diversity-weight",
        "0",
        "--save-checkpoint",
        str(directory / "weights.pt"),
        "--output",
        str(directory / "upstream.json"),
    ]
    args += ["--no-score-function"] if stage == "calibrate" else ["--sf-coef", "0.3"]
    if initialization is not None:
        args += ["--init-checkpoint", str(initialization)]
    return args


class OpenJevBackend:
    def __init__(self, config, directory):
        self.config, self.directory = config, Path(directory).resolve()
        self.model_root = self.directory / "model"
        self.stages = []

    def environment(self):
        source = validate_installation()
        torch = importlib.import_module("torch")
        if (
            int(os.environ.get("WORLD_SIZE", "1")) != 1
            or not torch.cuda.is_available()
            or not torch.cuda.is_bf16_supported()
        ):
            raise ValueError("OpenJev requires one BF16-capable CUDA GPU without DDP")
        self.device = f"cuda:{torch.cuda.current_device()}"
        return {
            "backend": "openjev",
            **source,
            "dependencies": dependency_versions(),
            "python": sys.version,
            "executable": sys.executable,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "visible_device_count": torch.cuda.device_count(),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "device": self.device,
            "world_size": 1,
            "precision": "float32_train_parameters_bfloat16_autocast",
            "generator_precision": "bfloat16",
            "reference_precision": "bfloat16",
            "optimizer": "upstream_AdamW_float32_state",
            "full_resume": False,
        }

    def prepare(self, data):
        c = self.config
        path = Path(c.model.name).expanduser()
        if path.is_dir():
            self.backbone = path.resolve()
            # A local file identity is observed; a configured Hub revision is
            # only a declaration unless resolution actually occurred on the Hub.
            revision = None
        else:
            if not c.model.revision or len(c.model.revision) != 40:
                raise ValueError("OpenJev Hub backbone requires a pinned 40-character revision")
            hub = importlib.import_module("huggingface_hub")
            self.backbone = Path(
                hub.snapshot_download(repo_id=c.model.name, revision=c.model.revision)
            )
            revision = self.backbone.name
        if any(
            (self.backbone / name).exists()
            for name in (
                "adapter_config.json",
                "unsloth_decision_config.json",
                "rl_agent_config.json",
                "openjev_artifact.json",
            )
        ):
            raise ValueError(
                "OpenJev model must be an ordinary Qwen3 backbone; use openjev.init_checkpoint for compatible weights"
            )
        transformers = importlib.import_module("transformers")
        cfg = transformers.AutoConfig.from_pretrained(str(self.backbone), local_files_only=True)
        if cfg.model_type != "qwen3":
            raise ValueError("OpenJev currently supports the ordinary Qwen3 causal architecture")
        self.tok = transformers.AutoTokenizer.from_pretrained(
            str(self.backbone), local_files_only=True
        )
        from decisionforge.label_tokens import allocate_labels

        self.training_max_options = max(
            len(option_keys(question.model_dump(mode="json", exclude_none=True)))
            for records in data.roles.values()
            for record in records
            for question in record.request.questions.values()
        )
        self.label_protocol = allocate_labels(
            self.tok,
            max(10, self.training_max_options),
            encoding_style="space_prefixed",
            legacy_count=10,
        )
        ids = self.label_protocol["token_ids"]
        if (self.backbone / "generation_config.json").is_file():
            generation_config = transformers.GenerationConfig.from_pretrained(
                str(self.backbone), local_files_only=True
            )
        else:
            generation_config = transformers.GenerationConfig.from_model_config(cfg)
        self.generation_config = generation_config.to_dict()
        self.splits, self.mapping = map_roles(data, labels=self.label_protocol["labels"])
        upstream = importlib.import_module("openjev_rlcd.train")
        from decisionforge.training.backends.openjev_task import register_task

        register_task(upstream.TASKS, self.splits, labels=self.label_protocol["labels"])
        suffix = self.tok.encode("\nAnswer:", add_special_tokens=False)
        limit = min(c.model.max_seq_length, cfg.max_position_embeddings)
        for items in self.splits.values():
            for item in items:
                check_prompt_budget(
                    self.tok,
                    upstream.prompt(self.tok, item, TASK_NAME),
                    limit,
                    c.openjev.max_new_tokens,
                    suffix,
                )
        self.limit = limit
        source_identity = file_identity(self.backbone)
        tokenizer_identity = {
            name: identity
            for name, identity in source_identity["files"].items()
            if name
            in {
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "added_tokens.json",
                "vocab.json",
                "merges.txt",
                "chat_template.jinja",
            }
        }
        backbone_config = {
            k: v for k, v in cfg.to_dict().items() if k not in {"_name_or_path", "_commit_hash"}
        }
        self.source_identity = {
            "actual_name": c.model.name,
            "actual_revision": revision,
            "declared_revision": c.model.revision,
            "tokenizer_identity": tokenizer_identity,
            "source_metadata": source_identity,
            "initialization": "new_run_from_weights",
            "init_checkpoint": None,
            "label_token_ids": ids,
            "label_protocol": self.label_protocol,
        }
        self.initialization = None
        if c.openjev.init_checkpoint:
            root = validate_artifact(c.openjev.init_checkpoint)
            previous = read_object(root / "openjev_artifact.json")
            if (
                previous.get("backbone_config") != backbone_config
                or previous.get("label_token_ids") != ids
                or previous.get("labels") != self.label_protocol["labels"]
                or (
                    previous.get("label_protocol") is not None
                    and previous["label_protocol"] != self.label_protocol
                )
                or previous.get("generation_config") != self.generation_config
                or previous.get("backbone", {}).get("tokenizer_identity") != tokenizer_identity
            ):
                raise ValueError("OpenJev initialization backbone/tokenizer is incompatible")
            self.initialization = root / "stages/reinforce/weights.pt"
            self.source_identity["init_checkpoint"] = artifact_identity(root)
        self.model_root.mkdir(parents=True, exist_ok=True)
        cfg.save_pretrained(self.model_root / "backbone")
        generation_config.save_pretrained(self.model_root / "backbone")
        self.tok.save_pretrained(self.model_root / "backbone")
        write_json(
            self.directory / "openjev_items.json",
            {
                "schema_version": MAPPING_VERSION_V2,
                "label_protocol": self.label_protocol,
                "public_protocol": PUBLIC_PROTOCOL,
                "roles": self.mapping,
            },
        )
        write_json(self.directory / "openjev_splits.json", self.splits)
        self.backbone_config = backbone_config
        self.data_identity = digest({"splits": self.splits, "mapping": self.mapping})
        return {
            "model": self.source_identity,
            "roles": self.mapping,
            "public_protocol": PUBLIC_PROTOCOL,
            "effective_max_seq_length": limit,
        }

    def train(self, checkpoint=None):
        if checkpoint is not None:
            raise ValueError("OpenJev does not support complete state resume")
        from decisionforge.training.distributed import run_process

        initialization = self.initialization
        for stage in ("calibrate", "reinforce"):
            destination = self.model_root / "stages" / stage
            destination.mkdir(parents=True, exist_ok=True)
            args = stage_arguments(self.config, stage, self.backbone, destination, initialization)
            plan = {
                "argv": args,
                "splits": str(self.directory / "openjev_splits.json"),
                "evidence": str(destination / "updates.json"),
                "max_seq_length": self.limit,
                "label_protocol": getattr(self, "label_protocol", None),
                "expected_init_sha256": inventory(initialization.parent, [initialization])[
                    initialization.name
                ]["sha256"]
                if initialization
                else None,
            }
            write_json(destination / "plan.json", plan)
            run_process(
                [
                    sys.executable,
                    "-m",
                    "decisionforge.training.backends.openjev_stage",
                    "--plan",
                    str(destination / "plan.json"),
                ],
                destination,
                env={**os.environ, "OPENJEV_DEVICE": self.device},
            )
            evidence = read_object(destination / "updates.json")
            expected_steps = int(args[args.index("--steps") + 1])
            if (
                evidence.get("actual_optimizer_steps") != expected_steps
                or evidence.get("parameter_updates", {}).get("max_abs_delta", 0) <= 0
            ):
                raise ValueError(f"OpenJev {stage} lacks expected actual optimizer updates")
            weights = destination / "weights.pt"
            info = inventory(self.model_root, [weights])[
                weights.relative_to(self.model_root).as_posix()
            ]
            init = (
                {"kind": "previous_stage", "sha256": self.stages[0]["sha256"]}
                if stage == "reinforce"
                else {
                    "kind": "new_run_weights" if initialization else "backbone",
                    "source": self.source_identity,
                }
            )
            self.stages.append(
                {
                    "name": stage,
                    "weights": weights.relative_to(self.model_root).as_posix(),
                    **info,
                    "initialization": init,
                    **evidence,
                    "argv": args,
                    "upstream_output": json.loads((destination / "upstream.json").read_text()),
                }
            )
            initialization = weights
        self.training_report = {
            "method": "openjev_calibrate_then_reinforce",
            "stages": self.stages,
            "actual_optimizer_steps": sum(s["actual_optimizer_steps"] for s in self.stages),
            "actual_optimizer_steps_basis": "observed_original_AdamW_step_calls",
            "parameter_precision": "float32",
            "compute_precision": "bfloat16_autocast",
            "generator_reference_precision": "bfloat16",
            "independent_calibration": False,
            "full_resume": False,
        }
        return self.training_report

    def evaluate(self, role, phase):
        if role not in {"validation", "test"}:
            raise ValueError("OpenJev diagnostics accept only validation/test roles")
        output = self.stages[-1]["upstream_output"]
        metrics = output["curve"][-1]["dev"] if role == "validation" else output["test"]
        return {
            "role": role,
            "phase": phase,
            "questions": min(
                len(self.splits["dev" if role == "validation" else "test"]),
                self.config.openjev.dev_cap
                if role == "validation"
                else self.config.openjev.test_cap,
            ),
            "source": "openjev_rlcd.train.evaluate",
            "metrics": metrics,
            "stage": "reinforce",
            "independent_calibration": False,
            "sampling_rationales": self.config.openjev.m_eval,
        }

    def save(self):
        manifest = {
            "schema_version": ARTIFACT_VERSION_V2,
            "artifact_format": "openjev_rlcd",
            "upstream_repo": UPSTREAM_REPO,
            "upstream_revision": UPSTREAM_REVISION,
            "dependencies": dependency_versions(),
            "backbone": self.source_identity,
            "backbone_config": self.backbone_config,
            "generation_config": self.generation_config,
            "data_identity": self.data_identity,
            "config_identity": digest(self.config.model_dump()),
            "labels": self.label_protocol["labels"],
            "label_token_ids": validate_labels(self.tok, self.label_protocol["labels"]),
            "label_protocol": self.label_protocol,
            "training_max_options": self.training_max_options,
            "mapping_version": MAPPING_VERSION_V2,
            "public_protocol": PUBLIC_PROTOCOL,
            "sampling": {**SAMPLING, "max_new_tokens": self.config.openjev.max_new_tokens},
            "max_seq_length": self.limit,
            "stages": self.stages,
            "full_resume": False,
            "files": inventory(self.model_root),
        }
        write_json(self.model_root / "openjev_artifact.json", manifest)
        return {
            "format": "openjev_rlcd",
            "model": str(self.model_root),
            "full_resume": False,
            "identity": artifact_identity(self.model_root),
        }

    def reload_probe(self, data):
        from decisionforge.eval.adapters.openjev_worker import OpenJevRuntime

        runtime = OpenJevRuntime({"path": str(self.model_root), "device": self.device})
        try:
            record = data.roles["test"][0]
            focus = next(iter(record.request.questions))
            response = runtime.predict(
                record.request.model_dump(mode="json", exclude_none=True), focus=focus
            )
            from decisionforge.training_free.anyjev.processing import json_safe

            write_json(self.directory / "openjev_reload_probe.json", json_safe(response))
            if response.get("openjev", {}).get("validation_error") or not response.get(
                "answers", {}
            ).get(focus):
                raise ValueError(
                    "OpenJev reload probe returned an illegal probability distribution"
                )
            return {
                "source": "exported_OpenJevRuntime_official_Rationales_sample_forward",
                "record_id": record.record_id,
                "question_id": focus,
                "response": response,
                "artifact_format": "openjev_rlcd",
                "runtime": runtime.metadata["runtime"],
            }
        finally:
            runtime.close()

    def close(self):
        self.tok = None
