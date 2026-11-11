"""Pinned Laya APIs and official notebook DDP orchestration; no SDK import at module load."""

from __future__ import annotations

import copy
import importlib
import json
import math
import os
import random
import shutil
import tempfile
from collections import Counter
from datetime import timedelta
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path

from decisionforge.eval.checkpoint import file_identity, read_object
from decisionforge.training.data import to_unsloth_row
from decisionforge.util import write_json

UPSTREAM_REPO = "https://github.com/NandhaKishorM/laya"
UPSTREAM_REVISION = "b9fe351ce81e2d974bca2bf219c1e4030c0a05ea"
TYPE_NAMES = {0: "choice", 1: "score", 2: "noul"}


def dependency_versions():
    result = {}
    for name in (
        "laya",
        "torch",
        "transformers",
        "safetensors",
        "numpy",
        "huggingface_hub",
        "tokenizers",
    ):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def validate_artifact(path):
    root = Path(path).expanduser().resolve()
    if any(
        (root / name).exists()
        for name in ("unsloth_decision_config.json", "adapter_config.json", "openjev_artifact.json")
    ):
        raise ValueError(f"Not an unambiguous Laya artifact: {root}")
    for name in (
        "rl_agent_config.json",
        "model.safetensors",
        "encoder/config.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/tokenizer.json",
    ):
        file = root / name
        if not file.is_file() or not file.stat().st_size:
            raise ValueError(f"Invalid Laya artifact: missing/empty {file}")
    cfg = read_object(root / "rl_agent_config.json")
    read_object(root / "encoder/config.json")
    read_object(root / "tokenizer/tokenizer_config.json")
    temperature = cfg.get("temperature", [1, 1, 1])
    if (
        not isinstance(temperature, list)
        or len(temperature) != 3
        or any(type(t) not in (int, float) or not math.isfinite(t) or t <= 0 for t in temperature)
    ):
        raise ValueError("Invalid Laya temperature: expected three finite positive scalars")
    return root


def artifact_identity(path):
    root = validate_artifact(path)
    return {"artifact_format": "laya", "checkpoint": file_identity(root)}


def stage_source(source, destination):
    """The upstream tokenizer compatibility fix may write; isolate it from shared inputs."""
    root = validate_artifact(source)
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    for name in ("rl_agent_config.json",):
        shutil.copy2(root / name, target / name)
    for name in ("encoder", "tokenizer"):
        shutil.copytree(root / name, target / name, dirs_exist_ok=True)
    (target / "model.safetensors").symlink_to(root / "model.safetensors")
    return target


def encode_roles(data, tok, api, config, truncation_check):
    items, reports = {}, {}
    for role, records in data.roles.items():
        items[role], mapping, skipped, truncated = [], [], [], []
        if role == "calibration" and not config.calibration.enabled:
            reports[role] = {"mapping": [], "skipped": [], "truncated": [], "usable_questions": 0}
            continue
        for record in records:
            row = to_unsloth_row(record)
            for qid, question in record.request.questions.items():
                entry = {
                    "record_id": record.record_id,
                    "question_id": qid,
                    "group_id": record.group_id,
                    "role": role,
                    "option_keys": list(question.criteria)
                    if question.type == "choice"
                    else ["false", "true"]
                    if question.type == "noul"
                    else [str(i) for i in range(len(question.criteria))],
                    "source_target": row["gold"][qid]["probabilities"],
                }
                detail = truncation_check(record, qid)
                if detail:
                    truncated.append({**entry, "reason": detail})
                    if config.data.on_truncated == "error":
                        raise ValueError(
                            f"Laya record_id={record.record_id} question={qid}: truncated input: {detail}"
                        )
                encoded, reasons = api.items_from_rows(
                    tok,
                    [{**row, "gold": {qid: row["gold"][qid]}}],
                    max_len=config.model.max_seq_length,
                    head_max_len=config.laya.head_max_length,
                )
                if len(encoded) != 1:
                    reason = reasons or {"missing_item": 1}
                    skipped.append({**entry, "reason": reason})
                    if config.data.on_skipped == "error":
                        raise ValueError(
                            f"Laya record_id={record.record_id} question={qid}: {reason}"
                        )
                    continue
                target = [entry["source_target"][k] for k in entry["option_keys"]]
                if len(encoded[0]["target"]) != len(target) or any(
                    abs(a - b) > 1e-6 for a, b in zip(encoded[0]["target"], target)
                ):
                    raise ValueError(
                        f"Laya record_id={record.record_id} question={qid}: upstream changed soft target"
                    )
                mapping.append({**entry, "item_index": len(items[role])})
                items[role].append(encoded[0])
        reports[role] = {
            "mapping": mapping,
            "skipped": skipped,
            "truncated": truncated,
            "usable_questions": len(items[role]),
        }
    if not items["train"]:
        raise ValueError("Laya has no usable training items")
    if config.calibration.enabled and len(items["calibration"]) < 10:
        raise ValueError("Laya calibration needs at least 10 usable questions after encoding")
    return items, reports


def partition_items(items, mapping, world_size, rank, on_skipped):
    retained = len(items) // world_size * world_size
    if not retained:
        raise ValueError("Laya DDP needs at least one usable item per rank")
    dropped = mapping[retained:]
    if dropped and on_skipped == "error":
        raise ValueError(
            "Laya encoded train item count must be divisible by world_size; "
            "set data.on_skipped=allow to record official notebook tail trimming"
        )
    return items[:retained][rank::world_size], {
        "original_items": len(items),
        "retained_items": retained,
        "items_per_rank": retained // world_size,
        "dropped": dropped,
        "sharding": "official_notebook_equal_rank_stride",
    }


def calibration_types(records, temperatures):
    counts = Counter(r[0] for r in records)
    return {
        name: {"questions": counts[qt], "fitted": counts[qt] >= 10, "temperature": temperatures[qt]}
        for qt, name in TYPE_NAMES.items()
    }


def input_loss(tok, state, qid, question, api, common, max_len, head_max_len, *, record_id=None):
    """Audit official encoded spans against complete public tokenization; never change encoding."""
    q = api.to_internal(qid, question)
    labels = (
        list(question["criteria"])
        if q["t"] == "choice"
        else ["false", "true"]
        if q["t"] == "noul"
        else [str(i) for i in range(len(question["criteria"]))]
    )
    options = common.render_options(q)
    # Pinned upstream removes this literal before tokenizing every model-text span.
    # This is unsupported content, not length truncation: never let on_truncated=allow
    # approve an altered public request. Inspect the same renderings upstream consumes,
    # including JSON object keys, without rejecting non-rendered bookkeeping such as qid.
    fields = [("state", common.serialize_state(state)), ("instructions", str(q["ins"]))]
    fields.extend((f"criteria[{label!r}]", option) for label, option in zip(labels, options))
    for field, text in fields:
        if tok.mask_token in text:
            record = f" record_id={record_id}" if record_id is not None else ""
            raise ValueError(
                f"Laya{record} question={qid} field={field}: literal tokenizer.mask_token="
                f"{tok.mask_token!r} is unsupported; pinned upstream removes this text"
            )
    head, markers, _ = common.build_head(tok, q, head_max_len)
    _ids, _markers, _stats, state_stats = common.build_sequence(
        tok, state, q, max_len, head_max_len, return_stats=True, return_truncation_stats=True
    )
    instruction = common.encode_text(
        tok, f"{q['t']} question: {q['ins']}", add_special_tokens=False
    )["input_ids"]
    instruction_dropped = max(0, len(instruction) - (markers[0] - 2))
    option_dropped = {}
    for i, option in enumerate(options):
        complete = common.encode_text(
            tok, " " + option, add_special_tokens=False
        )["input_ids"]
        stop = markers[i + 1] if i + 1 < len(markers) else len(head) - 1
        lost = max(0, len(complete) - (stop - markers[i] - 1))
        if lost:
            option_dropped[labels[i]] = lost
    beyond = max(0, len(head) - max_len)
    if state_stats["truncated"] or instruction_dropped or option_dropped or beyond:
        return {
            "state_tokens_dropped": state_stats["state_tokens_dropped"],
            "instruction_tokens_dropped": instruction_dropped,
            "option_tokens_dropped": option_dropped,
            "head_tokens_beyond_max_len": beyond,
        }
    return None


class LayaBackend:
    def __init__(self, config, directory):
        self.config, self.directory = config, Path(directory)
        self.rank = int(os.environ.get("RANK", "0"))
        self.world = config.distributed.world_size
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.items, self.mapping = {}, {}
        self.cfg, self.calibration = {}, {"fitted": False, "source": "inherited_checkpoint"}
        self.model = self.tok = self.staging = None
        self.group_active = False

    def environment(self):
        try:
            self.torch = importlib.import_module("torch")
            self.api = importlib.import_module("laya.train")
            self.common = importlib.import_module("laya.common")
            self.calibration_api = importlib.import_module("laya.calibrate")
        except ImportError as exc:
            raise ValueError("Missing Laya dependencies; run uv sync --extra train-laya") from exc
        direct = json.loads(distribution("laya").read_text("direct_url.json") or "{}")
        if direct.get("vcs_info", {}).get("commit_id") != UPSTREAM_REVISION:
            raise ValueError(
                "Laya installation must match uv sync --extra train-laya pinned upstream revision"
            )
        required = (
            "load_checkpoint",
            "train_model",
            "items_from_rows",
            "rlcd_loss",
            "save_checkpoint",
            "calibration_records",
            "encode_item",
            "sigma_at",
        )
        if any(not callable(getattr(self.api, name, None)) for name in required):
            raise ValueError("Installed Laya lacks pinned training APIs")
        torch = self.torch
        if not torch.cuda.is_available():
            raise ValueError(
                "Laya training requires CUDA; static validate-only works without GPU SDKs"
            )
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)
        if self.world > 1:
            torch.distributed.init_process_group(
                "nccl", timeout=timedelta(seconds=self.config.distributed.timeout_seconds)
            )
            self.group_active = True
        environment = {
            "backend": "laya",
            "upstream_repo": UPSTREAM_REPO,
            "upstream_revision": UPSTREAM_REVISION,
            "dependencies": dependency_versions(),
            "python": os.sys.version,
            "cuda": torch.version.cuda,
            "rank": self.rank,
            "world_size": self.world,
            "device": str(self.device),
            "gpu": torch.cuda.get_device_name(self.local_rank),
            "precision": "float32_parameters_fp16_autocast",
            "save_dtype": "float16",
        }
        if self.world > 1:
            per_rank = [None] * self.world
            torch.distributed.all_gather_object(per_rank, dict(environment))
            environment["ranks"] = per_rank
        return environment

    def _source_calibration(self):
        raw = {
            "temperature": copy.deepcopy(self.cfg.get("temperature", [1, 1, 1])),
            "temperature_by_options": copy.deepcopy(self.cfg.get("temperature_by_options", {})),
        }
        self.cfg["temperature"] = [self.common.clamp_temperature(t) for t in raw["temperature"]]
        self.cfg["temperature_by_options"] = {
            k: self.common.clamp_temperature(t) for k, t in raw["temperature_by_options"].items()
        }
        self.calibration = {
            "fitted": False,
            "source": "inherited_checkpoint_upstream_clamped",
            "raw": raw,
            "applied": {
                "temperature": self.cfg["temperature"],
                "temperature_by_options": self.cfg["temperature_by_options"],
            },
        }

    def _truncation(self, record, qid):
        return input_loss(
            self.tok,
            record.request.state,
            qid,
            record.request.questions[qid].model_dump(exclude_none=True),
            self.api,
            self.common,
            self.config.model.max_seq_length,
            self.config.laya.head_max_length,
            record_id=record.record_id,
        )

    def prepare(self, data):
        c = self.config
        source = c.laya.init_checkpoint or c.model.name
        if Path(source).expanduser().is_dir():
            resolved, revision = validate_artifact(source), None
        else:
            if c.laya.init_checkpoint:
                raise ValueError("Laya init_checkpoint requires an existing local Laya artifact")
            hub = importlib.import_module("huggingface_hub")
            resolved = Path(
                hub.snapshot_download(
                    repo_id=source,
                    revision=c.model.revision,
                    allow_patterns=[
                        "rl_agent_config.json",
                        "model.safetensors",
                        "encoder/*",
                        "tokenizer/*",
                    ],
                )
            )
            revision = resolved.name if len(resolved.name) == 40 else c.model.revision
        identity = artifact_identity(resolved)
        self.staging = tempfile.TemporaryDirectory(prefix="dforge-laya-")
        stage = stage_source(resolved, Path(self.staging.name) / "source")
        self.model, self.tok, self.cfg = self.api.load_checkpoint(str(stage))
        source_cfg = copy.deepcopy(self.cfg)
        self._source_calibration()
        if self.cfg.get("option_layout", "sequential") != "sequential":
            raise ValueError(
                "Laya integration currently requires the upstream sequential option layout"
            )
        self.cfg.update(max_len=c.model.max_seq_length, head_max_len=c.laya.head_max_length)
        self.items, self.mapping = encode_roles(data, self.tok, self.api, c, self._truncation)
        self.train_items, self.partition = partition_items(
            self.items["train"],
            self.mapping["train"]["mapping"],
            self.world,
            self.rank,
            c.data.on_skipped,
        )
        self.before = {}
        for group, encoder in (("encoder", True), ("head", False)):
            params = [
                (n, p)
                for n, p in self.model.named_parameters()
                if p.requires_grad and n.startswith("encoder.") == encoder
            ]
            if not params:
                raise ValueError(f"Laya has no trainable {group} parameters")
            self.before[group] = {
                n: p.detach().flatten()[:256].float().cpu().clone() for n, p in params
            }
        self.model.to(self.device)
        if not self.rank:
            write_json(
                self.directory / "laya_items.json", {"roles": self.mapping, "ddp": self.partition}
            )
        self.source_identity = {
            "actual_name": source,
            "actual_revision": revision,
            "source_metadata": identity,
            "initialization": "new_run_from_weights",
            "init_checkpoint": c.laya.init_checkpoint,
            "checkpoint_config": source_cfg,
        }
        return {"model": self.source_identity, "roles": self.mapping, "distributed": self.partition}

    def _train_config(self):
        c, t, settings = self.config, self.config.training, self.config.laya
        return self.api.TrainConfig(
            epochs=int(t.num_train_epochs),
            micro_batch=t.batch_size,
            grad_accum=t.gradient_accumulation_steps,
            encoder_lr=t.learning_rate,
            head_lr=t.head_learning_rate,
            min_lr=settings.min_learning_rate,
            weight_decay=t.weight_decay,
            grad_clip=settings.gradient_clip,
            loss="rlcd",
            rl_samples=settings.rl_samples,
            sigma_start=settings.sigma_start,
            sigma_end=settings.sigma_end,
            w_sph=settings.w_sph,
            w_rps=settings.w_rps,
            seed=t.seed,
            max_len=c.model.max_seq_length,
            head_max_len=settings.head_max_length,
            amp=True,
            gradient_checkpointing=t.gradient_checkpointing,
            log_every=t.logging_steps,
            calib_max=0,
            calib_frac=0.0,
            calib_seed=t.seed,
        )

    def _epoch_save(self, epoch, loss):
        if not self.rank:
            self.api.save_checkpoint(
                self.model, self.tok, self.cfg, str(self.directory / "checkpoint_latest")
            )
            write_json(
                self.directory / "checkpoint_latest" / "checkpoint_meta.json",
                {
                    "epoch": epoch + 1,
                    "loss": loss,
                    "artifact_format": "laya",
                    "full_resume": False,
                    "phase": "epoch_weights_before_independent_fit",
                    "upstream_revision": UPSTREAM_REVISION,
                },
            )

    def train(self, checkpoint=None):
        if checkpoint is not None:
            raise ValueError("Laya does not support full state resume")
        config = self._train_config()
        actual_steps = 0

        def stepped(optimizer, args, kwargs):
            nonlocal actual_steps
            if isinstance(optimizer, self.torch.optim.AdamW):
                actual_steps += 1

        # Observational PyTorch hook: GradScaler may skip optimizer.step on overflow.
        # Counting only real calls avoids reporting the planned update budget as evidence.
        hook = importlib.import_module("torch.optim.optimizer").register_optimizer_step_post_hook(
            stepped
        )
        try:
            if self.world == 1:
                history = self.api.train_model(
                    self.model,
                    self.tok,
                    self.train_items,
                    config,
                    self.device,
                    self.config.model.max_seq_length,
                    self.config.laya.head_max_length,
                    on_epoch_end=self._epoch_save,
                )
            else:
                history = self._train_ddp(config)
        finally:
            hook.remove()
        if any(not math.isfinite(loss) for loss in history):
            raise ValueError("Laya training produced nonfinite loss")
        updates = (
            math.ceil(math.ceil(len(self.train_items) / config.micro_batch) / config.grad_accum)
            * config.epochs
        )
        probes = {}
        named = dict(self.model.named_parameters())
        for group, values in self.before.items():
            deltas = {
                n: float((named[n].detach().flatten()[:256].float().cpu() - v).abs().max().item())
                for n, v in values.items()
            }
            probes[group] = {
                "trainable_parameters": sum(
                    p.numel()
                    for n, p in named.items()
                    if p.requires_grad and n.startswith("encoder.") == (group == "encoder")
                ),
                "sampled_tensors": len(deltas),
                "max_abs_delta": max(deltas.values()),
                "scope": "first_256_values_per_tensor",
            }
        self.training_report = {
            "loss_history": history,
            "optimizer_updates": updates,
            "optimizer_updates_basis": "planned_per_rank_step_budget",
            "actual_optimizer_steps": actual_steps,
            "actual_optimizer_steps_basis": "PyTorch_AdamW_step_post_hook",
            "calibration_assignment": "preassigned_group_disjoint_roles_no_upstream_split",
            "scheduler_T_max": updates
            if self.world == 1
            else max(
                1, len(self.train_items) // (config.micro_batch * config.grad_accum) * config.epochs
            ),
            "scheduler_source": "laya.train.train_model"
            if self.world == 1
            else "official_two_T4_notebook_floor_budget",
            "effective_batch_up_to": config.micro_batch * config.grad_accum * self.world,
            "parameter_updates": probes,
            "distributed": self.partition,
            "upstream_config": config.__dict__,
            "full_resume": False,
        }
        if self.world > 1:
            ranks = [None] * self.world
            self.torch.distributed.all_gather_object(
                ranks,
                {
                    "rank": self.rank,
                    "device": str(self.device),
                    "actual_optimizer_steps": actual_steps,
                    "parameter_updates": probes,
                },
            )
            self.training_report["ranks"] = ranks
        return self.training_report

    def _train_ddp(self, config):
        """Orchestration from Laya's pinned official two-T4 notebook; algorithms remain upstream."""
        torch, model = self.torch, self.model
        if config.gradient_checkpointing:
            model.encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        model.head_checkpointing = config.gradient_checkpointing
        model.train()
        ddp = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[self.local_rank], find_unused_parameters=True
        )
        enc = [
            p for n, p in model.named_parameters() if n.startswith("encoder.") and p.requires_grad
        ]
        head = [
            p
            for n, p in model.named_parameters()
            if not n.startswith("encoder.") and p.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            [{"params": enc, "lr": config.encoder_lr}, {"params": head, "lr": config.head_lr}],
            weight_decay=config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(
                1, len(self.train_items) // (config.micro_batch * config.grad_accum) * config.epochs
            ),
            eta_min=config.min_lr,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        torch.manual_seed(config.seed + self.rank)
        my_items, history = list(self.train_items), []
        for epoch in range(config.epochs):
            random.Random(config.seed + epoch + self.rank).shuffle(my_items)
            optimizer.zero_grad(set_to_none=True)
            total, n_batches = 0.0, 0
            sigma = self.api.sigma_at(epoch, config.epochs, config.sigma_start, config.sigma_end)
            steps = math.ceil(len(my_items) / config.micro_batch)
            for start in range(0, len(my_items), config.micro_batch):
                chunk = [
                    self.api.encode_item(
                        self.tok,
                        item,
                        self.config.model.max_seq_length,
                        self.config.laya.head_max_length,
                    )
                    for item in my_items[start : start + config.micro_batch]
                ]
                batch = self.common.collate_items([chunk], self.tok.pad_token_id)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits, _act = ddp(
                        *(
                            batch[k].to(self.device)
                            for k in (
                                "input_ids",
                                "attention_mask",
                                "marker_pos",
                                "marker_mask",
                                "qtype",
                            )
                        )
                    )
                loss = self.api.rlcd_loss(
                    logits.float(),
                    batch["target"].to(self.device),
                    batch["marker_mask"].to(self.device),
                    batch["qtype"].to(self.device),
                    sigma,
                    samples=config.rl_samples,
                    w_sph=config.w_sph,
                    w_rps=config.w_rps,
                )
                if not torch.isfinite(loss).item():
                    raise ValueError("Laya DDP training produced nonfinite loss")
                window = min(
                    config.grad_accum, steps - (n_batches // config.grad_accum) * config.grad_accum
                )
                scaler.scale(loss / window).backward()
                n_batches += 1
                if n_batches % config.grad_accum == 0 or start + config.micro_batch >= len(
                    my_items
                ):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(ddp.parameters(), config.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                total += loss.item()
            # Report a true all-rank mean, then every rank observes save completion.
            mean = torch.tensor(total / n_batches, device=self.device)
            torch.distributed.all_reduce(mean)
            mean = mean.item() / self.world
            history.append(mean)
            torch.distributed.barrier()
            self._epoch_save(epoch, mean)
            torch.distributed.barrier()
        del ddp, optimizer, scheduler, scaler
        model.eval()
        return history

    def finish_distributed(self):
        if self.group_active:
            self.torch.distributed.barrier()
            self.torch.distributed.destroy_process_group()
            self.group_active = False

    def _records(self, role):
        return self.api.calibration_records(
            self.model,
            self.tok,
            self.items[role],
            self.device,
            self.config.model.max_seq_length,
            self.config.laya.head_max_length,
            batch_size=self.config.training.batch_size,
        )

    def evaluate(self, role, phase):
        records = self._records(role)
        result = (
            self.api.evaluate_records(
                records, self.cfg.get("temperature"), self.cfg.get("temperature_by_options")
            )
            if records
            else {}
        )
        return {
            "role": role,
            "phase": phase,
            "questions": len(records),
            "source": "laya.train.evaluate_records",
            "metrics": result,
            "calibration_source": self.calibration.get("source"),
            "temperature": self.cfg.get("temperature"),
            "temperature_by_options": self.cfg.get("temperature_by_options", {}),
        }

    def calibrate(self):
        records = self._records("calibration")
        fitted = self.api.fit_temperature_map(records)
        self.cfg["temperature"] = fitted["temperature"]
        self.cfg.pop("temperature_by_options", None)
        if fitted["temperature_by_options"]:
            self.cfg["temperature_by_options"] = fitted["temperature_by_options"]
        self.calibration = {
            "fitted": any(
                v["fitted"] for v in calibration_types(records, fitted["temperature"]).values()
            ),
            "source": "independent_calibration_role",
            "types": calibration_types(records, fitted["temperature"]),
            "result": fitted,
            "groups": sorted({m["group_id"] for m in self.mapping["calibration"]["mapping"]}),
            "upstream_report": self.api.calibration_report(records, fitted["temperature"]),
        }
        return self.calibration

    def calibration_state(self):
        return {
            "temperature": self.cfg.get("temperature", [1, 1, 1]),
            "temperature_by_options": self.cfg.get("temperature_by_options", {}),
            "evidence": self.calibration,
        }

    def restore_calibration(self, state):
        self.cfg["temperature"] = state["temperature"]
        self.cfg.pop("temperature_by_options", None)
        if state["temperature_by_options"]:
            self.cfg["temperature_by_options"] = state["temperature_by_options"]
        self.calibration = state["evidence"]

    def save(self):
        self.cfg.update(fine_tuned=True)
        self.cfg["training"] = {
            "laya_train": self._train_config().__dict__,
            "dforge_report": self.training_report,
        }
        self.cfg["dforge"] = {
            "upstream_revision": UPSTREAM_REVISION,
            "initialization": self.source_identity,
            "calibration": self.calibration,
            "training": self._train_config().__dict__,
            "full_resume": False,
        }
        target = self.directory / "model"
        self.api.save_checkpoint(self.model, self.tok, self.cfg, str(target))
        return {
            "format": "laya",
            "model": str(target),
            "phase": "calibrated" if self.config.calibration.enabled else "trained",
            "save_api": "laya.train.save_checkpoint",
            "weights_dtype": "float16",
            "full_resume": False,
            "identity": artifact_identity(target),
        }

    def reload_probe(self, data):
        from decisionforge.eval.adapters.laya_worker import agent_input_loss, predict_public

        laya = importlib.import_module("laya")
        # Release the training allocation before loading the saved model.
        self.model = None
        self.torch.cuda.empty_cache()
        with laya.load(
            str(self.directory / "model"), device=str(self.device), backend="eager"
        ) as agent:
            record = next(
                iter(data.roles["test"] or data.roles["validation"] or data.roles["train"])
            )
            # Canonical eval Request deliberately contains one decision question.
            qid = next(iter(record.request.questions))
            public = {
                "state": record.request.state,
                "questions": {
                    "decision": record.request.questions[qid].model_dump(exclude_none=True)
                },
            }
            response = predict_public(
                agent, public, input_check=lambda request: agent_input_loss(agent, request)
            )
            runtime = {
                "device": "cpu" if agent.cpu_fallback_count else str(agent.device),
                "precision": "torch.float32" if agent.cpu_fallback_count else str(agent.dtype),
                "cpu_fallbacks": agent.cpu_fallback_count,
                "fallback_reason": agent.last_fallback_reason,
            }
        return {
            "source": "laya.load/Agent.system_one",
            "record_id": record.record_id,
            "question_id": qid,
            "response": response,
            "artifact_format": "laya",
            "runtime": runtime,
        }

    def close(self):
        if self.group_active:
            self.torch.distributed.destroy_process_group()
            self.group_active = False
        self.model = self.tok = None
        if self.staging is not None:
            self.staging.cleanup()
            self.staging = None
        if hasattr(self, "torch") and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
