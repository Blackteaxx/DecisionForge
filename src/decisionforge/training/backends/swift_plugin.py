"""ms-swift external plugin. Imported only inside its isolated SDK process."""

import json
import os
from pathlib import Path

from swift.callbacks.base import TrainerCallback
from swift.callbacks.mapping import callbacks_map
from swift.callbacks.temperature_calibration import TemperatureCalibrationCallback
from swift.dataset import DatasetMeta, register_dataset
from swift.dataset.preprocessor.decision import OmniJevPreprocessor
from swift.model.decision_head import get_scoring_head

from decisionforge.training.backends.swift_routes import get_route
from decisionforge.training.backends.swift_routes.sdk import (
    preprocessor_for,
    register_templates,
    temperature_buffers,
)
from decisionforge.util import write_json


class UpdateEvidence(TrainerCallback):
    """Observe optimizer calls and small weight samples without changing training."""

    def _groups(self):
        return (
            ("backbone", "head")
            if getattr(self, "finetuning", "lora") == "full"
            else ("adapter", "head")
        )

    def _belongs(self, name, group):
        if group == "backbone":
            return "scoring_head" not in name
        return ("lora_" if group == "adapter" else "scoring_head") in name

    def on_train_begin(self, args, state, control, **kwargs):
        from torch.optim.optimizer import register_optimizer_step_post_hook

        self.steps = 0
        self.finetuning = os.environ.get("DFORGE_SWIFT_FINETUNING", "lora")
        self.before = {}
        self.observed_parameters = {}
        self.first_step_gradients = None
        for name, parameter in self.trainer.model.named_parameters():
            if parameter.requires_grad and any(
                self._belongs(name, group) for group in self._groups()
            ):
                self.before[name] = parameter.detach().flatten()[:256].float().cpu().clone()
                self.observed_parameters[name] = parameter
        self.hook = register_optimizer_step_post_hook(self.stepped)
        head = get_scoring_head(self.trainer.model)
        for buffer in temperature_buffers(head).values():
            buffer.fill_(1)

    def stepped(self, optimizer, args, kwargs):
        import torch

        # Accelerate may wrap the optimizer. Count only the underlying update,
        # matching the explicit native AdamW configuration rather than wrappers.
        if isinstance(optimizer, torch.optim.AdamW):
            self.steps += 1
            if self.steps == 1:
                self.first_step_gradients = {}
                for group in self._groups():
                    parameters = [
                        p
                        for name, p in self.observed_parameters.items()
                        if self._belongs(name, group)
                    ]
                    gradients = [p.grad for p in parameters if p.grad is not None]
                    maximum = max((g.detach().abs().max().item() for g in gradients), default=0)
                    self.first_step_gradients[group] = {
                        "trainable_tensors": len(parameters),
                        "tensors_with_gradient": len(gradients),
                        "max_abs_gradient": maximum,
                    }
                    if not gradients or not maximum > 0:
                        raise ValueError(
                            f"No nonzero {group} gradient at the first AdamW update; "
                            "refusing a run with a disconnected training graph"
                        )

    def on_save(self, args, state, control, **kwargs):
        checkpoint = Path(
            getattr(state, "last_model_checkpoint", None)
            or Path(args.output_dir) / f"checkpoint-{state.global_step}"
        )
        route = os.environ.get("DFORGE_SWIFT_MODEL_TYPE", "omnijev")
        if route != "omnijev":
            get_scoring_head(self.trainer.model).save_pretrained(str(checkpoint), save_weights=True)
        required = ["scoring_head.safetensors", "scoring_head.json"]
        if os.environ.get("DFORGE_SWIFT_FINETUNING", "lora") == "lora":
            required.append("adapter_model.safetensors")
        elif (
            not any(checkpoint.glob("model*.safetensors"))
            and not (checkpoint / "pytorch_model.bin").exists()
        ):
            raise ValueError("Native full checkpoint is missing backbone weights")
        for name in required:
            if not (checkpoint / name).is_file():
                raise ValueError(f"Native checkpoint is missing {name}: {checkpoint}")
        source = Path(
            os.environ.get("DFORGE_SWIFT_LINEAGE")
            or Path(os.environ["DFORGE_SWIFT_DATA"]).parent / "swift_lineage.json"
        )
        write_json(checkpoint / "swift_lineage.json", json.loads(source.read_text()))

    def on_train_end(self, args, state, control, **kwargs):
        self.hook.remove()
        names = dict(self.trainer.model.named_parameters())
        groups = {}
        for group in self._groups():
            values = [
                (names[name].detach().flatten()[:256].float().cpu() - before).abs().max().item()
                for name, before in self.before.items()
                if self._belongs(name, group)
            ]
            groups[group] = {
                "sampled_tensors": len(values),
                "max_abs_delta": max(values, default=0),
                "scope": "first_256_values_per_trainable_tensor",
            }
            if not values or max(values) == 0:
                raise ValueError(f"No observed {group} parameter update")
        if self.steps < 1:
            raise ValueError("No actual optimizer update occurred")
        write_json(
            Path(args.output_dir) / "dforge_update_evidence.json",
            {
                "actual_optimizer_steps": self.steps,
                "first_step_gradients": self.first_step_gradients,
                "parameter_updates": groups,
            },
        )


class IndependentCalibration(TemperatureCalibrationCallback):
    """Use ms-swift fitting once, on the explicitly assigned calibration role.

    The upstream callback collects serving logits. Reset only post-hoc temperatures
    during collection so the new fit replaces, rather than stacks on, shipped values.
    Learned log_tau parameters and the shipped noul bias remain model parameters.
    """

    def _collect_logits_and_targets(self):
        if hasattr(self, "_final_collection"):
            return self._final_collection
        head = get_scoring_head(self.trainer.model)
        buffers = temperature_buffers(head)
        before = {name: value.clone() for name, value in buffers.items()}
        try:
            for buffer in buffers.values():
                buffer.fill_(1)
            return super()._collect_logits_and_targets()
        finally:
            for name, value in before.items():
                getattr(head, name).copy_(value)

    def on_train_end(self, args, state, control, **kwargs):
        if os.environ["DFORGE_SWIFT_CALIBRATE"] == "0":
            return
        self._final_collection = self._collect_logits_and_targets()
        items = self._flatten_collected(self._final_collection)
        if len(items) < self.min_items:
            raise ValueError("Insufficient encoded calibration questions; refusing a silent no-op")
        try:
            super().on_train_end(args, state, control, **kwargs)
        finally:
            del self._final_collection
        head = get_scoring_head(self.trainer.model)
        head_t, relative = self._served_temperatures(head)
        write_json(
            Path(args.output_dir) / "dforge_calibration.json",
            {
                "owner": "ms-swift TemperatureCalibrationCallback",
                "role": "calibration",
                "questions": len(items),
                "head_temperature": head_t,
                "relative_temperatures": relative,
                "before": self._in_sample_metrics(items, 1.0, {}),
                "after": self._in_sample_metrics(items, head_t, relative),
                "meaning": "in-sample fitting diagnostics; independent test is never fitted",
            },
        )


callbacks_map["temperature_calibration"] = IndependentCalibration
callbacks_map["dforge_update_evidence"] = UpdateEvidence
register_templates()
route_name = os.environ.get("DFORGE_SWIFT_MODEL_TYPE", "omnijev")
route = get_route(route_name)
if route.noncommercial and os.environ.get("DFORGE_SWIFT_USAGE") != "noncommercial_research":
    raise ValueError("StartLux plugin requires explicit noncommercial research usage")
if os.environ.get("DFORGE_SWIFT_DATA"):
    root = Path(os.environ["DFORGE_SWIFT_DATA"])
    for role in ("train", "validation", "calibration", "test"):
        path = root / f"{role}.jsonl"
        if path.is_file() and path.stat().st_size:
            register_dataset(
                DatasetMeta(
                    dataset_name=f"dforge_{role}",
                    dataset_path=str(path),
                    preprocess_func=OmniJevPreprocessor()
                    if route_name == "omnijev"
                    else preprocessor_for(route_name),
                ),
                exist_ok=True,
            )
