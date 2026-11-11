"""Internal isolated task registration; invoke the pinned upstream CLI main unchanged."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from pathlib import Path

from decisionforge.training.backends.openjev import inventory, validate_installation
from decisionforge.training.backends.openjev_task import (
    TASK_NAME,
    check_prompt_budget,
    public_item,
    register_task,
    validate_labels,
)
from decisionforge.util import write_json


class OptimizerObservation:
    """Observe successful original AdamW steps without changing updates or RNG."""

    def __init__(self, torch):
        self.torch, self.steps, self.before, self.parameters = torch, 0, {}, {}
        hooks = importlib.import_module("torch.optim.optimizer")
        self.pre = hooks.register_optimizer_step_pre_hook(self._before)
        self.post = hooks.register_optimizer_step_post_hook(self._after)

    def _before(self, optimizer, args, kwargs):
        if not isinstance(optimizer, self.torch.optim.AdamW) or self.before:
            return
        for gi, group in enumerate(optimizer.param_groups):
            for pi, parameter in enumerate(group["params"]):
                if parameter.requires_grad:
                    key = f"group{gi}.parameter{pi}"
                    self.parameters[key] = parameter
                    self.before[key] = parameter.detach().flatten()[:256].float().cpu().clone()

    def _after(self, optimizer, args, kwargs):
        if isinstance(optimizer, self.torch.optim.AdamW):
            self.steps += 1
            counters = [
                float(state["step"].item()) for state in optimizer.state.values() if "step" in state
            ]
            if not counters or any(
                not math.isfinite(step) or step != self.steps for step in counters
            ):
                raise ValueError(
                    "OpenJev AdamW state counters do not match successful observed steps"
                )
            self.state_steps = sorted(set(counters))

    def result(self):
        deltas = {
            key: float(
                (parameter.detach().flatten()[:256].float().cpu() - self.before[key])
                .abs()
                .max()
                .item()
            )
            for key, parameter in self.parameters.items()
        }
        if not deltas or any(not math.isfinite(value) for value in deltas.values()):
            raise ValueError("OpenJev optimizer observation has missing/nonfinite parameter values")
        return {
            "actual_optimizer_steps": self.steps,
            "actual_optimizer_steps_basis": "original_PyTorch_AdamW_step_pre_post_hooks",
            "optimizer_state_steps": self.state_steps,
            "parameter_updates": {
                "scope": "first_256_values_per_trainable_tensor_before_first_step_vs_final",
                "sampled_tensors": len(deltas),
                "trainable_parameters": sum(p.numel() for p in self.parameters.values()),
                "max_abs_delta": max(deltas.values()),
                "changed_tensors": sum(value > 0 for value in deltas.values()),
            },
        }

    def close(self):
        self.pre.remove()
        self.post.remove()


def run_stage(plan):
    validate_installation()
    torch = importlib.import_module("torch")
    upstream = importlib.import_module("openjev_rlcd.train")
    transformers = importlib.import_module("transformers")
    args = plan["argv"]
    backbone = args[args.index("--model") + 1]
    tok = transformers.AutoTokenizer.from_pretrained(backbone, local_files_only=True)
    protocol = plan.get("label_protocol")
    if protocol is None:
        validate_labels(tok)
        labels = None
    else:
        from decisionforge.label_tokens import validate_label_protocol

        validate_label_protocol(tok, protocol)
        if protocol["encoding_style"] != "space_prefixed" or protocol["legacy_count"] != 10:
            raise ValueError("OpenJev stage label protocol is incompatible")
        labels = protocol["labels"]
    splits = json.loads(Path(plan["splits"]).read_text())
    register_task(upstream.TASKS, splits, labels=labels)
    suffix = tok.encode("\nAnswer:", add_special_tokens=False)
    for items in splits.values():
        for item in items:
            expected = public_item(
                item["public_request"], item["focus"], item["uid"], labels=labels
            )
            if any(item.get(key) != value for key, value in expected.items()):
                raise ValueError("OpenJev stage item mapping differs from validated label protocol")
            check_prompt_budget(
                tok,
                upstream.prompt(tok, item, TASK_NAME),
                plan["max_seq_length"],
                int(args[args.index("--max-new-tokens") + 1]),
                suffix,
            )
    if "--init-checkpoint" in args:
        path = Path(args[args.index("--init-checkpoint") + 1])
        if inventory(path.parent, [path])[path.name]["sha256"] != plan["expected_init_sha256"]:
            raise ValueError("OpenJev stage initialization digest changed before loading")
    # main() is the original implementation: model/gen/ref construction, loss,
    # sampling, optimizer, train/evaluation loops and save are all unchanged.
    observer = OptimizerObservation(torch)
    previous = sys.argv
    try:
        sys.argv = ["openjev_rlcd.train", *args]
        upstream.main()
        evidence = observer.result()
        expected = int(args[args.index("--steps") + 1])
        if (
            evidence["actual_optimizer_steps"] != expected
            or evidence["parameter_updates"]["max_abs_delta"] <= 0
        ):
            raise ValueError("OpenJev stage did not perform expected measured optimizer updates")
        write_json(Path(plan["evidence"]), evidence)
    finally:
        sys.argv = previous
        observer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Internal pinned OpenJev CLI adapter")
    parser.add_argument("--plan", required=True)
    run_stage(json.loads(Path(parser.parse_args().plan).read_text()))
