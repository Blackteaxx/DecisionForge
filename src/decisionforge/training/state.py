"""Read-only fingerprints for observing state restored by the upstream Trainer."""

from __future__ import annotations

import hashlib
import json


def state_fingerprint(value, torch, *, sample_values=None):
    def encode(item):
        if hasattr(item, "detach") and hasattr(item, "shape"):
            tensor = item.detach().flatten()
            if sample_values is not None:
                tensor = tensor[:sample_values]
            raw = tensor.cpu().contiguous().view(torch.uint8).numpy().tobytes()
            return {"tensor": str(item.dtype), "shape": list(item.shape), "bytes": raw.hex()}
        if hasattr(item, "tobytes") and hasattr(item, "dtype"):
            return {
                "array": str(item.dtype),
                "shape": list(item.shape),
                "bytes": item.tobytes().hex(),
            }
        if isinstance(item, dict):
            return {"dict": [[str(key), encode(item[key])] for key in sorted(item, key=str)]}
        if isinstance(item, (tuple, list)):
            return [encode(part) for part in item]
        if item is None or type(item) in (str, int, float, bool):
            return item
        raise ValueError(f"Unsupported training state value: {type(item).__name__}")

    text = json.dumps(encode(value), sort_keys=True, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def optimizer_summary(value, torch):
    steps = []
    for state in value.get("state", {}).values():
        step = state.get("step")
        if step is not None:
            steps.append(float(step.item() if hasattr(step, "item") else step))
    return {
        "initialized_parameters": len(value.get("state", {})),
        "step_min": min(steps) if steps else None,
        "step_max": max(steps) if steps else None,
        "sample_values_per_tensor": 64,
        "state_sample_sha256": state_fingerprint(value, torch, sample_values=64),
    }
