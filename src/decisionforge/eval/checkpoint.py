"""CPU-only checkpoint checks and content identity; never imports a model SDK."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from decisionforge.eval.data import parse_json

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
    "triton",
    "tokenizers",
    "huggingface_hub",
)


def read_object(path):
    try:
        value = parse_json(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object")
        json.dumps(value, allow_nan=False)
        return value
    except (ValueError, OSError) as exc:
        raise ValueError(f"Invalid checkpoint configuration {path}: {exc}") from exc


def indexed_weights(root):
    index = root / "model.safetensors.index.json"
    if not index.is_file():
        return []
    weight_map = read_object(index).get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Invalid checkpoint weight_map in {index}")
    names = list(weight_map.values())
    if any(
        not isinstance(name, str) or not name or Path(name).anchor or ".." in Path(name).parts
        for name in names
    ):
        raise ValueError(f"Checkpoint shard names must be relative paths within {root}")
    return [root / name for name in sorted(set(names))]


def validate_checkpoint(path):
    path = Path(path).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"Local decision checkpoint directory does not exist: {path}")
    required = (
        "joint_head.safetensors",
        "joint_head_config.json",
        "unsloth_decision_config.json",
        "tokenizer_config.json",
    )
    for name in required:
        file = path / name
        if not file.is_file() or file.stat().st_size == 0:
            raise ValueError(
                f"Not a saved decision checkpoint: missing/empty {file}; choose run/model or run/merged"
            )
    decision = read_object(path / "unsloth_decision_config.json")
    if decision.get("layout") != "clef":
        raise ValueError("unsloth_local supports saved Clef decision checkpoints")
    read_object(path / "joint_head_config.json")
    adapter = path / "adapter_config.json"
    if adapter.is_file():
        read_object(adapter)
        weights = [path / "adapter_model.safetensors"]
    else:
        if not (path / "config.json").is_file():
            raise ValueError("Decision checkpoint has neither adapter_config.json nor config.json")
        read_object(path / "config.json")
        index = path / "model.safetensors.index.json"
        if index.is_file():
            weights = indexed_weights(path)
        else:
            weights = [path / "model.safetensors"]
    if not weights or any(not p.is_file() or p.stat().st_size == 0 for p in weights):
        raise ValueError(f"Missing/empty decision model weights in {path}")
    return path


def file_identity(path):
    root = Path(path).resolve()
    files = {}
    candidates = {
        file
        for file in root.rglob("*")
        if file.is_file()
        and not any(part.startswith(".") for part in file.relative_to(root).parts)
        and file.suffix in {".json", ".safetensors", ".bin", ".model", ".txt", ".jinja", ".py"}
    }
    # Indexed shards may live in symlinked directories or use nonstandard names.
    # Hash their actual contents even when the directory walk does not find them.
    candidates.update(indexed_weights(root))
    for file in sorted(candidates):
        before = file.stat()
        hasher = hashlib.sha256()
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                hasher.update(block)
        after = file.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ino,
        ):
            raise ValueError(f"Checkpoint changed while computing inference identity: {file}")
        files[file.relative_to(root).as_posix()] = {
            "size": after.st_size,
            "sha256": hasher.hexdigest(),
        }
    return {"path": str(root), "files": files}


def checkpoint_identity(path):
    root = validate_checkpoint(path)
    result = {"checkpoint": file_identity(root)}
    adapter = root / "adapter_config.json"
    if adapter.is_file():
        info = read_object(adapter)
        base = info.get("base_model_name_or_path")
        result["base_model"] = {"name": base, "revision": info.get("revision")}
        if isinstance(base, str):
            base_path = Path(base).expanduser()
            if not base_path.is_absolute() and (root / base_path).is_dir():
                base_path = root / base_path
            if base_path.is_dir():
                result["base_model"]["local_content"] = file_identity(base_path)
    return result


def dependency_versions():
    result = {}
    for package in PACKAGES:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result
