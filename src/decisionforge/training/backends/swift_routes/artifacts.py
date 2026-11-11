"""Versioned multi-route artifacts; legacy OmniJev v1 stays on its original loader."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from decisionforge.util import write_json

from . import SWIFT_REVISION, get_route

SCHEMA = "decisionforge.swift.decision.v1"
MARKER = "dforge_swift_artifact.json"


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sources(route):
    return {
        "base_model": {"name": route.model, "revision": route.revision},
        "adapter": (
            {
                "name": route.adapter,
                "revision": route.adapter_revision,
                "subdirectory": route.adapter_subdir,
            }
            if route.adapter
            else None
        ),
        "ms_swift": SWIFT_REVISION,
    }


def published(path, route):
    return str(path) in {route.model, route.adapter} and route.name != "omnijev"


def validate(path, model_type=None):
    root = Path(path).expanduser().resolve()
    marker = root / MARKER
    if not marker.is_file():
        if model_type and published(path, get_route(model_type)):
            return Path(path)
        raise ValueError(
            "Swift route requires its fixed published source ID or a verified exported artifact"
        )
    manifest = json.loads(marker.read_text())
    if manifest.get("schema_version") != SCHEMA:
        raise ValueError("Unsupported multi-route Swift artifact schema")
    route = get_route(manifest.get("model_type"))
    if model_type and route.name != model_type:
        raise ValueError("Swift artifact model_type disagrees with configured route")
    if manifest.get("finetuning") not in ({"lora", "full"} if route.merged else {"lora"}):
        raise ValueError("Swift artifact has an incompatible finetuning mode")
    lineage = manifest.get("lineage", {})
    if any(lineage.get(k) != value for k, value in sources(route).items()):
        raise ValueError("Swift artifact source lineage is incomplete or incompatible")
    if route.noncommercial and lineage.get("usage") != "noncommercial_research":
        raise ValueError("StartLux artifact lacks its noncommercial research declaration")
    files = manifest.get("files", {})
    required = {"scoring_head.safetensors", "scoring_head.json", "swift_lineage.json"}
    required.update(asset[2] for asset in route.assets)
    if manifest["finetuning"] == "lora":
        required.update({"adapter_config.json", "adapter_model.safetensors"})
    else:
        required.add("config.json")
        if not any(
            name.startswith(("model", "pytorch_model")) and name.endswith((".safetensors", ".bin"))
            for name in files
        ):
            raise ValueError("Full Swift artifact is missing backbone weights")
    if not required <= set(files):
        raise ValueError(f"Incomplete Swift {route.name} artifact: {sorted(required - set(files))}")
    for name, expected in files.items():
        candidate = (root / name).resolve()
        if (
            not candidate.is_relative_to(root)
            or not candidate.is_file()
            or file_hash(candidate) != expected
        ):
            raise ValueError(f"Swift artifact hash mismatch: {name}")
    if json.loads((root / "swift_lineage.json").read_text()) != lineage:
        raise ValueError("Swift artifact embedded and file lineage differ")
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        if (root / name).exists():
            weights = json.loads((root / name).read_text()).get("weight_map", {})
            if not weights or not set(weights.values()) <= set(files):
                raise ValueError("Incomplete full Swift backbone shard index")
    return root


def write_manifest(root, route, finetuning, lineage):
    root = Path(root)
    write_json(root / "swift_lineage.json", lineage)
    write_json(
        root / MARKER,
        {
            "schema_version": SCHEMA,
            "model_type": route.name,
            "finetuning": finetuning,
            "lineage": lineage,
            "files": {
                p.relative_to(root).as_posix(): file_hash(p)
                for p in sorted(root.rglob("*"))
                if p.is_file() and p.name != MARKER
            },
        },
    )


def source_roots(route):
    """Weight downloads are confined to a real, GPU-authorized training/inference path."""
    from huggingface_hub import snapshot_download

    model = Path(snapshot_download(route.model, revision=route.revision))
    adapter = None
    if route.adapter:
        adapter = (
            Path(snapshot_download(route.adapter, revision=route.adapter_revision))
            / route.adapter_subdir
        )
    return model, adapter
