"""Pinned source bundles; downloading is separate from offline conversion."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import httpx
import yaml

from data.benchmarks._shared.converters import source_info
from decisionforge.eval.jevbench import JEVBENCH_PIN
from decisionforge.util import write_json

TYPED_PIN = "e039ebffcc280174dd354227424fb2b249f191de"
CLINC_PIN = "155b9c710419136e17307b80d0a13e68cd46b4ec"
BANKING_PIN = "57ec275d8078af65b7731c2a98be812d844a6d6b"
SPLITS = {
    "jevbench": ("public",),
    "typed_decisions": ("test", "train"),
    "banking77": ("test", "train"),
    "clinc150_oos": ("test", "train", "validation"),
}
DATASETS = tuple(SPLITS)


def fetch_dataset(dataset, output, split=None, transport=None):
    output = Path(output).resolve()
    if dataset not in DATASETS:
        raise ValueError("Unknown evaluation dataset")
    split = split or ("public" if dataset == "jevbench" else "test")
    if split not in SPLITS[dataset]:
        raise ValueError("Unsupported dataset split")
    if dataset == "jevbench":
        revision, repo, license_name = JEVBENCH_PIN, "fstandhartinger/jevbench", "MIT"
        root = f"https://raw.githubusercontent.com/{repo}/{revision}/datasets/public"
        urls = {f"{tier}.jsonl": f"{root}/{tier}.jsonl" for tier in ("original", "easy", "hard")}
    elif dataset == "banking77":
        revision, repo, license_name = BANKING_PIN, "PolyAI-LDN/task-specific-datasets", "CC-BY-4.0"
        root = f"https://raw.githubusercontent.com/{repo}/{revision}/banking_data"
        urls = {f"{split}.csv": f"{root}/{split}.csv", "labels.json": f"{root}/categories.json"}
    else:
        revision = TYPED_PIN if dataset == "typed_decisions" else CLINC_PIN
        repo = "LocalLLaMA/typed-decisions" if dataset == "typed_decisions" else "clinc/clinc_oos"
        license_name = (
            "Apache-2.0" if dataset == "typed_decisions" else "unspecified in dataset card"
        )
        root = f"https://huggingface.co/datasets/{repo}/resolve/{revision}"
        config = "all" if dataset == "typed_decisions" else "plus"
        urls = {
            f"{split}.parquet": f"{root}/{config}/{split}-00000-of-00001.parquet",
            "README.md": f"{root}/README.md",
        }
    if (output / "source.json").exists():
        existing = source_info(output)
        if (existing["dataset"], existing["revision"], existing["split"]) != (
            dataset,
            revision,
            split,
        ):
            raise ValueError("Existing dataset source bundle is incompatible")
        return existing
    if output.exists() and any(output.iterdir()):
        raise ValueError("Dataset source output must be empty or a verified source bundle")
    output.mkdir(parents=True, exist_ok=True)
    files = []
    with httpx.Client(timeout=120, follow_redirects=True, transport=transport) as client:
        for name, url in urls.items():
            response = client.get(url)
            response.raise_for_status()
            data = response.content
            temporary = output / f"{name}.tmp"
            with temporary.open("wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(output / name)
            files.append({"name": name, "url": url, "sha256": hashlib.sha256(data).hexdigest()})
    if dataset == "clinc150_oos":
        text = (output / "README.md").read_text(encoding="utf-8")
        card = yaml.safe_load(text.split("---", 2)[1])
        config_info = next(x for x in card["dataset_info"] if x["config_name"] == "plus")
        feature = next(x for x in config_info["features"] if x["name"] == "intent")
        names = feature["dtype"]["class_label"]["names"]
        labels = [names[str(i)] for i in range(len(names))] if isinstance(names, dict) else names
        write_json(output / "labels.json", labels)
        files.append(
            {
                "name": "labels.json",
                "url": urls["README.md"] + "#dataset_info/plus/intent",
                "sha256": hashlib.sha256((output / "labels.json").read_bytes()).hexdigest(),
            }
        )
    manifest = {
        "dataset": dataset,
        "repository": repo,
        "revision": revision,
        "split": split,
        "license": license_name,
        "files": files,
    }
    write_json(output / "source.json", manifest)
    return manifest
