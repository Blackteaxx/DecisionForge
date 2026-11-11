"""Pinned ScienceQA image-subset preparation; no model calls or credentials."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from PIL import Image

from decisionforge.eval.models import Sample
from decisionforge.models import TrainingExample
from decisionforge.util import write_json

DATASET = "derek-thomas/ScienceQA"
REVISION = "f18b0a70359ebfb41f658fd564208d0355b013f4"
FILES = {
    "train": "data/train-00000-of-00001-1028f23e353fbe3e.parquet",
    "validation": "data/validation-00000-of-00001-6c7328ff6c84284c.parquet",
    "test": "data/test-00000-of-00001-f0e719df791966ff.parquet",
}
MAX_PIXELS = 768 * 28 * 28


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(root):
    import shutil

    from huggingface_hub import hf_hub_download

    manifest = json.loads(Path(__file__).with_name("source.json").read_text())
    manifest["files"] = {}
    for split, name in FILES.items():
        path = Path(hf_hub_download(DATASET, name, repo_type="dataset", revision=REVISION))
        target = root / "raw" / f"{split}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        manifest["files"][split] = {
            "path": target.name,
            "url": f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{name}",
            "sha256": sha(target),
            "bytes": target.stat().st_size,
        }
    write_json(root / "raw/source.json", manifest)
    return manifest


def convert(root, *, splits=None):
    import pyarrow.parquet as pq

    root = Path(root).resolve()
    source = json.loads((root / "raw/source.json").read_text())
    if source["source_revision"] != REVISION:
        raise ValueError("ScienceQA source revision changed")
    processed = root / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    training, split_counts = [], {}
    for split in splits or FILES:
        path = root / "raw" / source["files"][split]["path"]
        if sha(path) != source["files"][split]["sha256"]:
            raise ValueError(f"Source SHA-256 mismatch: {path}")
        rows, transforms, excluded = [], [], 0
        table = pq.ParquetFile(path)
        for batch in table.iter_batches(batch_size=64):
            for index, row in enumerate(batch.to_pylist()):
                # Monotonic index includes text-only rows, preserving source row mapping.
                position = len(rows) + excluded
                image = row["image"]
                if image is None:
                    excluded += 1
                    continue
                if not isinstance(image, dict) or not image.get("bytes"):
                    raise ValueError(f"{split}:{position}: expected embedded image bytes")
                identifier = f"scienceqa-img:{split}:{position}"
                im = Image.open(io.BytesIO(image["bytes"])).convert("RGB")
                original_size = im.size
                if im.width * im.height > MAX_PIXELS:
                    factor = math.sqrt(MAX_PIXELS / (im.width * im.height))
                    im = im.resize(
                        (max(1, int(im.width * factor)), max(1, int(im.height * factor))),
                        Image.Resampling.LANCZOS,
                    )
                target = processed / "images" / split / f"{position}.png"
                target.parent.mkdir(parents=True, exist_ok=True)
                im.save(target)
                image_hash = sha(target)
                choices = row["choices"]
                if not 2 <= len(choices) <= 5 or any(len(choice) > 200 for choice in choices):
                    raise ValueError(f"{identifier}: unsupported options; refusing to truncate")
                if type(row["answer"]) is not int or not 0 <= row["answer"] < len(choices):
                    raise ValueError(f"{identifier}: invalid original answer")
                request = {
                    "state": row.get("hint") or "",
                    "images": [{"path": os.path.relpath(target, processed), "sha256": image_hash}],
                    "questions": {
                        "decision": {
                            "type": "choice",
                            "instructions": row["question"],
                            "criteria": {str(i): value for i, value in enumerate(choices)},
                        }
                    },
                }
                sample = Sample.model_validate(
                    {
                        "schema_version": "decisionforge.eval.v1",
                        "id": identifier,
                        "group_id": identifier,
                        "split": split,
                        "subset": "scienceqa_img",
                        "request": request,
                        "target": {"kind": "hard", "value": str(row["answer"])},
                        "metadata": {
                            "source_revision": REVISION,
                            "source_row": position,
                            "target_source": "ScienceQA original answer",
                            "subject": row.get("subject"),
                        },
                    }
                )
                rows.append(sample.model_dump(mode="json"))
                transforms.append(
                    {
                        "source_row": position,
                        "source_image_sha256": hashlib.sha256(image["bytes"]).hexdigest(),
                        "image_sha256": image_hash,
                        "original_size": original_size,
                        "prepared_size": im.size,
                    }
                )
                # Calibration comes only from official train, deterministic 10% by row ID.
                role = (
                    "calibration"
                    if split == "train"
                    and int(hashlib.sha256(identifier.encode()).hexdigest()[:8], 16) % 10 == 0
                    else split
                )
                number = (list(FILES).index(split) + 1) * 1_000_000 + position + 1
                target_probs = {str(i): float(i == row["answer"]) for i in range(len(choices))}
                example = TrainingExample.model_validate(
                    {
                        "schema_version": "decisionforge.training.v3",
                        "record_id": number,
                        "group_id": number,
                        "split": role,
                        "request": request,
                        "supervision": {
                            "answers": {
                                "decision": {
                                    "type": "choice",
                                    "choice": str(row["answer"]),
                                    "confidence": 1.0,
                                    "probabilities": target_probs,
                                }
                            }
                        },
                    }
                )
                training.append(example.model_dump(mode="json"))
        output = processed / f"{split}.jsonl"
        output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        write_json(
            processed / f"{split}.jsonl.manifest.json",
            {
                "schema_version": "decisionforge.scienceqa_img.prepare.v1",
                "source": source,
                "split": split,
                "records": len(rows),
                "excluded_text_only": excluded,
                "original_rows": len(rows) + excluded,
                "output_sha256": sha(output),
                "image_policy": "RGB PNG; explicit aspect-preserving resize to max 602112 pixels; no crop",
                "images": transforms,
                "rules": "retain question, choices and hint verbatim; never expose lecture or solution; full official image subset",
            },
        )
        split_counts[split] = len(rows)
    (processed / "training.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in training)
    )
    write_json(
        processed / "training.jsonl.manifest.json",
        {
            "source": source,
            "records": len(training),
            "sha256": sha(processed / "training.jsonl"),
            "calibration_assignment": "official train only; sha256(source ID) prefix modulo 10 == 0",
            "split_counts": {
                role: sum(row["split"] == role for row in training)
                for role in ("train", "validation", "calibration", "test")
            },
        },
    )
    return split_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--fetch-only", action="store_true")
    modes.add_argument("--convert-only", action="store_true")
    parser.add_argument("--split", choices=list(FILES), action="append")
    args = parser.parse_args()
    if not args.convert_only:
        fetch(args.data_dir.resolve())
    if not args.fetch_only:
        print(json.dumps(convert(args.data_dir.resolve(), splits=args.split)))


if __name__ == "__main__":
    main()
