"""Repository data preparation entry points; this module is not part of the library."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import httpx
import yaml

from data.benchmarks._shared.converters import (
    convert_intents,
    convert_jevbench,
    convert_typed_decisions,
)
from data.benchmarks._shared.sources import SPLITS, fetch_dataset


def prepare_dataset(
    dataset: str, directory: Path, split: str, *, fetch_only=False, convert_only=False
):
    if fetch_only and convert_only:
        raise ValueError("fetch-only and convert-only are mutually exclusive")
    if split not in SPLITS[dataset]:
        raise ValueError("Unsupported dataset split")
    raw = directory / "raw" / split
    processed = directory / "processed"
    source = None if convert_only else fetch_dataset(dataset, raw, split)
    if fetch_only:
        return {"dataset": dataset, "raw": str(raw), "source": source}
    output = processed / f"{split}.jsonl"
    if dataset == "jevbench":
        manifest = convert_jevbench(raw, output)
        for tier in ("original", "easy", "hard"):
            path = raw / f"{tier}.jsonl"
            if path.exists():
                convert_jevbench(path, processed / f"{tier}.jsonl")
    elif dataset == "typed_decisions":
        manifest = convert_typed_decisions(raw, output)
    else:
        manifest = convert_intents(dataset, raw, output)
    return {"dataset": dataset, "raw": str(raw), "processed": str(output), "conversion": manifest}


def main(dataset: str, dataset_directory: Path, argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=f"Prepare pinned {dataset} data without model calls.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", type=Path, default=dataset_directory, help="Dataset cache directory"
    )
    parser.add_argument("--split", choices=SPLITS[dataset], default=SPLITS[dataset][0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fetch-only", action="store_true", help="Download and verify raw data only")
    mode.add_argument(
        "--convert-only", action="store_true", help="Convert existing raw data offline"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        result = prepare_dataset(
            dataset,
            args.data_dir.resolve(),
            args.split,
            fetch_only=args.fetch_only,
            convert_only=args.convert_only,
        )
    except (ValueError, OSError, httpx.HTTPError, yaml.YAMLError) as exc:
        parser.exit(1, f"Data preparation failed: {exc}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0
