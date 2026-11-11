#!/usr/bin/env python3
"""Reproduce CPU-only exact-template evidence; never fetch/load model weights."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from decisionforge.training.backends.swift_routes import get_route
from decisionforge.training.backends.swift_routes.sdk import metadata_snapshot
from decisionforge.training.backends.swift_runtime import installation

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    evidence = {
        "schema_version": "decisionforge.swift.cpu-routes.v1",
        "upstream": installation(),
        "no_gpu": True,
        "no_model_weights": True,
        "routes": {},
    }
    for name in ("clef", "kev", "jev", "startlux"):
        route = get_route(name)
        _, files = metadata_snapshot(route)
        config = ROOT / f"examples/training/swift/{name}.yaml"
        command = [
            sys.executable,
            "-m",
            "decisionforge",
            "train",
            "run", "--config",
            str(config),
            "--validate-only",
        ]
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        (args.output / f"{name}.stderr.log").write_text(result.stderr)
        (args.output / f"{name}.stdout.json").write_text(result.stdout)
        if result.returncode:
            raise RuntimeError(f"{name} validate-only failed: {result.stderr}")
        report = json.loads(result.stdout)
        evidence["routes"][name] = {
            "exit_code": result.returncode,
            "model": route.model,
            "revision": route.revision,
            "metadata_files_sha256": files,
            "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            "report": report,
        }
    (args.output / "encoding-evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
