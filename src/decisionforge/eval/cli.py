from __future__ import annotations

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from decisionforge.eval.models import load_config
from decisionforge.eval.registry import load_registry
from decisionforge.eval.runner import run_evaluation


def execute(args):
    if args.command == "eval-list":
        registry = load_registry()
        return {
            "benchmarks": [
                {
                    "name": b.name,
                    "version": b.version,
                    "supported_types": sorted(b.supported_types),
                    "default_metrics": getattr(b, "default_metrics", {}),
                }
                for b in sorted(registry.benchmarks.values(), key=lambda b: b.name)
            ],
            "metrics": [
                {"name": m.name, "version": m.version, "direction": m.direction, "unit": m.unit}
                for m in sorted(registry.metrics.values(), key=lambda m: m.name)
            ],
        }
    config = load_config(args.config)
    if args.run_dir:
        config.output = str(Path(args.run_dir).resolve())
    if args.env_file and not (args.score_only or args.validate_only):
        if not Path(args.env_file).is_file():
            raise ValueError("Credential environment file does not exist")
        load_dotenv(args.env_file, override=False)
    return asyncio.run(
        run_evaluation(
            config, resume=args.resume, score_only=args.score_only, validate_only=args.validate_only
        )
    )
