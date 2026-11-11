"""Run generation -> training -> evaluation using each stage's actual output path."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv  # noqa: E402

from decisionforge.config import load_config as load_generation  # noqa: E402
from decisionforge.eval.models import load_config as load_evaluation  # noqa: E402
from decisionforge.training.config import load_config as load_training  # noqa: E402
from decisionforge.util import json_text  # noqa: E402
from decisionforge.workflow import run_workflow  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-config", required=True)
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--eval-config", required=True)
    parser.add_argument("--resume", action="store_true", help="Resume training/eval if present")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    result = asyncio.run(
        run_workflow(
            load_generation(args.generation_config),
            load_training(args.training_config),
            load_evaluation(args.eval_config),
            resume=args.resume,
        )
    )
    print(json_text(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
