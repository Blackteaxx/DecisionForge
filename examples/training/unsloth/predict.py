"""Reload a saved adapter or merged decision model and predict public JEV requests."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from decisionforge.training.unsloth_backend import native_prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--request",
        required=True,
        type=Path,
        help='JSON object containing only "state" and "questions"',
    )
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    if set(request) != {"state", "questions"}:
        parser.error("Provide a public request with only state and questions, no supervision")
    # Required initialization order: Unsloth before torch/Transformers.
    unsloth = importlib.import_module("unsloth")
    torch = importlib.import_module("torch")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model, tokenizer = unsloth.FastDecisionModel.from_pretrained(
        args.model,
        dtype=dtype,
        load_in_4bit=args.load_in_4bit,
        use_gradient_checkpointing=False,
        local_files_only=Path(args.model).is_dir(),
    )
    answers = unsloth.FastDecisionModel.predict(model, tokenizer, **request)
    print(
        json.dumps(native_prediction(answers, request["questions"]), indent=2, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
