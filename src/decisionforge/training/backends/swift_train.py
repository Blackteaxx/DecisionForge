"""Small subprocess entry point so ms-swift owns its normal trainer lifecycle."""

from __future__ import annotations

import json
import os
from pathlib import Path


def main():
    from swift.pipelines.train.sft import SwiftSft

    class CheckedSwiftSft(SwiftSft):
        def _prepare_dataset(self):
            train, validation = super()._prepare_dataset()
            root = Path(os.environ["DFORGE_SWIFT_DATA"])
            expected_train = sum(bool(line.strip()) for line in (root / "train.jsonl").open())
            expected_calibration = (
                sum(bool(line.strip()) for line in (root / "calibration.jsonl").open())
                if os.environ["DFORGE_SWIFT_CALIBRATE"] == "1"
                else 0
            )
            if (
                len(train) != expected_train
                or (len(validation) if validation is not None else 0) != expected_calibration
            ):
                raise ValueError(
                    "ms-swift changed dataset cardinality; refusing deletion/subsampling"
                )
            return train, validation

        def train(self, trainer):
            self.dforge_trainer = trainer
            return super().train(trainer)

    pipeline = CheckedSwiftSft()
    result = pipeline.main()
    # SwiftSft returns a durable train_msg; retain it outside the checkpoint so
    # DecisionForge can prove which checkpoint was selected.
    output = os.environ.get("SWIFT_OUTPUT_DIR")
    if output and isinstance(result, dict):
        final = Path(output) / "dforge_final"
        pipeline.dforge_trainer.save_model(str(final))
        if os.environ.get("DFORGE_SWIFT_MODEL_TYPE", "omnijev") != "omnijev":
            from swift.model.decision_head import get_scoring_head

            get_scoring_head(pipeline.dforge_trainer.model).save_pretrained(
                str(final), save_weights=True
            )
        result["last_model_checkpoint"] = str(final)
        with open(os.path.join(output, "dforge_training.json"), "w", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
    return result


if __name__ == "__main__":
    main()
