"""Validation selection must use complete committed checkpoints from finished raw training."""

import json

import pytest


def test_selection_discovers_only_committed_checkpoints_in_step_order(tmp_path):
    from examples.domain.medical_triage.scripts.v1.select_model import (
        checkpoints_for_selection,
    )

    (tmp_path / "manifest.json").write_text(json.dumps({"status": "completed"}))
    (tmp_path / "config.yaml").write_text("calibration: {enabled: false}\n")
    for step in (200, 100, 150):
        p = tmp_path / f"checkpoints/checkpoint-{step}"
        p.mkdir(parents=True)
        if step != 150:
            (p / "dforge_checkpoint.json").write_text("{}")
    assert [p.name for p in checkpoints_for_selection(tmp_path)] == [
        "checkpoint-100",
        "checkpoint-200",
    ]


def test_selection_refuses_nonterminal_training_or_automatic_last_checkpoint_calibration(tmp_path):
    from examples.domain.medical_triage.scripts.v1.select_model import (
        checkpoints_for_selection,
    )

    for status, enabled in [("running", False), ("completed", True)]:
        (tmp_path / "manifest.json").write_text(json.dumps({"status": status}))
        (tmp_path / "config.yaml").write_text(
            "calibration: {enabled: " + str(enabled).lower() + "}\n"
        )
        with pytest.raises(ValueError, match="completed|calibrat"):
            checkpoints_for_selection(tmp_path)
