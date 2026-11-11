"""Only independent calibration records may enter selected-checkpoint temperature fitting."""

import pytest

from decisionforge.models import TrainingExample
from examples.domain.medical_triage.scripts.prepare_benchmarks import (
    PAIR_CRITERIA,
    TRIAGE_CRITERIA,
)


def record(identity, role, group, pair=False):
    criteria = PAIR_CRITERIA if pair else TRIAGE_CRITERIA
    return TrainingExample.model_validate(
        {
            "schema_version": "decisionforge.training.v3",
            "record_id": identity,
            "group_id": group,
            "split": role,
            "request": {
                "state": f"Distinct synthetic facts for patient {identity}",
                "questions": {
                    "q1": {
                        "type": "choice",
                        "instructions": "Choose the current action.",
                        "criteria": criteria,
                    }
                },
            },
            "supervision": {
                "answers": {
                    "q1": {
                        "type": "choice",
                        "choice": "A",
                        "confidence": 1,
                        "probabilities": {k: float(k == "A") for k in criteria},
                    }
                }
            },
        }
    )


def test_calibration_admission_excludes_train_validation_and_test():
    from examples.domain.medical_triage.scripts.v1.calibrate import calibration_records

    calibration = [record(i, "calibration", 10, pair=bool(i % 2)) for i in range(1, 11)]
    all_rows = [
        *calibration,
        record(11, "train", 20),
        record(12, "validation", 30),
        record(13, "test", 40),
    ]
    assert calibration_records(all_rows) == calibration


def test_calibration_admission_refuses_cross_role_source_group():
    from examples.domain.medical_triage.scripts.v1.calibrate import calibration_records

    rows = [record(i, "calibration", 10, pair=bool(i % 2)) for i in range(1, 11)]
    rows.append(record(11, "train", 10))
    with pytest.raises(ValueError, match="group|role"):
        calibration_records(rows)


def test_calibration_requires_enough_examples_and_both_joint_tasks():
    from examples.domain.medical_triage.scripts.v1.calibrate import calibration_records

    with pytest.raises(ValueError, match="10|enough"):
        calibration_records([record(1, "calibration", 10)])
    with pytest.raises(ValueError, match="both|joint"):
        calibration_records([record(i, "calibration", 10) for i in range(1, 11)])
