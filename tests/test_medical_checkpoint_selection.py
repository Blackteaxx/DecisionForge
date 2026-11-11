"""Checkpoint selection must use independent, equally weighted tasks, not test scores."""

import pytest


def candidate(name, step, triage, pair, brier=0.4):
    return {
        "checkpoint": name,
        "step": step,
        "split": "validation",
        "status": "completed",
        "tasks": {
            "triage_action": {
                "accuracy": triage,
                "brier": brier,
                "count": 100,
                "data_sha256": "fixed-triage",
            },
            "message_priority": {
                "accuracy": pair,
                "brier": brier,
                "count": 10,
                "data_sha256": "fixed-pairs",
            },
        },
    }


def test_selection_weights_tasks_equally_despite_different_sample_counts():
    from examples.domain.medical_triage.scripts.v1.experiment import select_checkpoint

    result = select_checkpoint(
        [candidate("pooled-winner", 100, 0.9, 0.1), candidate("balanced-winner", 200, 0.6, 0.6)]
    )
    assert result["checkpoint"] == "balanced-winner"
    assert result["selection_score"] == 0.6


def test_selection_breaks_accuracy_ties_by_brier_then_earlier_step():
    from examples.domain.medical_triage.scripts.v1.experiment import select_checkpoint

    result = select_checkpoint(
        [
            candidate("worse-brier", 10, 0.7, 0.7, 0.5),
            candidate("later", 200, 0.7, 0.7, 0.3),
            candidate("earlier", 100, 0.7, 0.7, 0.3),
        ]
    )
    assert result["checkpoint"] == "earlier"


@pytest.mark.parametrize("change", [{"split": "test"}, {"status": "running"}])
def test_selection_refuses_test_derived_or_nonterminal_results(change):
    from examples.domain.medical_triage.scripts.v1.experiment import select_checkpoint

    bad = dict(candidate("not-eligible", 100, 0.9, 0.9), **change)
    with pytest.raises(ValueError, match="validation|completed"):
        select_checkpoint([bad])


def test_selection_refuses_different_validation_inputs_across_checkpoints():
    from examples.domain.medical_triage.scripts.v1.experiment import select_checkpoint

    first = candidate("first", 100, 0.7, 0.7)
    second = candidate("second", 200, 0.8, 0.8)
    second["tasks"]["message_priority"]["data_sha256"] = "different-data"
    with pytest.raises(ValueError, match="same|frozen"):
        select_checkpoint([first, second])


def test_validation_questions_and_gold_are_canonicalized_by_content():
    from examples.domain.medical_triage.scripts.v1.experiment import evaluation_samples

    record = {
        "record_id": 1,
        "group_id": 5,
        "split": "validation",
        "request": {
            "state": "Current patient facts",
            "questions": {
                "q1": {
                    "type": "choice",
                    "instructions": "Choose the appropriate current action.",
                    "criteria": {
                        "A": "Offer self-care, monitoring, reassurance, or conditional safety-netting.",
                        "B": "Recommend urgent or emergency-level evaluation now.",
                        "C": "Ask the smallest necessary clarification; current information is insufficient.",
                        "D": "Recommend current nonurgent professional evaluation or care.",
                    },
                }
            },
        },
        "supervision": {"answers": {"q1": {"choice": "B"}}},
    }
    samples = evaluation_samples([record], {1: "triage_action"}, "validation")
    assert samples[0]["target"] == {"kind": "hard", "value": "D"}
    assert samples[0]["request"]["questions"]["decision"]["criteria"]["D"] == (
        "Recommend urgent or emergency-level evaluation now."
    )
    assert samples[0]["request"]["state"] == "Current patient facts"
    assert samples[0]["split"] == "validation"
