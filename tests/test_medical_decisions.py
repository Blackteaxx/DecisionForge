"""Accuracy must not change its denominator or silently replace native Choice."""

import sys
from pathlib import Path

import pytest

from decisionforge.eval.models import Sample
from decisionforge.eval.registry import load_registry


@pytest.fixture
def medical_registry(monkeypatch):
    extension = Path(__file__).parents[1] / "examples/domain/medical_triage/benchmark_ext"
    monkeypatch.syspath_prepend(str(extension))
    from medical_triage_benchmark import register

    registry = load_registry()
    register(registry)
    yield registry
    for name in list(sys.modules):
        if name == "medical_triage_benchmark" or name.startswith("medical_triage_benchmark."):
            sys.modules.pop(name)


def sample(i, gold="A"):
    return Sample.model_validate(
        {
            "schema_version": "decisionforge.eval.v1",
            "id": str(i),
            "group_id": None,
            "split": "test",
            "subset": "triage_action",
            "request": {
                "state": "A patient message",
                "questions": {
                    "decision": {
                        "type": "choice",
                        "instructions": "Choose.",
                        "criteria": {"A": "First action", "B": "Second action"},
                    }
                },
            },
            "target": {"kind": "hard", "value": gold},
            "metadata": {},
        }
    )


def prediction(i, choice="A", probabilities=None, transport=True):
    return {
        "id": str(i),
        "transport_ok": transport,
        "protocol_ok": transport,
        "terminal": True,
        "answer": {"choice": choice, "probabilities": probabilities or {"A": 0.1, "B": 0.9}},
    }


def test_native_choice_conflict_failure_and_absent_prediction_keep_full_denominator(medical_registry):
    registry = medical_registry
    result = registry.benchmark("medical_decisions").evaluate(
        [sample(1), sample(2), sample(3)],
        [prediction(1), prediction(2, transport=False)],
        registry,
        None,
    )
    assert result["metrics"]["medical_accuracy"]["value"] == pytest.approx(1 / 3)
    assert result["metrics"]["medical_accuracy"]["eligible"] == 3
    assert result["metrics"]["medical_accuracy"]["failed"] == 2
    assert result["counts"]["planned"] == 3
    assert result["counts"]["unattempted"] == 1
    assert result["counts"]["choice_argmax_disagreements"] == 1
    assert result["decision_rule"] == "native_choice"
    assert result["metrics"]["medical_macro_f1"]["value"] == pytest.approx(0.25)
    assert result["partial"] is True


def test_probability_metrics_report_coverage_and_never_renormalize(medical_registry):
    registry = medical_registry
    result = registry.benchmark("medical_decisions").evaluate(
        [sample(1), sample(2)],
        [
            prediction(1, probabilities={"A": 0.2, "B": 0.2}),
            prediction(2, choice="A", probabilities={"A": 0.75, "B": 0.25}),
        ],
        registry,
        None,
    )
    assert result["metrics"]["medical_brier"]["value"] == pytest.approx(0.125)
    assert result["metrics"]["medical_brier"]["used"] == 1
    assert result["metrics"]["medical_brier"]["eligible"] == 2
    assert result["metrics"]["medical_brier"]["coverage"] == 0.5
    assert result["counts"]["renormalized"] == 0
    assert result["counts"]["strict_valid"] == 1
    assert result["metrics"]["medical_ece"]["value"] == 0.25


def test_invalid_native_choice_is_not_replaced_by_good_probabilities(medical_registry):
    registry = medical_registry
    result = registry.benchmark("medical_decisions").evaluate(
        [sample(1)], [prediction(1, choice="X", probabilities={"A": 1.0, "B": 0.0})], registry, None
    )
    assert result["metrics"]["medical_accuracy"]["value"] == 0
    assert result["metrics"]["medical_macro_f1"]["per_class"]["A"]["fn"] == 1


def test_exclusion_flag_cannot_shrink_official_benchmark(medical_registry):
    registry = medical_registry
    row = sample(1).model_copy(update={"metadata": {"exclude_reason": "hard case"}})
    with pytest.raises(ValueError, match="exclu"):
        registry.benchmark("medical_decisions").evaluate([row], [], registry, None)
