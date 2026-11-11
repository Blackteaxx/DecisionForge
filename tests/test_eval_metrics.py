import pytest

from decisionforge.eval.benchmarks import GenericBenchmark
from decisionforge.eval.metrics import Metric
from decisionforge.eval.models import Sample
from decisionforge.eval.registry import Registry, load_registry
from tests.eval_fixtures import prediction, sample


def evaluate(items, predictions, metrics=None):
    registry = load_registry()
    return GenericBenchmark().evaluate(
        [Sample.model_validate(s) for s in items], predictions, registry, metrics or {}
    )


def test_accuracy_failure_denominator_and_calibration_probability():
    result = evaluate(
        [sample(identity="one"), sample(identity="two")],
        [prediction(), prediction("two", valid=False)],
    )
    assert result["metrics"]["accuracy"]["value"] == 0.5
    assert result["metrics"]["brier"]["value"] == pytest.approx(0.18)
    assert result["metrics"]["ece"]["value"] == pytest.approx(0.3)
    assert result["metrics"]["ece"]["used"] == 1
    assert result["metrics"]["ece"]["failed"] == 1


def test_soft_reference_and_failure_keep_identity():
    result = evaluate(
        [sample(target="soft"), sample(target="soft", identity="two")],
        [prediction(), prediction("two", valid=False)],
    )["metrics"]
    assert result["soft_accuracy"]["value"] == 0.4
    assert result["soft_brier"]["value"] == pytest.approx(0.02)
    assert result["soft_ece"]["value"] == pytest.approx(0.1)
    assert result["accuracy"]["value"] is None


def test_noul_brier_two_class_and_scalar_mae():
    result = evaluate([sample("noul")], [prediction(kind="noul")])
    assert result["metrics"]["brier"]["value"] == pytest.approx(0.08)
    result = evaluate([sample("score", "scalar")], [prediction(kind="score")])
    assert result["metrics"]["mae"]["value"] == pytest.approx(0.1)
    assert result["metrics"]["brier"]["value"] is None


def test_ece_boundaries_and_unattempted_are_not_zero():
    result = evaluate(
        [sample(identity="zero"), sample(identity="one"), sample(identity="unattempted")],
        [
            prediction("zero", probabilities={"A": 1, "B": 0}),
            prediction("one", probabilities={"A": 0, "B": 1}),
        ],
        {"ece": {"bins": 10}},
    )
    metric = result["metrics"]["ece"]
    assert metric["value"] == 0.5
    assert metric["bins"][-1]["n"] == 2
    assert result["counts"]["unattempted"] == 1
    assert result["partial"] is True


def test_nonunit_probability_total_is_scored_using_original_values():
    original = prediction(probabilities={"A": 0.3, "B": 0.8})
    result = evaluate([sample()], [original])
    assert original["answer"]["probabilities"]["B"] == 0.8
    assert result["metrics"]["accuracy"]["value"] == 1
    assert result["metrics"]["brier"]["value"] == pytest.approx(0.13)
    assert result["counts"]["scoring_invalid"] == 0


def test_registry_custom_aggregate_and_preflight_parameters():
    registry = Registry()
    registry.register_metric(
        Metric(
            "custom",
            "1",
            "higher",
            "ratio",
            lambda rows, params: {"value": 0.42, "used": len(rows)},
        )
    )
    with pytest.raises(ValueError, match="duplicate"):
        registry.register_metric(registry.metric("custom"))
    with pytest.raises(ValueError, match="Unknown"):
        registry.metric("missing")
    loaded = load_registry()
    with pytest.raises(ValueError):
        GenericBenchmark().validate(loaded, {"ece": {"bins": 0}})
    with pytest.raises(ValueError):
        GenericBenchmark().validate(loaded, {"ece": {"bins": True}})
    result = loaded.benchmark("demo").evaluate(
        [Sample.model_validate(sample())], [prediction()], loaded, {}
    )
    assert "group_agreement" in result["metrics"]


def test_documented_entry_point_plugins(monkeypatch):
    from types import SimpleNamespace

    import decisionforge.eval.registry as module
    from extensions.example_plugin.jev_eval_example import benchmark, group_metric

    def entries(*, group):
        factory = group_metric if group.endswith("metrics") else benchmark
        return [SimpleNamespace(load=lambda: factory)]

    monkeypatch.setattr(module, "entry_points", entries)
    registry = load_registry()
    items = [sample(identity=i) for i in ("one", "two")]
    for item in items:
        item["group_id"] = "pair"
    result = registry.benchmark("custom_groups").evaluate(
        [Sample.model_validate(s) for s in items],
        [prediction(i) for i in ("one", "two")],
        registry,
        {},
    )
    assert result["metrics"]["unanimous_groups"]["value"] == 1
    assert result["metrics"]["unanimous_groups"]["unit_of_aggregation"] == "group"


@pytest.mark.parametrize("target,expected", [("hard", 0.4), ("soft", 0.0), ("scalar", 0.1)])
def test_native_mae_remains_usable_when_categorical_distribution_is_invalid(target, expected):
    items = [sample("score", target)]
    pred = prediction(kind="score", probabilities={"0": -0.1, "1": 0.1, "2": 0.1})
    result = evaluate(items, [pred])
    assert result["counts"]["scoring_invalid"] == 1
    assert result["metrics"]["mae"]["value"] == pytest.approx(expected)
    assert result["metrics"]["mae"]["used"] == 1
    assert result["metrics"]["mae"]["failed"] == 0
    from decisionforge.eval.jevbench import JevBench

    registry = load_registry()
    hard = Sample.model_validate(sample("score"))
    assert JevBench().evaluate([hard], [pred], registry, {})["metrics"]["mae"]["value"] is None


def test_extreme_integer_probability_is_invalid_without_overflow():
    pred = prediction(probabilities={"A": 10**400, "B": 0})
    result = evaluate([sample()], [pred])
    assert result["counts"]["scoring_invalid"] == 1
    assert result["metrics"]["accuracy"]["value"] == 0
