from pathlib import Path

import pytest

from data.benchmarks._shared.converters import convert_jevbench
from decisionforge.eval.data import load_samples
from decisionforge.eval.jevbench import JevBench
from decisionforge.eval.registry import load_registry
from tests.eval_fixtures import prediction, sample
from tests.fixtures.jevbench_oracle.scoring import score_task
from tests.fixtures.jevbench_oracle.summarize import summarize
from tests.fixtures.jevbench_oracle.tasks import load_jsonl

FIXTURE = Path(__file__).parent / "fixtures/jevbench_oracle/original.jsonl"


def assert_oracle_equal(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_oracle_equal(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected, strict=True):
            assert_oracle_equal(a, e)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, abs=1e-12)
    else:
        assert actual == expected


def test_public_conversion_and_upstream_summary(tmp_path):
    converted = tmp_path / "converted.jsonl"
    manifest = convert_jevbench(FIXTURE, converted)
    samples, tasks = load_samples(converted), load_jsonl(str(FIXTURE))
    assert len(samples) == len(tasks) == manifest["records"]
    assert {s.question.type for s in samples} == {"choice", "noul", "score"}
    predictions, oracle_records = [], []
    for i, (s, t) in enumerate(zip(samples, tasks, strict=True)):
        labels = s.question.labels
        p = {k: 1 / len(labels) for k in labels}
        # Native Noul has no rounded distribution.
        if s.question.type == "noul":
            p = {"false": 0.2, "true": 0.8}
        pred = prediction(s.id, s.question.type, p)
        if s.question.type == "noul":
            pred["answer"]["noul"] = p["true"]
        predictions.append(pred)
        native = {"no": p["false"], "yes": p["true"]} if t.question["type"] == "noul" else p
        scored = score_task(native, t)
        oracle_records.append(
            {
                "task_id": t.id,
                "ok": True,
                "model": "fixture",
                "probs_source": "native",
                "cost_usd": None,
                "cost_basis": "unknown",
                "latency_s": None,
                **scored,
            }
        )
    result = JevBench().evaluate(samples, predictions, load_registry(), {})
    assert_oracle_equal(result["upstream_summary"], summarize(tasks, oracle_records))
    assert not any(d["renormalized"] for d in result["diagnostics"].values())


@pytest.mark.parametrize("total", [0.0, 1.0, 0.9995, 0.99, 0.8, 1.75])
def test_nonunit_vectors_keep_raw_values_and_lexicographic_ties(total):
    raw = sample()
    raw["metadata"].update(family="intent", labels=["A", "B"], provenance={})
    from decisionforge.eval.models import Sample

    s = Sample.model_validate(raw)
    pred = prediction(probabilities={"A": total / 2, "B": total / 2})
    row = JevBench().prepare(s, pred)
    assert row["scoring_valid"] is True
    assert row["strict_valid"] is True
    assert row["label"] == "A"
    assert row["probabilities"] == {"A": total / 2, "B": total / 2}
    assert row["renormalized"] is False
    assert pred["answer"]["probabilities"]["A"] == total / 2


def test_converter_rejects_wrong_source_labels_without_output(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '{"id":"x","state":"text","question":{"type":"noul","instructions":"Q"},'
        '"expected":"maybe","labels":["no","yes"],"family":"policy","split":"public"}'
    )
    output = tmp_path / "out.jsonl"
    with pytest.raises(ValueError):
        convert_jevbench(path, output)
    assert not output.exists()


def test_null_exclusion_failure_and_pair_boundaries_match_oracle(tmp_path):
    from decisionforge.eval.data import write_jsonl

    source = []
    for i in range(8):
        score = i < 2
        source.append(
            {
                "id": str(i),
                "family": "score" if score else "intent",
                "state": "Content",
                "question": {
                    "type": "score" if score else "choice",
                    "instructions": "Decide",
                    "criteria": ["low", "mid", "high"] if score else {"A": "a", "B": "b"},
                },
                "labels": ["0", "1", "2"] if score else ["A", "B"],
                "expected": None if score else "B",
                "split": "public",
                "group": "null-score-pair"
                if i < 2
                else "triple"
                if i < 5
                else "partial-pair"
                if i < 7
                else None,
                "provenance": {"exclude_reason": "ambiguous"} if i == 2 else {},
            }
        )
    path, converted = tmp_path / "source.jsonl", tmp_path / "converted.jsonl"
    write_jsonl(path, source)
    convert_jevbench(path, converted)
    tasks, samples = load_jsonl(str(path)), load_samples(converted)
    predictions, oracle = [], []
    for i, (t, s) in enumerate(zip(tasks, samples, strict=True)):
        if i == 6:
            continue
        p = (
            ({"0": 1.0, "1": 0.0, "2": 0.0} if i == 0 else {"0": 0.0, "1": 0.0, "2": 1.0})
            if i < 2
            else {"A": 0.1, "B": 0.9}
        )
        valid = i != 7
        pred = prediction(s.id, s.question.type, p, valid=valid)
        pred.update(attempt_latency_ms=10.0, cost_usd=None)
        predictions.append(pred)
        oracle.append(
            {
                "task_id": t.id,
                "ok": valid,
                "model": "fixture",
                "probs_source": "native",
                "latency_s": 0.01,
                "cost_usd": None,
                "cost_basis": "unknown",
                **score_task(p if valid else None, t),
            }
        )
    result = JevBench().evaluate(samples, predictions, load_registry(), {})
    assert_oracle_equal(result["upstream_summary"], summarize(tasks, oracle))
