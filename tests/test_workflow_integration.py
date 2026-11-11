"""Generation -> training -> held-out local evaluation, offline."""

import json
from pathlib import Path

import pytest
from conftest import FixtureProvider
from test_eval_local import checkpoint_at
from test_training_runner import FakeBackend

from decisionforge.eval.data import load_samples
from decisionforge.eval.local_adapter import LocalAdapter
from decisionforge.eval.models import EvalConfig
from decisionforge.eval.runner import run_evaluation
from decisionforge.exporter import read_training_records
from decisionforge.pipeline import Pipeline
from decisionforge.storage import Store
from decisionforge.training.config import TrainConfig
from decisionforge.training.runner import run_training


def decision_worker(connection, options):
    from decisionforge.eval.local_worker import answer_from_probabilities

    connection.send({"kind": "ready", "metadata": {"model": options["path"]}})
    while True:
        message = connection.recv()
        if message["kind"] == "close":
            break
        request = message["request"]
        assert set(request) == {"state", "questions"}
        question = request["questions"]["decision"]
        labels = (
            list(question["criteria"])
            if question["type"] == "choice"
            else ["false", "true"]
            if question["type"] == "noul"
            else [str(i) for i in range(len(question["criteria"]))]
        )
        answer = answer_from_probabilities(question, dict.fromkeys(labels, 1 / len(labels)))
        connection.send(
            {
                "kind": "prediction",
                "response": {"model": options["path"], "answers": {"decision": answer}},
                "input_tokens": 16,
            }
        )
    connection.close()


@pytest.mark.asyncio
async def test_generated_data_connects_training_and_heldout_eval(config, tmp_path):
    rows = [
        {"question": f"Case {i}: What is 2+2?", "answer": "4", "group_id": f"case-{i}"}
        for i in range(12)
    ]
    Path(config.input.path).write_text(json.dumps(rows), encoding="utf-8")
    config.task.modes = ["choice", "noul", "score"]
    directory = Path(config.output.directory)
    store = Store(directory)
    try:
        result = await Pipeline(config, store, FixtureProvider()).run()
    finally:
        store.close()
    assert result["accepted"] == 36
    training = TrainConfig(
        data={
            "path": str(directory / "accepted.json"),
            "auto_holdout": {"validation_fraction": 0.25},
        },
        calibration={"enabled": False},
        output={"directory": str(tmp_path / "training")},
    )
    FakeBackend.fail = None
    FakeBackend.calls = []
    trained = run_training(training, backend_factory=FakeBackend)
    assert trained["status"] == "completed"
    run = Path(training.output.directory)
    original = list(read_training_records(run / "samples.jsonl"))
    train = list(read_training_records(run / "data/train.jsonl"))
    heldout = list(read_training_records(run / "data/validation.jsonl"))
    assert len(train) == 27 and len(heldout) == 9
    assert {r.split for r in original} == {"train"}
    assert {r.split for r in heldout} == {"validation"}
    assert not ({r.group_id for r in train} & {r.group_id for r in heldout})
    assert {q.type for r in heldout for q in r.request.questions.values()} == {
        "choice",
        "noul",
        "score",
    }
    samples = load_samples(run / "data/validation.jsonl")
    assert len(samples) == 9 and {s.split for s in samples} == {"validation"}
    model = checkpoint_at(run / "model")  # Format fixture, not learned weights.
    evaluation = EvalConfig(
        data={"path": str(run / "data/validation.jsonl"), "splits": ["validation"]},
        model={"adapter": "unsloth_local", "path": str(model)},
        output=str(tmp_path / "evaluation"),
    )

    async def factory(model, execution, transport=None):
        return await LocalAdapter(model, execution, worker_target=decision_worker).start()

    evaluated = await run_evaluation(evaluation, adapter_factory=factory)
    report = json.loads(
        (Path(evaluation.output) / evaluated["report"] / "metrics.json").read_text()
    )
    assert report["counts"]["strict_valid"] == 9
    assert report["counts"]["scoring_invalid"] == 0
    assert report["performance"]["cost_basis"] == "local_compute_unpriced"
