"""Merged contracts and orchestration, without paid APIs or learned model weights."""

import json
from pathlib import Path

import httpx
import pytest
from conftest import FixtureProvider
from test_eval_local import checkpoint_at
from test_training_data import example
from test_training_runner import FakeBackend

from decisionforge import workflow
from decisionforge.eval.adapter import JevAdapter
from decisionforge.eval.data import load_samples
from decisionforge.eval.local_worker import answer_from_probabilities
from decisionforge.eval.models import EvalConfig, ModelConfig
from decisionforge.eval.runner import run_evaluation
from decisionforge.models import TrainingExample
from decisionforge.pipeline import Pipeline
from decisionforge.storage import Store
from decisionforge.training.config import TrainConfig
from decisionforge.training.data import to_unsloth_row
from decisionforge.training.runner import run_training


def response(request):
    payload = json.loads(request.content)
    chat = "messages" in payload
    public = json.loads(payload["messages"][1]["content"]) if chat else payload
    assert "supervision" not in public and "target" not in public
    question = public["questions"]["decision"]
    labels = (
        list(question["criteria"])
        if question["type"] == "choice"
        else ["false", "true"]
        if question["type"] == "noul"
        else [str(i) for i in range(len(question["criteria"]))]
    )
    answer = answer_from_probabilities(question, dict.fromkeys(labels, 1 / len(labels)))
    if chat and question["type"] == "noul":
        # Chat's closed response schema only exposes the native scalar Noul value.
        answer.pop("probabilities")
    value = {"model": "fixture", "answers": {"decision": answer}}
    if chat:
        value = {
            "model": "fixture",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({"answers": value["answers"]}),
                    },
                }
            ],
        }
    return httpx.Response(200, json=value)


async def generate_fixture(config):
    store = Store(Path(config.output.directory))
    try:
        result = await Pipeline(config, store, FixtureProvider()).run()
        return {**result, "run_directory": config.output.directory}
    finally:
        store.close()


def setup_configs(config, tmp_path):
    Path(config.input.path).write_text(
        json.dumps(
            [
                {"question": f"Case {i}: What is 2+2?", "answer": "4", "group_id": f"case-{i}"}
                for i in range(12)
            ]
        )
    )
    config.task.modes = ["choice", "noul", "score"]
    config.output.validation_fraction = 0.25
    training = TrainConfig(
        data={"path": str(tmp_path / "wrong-old-generation.json")},
        calibration={"enabled": False},
        output={"directory": str(tmp_path / "training")},
    )
    evaluation = EvalConfig(
        data={"path": str(tmp_path / "wrong-old-validation.jsonl"), "splits": ["validation"]},
        model={"adapter": "unsloth_local", "path": str(tmp_path / "wrong-old-model")},
        output=str(tmp_path / "evaluation"),
    )
    return training, evaluation


@pytest.mark.parametrize("versioned", [False, True])
async def test_workflow_uses_actual_outputs_and_resumes_completed_stages(
    config, tmp_path, monkeypatch, versioned
):
    training, evaluation = setup_configs(config, tmp_path)
    expected = Path(config.output.directory + ("_v_fixture" if versioned else ""))

    async def generate(current):
        # execute() may change the config it was given when selecting a version.
        current.output.directory = str(expected)
        return await generate_fixture(current)

    def train(current, **kwargs):
        FakeBackend.fail, FakeBackend.calls = None, []
        result = run_training(current, backend_factory=FakeBackend, **kwargs)
        checkpoint_at(Path(current.output.directory) / "model")
        return result

    async def factory(model, execution, transport=None):
        adapter = JevAdapter(ModelConfig(api_key_env=None), execution, httpx.MockTransport(response))
        adapter.metadata = {"model": model.path}
        return adapter

    async def evaluate(current, **kwargs):
        assert current.model.path == str(Path(current.data.path).parents[1] / "model")
        return await run_evaluation(current, adapter_factory=factory, **kwargs)

    monkeypatch.setattr(workflow, "generate", generate)
    monkeypatch.setattr(workflow, "run_training", train)
    monkeypatch.setattr(workflow, "run_evaluation", evaluate)
    result = await workflow.run_workflow(config, training, evaluation)
    assert result["generation"]["run_directory"] == str(expected)
    assert result["generation"]["accepted"] == 36
    train_dir = Path(result["training"]["run_directory"])
    assert (train_dir != Path(training.output.directory)) == versioned
    saved = json.loads((train_dir / "manifest.json").read_text())
    assert saved["input_path"] == str(expected / "accepted.json")
    heldout = load_samples(train_dir / "data/validation.jsonl")
    assert len(heldout) == 9
    assert {s.question.type for s in heldout} == {"choice", "noul", "score"}
    before = (expected / "accepted.json").read_bytes()

    def no_training(*args, **kwargs):
        pytest.fail("Completed training must not be repeated on resume")

    monkeypatch.setattr(workflow, "run_training", no_training)
    again = await workflow.run_workflow(config, training, evaluation, resume=True)
    assert again["training"]["reused"]
    assert (expected / "accepted.json").read_bytes() == before
    assert config.output.directory != str(expected) or not versioned
    assert training.data.path.endswith("wrong-old-generation.json")
    # Reusing completed training must still reject a changed effective configuration.
    training.training.learning_rate *= 2
    with pytest.raises(ValueError, match="configuration changed"):
        await workflow.run_workflow(config, training, evaluation, resume=True)


async def test_incomplete_generation_does_not_start_training(config, tmp_path, monkeypatch):
    training, evaluation = setup_configs(config, tmp_path)

    async def paused(current):
        return {"status": "budget_exhausted", "run_directory": current.output.directory}

    monkeypatch.setattr(workflow, "generate", paused)
    with pytest.raises(ValueError, match="resume it before training"):
        await workflow.run_workflow(config, training, evaluation)
    assert not Path(training.output.directory).exists()


@pytest.mark.parametrize("adapter", ["jev", "chat_api"])
async def test_generated_v3_is_accepted_by_merged_http_adapters(config, tmp_path, adapter):
    setup_configs(config, tmp_path)
    generated = await generate_fixture(config)
    evaluation = EvalConfig(
        data={
            "path": str(Path(generated["run_directory"]) / "accepted.json"),
            "splits": ["validation"],
        },
        model={"adapter": adapter, "model": "fixture", "api_key_env": None},
        output=str(tmp_path / adapter),
    )
    result = await run_evaluation(evaluation, transport=httpx.MockTransport(response))
    report = json.loads((Path(evaluation.output) / result["report"] / "metrics.json").read_text())
    assert report["counts"]["strict_valid"] == 9
    assert report["counts"]["scoring_invalid"] == 0
    assert report["prediction_provenance"]["method"] == (
        "chat_baseline" if adapter == "chat_api" else "native"
    )


@pytest.mark.parametrize("extension", [".json", ".jsonl"])
def test_multi_question_soft_targets_match_training_and_eval(tmp_path, extension):
    record = example(split="validation")
    path = tmp_path / ("accepted" + extension)
    path.write_text(json.dumps([record] if extension == ".json" else record) + "\n")
    train_row = to_unsloth_row(TrainingExample.model_validate(record))
    samples = load_samples(path)
    assert len(samples) == 3
    assert {s.group_id for s in samples} == {"1"}
    assert {s.split for s in samples} == {"validation"}
    for sample in samples:
        name = sample.id.split(":", 1)[1]
        assert sample.request.state == record["request"]["state"]
        assert sample.target.kind == "soft"
        assert sample.target.probabilities == train_row["gold"][name]["probabilities"]
        assert sample.question.model_dump(exclude_none=True) == (
            sample.question.model_validate(record["request"]["questions"][name])
            .model_dump(exclude_none=True)
        )
