import json

import pytest

from decisionforge.eval.data import load_samples, select_samples
from decisionforge.eval.models import EvalConfig, Sample, load_config
from tests.eval_fixtures import sample


@pytest.mark.parametrize(
    "kind,target",
    [
        ("choice", "hard"),
        ("choice", "soft"),
        ("noul", "hard"),
        ("noul", "soft"),
        ("score", "hard"),
        ("score", "soft"),
        ("score", "scalar"),
        ("choice", None),
    ],
)
def test_native_contracts(kind, target):
    value = Sample.model_validate(sample(kind, target))
    assert value.request.state["condition"] == "limit"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.update(extra=1),
        lambda s: s["request"]["questions"].update(other={"type": "noul"}),
        lambda s: s["target"].update(probabilities={"A": 1}),
        lambda s: s["target"].update(value=3),
        lambda s: s["request"].update(model="secret"),
    ],
)
def test_rejects_incompatible_contract(mutate):
    value = sample()
    mutate(value)
    with pytest.raises(ValueError):
        Sample.model_validate(value)


def test_soft_and_scalar_do_not_guess_labels():
    value = sample("noul")
    value["target"]["value"] = 1
    with pytest.raises(ValueError):
        Sample.model_validate(value)
    value = sample("choice", "soft")
    value["target"]["probabilities"] = {"A": 0.1, "B": 0.8}
    with pytest.raises(ValueError):
        Sample.model_validate(value)
    value["target"]["probabilities"] = {"A": float("nan"), "B": 1}
    with pytest.raises(ValueError):
        Sample.model_validate(value)
    with pytest.raises(ValueError):
        Sample.model_validate(sample("choice", "scalar"))


def test_optional_native_instructions_and_repeated_descriptions():
    value = sample()
    del value["request"]["questions"]["decision"]["instructions"]
    value["request"]["questions"]["decision"]["criteria"] = {"A": None, "B": None}
    assert Sample.model_validate(value).request.questions["decision"].instructions is None


def test_full_file_preflight_and_duplicate_keys(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps(sample()) + "\n" + json.dumps(sample()) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_samples(path)
    path.write_text('{"id":"first","id":"second"}\n')
    with pytest.raises(ValueError, match="line 1"):
        load_samples(path)
    bad = sample(identity="bad")
    bad["split"] = ""
    path.write_text(json.dumps(sample()) + "\n" + json.dumps(bad) + "\n")
    with pytest.raises(ValueError, match="bad"):
        select_samples(load_samples(path), EvalConfig(data={"path": str(path), "limit": 1}).data)


def test_config_paths_and_invalid_operational_values(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("data:\n  path: data.jsonl\noutput: runs/first\n")
    config = load_config(path)
    assert config.data.path == str(tmp_path / "data.jsonl")
    assert config.output == str(tmp_path / "runs/first")
    for settings in ({"timeout": float("inf")}, {"concurrency": 0}, {"max_attempts": True}):
        with pytest.raises(ValueError):
            EvalConfig(data={"path": "data.jsonl"}, execution=settings)
