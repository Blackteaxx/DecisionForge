import json
import subprocess
import sys

import pytest
from pydantic import ValidationError

from decisionforge.jevmath import choice_confidence, score_confidence


def example(i=1, split="train", group=None):
    return {
        "schema_version": "decisionforge.training.v3",
        "record_id": i,
        "group_id": group or i,
        "split": split,
        "request": {
            "state": {"text": f"case {i}"},
            "questions": {
                "decision": {
                    "type": "choice",
                    "instructions": ["Select"],
                    "criteria": {"b": {"text": "B"}, "a": None},
                },
                "rating": {"type": "score", "instructions": "Rate", "criteria": ["low", "high"]},
                "flag": {"type": "noul", "instructions": "Urgent?"},
            },
        },
        "supervision": {
            "answers": {
                "decision": {
                    "type": "choice",
                    "choice": "b",
                    "confidence": choice_confidence([0.7, 0.3]),
                    "probabilities": {"b": 0.7, "a": 0.3},
                },
                "rating": {
                    "type": "score",
                    "score": 0.4,
                    "confidence": score_confidence([0.6, 0.4]),
                    "probabilities": {"0": 0.6, "1": 0.4},
                    "legend": {"0": "low", "1": "high"},
                },
                "flag": {"type": "noul", "noul": 0.2},
            }
        },
    }


def make_config(tmp_path, rows=None, **overrides):
    from decisionforge.training.config import TrainConfig

    source = tmp_path / "data.jsonl"
    source.write_text(
        "".join(json.dumps(x) + "\n" for x in (rows or [example(i) for i in range(1, 101)]))
    )
    return TrainConfig.model_validate(
        {
            "data": {"path": str(source)},
            "model": {"name": "local/model"},
            "output": {"directory": str(tmp_path / "run")},
            **overrides,
        }
    )


def test_soft_multi_question_mapping():
    from decisionforge.models import TrainingExample
    from decisionforge.training.data import to_unsloth_row

    row = to_unsloth_row(TrainingExample.model_validate(example()))
    assert row["state"] == {"text": "case 1"}
    assert list(row["questions"]) == ["decision", "rating", "flag"]
    assert list(row["questions"]["decision"]["criteria"]) == ["b", "a"]
    assert row["gold"] == {
        "decision": {"probabilities": {"b": 0.7, "a": 0.3}},
        "rating": {"probabilities": {"0": 0.6, "1": 0.4}},
        "flag": {"probabilities": {"false": 0.8, "true": 0.2}},
    }
    assert set(row) == {"state", "questions", "gold"}


def test_config_strict_and_termination(tmp_path):
    from decisionforge.training.config import TrainConfig

    for patch in (
        {"backend": "unknown"},
        {"training": {"max_steps": 3}},
        {"training": {"max_steps": None, "num_train_epochs": None}},
        {"training": {"learning_rate": float("nan")}},
        {"model": {"kwargs": {}}},
    ):
        with pytest.raises(ValidationError):
            make_config(tmp_path, **patch)
    c = make_config(tmp_path, training={"max_steps": 3, "num_train_epochs": None})
    assert c.training.max_steps == 3
    assert TrainConfig.model_validate(c.model_dump()) == c


def test_config_paths(tmp_path):
    from decisionforge.training.config import load_config

    p = tmp_path / "cfg.yaml"
    p.write_text("data: {path: data.jsonl}\nmodel: {name: ./model}\noutput: {directory: runs}\n")
    c = load_config(p)
    assert c.data.path == str(tmp_path / "data.jsonl")
    assert c.model.name == str(tmp_path / "model")
    assert c.output.directory == str(tmp_path / "runs")


def test_deterministic_group_holdout_preserves_existing(tmp_path):
    from decisionforge.training.data import prepare_data

    rows = [example(i, group=(i + 1) // 2) for i in range(1, 81)]
    rows += [example(81, "validation", 50), example(82, "calibration", 51), example(83, "test", 52)]
    # Add enough existing calibration questions for official backend minimum.
    rows += [example(i, "calibration", i) for i in range(84, 88)]
    c = make_config(tmp_path, rows)
    a, b = prepare_data(c), prepare_data(c)
    assert a.assignments == b.assignments
    assert [r.record_id for r in a.roles["train"]] == list(range(1, 81))
    assert [r.record_id for r in a.roles["test"]] == [83]
    assert a.report["total"]["questions"] == 87 * 3
    c = make_config(tmp_path)
    a = prepare_data(c)
    assert {k: len(v) for k, v in a.roles.items()} == {
        "train": 80,
        "validation": 10,
        "calibration": 10,
        "test": 0,
    }
    assert not (
        set(r.group_id for r in a.roles["train"]) & set(r.group_id for r in a.roles["calibration"])
    )


def test_disabled_calibration_and_small_holdout(tmp_path):
    from decisionforge.training.data import prepare_data

    c = make_config(tmp_path, calibration={"enabled": False})
    p = prepare_data(c)
    assert len(p.roles["train"]) == 90 and not p.roles["calibration"]
    with pytest.raises(ValueError, match="holdout|划分"):
        prepare_data(make_config(tmp_path, [example()]))


@pytest.mark.parametrize(
    "case,match",
    [
        ("duplicate_id", "record_id=1.*Duplicate record"),
        ("duplicate_request", "record_id=2.*Duplicate public"),
        ("group", "record_id=2.*group"),
        ("unknown", "record_id=2.*Unknown split"),
        ("bad_label", "record_id=2.*decision"),
    ],
)
def test_located_data_errors(tmp_path, case, match):
    from decisionforge.training.data import prepare_data

    rows = [example(1), example(2)]
    if case == "duplicate_id":
        rows[1]["record_id"] = 1
    if case == "duplicate_request":
        rows[1]["request"] = rows[0]["request"]
    if case == "group":
        rows[1].update(group_id=1, split="test")
    if case == "unknown":
        rows[1]["split"] = "dev"
    if case == "bad_label":
        del rows[1]["supervision"]["answers"]["decision"]
    with pytest.raises(ValueError, match=match):
        prepare_data(make_config(tmp_path, rows))


def test_jsonl_error_actual_line(tmp_path):
    from decisionforge.training.data import prepare_data

    c = make_config(tmp_path)
    from pathlib import Path

    Path(c.data.path).write_text("\n" + json.dumps(example()) + "\n{bad json}\n")
    with pytest.raises(ValueError, match=r"data.jsonl.*line 3"):
        prepare_data(c)


def test_cli_cpu_import_isolation(tmp_path):
    make_config(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text(
        f"data: {{path: {tmp_path / 'data.jsonl'}}}\nmodel: {{name: missing-model}}\noutput: {{directory: {tmp_path / 'run'}}}\n"
    )
    code = """import sys
class Reject:
 def find_spec(self, name, *args):
  if name.split('.')[0] in {'torch','unsloth','transformers'}: raise AssertionError(name)
sys.meta_path.insert(0, Reject())
from decisionforge.cli import main
raise SystemExit(main(sys.argv[1:]))"""
    result = subprocess.run(
        [sys.executable, "-c", code, "train", "validate", "--config", str(config)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert '"status": "validated"' in result.stdout
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("legacy_version", [1, 2])
@pytest.mark.parametrize("extension", ["json", "jsonl"])
def test_legacy_training_uses_same_preflight(config, tmp_path, legacy_version, extension):
    from decisionforge.generation.components import compile_record, initial_draft
    from decisionforge.generation.intake import normalize
    from decisionforge.models import LegacyTrainingExample
    from decisionforge.training.config import TrainConfig
    from decisionforge.training.data import prepare_data, to_unsloth_row

    source = normalize({"question": "What is 2+2?", "answer": "4", "options": ["4", "5"]}, config)
    record = compile_record(
        source, initial_draft(source, "choice", None), "choice", "one", "one:choice", config
    )
    legacy = record if legacy_version == 1 else LegacyTrainingExample.from_record(record)
    path = tmp_path / f"legacy.{extension}"
    rows = [legacy.model_dump(), example(1001, "validation")]
    path.write_text(
        json.dumps(rows) if extension == "json" else "".join(json.dumps(r) + "\n" for r in rows)
    )
    c = TrainConfig(data={"path": str(path)}, calibration={"enabled": False})
    prepared = prepare_data(c)
    assert len(prepared.roles["train"]) == 1
    assert prepared.records[0].schema_version == "decisionforge.training.v3"
    gold = to_unsloth_row(prepared.records[0])["gold"]
    assert list(gold) == ["q1"]
    values = list(gold["q1"]["probabilities"].values())
    assert values.count(1.0) == 1 and sum(values) == 1.0
    assert set(values) == {0.0, 1.0}


def test_calibration_disabled_retains_existing_role(tmp_path):
    from decisionforge.training.data import prepare_data

    rows = [example(1), example(2, "validation"), example(3, "calibration")]
    c = make_config(tmp_path, rows, calibration={"enabled": False})
    prepared = prepare_data(c)
    assert [r.record_id for r in prepared.roles["train"]] == [1]
    assert [r.record_id for r in prepared.roles["calibration"]] == [3]


def test_json_array_error_and_schema_line_locations(tmp_path):
    from pathlib import Path

    from decisionforge.training.data import prepare_data

    c = make_config(tmp_path)
    c.data.path = str(tmp_path / "broken.json")
    Path(c.data.path).write_text("{broken")
    with pytest.raises(ValueError, match=r"broken.json.*line 1"):
        prepare_data(c)
    c.data.path = str(tmp_path / "broken.jsonl")
    bad = example(2)
    bad["supervision"]["answers"]["flag"]["noul"] = -1
    Path(c.data.path).write_text("\n\n" + json.dumps(bad) + "\n")
    with pytest.raises(ValueError, match=r"broken.jsonl.*line 3.*record_id=2.*flag"):
        prepare_data(c)


@pytest.mark.parametrize("role", ["train", "validation", "calibration", "test"])
def test_canonical_split_roles_cannot_be_remapped(tmp_path, role):
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="canonical|Canonical"):
        make_config(
            tmp_path,
            data={"path": "unused", "split_map": {role: "train" if role != "train" else "test"}},
        )
