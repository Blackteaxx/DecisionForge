import pytest
from test_training_data import example

from decisionforge.models import TrainingExample


def test_encoding_reports_questions_vs_items_and_affected_records():
    from decisionforge.training.unsloth_backend import encoding_report

    records = [TrainingExample.model_validate(example(i)) for i in (1, 2)]
    items = [
        {
            "row": 0,
            "input_ids": [1, 2],
            "source": {"questions": {"flag": {}, "rating": {}}},
            "qtypes": [0, 2],
            "targets": [[0.8, 0.2], [0.6, 0.4]],
            "labels": [0, 0],
        }
    ]
    report = encoding_report(
        records, items, {"total": 6, "skipped": 4, "truncated": 1, "reason": "bad encoding"}, 2
    )
    assert report["items"] == 1 and report["questions"] == 2
    assert report["types"] == {"noul": 1, "score": 1}
    assert report["skipped"] == [
        {"record_id": 1, "questions": ["decision"]},
        {"record_id": 2, "questions": ["decision", "rating", "flag"]},
    ]
    assert report["truncated"] == [{"record_id": 1, "questions": ["flag", "rating"]}]


def test_encoding_policy_and_calibration_floor():
    from decisionforge.training.config import TrainConfig
    from decisionforge.training.unsloth_backend import check_encoding

    c = TrainConfig(data={"path": "data.jsonl"})
    report = {"items": 3, "questions": 9, "skipped": [], "truncated": []}
    with pytest.raises(ValueError, match="10.*disable"):
        check_encoding(c, "calibration", report)
    for key in ("skipped", "truncated"):
        report = {
            "items": 3,
            "questions": 15,
            "skipped": [],
            "truncated": [],
            key: [{"record_id": 1}],
        }
        with pytest.raises(ValueError, match=key):
            check_encoding(c, "train", report)
        setattr(c.data, "on_" + key, "allow")
        check_encoding(c, "train", report)
    with pytest.raises(ValueError, match="All.*skipped"):
        check_encoding(c, "train", {"items": 0, "questions": 0})


def test_type_subset_preserves_encoded_input_and_spans():
    from dataclasses import dataclass

    from decisionforge.training.unsloth_backend import type_items

    @dataclass(frozen=True)
    class Record:
        input_ids: tuple
        questions: tuple

    item = {
        "row": 0,
        "input_ids": [5, 6, 7],
        "record": Record((5, 6, 7), ("n", "c", "s")),
        "qtypes": [0, 1, 2],
        "targets": [[0.2, 0.8], [0.3, 0.7], [0.6, 0.4]],
        "labels": [1, 1, 0],
        "source": {"state": "public", "questions": {"n": {}, "c": {}, "s": {}}},
    }
    selected = type_items([item], "choice")
    assert len(selected) == 1
    assert selected[0]["input_ids"] == item["input_ids"]
    assert selected[0]["record"].questions == ("c",)
    assert selected[0]["targets"] == [[0.3, 0.7]]
    assert selected[0]["qtypes"] == [1]
    assert item["record"].questions == ("n", "c", "s")


def test_official_clef_checkpoint_layout(tmp_path, monkeypatch):
    import json
    import sys
    from types import SimpleNamespace

    from decisionforge.training.config import TrainConfig
    from decisionforge.training.unsloth_backend import UnslothBackend

    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    # Layout observed from actual DecisionTrainer GPU checkpoint, not the Laya layout.
    for name in (
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "adapter_model.safetensors",
        "adapter_config.json",
        "joint_head.safetensors",
        "joint_head_config.json",
        "unsloth_decision_config.json",
    ):
        values = {
            "trainer_state.json": {
                "global_step": 1,
                "max_steps": 3,
                "log_history": [{"loss": 0.5}],
            },
            "adapter_config.json": {
                "peft_type": "LORA",
                "base_model_name_or_path": "model",
                "r": 8,
            },
            "joint_head_config.json": {
                "hidden_size": 16,
                "width": 16,
                "heads": 2,
                "feedforward": 32,
                "routing_layers": 2,
                "layers": 4,
            },
            "unsloth_decision_config.json": {
                "layout": "clef",
                "max_len": 2048,
                "temperature": [1, 1, 1],
            },
        }
        (checkpoint / name).write_text(json.dumps(values.get(name, {"dummy": True})))

    class Tensors:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def keys(self):
            return ["weights"]

        def get_tensor(self, key):
            return [1]

    monkeypatch.setitem(
        sys.modules, "safetensors", SimpleNamespace(safe_open=lambda *a, **k: Tensors())
    )

    def load(path, **kwargs):
        return {
            "optimizer.pt": {"state": {0: {}}, "param_groups": [{}]},
            "scheduler.pt": {"last_epoch": 1},
            "rng_state.pth": dict.fromkeys(("python", "numpy", "cpu", "cuda")),
        }[path.name]

    backend = UnslothBackend(TrainConfig(data={"path": "unused"}), tmp_path)
    backend.torch = SimpleNamespace(load=load, float16="fp16")
    backend.dtype = "bf16"
    assert backend.validate_checkpoint(checkpoint, require_marker=False) == 1
    for name in ("adapter_config.json", "joint_head_config.json", "unsloth_decision_config.json"):
        path = checkpoint / name
        original = path.read_text()
        path.write_text("X" + original[1:])
        with pytest.raises(ValueError, match=name):
            backend.validate_checkpoint(checkpoint, require_marker=False)
        path.write_text(original)
    state_path = checkpoint / "trainer_state.json"
    original_state = state_path.read_text()
    state_path.write_text(json.dumps({"global_step": 1, "max_steps": 3, "log_history": []}))
    with pytest.raises(ValueError, match="loss"):
        backend.validate_checkpoint(checkpoint, require_marker=False)
    state_path.write_text(original_state)
    (checkpoint / "optimizer.pt").unlink()
    with pytest.raises(ValueError, match="optimizer.pt"):
        backend.validate_checkpoint(checkpoint, require_marker=False)


def test_calibration_config_survives_head_temperature_folding():
    from decisionforge.training.unsloth_backend import verify_calibration_reload

    original = {"temperature": [0.5, 1.1, 0.9], "head_temperature": 0.3}
    reloaded = {"temperature": [0.5, 1.1, 0.9], "folded_temperature": 0.3}
    verify_calibration_reload(original, reloaded)
    with pytest.raises(ValueError, match="calibration"):
        verify_calibration_reload(original, {"temperature": [1, 1, 1], "folded_temperature": 0.3})
    with pytest.raises(ValueError, match="calibration"):
        verify_calibration_reload(
            original, {"temperature": [0.5, 1.1, 0.9], "folded_temperature": 1}
        )


def test_unfitted_types_keep_effective_existing_temperatures():
    from decisionforge.training.unsloth_backend import retain_unfitted_calibration

    before = {
        "temperature": [0.5, 2, 3],
        "head_temperature": 0.5,
        "temperature_by_options": {"score:3-5": 4, "choice:3-5": 2},
    }
    after = {"temperature": [2, 1, 1], "head_temperature": 0.2}
    adjusted = retain_unfitted_calibration(before, after, [0], lambda t: min(5.0, max(0.5, t)))
    assert after["head_temperature"] == 0.5
    assert after["temperature"][0] == pytest.approx(0.8)
    assert after["temperature"][1:] == [2, 3]
    assert after["temperature_by_options"] == {"score:3-5": 4}
    assert adjusted["shared_head_temperature_retained"] is True
    assert (
        after["head_temperature"] * after["temperature"][1]
        == before["head_temperature"] * before["temperature"][1]
    )


def test_sparse_logging_still_records_initial_training_loss(tmp_path):
    from types import SimpleNamespace

    from decisionforge.training.config import TrainConfig
    from decisionforge.training.unsloth_backend import UnslothBackend

    config = TrainConfig(
        data={"path": "unused"},
        training={"num_train_epochs": None, "max_steps": 2, "logging_steps": 100},
    )
    backend = UnslothBackend(config, tmp_path)
    backend.Callback = type("Callback", (), {})
    backend.TrainingArguments = lambda **kwargs: SimpleNamespace(**kwargs)
    backend.Trainer = lambda **kwargs: SimpleNamespace(args=kwargs["args"])
    backend.torch = SimpleNamespace(bfloat16="bf16", float16="fp16")
    backend.dtype = "bf16"
    backend.model = backend.tokenizer = object()
    backend.items = {"train": ["encoded"]}
    backend._create_trainer()
    assert backend.trainer.args.logging_first_step is True
