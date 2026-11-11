from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from test_training_data import make_config

from decisionforge.training.config import TrainConfig, load_config
from decisionforge.training.data import prepare_data


def test_laya_defaults_and_saved_roundtrip():
    c = TrainConfig(backend="laya", data={"path": "unused"})
    assert c.finetuning == "full" and c.lora is None
    assert c.model.name == "convaiinnovations/laya" and not c.model.load_in_4bit
    assert c.training.warmup_steps == 0 and c.training.learning_rate == 2.5e-5
    assert c.training.num_train_epochs == 4 and c.training.precision == "fp16"
    assert TrainConfig.model_validate(c.model_dump()) == c


def test_laya_example_preserves_existing_calibration_groups():
    from pathlib import Path

    c = load_config(Path(__file__).resolve().parents[1] / "examples/training/laya/ddp.yaml")
    assert prepare_data(c).report["roles"]["calibration"]["types"] == {
        "choice": 16,
        "noul": 14,
        "score": 20,
    }


@pytest.mark.parametrize(
    "patch",
    [
        {"training": {"num_train_epochs": 1.5}},
        {"training": {"num_train_epochs": None, "max_steps": 2}},
        {"training": {"precision": "bf16"}},
        {"training": {"warmup_steps": 2}},
        {"training": {"lr_scheduler_type": "linear"}},
        {"finetuning": "lora"},
        {"model": {"load_in_4bit": True}},
        {"output": {"save_merged": True}},
        {"laya": {"sigma_end": 0}},
    ],
)
def test_laya_rejects_unsupported_explicit_settings(patch):
    with pytest.raises(ValidationError, match="Laya|laya|greater than 0"):
        TrainConfig(backend="laya", data={"path": "unused"}, **patch)


def test_laya_init_path_and_dispatch(tmp_path):
    from decisionforge.training.backend import get_backend

    p = tmp_path / "config.yaml"
    p.write_text("backend: laya\ndata: {path: samples.jsonl}\nlaya: {init_checkpoint: ./source}\n")
    c = load_config(p)
    assert c.laya.init_checkpoint == str(tmp_path / "source")
    assert type(get_backend(c, tmp_path)).__name__ == "LayaBackend"


def test_laya_resume_rejected_before_source_and_output(tmp_path):
    from decisionforge.training.runner import run_training

    c = TrainConfig(
        backend="laya", data={"path": "missing"}, output={"directory": str(tmp_path / "out")}
    )
    with pytest.raises(ValueError, match="resume"):
        run_training(c, resume=True)
    assert not (tmp_path / "out").exists()


def fake_encoder():
    def items_from_rows(tok, rows, max_len, head_max_len):
        (row,) = rows
        # Full request stays intact, only one question's target is selected.
        assert len(row["questions"]) == 3
        (qid,) = row["gold"]
        q = row["questions"][qid]
        keys = (
            list(q["criteria"])
            if q["type"] == "choice"
            else ["0", "1"]
            if q["type"] == "score"
            else ["false", "true"]
        )
        return [
            {
                "q": {"t": q["type"]},
                "target": [row["gold"][qid]["probabilities"][k] for k in keys],
                "qtype": {"choice": 0, "score": 1, "noul": 2}[q["type"]],
                "k": 2,
            }
        ], {}

    return SimpleNamespace(items_from_rows=items_from_rows)


def test_role_mapping_preserves_all_soft_targets_and_groups(tmp_path):
    from decisionforge.training.backends.laya import encode_roles

    c = make_config(tmp_path, backend="laya")
    data = prepare_data(c)
    items, evidence = encode_roles(data, object(), fake_encoder(), c, lambda r, q: None)
    assert [it["target"] for it in items["train"][:3]] == [[0.7, 0.3], [0.6, 0.4], [0.8, 0.2]]
    assert [m["question_id"] for m in evidence["train"]["mapping"][:3]] == [
        "decision",
        "rating",
        "flag",
    ]
    train_groups = {m["group_id"] for m in evidence["train"]["mapping"]}
    calib_groups = {m["group_id"] for m in evidence["calibration"]["mapping"]}
    assert not train_groups & calib_groups
    assert len(items["calibration"]) == 30


def test_skipped_question_is_localized_and_target_repair_rejected(tmp_path):
    from decisionforge.training.backends.laya import encode_roles

    c = make_config(
        tmp_path, backend="laya", data={"path": str(tmp_path / "data.jsonl"), "on_skipped": "allow"}
    )
    data = prepare_data(c)
    api = fake_encoder()
    original = api.items_from_rows

    def skip(tok, rows, **kw):
        if "rating" in rows[0]["gold"]:
            return [], {"options_collapsed": 1}
        return original(tok, rows, **kw)

    api.items_from_rows = skip
    items, report = encode_roles(data, object(), api, c, lambda r, q: None)
    assert len(items["train"]) == 160
    assert {x["question_id"] for x in report["train"]["skipped"]} == {"rating"}
    c.data.on_skipped = "error"
    with pytest.raises(ValueError, match="rating.*options_collapsed"):
        encode_roles(data, object(), api, c, lambda r, q: None)

    def repaired(tok, rows, **kw):
        result, skipped = original(tok, rows, **kw)
        result[0]["target"] = [0.5, 0.5]
        return result, skipped

    api.items_from_rows = repaired
    with pytest.raises(ValueError, match="target"):
        encode_roles(data, object(), api, c, lambda r, q: None)


def test_ddp_tail_uses_encoded_questions_and_records_every_drop():
    from decisionforge.training.backends.laya import partition_items

    mapping = [
        {"record_id": 1, "question_id": x, "group_id": 1, "role": "train"} for x in ("a", "b", "c")
    ]
    with pytest.raises(ValueError, match="divisible"):
        partition_items([10, 20, 30], mapping, 2, 0, "error")
    shard, report = partition_items([10, 20, 30], mapping, 2, 1, "allow")
    assert shard == [20] and report["dropped"] == [mapping[2]]
    assert report["retained_items"] == 2 and report["items_per_rank"] == 1
    with pytest.raises(ValueError, match="rank|items"):
        partition_items([10], mapping[:1], 2, 0, "allow")


def artifact_at(root):
    (root / "encoder").mkdir(parents=True)
    (root / "tokenizer").mkdir()
    (root / "encoder/config.json").write_text('{"model_type":"modernbert"}')
    (root / "tokenizer/tokenizer_config.json").write_text("{}")
    (root / "tokenizer/tokenizer.json").write_text("{}")
    (root / "rl_agent_config.json").write_text('{"temperature":[1,1,1],"max_len":512}')
    (root / "model.safetensors").write_bytes(b"fixture")
    return root


def test_laya_artifact_identity_changes_and_rejects_other_formats(tmp_path):
    from decisionforge.training.backends.laya import artifact_identity, validate_artifact

    root = artifact_at(tmp_path / "model")
    before = artifact_identity(root)
    (root / "model.safetensors").write_bytes(b"changed")
    assert artifact_identity(root) != before
    (root / "unsloth_decision_config.json").write_text("{}")
    with pytest.raises(ValueError, match="Laya|laya"):
        validate_artifact(root)


def test_calibration_only_uses_assigned_role_and_type_order(tmp_path):
    from decisionforge.training.backends.laya import LayaBackend

    c = make_config(tmp_path, backend="laya")
    b = LayaBackend(c, tmp_path)
    b.items = {"calibration": ["independent"] * 30, "train": ["never"]}
    b.mapping = {"calibration": {"mapping": [{"group_id": 90}] * 30}}
    b.cfg = {"temperature": [2, 3, 4], "temperature_by_options": {"old": 2}}
    b.model = b.tok = b.device = object()

    def records(model, tok, items, device, max_len, head_max_len, **kwargs):
        assert items == ["independent"] * 30
        return [(qt, [1, 0], [0.7, 0.3], 2) for qt in (0, 1, 2) for _ in range(10)]

    b.api = SimpleNamespace(
        calibration_records=records,
        calibration_report=lambda rec, temps: {},
        fit_temperature_map=lambda rec: {
            "temperature": [1.1, 1.2, 1.3],
            "temperature_by_options": {},
            "n_by_bucket": {},
            "temperature_clamped": {},
        },
    )
    result = b.calibrate()
    assert result["types"]["score"] == {"questions": 10, "fitted": True, "temperature": 1.2}
    assert result["types"]["noul"]["temperature"] == 1.3
    assert b.cfg["temperature"] == [1.1, 1.2, 1.3] and "temperature_by_options" not in b.cfg


def test_calibration_underfloor_reports_unfitted(tmp_path):
    from decisionforge.training.backends.laya import calibration_types

    records = [(0, [1, 0], [0.7, 0.3], 2)] * 9
    assert calibration_types(records, [1, 1, 1])["choice"] == {
        "questions": 9,
        "fitted": False,
        "temperature": 1,
    }


def test_laya_environment_missing_dependency_has_install_guidance(tmp_path, monkeypatch):
    from decisionforge.training.backends import laya

    b = laya.LayaBackend(TrainConfig(backend="laya", data={"path": "unused"}), tmp_path)

    def missing(name):
        raise ImportError(name)

    monkeypatch.setattr(laya.importlib, "import_module", missing)
    with pytest.raises(ValueError, match="uv sync --extra train-laya"):
        b.environment()


def test_init_artifact_rejected_before_new_run_writes(tmp_path):
    from decisionforge.training.runner import run_training

    c = make_config(tmp_path, backend="laya", laya={"init_checkpoint": str(tmp_path / "missing")})
    with pytest.raises(ValueError, match="Laya"):
        run_training(c)
    assert not (tmp_path / "run").exists()


def test_laya_backend_defaults_accept_typed_child_models():
    from decisionforge.training.config import ModelConfig, TrainingConfig

    c = TrainConfig(
        backend="laya",
        data={"path": "unused"},
        model=ModelConfig(name="local", load_in_4bit=False),
        training=TrainingConfig(
            num_train_epochs=1.0, warmup_steps=0, save_steps=None, precision="fp16"
        ),
    )
    assert c.model.name == "local" and c.training.num_train_epochs == 1.0


def test_source_staging_does_not_mutate_inputs(tmp_path):
    from decisionforge.training.backends.laya import artifact_identity, stage_source

    root = artifact_at(tmp_path / "source")
    before = artifact_identity(root)
    target = stage_source(root, tmp_path / "stage")
    (target / "tokenizer/tokenizer_config.json").write_text('{"fixed":true}')
    assert artifact_identity(root) == before
    assert (target / "model.safetensors").read_bytes() == b"fixture"


def test_inherited_calibration_uses_upstream_clamp_and_records_original(tmp_path):
    from decisionforge.training.backends.laya import LayaBackend

    b = LayaBackend(TrainConfig(backend="laya", data={"path": "unused"}), tmp_path)
    b.cfg = {"temperature": [1.0, 2.0, 3.0], "temperature_by_options": {"choice:11+": 0.1}}
    b.common = SimpleNamespace(clamp_temperature=lambda t: max(0.5, min(5.0, t)))
    b._source_calibration()
    assert b.cfg["temperature_by_options"]["choice:11+"] == 0.5
    assert b.calibration["raw"]["temperature_by_options"]["choice:11+"] == 0.1
    assert b.calibration["source"] == "inherited_checkpoint_upstream_clamped"


def test_nested_laya_eval_example_keeps_paths_and_both_glob_boundaries():
    from pathlib import Path

    from decisionforge.eval.models import load_config as load_eval

    root = Path(__file__).resolve().parents[1]
    path = root / "examples/training/laya/eval.yaml"
    config = load_eval(path)
    assert config.model.path == str(root / "outputs/train/laya-single/model")
    assert config.data.path == str(root / "outputs/train/laya-single/data/test.jsonl")
    assert path not in list((root / "examples/training").glob("*.yaml"))
    assert path not in list((root / "examples/training/eval").glob("*.yaml"))
