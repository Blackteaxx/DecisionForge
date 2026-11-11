import json
from types import SimpleNamespace

import pytest

from decisionforge.training.backends.unsloth import UnslothBackend
from decisionforge.training.config import TrainConfig
from decisionforge.training.data import PreparedData


def full_config(**kwargs):
    return TrainConfig(
        data={"path": "unused"}, finetuning="full", model={"load_in_4bit": False}, **kwargs
    )


class Parameter:
    requires_grad = True

    def numel(self):
        return 8

    def detach(self):
        return self

    def flatten(self):
        return self

    def __getitem__(self, key):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def clone(self):
        return self


@pytest.mark.parametrize("mode", ["lora", "full"])
def test_official_mode_flags_and_trainable_scopes(tmp_path, mode):
    c = full_config() if mode == "full" else TrainConfig(data={"path": "unused"})
    backend = UnslothBackend(c, tmp_path)
    names = (
        ["encoder.weight", "encoder.lm_head.weight", "head.weight"]
        if mode == "full"
        else ["encoder.lora_A", "head.weight"]
    )
    model = SimpleNamespace(
        is_clef=True,
        encoder=SimpleNamespace(config=SimpleNamespace()),
        decision_config={},
        named_parameters=lambda: [(n, Parameter()) for n in names],
        save_pretrained=lambda *a: None,
        save_pretrained_merged=lambda *a: None,
    )
    calls = []
    backend.api = SimpleNamespace(
        from_pretrained=lambda **kw: (calls.append(kw) or model, object()),
        get_peft_model=lambda model, **kw: calls.append("peft") or model,
    )
    backend.dtype = "bf16"
    backend.set_seed = lambda seed: None
    backend._create_trainer = lambda: None
    report = backend.prepare(PreparedData([], [], {}))
    assert calls[0]["full_finetuning"] == (mode == "full")
    assert ("peft" in calls) == (mode == "lora")
    assert set(backend.before) == ({"backbone", "head"} if mode == "full" else {"lora", "head"})
    assert report["trainable_parameters"]["total"] == (24 if mode == "full" else 16)
    assert report["trainable_parameters"]["groups"]["head"]["count"] == 8


def checkpoint(tmp_path, monkeypatch, world=1, sharded=False):
    import sys

    c = full_config(distributed={"strategy": "ddp", "world_size": world} if world > 1 else {})
    backend = UnslothBackend(c, tmp_path)
    p = tmp_path / "checkpoint-1"
    p.mkdir()
    data = {
        "trainer_state.json": {"global_step": 1, "max_steps": 3, "log_history": [{"loss": 0.5}]},
        "config.json": {"model_type": "qwen3", "hidden_size": 16},
        "joint_head_config.json": {
            "hidden_size": 16,
            "width": 16,
            "heads": 2,
            "feedforward": 32,
            "routing_layers": 2,
            "layers": 4,
        },
        "unsloth_decision_config.json": {"layout": "clef", "max_len": 256},
    }
    weights = (
        ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
        if sharded
        else ["model.safetensors"]
    )
    if sharded:
        data["model.safetensors.index.json"] = {"weight_map": {"a": weights[0], "b": weights[1]}}
    for name, value in data.items():
        (p / name).write_text(json.dumps(value))
    rngs = [f"rng_state_{r}.pth" for r in range(world)] if world > 1 else ["rng_state.pth"]
    for name in ["optimizer.pt", "scheduler.pt", "joint_head.safetensors", *rngs, *weights]:
        (p / name).write_text("saved")

    def load(path, **kw):
        if path.name == "optimizer.pt":
            return {"state": {0: {}}, "param_groups": [{}]}
        if path.name == "scheduler.pt":
            return {"last_epoch": 1}
        return dict.fromkeys(["python", "numpy", "cpu", "cuda"])

    class Tensors:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def keys(self):
            return ["a", "b", "head"]

        def get_tensor(self, key):
            return [1]

    monkeypatch.setitem(
        sys.modules, "safetensors", SimpleNamespace(safe_open=lambda *a, **k: Tensors())
    )
    backend.torch = SimpleNamespace(load=load, float16="fp16")
    backend.dtype = "bf16"
    backend.trainer = SimpleNamespace(state=SimpleNamespace(global_step=1))
    backend.barrier = lambda: None
    return backend, p


@pytest.mark.parametrize("world,sharded", [(1, False), (2, False), (2, True)])
def test_full_checkpoint_complete_state_and_identity(tmp_path, monkeypatch, world, sharded):
    backend, p = checkpoint(tmp_path, monkeypatch, world, sharded)
    assert backend.validate_checkpoint(p, require_marker=False) == 1
    backend._mark_checkpoint(p)
    marker = json.loads((p / "dforge_checkpoint.json").read_text())
    assert marker["finetuning"] == "full" and marker["world_size"] == world
    assert backend.validate_checkpoint(p) == 1
    backend.config.distributed.world_size = world + 1
    with pytest.raises(ValueError, match="world.size|RNG|rng_state"):
        backend.validate_checkpoint(p)


def test_full_checkpoint_rejects_missing_rank_and_shard_traversal(tmp_path, monkeypatch):
    backend, p = checkpoint(tmp_path, monkeypatch, 2, True)
    (p / "rng_state_1.pth").unlink()
    with pytest.raises(ValueError, match="rng_state_1"):
        backend.validate_checkpoint(p, require_marker=False)
    (p / "rng_state_1.pth").write_text("saved")
    index = p / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"a": "../outside.safetensors"}}))
    with pytest.raises(ValueError, match="shard|path"):
        backend.validate_checkpoint(p, require_marker=False)


def test_nonzero_checkpoint_synchronizes_without_writing_marker(tmp_path, monkeypatch):
    backend, p = checkpoint(tmp_path, monkeypatch, 2)
    monkeypatch.setenv("RANK", "1")
    calls = []
    backend.barrier = lambda: calls.append("barrier")
    backend._mark_checkpoint(p)
    assert calls == ["barrier", "barrier"]
    assert not (p / "dforge_checkpoint.json").exists()


def test_checkpoint_marker_cannot_cross_data_identity(tmp_path, monkeypatch):
    backend, p = checkpoint(tmp_path, monkeypatch)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"identity": {"samples.jsonl": "a", "splits.json": "b", "config": "c"}})
    )
    backend._mark_checkpoint(p)
    manifest.write_text(
        json.dumps({"identity": {"samples.jsonl": "changed", "splits.json": "b", "config": "c"}})
    )
    with pytest.raises(ValueError, match="identity"):
        backend.validate_checkpoint(p)


def test_full_index_rejects_non_string_shards(tmp_path, monkeypatch):
    backend, p = checkpoint(tmp_path, monkeypatch, 1, True)
    (p / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": None, "b": 2}}))
    with pytest.raises(ValueError, match="shard"):
        backend.validate_checkpoint(p, require_marker=False)


def test_full_checkpoint_rejects_unindexed_backbone_shard(tmp_path, monkeypatch):
    backend, p = checkpoint(tmp_path, monkeypatch, 1, True)
    (p / "model-extra.safetensors").write_text("unexpected")
    with pytest.raises(ValueError, match="shard"):
        backend.validate_checkpoint(p, require_marker=False)


def test_fp16_checkpoint_rejects_invalid_scaler_state(tmp_path, monkeypatch):
    backend, p = checkpoint(tmp_path, monkeypatch)
    backend.dtype = "fp16"
    (p / "scaler.pt").write_text("present but invalid scaler state")
    with pytest.raises(ValueError, match="scaler"):
        backend.validate_checkpoint(p, require_marker=False)


def test_actual_resume_state_audit_checks_step_optimizer_and_scheduler(tmp_path):
    from decisionforge.training.state import optimizer_summary, state_fingerprint

    backend = UnslothBackend(full_config(), tmp_path)
    backend.torch = SimpleNamespace()
    optimizer = {
        "state": {0: {"step": 2, "exp_avg": [0.1, 0.2]}},
        "param_groups": [{"lr": 0.01, "params": [0]}],
    }
    scheduler = {"last_epoch": 2, "_last_lr": [0.01]}
    p = tmp_path / "checkpoint-2"
    backend._resume_checkpoint = p
    backend._checkpoint_audits[str(p)] = {
        "global_step": 2,
        "optimizer": optimizer_summary(optimizer, backend.torch),
        "scheduler_sha256": state_fingerprint(scheduler, backend.torch),
    }
    backend.trainer = SimpleNamespace(
        optimizer=SimpleNamespace(state_dict=lambda: optimizer),
        lr_scheduler=SimpleNamespace(state_dict=lambda: scheduler),
    )
    backend._record_restored_state(SimpleNamespace(global_step=2))
    event = backend.resume_state[-1]
    assert event["event"] == "optimizer_scheduler_restored" and event["matches_checkpoint"]
    assert event["actual"]["optimizer"]["step_min"] == 2
    scheduler["last_epoch"] = 0
    with pytest.raises(ValueError, match="restored"):
        backend._record_restored_state(SimpleNamespace(global_step=2))


def test_rng_audit_observes_upstream_loader_after_restore(tmp_path):
    from decisionforge.training.state import state_fingerprint

    backend = UnslothBackend(full_config(), tmp_path)
    expected = {"python": [1, 2], "numpy": [3, 4], "cpu": [5, 6], "cuda": [7, 8]}
    calls = []
    backend.torch = SimpleNamespace(load=lambda *a, **k: expected)
    backend.trainer = SimpleNamespace(_load_rng_state=lambda checkpoint: calls.append("upstream"))
    backend._rng_state = lambda: calls.append("observe") or expected
    backend._install_rng_audit()
    backend.trainer._load_rng_state(tmp_path / "checkpoint-2")
    assert calls == ["upstream", "observe"]
    event = backend.resume_state[-1]
    assert event["event"] == "rng_restored" and event["matches_checkpoint"]
    assert event["actual_sha256"] == state_fingerprint(expected, backend.torch)
