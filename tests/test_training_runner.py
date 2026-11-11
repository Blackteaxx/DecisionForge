import json
from pathlib import Path

import pytest
from test_training_data import make_config

from decisionforge.training.data import prepare_data


class FakeBackend:
    fail = None
    calls = []

    def __init__(self, config, directory):
        self.config, self.directory = config, directory

    def environment(self):
        return {"packages": {"fake": "1"}, "gpu": "test"}

    def prepare(self, data):
        self.calls.append("prepare")
        return {"roles": {k: {"items": len(v)} for k, v in data.roles.items()}}

    def validate_checkpoint(self, checkpoint):
        if not (checkpoint / "optimizer.pt").is_file():
            raise ValueError("Corrupt complete checkpoint")
        return json.loads((checkpoint / "trainer_state.json").read_text())["global_step"]

    def train(self, checkpoint=None):
        self.calls.append(("train", str(checkpoint) if checkpoint else None))
        if self.fail == "train":
            self.save_final_checkpoint()
            raise KeyboardInterrupt()
        return {"training_loss": 0.5, "global_step": 3}

    def save_final_checkpoint(self):
        p = (
            Path(self.config.output.checkpoint_directory or self.directory / "checkpoints")
            / "checkpoint-3"
        )
        p.mkdir(parents=True, exist_ok=True)
        (p / "optimizer.pt").write_text("optimizer")
        (p / "trainer_state.json").write_text(
            json.dumps(
                {
                    "global_step": 3,
                    "max_steps": self.config.training.max_steps or 10,
                    "log_history": [{"loss": 0.5}],
                }
            )
        )
        (p / "dforge_checkpoint.json").write_text('{"global_step": 3}')
        if self.fail == "final_checkpoint":
            raise KeyboardInterrupt("interrupted after final checkpoint but before stage commit")
        return p

    def load_final_checkpoint(self, checkpoint):
        self.calls.append(("load_final", str(checkpoint)))

    def evaluate(self, role, phase):
        self.calls.append((role, phase))
        return {"role": role, "phase": phase, "raw": {"accuracy": 0.6}}

    def calibrate(self):
        self.calls.append("calibrate")
        return {"fitted_types": [0, 1, 2]}

    def calibration_state(self):
        return {"temperature": [1, 1, 1]}

    def restore_calibration(self, state):
        self.calls.append("restore_calibration")

    def save(self):
        self.calls.append("save")
        if self.fail == "save":
            raise RuntimeError("export interrupted")
        return {"model": "model", "merged": "merged"}

    def reload_probe(self, data):
        self.calls.append("reload")
        return {"valid": True}

    def close(self):
        self.calls.append("close")


def run(config, **kwargs):
    from decisionforge.training.runner import run_training

    return run_training(config, backend_factory=FakeBackend, **kwargs)


def test_separate_checkpoint_root_and_resume(tmp_path):
    config = make_config(tmp_path)
    config.output.checkpoint_directory = str(tmp_path / "shared-checkpoints" / "named-model")
    FakeBackend.fail = "train"
    with pytest.raises(KeyboardInterrupt):
        run(config)
    checkpoint = Path(config.output.checkpoint_directory) / "checkpoint-3"
    assert checkpoint.is_dir()
    FakeBackend.fail = None
    result = run(config, resume=True)
    assert result["status"] == "completed"
    manifest = json.loads((Path(config.output.directory) / "manifest.json").read_text())
    assert (Path(config.output.directory) / manifest["final_checkpoint"]).resolve() == checkpoint


@pytest.fixture(autouse=True)
def reset():
    FakeBackend.calls = []
    FakeBackend.fail = None


def test_validate_only_no_directory(tmp_path):
    c = make_config(tmp_path)
    assert run(c, validate_only=True)["status"] == "validated"
    assert not Path(c.output.directory).exists()
    assert not FakeBackend.calls


@pytest.mark.parametrize("field", ["actual_name", "actual_revision"])
def test_resume_rejects_changed_loaded_backbone_before_updates(tmp_path, field):
    from decisionforge.training.runner import run_training

    class Identified(FakeBackend):
        identity = {"actual_name": "fixture/model", "actual_revision": "revision-a"}

        def prepare(self, data):
            return {**super().prepare(data), "model": dict(self.identity)}

    c = make_config(tmp_path)
    FakeBackend.fail = "train"
    with pytest.raises(KeyboardInterrupt):
        run_training(c, backend_factory=Identified)
    manifest = Path(c.output.directory) / "manifest.json"
    original = json.loads(manifest.read_text())["loaded_model"]
    FakeBackend.fail = None
    FakeBackend.calls = []
    Identified.identity[field] = "changed-backbone"
    with pytest.raises(ValueError, match="backbone"):
        run_training(c, resume=True, backend_factory=Identified)
    assert not any(
        isinstance(call, tuple) and call[0] in {"train", "load_final"} for call in FakeBackend.calls
    )
    assert json.loads(manifest.read_text())["loaded_model"] == original


def test_resume_pins_resolved_hf_revision_without_changing_saved_config(tmp_path):
    from decisionforge.training.runner import run_training

    class Revisioned(FakeBackend):
        current_revision = "revision-a"
        requested = []

        def prepare(self, data):
            self.requested.append(self.config.model.revision)
            return {
                **super().prepare(data),
                "model": {
                    "actual_name": self.config.model.name,
                    "actual_revision": self.config.model.revision or self.current_revision,
                },
            }

    c = make_config(tmp_path)
    FakeBackend.fail = "train"
    with pytest.raises(KeyboardInterrupt):
        run_training(c, backend_factory=Revisioned)
    Revisioned.current_revision = "revision-b"
    FakeBackend.fail = None
    run_training(c, resume=True, backend_factory=Revisioned)
    assert Revisioned.requested == [None, "revision-a"]
    assert c.model.revision is None
    assert (
        json.loads(Path(c.output.directory, "manifest.json").read_text())["model"]["revision"]
        is None
    )


def test_resume_rejects_mutated_local_backbone_metadata(tmp_path):
    import os

    from decisionforge.training import unsloth_backend
    from decisionforge.training.runner import run_training

    assert callable(getattr(unsloth_backend, "local_model_metadata", None))

    class Local(FakeBackend):
        def prepare(self, data):
            return {
                **super().prepare(data),
                "model": {
                    "actual_name": self.config.model.name,
                    "actual_revision": None,
                    "source_metadata": unsloth_backend.local_model_metadata(self.config.model.name),
                },
            }

    c = make_config(tmp_path)
    base = tmp_path / "base"
    base.mkdir()
    weights = base / "model.safetensors"
    weights.write_bytes(b"aaaa")
    c.model.name = str(base)
    FakeBackend.fail = "train"
    with pytest.raises(KeyboardInterrupt):
        run_training(c, backend_factory=Local)
    before = weights.stat()
    weights.write_bytes(b"bbbb")
    os.utime(weights, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
    FakeBackend.fail = None
    FakeBackend.calls = []
    with pytest.raises(ValueError, match="backbone"):
        run_training(c, resume=True, backend_factory=Local)
    assert not any(isinstance(call, tuple) and call[0] == "train" for call in FakeBackend.calls)


def test_complete_artifacts_and_completed_resume_rejected(tmp_path):
    c = make_config(tmp_path)
    assert run(c)["status"] == "completed"
    p = Path(c.output.directory)
    for name in (
        "manifest.json",
        "config.yaml",
        "samples.jsonl",
        "splits.json",
        "data_report.json",
        "report.json",
        "summary.md",
    ):
        assert (p / name).is_file()
    m = json.loads((p / "manifest.json").read_text())
    assert m["status"] == "completed" and m["stage"] == "completed"
    assert (p / "checkpoints/checkpoint-3/optimizer.pt").exists()
    assert "calibrate" in FakeBackend.calls
    with pytest.raises(ValueError, match="completed"):
        run(c, resume=True)
    with pytest.raises(ValueError, match="resume|empty"):
        run(c)


def test_export_failure_resume_does_not_retrain_or_read_input(tmp_path):
    c = make_config(tmp_path)
    expected = prepare_data(c).assignments
    FakeBackend.fail = "save"
    with pytest.raises(ValueError, match="export interrupted"):
        run(c)
    p = Path(c.output.directory)
    assert json.loads((p / "manifest.json").read_text())["stage"] == "calibrated"
    Path(c.data.path).write_text("destroyed original data")
    FakeBackend.fail = None
    FakeBackend.calls = []
    assert run(c, resume=True)["status"] == "completed"
    assert not any(isinstance(call, tuple) and call[0] == "train" for call in FakeBackend.calls)
    assert any(isinstance(call, tuple) and call[0] == "load_final" for call in FakeBackend.calls)
    assert json.loads((p / "splits.json").read_text())["assignments"] == expected


def test_real_checkpoint_required_and_semantics_checked(tmp_path):
    c = make_config(tmp_path)
    FakeBackend.fail = "train"
    with pytest.raises(KeyboardInterrupt):
        run(c)
    p = Path(c.output.directory)
    assert json.loads((p / "manifest.json").read_text())["status"] == "interrupted"
    changed = c.model_copy(deep=True)
    changed.training.learning_rate *= 2
    with pytest.raises(ValueError, match="incompatible"):
        run(changed, resume=True)
    (p / "checkpoints/checkpoint-3/optimizer.pt").unlink()
    with pytest.raises(ValueError, match="Corrupt"):
        run(c, resume=True)


def test_resume_training_uses_checkpoint(tmp_path):
    c = make_config(tmp_path)
    FakeBackend.fail = "train"
    with pytest.raises(KeyboardInterrupt):
        run(c)
    FakeBackend.fail = None
    FakeBackend.calls = []
    run(c, resume=True)
    assert (
        "train",
        str(Path(c.output.directory) / "checkpoints/checkpoint-3"),
    ) in FakeBackend.calls


def test_lock_and_nonempty_directory(tmp_path):
    from filelock import FileLock, Timeout

    c = make_config(tmp_path)
    p = Path(c.output.directory)
    p.mkdir()
    (p / "unrelated.txt").write_text("preserve")
    with pytest.raises(ValueError, match="empty"):
        run(c)
    with FileLock(str(p / ".train.lock")):
        with pytest.raises(Timeout):
            run(c)


def test_snapshot_split_corruption_rejected_without_source_read(tmp_path):
    c = make_config(tmp_path)
    FakeBackend.fail = "train"
    with pytest.raises(KeyboardInterrupt):
        run(c)
    directory = Path(c.output.directory)
    p = directory / "splits.json"
    splits = json.loads(p.read_text())
    splits["assignments"][0]["record_id"] = 999999
    p.write_text(json.dumps(splits))
    Path(c.data.path).unlink()
    with pytest.raises(ValueError, match="snapshot|assignments"):
        run(c, resume=True)


def test_failure_report_has_error_and_resume_clears_current_error(tmp_path):
    c = make_config(tmp_path)
    FakeBackend.fail = "save"
    with pytest.raises(ValueError):
        run(c)
    directory = Path(c.output.directory)
    report = json.loads((directory / "report.json").read_text())
    assert report["error"]["message"] == "export interrupted"
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["ended_at"] >= manifest["started_at"]
    FakeBackend.fail = None
    run(c, resume=True)
    assert "error" not in json.loads((directory / "report.json").read_text())
    assert "error" not in json.loads((directory / "manifest.json").read_text())


def test_terminal_checkpoint_before_stage_commit_never_retrains(tmp_path):
    c = make_config(tmp_path, training={"num_train_epochs": None, "max_steps": 3})
    FakeBackend.fail = "final_checkpoint"
    with pytest.raises(KeyboardInterrupt):
        run(c)
    directory = Path(c.output.directory)
    assert json.loads((directory / "manifest.json").read_text())["stage"] == "training"
    before = (directory / "checkpoints/checkpoint-3/optimizer.pt").read_bytes()
    FakeBackend.fail = None
    FakeBackend.calls = []
    assert run(c, resume=True)["status"] == "completed"
    assert not any(isinstance(call, tuple) and call[0] == "train" for call in FakeBackend.calls)
    assert before == (directory / "checkpoints/checkpoint-3/optimizer.pt").read_bytes()


def test_allowed_logging_override_updates_effective_config(tmp_path):
    import yaml

    c = make_config(tmp_path)
    FakeBackend.fail = "save"
    with pytest.raises(ValueError):
        run(c)
    FakeBackend.fail = None
    c.training.logging_steps = 100
    run(c, resume=True)
    saved = yaml.safe_load((Path(c.output.directory) / "config.yaml").read_text())
    assert saved["training"]["logging_steps"] == 100
