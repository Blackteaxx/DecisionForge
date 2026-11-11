import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError
from test_training_data import make_config

from decisionforge.training.config import TrainConfig


def test_full_config_roundtrip_and_legacy_lora():
    old = TrainConfig(data={"path": "unused"})
    assert old.finetuning == "lora" and old.lora.rank == 16
    full = TrainConfig(data={"path": "unused"}, finetuning="full", model={"load_in_4bit": False})
    assert full.lora is None
    assert TrainConfig.model_validate(full.model_dump()) == full
    for patch in ({"model": {"load_in_4bit": True}}, {"lora": {"rank": 8}}):
        with pytest.raises(ValidationError, match="full"):
            TrainConfig(data={"path": "unused"}, finetuning="full", **patch)


@pytest.mark.parametrize(
    "settings",
    [
        {"strategy": "single", "world_size": 2},
        {"strategy": "ddp", "world_size": 1},
        {"strategy": "ddp", "world_size": "2"},
        {"strategy": "fsdp", "world_size": 2},
        {"strategy": "ddp", "world_size": 2, "nnodes": 2},
    ],
)
def test_distributed_config_rejects_unsupported_settings(settings):
    with pytest.raises(ValidationError):
        TrainConfig(data={"path": "unused"}, distributed=settings)


def test_capabilities_and_unsupported_resume_are_gpu_free(tmp_path):
    from decisionforge.training.backend import capabilities, preflight
    from decisionforge.training.runner import run_training

    assert capabilities("unsloth").full_resume
    assert capabilities("laya").distributed and not capabilities("laya").full_resume
    assert not capabilities("openjev").distributed
    for name in ("laya", "openjev"):
        c = TrainConfig(
            data={"path": "missing"},
            backend=name,
            finetuning="full",
            model={"load_in_4bit": False},
            calibration={"enabled": False},
            output={"directory": str(tmp_path / name)},
        )
        with pytest.raises(ValueError, match="resume"):
            run_training(c, resume=True)
        assert not Path(c.output.directory).exists()
    c.distributed.strategy, c.distributed.world_size = "ddp", 2
    with pytest.raises(ValueError, match="distributed|DDP"):
        preflight(c)


def test_full_cli_static_preflight_without_gpu_sdks(tmp_path):
    c = make_config(
        tmp_path,
        finetuning="full",
        model={"load_in_4bit": False},
        distributed={"strategy": "ddp", "world_size": 2},
    )
    cfg = tmp_path / "full.json"
    cfg.write_text(json.dumps(c.model_dump()))
    code = """import sys
class Reject:
 def find_spec(self,name,*args):
  if name.split('.')[0] in {'torch','unsloth','transformers','laya','openjev_rlcd'}: raise AssertionError(name)
sys.meta_path.insert(0,Reject())
from decisionforge.cli import main
raise SystemExit(main(sys.argv[1:]))"""
    result = subprocess.run(
        [sys.executable, "-c", code, "train", "validate", "--config", str(cfg)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert '"status": "validated"' in result.stdout
    assert not Path(c.output.directory).exists()
