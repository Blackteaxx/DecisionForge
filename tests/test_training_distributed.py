import json
import os
import signal
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from filelock import FileLock, Timeout
from test_training_data import make_config
from test_training_runner import FakeBackend


def test_coordinator_holds_lock_and_snapshot_before_launch(tmp_path, monkeypatch):
    import decisionforge.training.distributed as distributed
    from decisionforge.training.runner import run_training

    c = make_config(tmp_path, distributed={"strategy": "ddp", "world_size": 2})

    def launch(config, directory, resume):
        assert not resume
        assert (directory / "samples.jsonl").exists() and (directory / "splits.json").exists()
        with pytest.raises(Timeout):
            with FileLock(str(directory / ".train.lock"), timeout=0):
                pass
        return {"status": "completed", "fixture": True}

    monkeypatch.setattr(distributed, "launch_distributed", launch)
    assert run_training(c)["fixture"]


def test_direct_torchrun_public_runner_rejected_without_writes(tmp_path, monkeypatch):
    from decisionforge.training.runner import run_training

    c = make_config(tmp_path)
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="coordinator|torchrun"):
        run_training(c, backend_factory=FakeBackend)
    assert not Path(c.output.directory).exists()


def test_weight_only_backend_has_no_fake_checkpoint_or_calibration(tmp_path, monkeypatch):
    import decisionforge.training.backend as contract
    from decisionforge.training.runner import run_training

    cap = replace(
        contract.capabilities("unsloth"),
        full_resume=False,
        calibration="none",
        artifact_format="weight_fixture",
        diagnostics_source="fixture upstream",
    )
    monkeypatch.setattr(contract, "capabilities", lambda name: cap)

    class WeightOnly(FakeBackend):
        def save_final_checkpoint(self):
            raise AssertionError("fake Trainer checkpoint")

        def calibrate(self):
            raise AssertionError("unsupported calibration")

        def calibration_state(self):
            raise AssertionError("unsupported calibration state")

    c = make_config(tmp_path, calibration={"enabled": False})
    run_training(c, backend_factory=WeightOnly)
    p = Path(c.output.directory)
    assert not (p / "calibration.json").exists()
    manifest = json.loads((p / "manifest.json").read_text())
    assert "final_checkpoint" not in manifest
    assert manifest["capabilities"]["artifact_format"] == "weight_fixture"
    assert "Unsloth" not in (p / "summary.md").read_text()


def test_snapshot_bytes_identity_rejects_mutation(tmp_path):
    from decisionforge.training.runner import run_training

    c = make_config(tmp_path)

    class Interrupt(FakeBackend):
        fail = "train"

    with pytest.raises(KeyboardInterrupt):
        run_training(c, backend_factory=Interrupt)
    p = Path(c.output.directory) / "samples.jsonl"
    p.write_text(p.read_text().replace('case 1"', 'other1"'))
    with pytest.raises(ValueError, match="identity|snapshot"):
        run_training(c, resume=True, backend_factory=FakeBackend)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper process supervision")
@pytest.mark.parametrize(
    "cancel_signal,expected_code", [(signal.SIGINT, 130), (signal.SIGTERM, 143)]
)
def test_owned_worker_failure_and_interrupt_cleanup(tmp_path, cancel_signal, expected_code):
    from decisionforge.training.distributed import run_process

    with pytest.raises(ValueError, match="exit.*7"):
        run_process([sys.executable, "-c", "raise SystemExit(7)"], tmp_path)
    child = tmp_path / "child.py"
    pidfile = tmp_path / "pids.json"
    child.write_text("""import json,os,subprocess,sys,time
p=subprocess.Popen([sys.executable,'-c','import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(100)'],start_new_session=True)
open(sys.argv[1],'w').write(json.dumps([os.getpid(),p.pid]))
time.sleep(100)
""")
    harness = """import os,signal,sys,threading,time
from pathlib import Path
from decisionforge.training.distributed import run_process
p=Path(sys.argv[2])
def interrupt():
 for i in range(100):
  if p.exists(): break
  time.sleep(.05)
 os.kill(os.getpid(),int(sys.argv[3]))
threading.Thread(target=interrupt,daemon=True).start()
try: run_process([sys.executable,sys.argv[1],str(p)],p.parent)
except KeyboardInterrupt: raise SystemExit(130)
"""
    result = subprocess.run(
        [sys.executable, "-c", harness, str(child), str(pidfile), str(int(cancel_signal))],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == expected_code, result.stderr
    pids = json.loads(pidfile.read_text())
    try:
        for pid in pids:
            assert not Path(f"/proc/{pid}").exists(), "cancellation must reap every owned process"
    finally:
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_two_worker_snapshot_handoff_and_rank_zero_lifecycle(tmp_path, monkeypatch):
    import decisionforge.training.distributed as distributed
    from decisionforge.training.runner import run_training

    c = make_config(tmp_path, distributed={"strategy": "ddp", "world_size": 2})
    harness = tmp_path / "workers.py"
    harness.write_text("""import os,sys,time
from pathlib import Path
sys.path.insert(0,sys.argv[3])
from test_training_runner import FakeBackend
import decisionforge.training.backend as contract
import decisionforge.training.runner as runner
from decisionforge.training.distributed import worker
rank=int(sys.argv[2]);directory=Path(sys.argv[1])
os.environ.update(RANK=str(rank),LOCAL_RANK=str(rank),WORLD_SIZE='2')
def rendezvous(name):
 (directory / (name+str(rank))).touch()
 for i in range(500):
  if all((directory / (name+str(r))).exists() for r in range(2)):return
  time.sleep(.01)
 raise RuntimeError('peer timeout')
class Backend(FakeBackend):
 def train(self,checkpoint=None):
  rendezvous('train')
  return super().train(checkpoint)
 def save_final_checkpoint(self):
  if rank==0: result=super().save_final_checkpoint()
  rendezvous('save')
  return directory/'checkpoints/checkpoint-3'
 def finish_distributed(self):rendezvous('finish')
 def evaluate(self,role,phase):
  assert rank==0 and (directory/'finish1').exists()
  return super().evaluate(role,phase)
 def save(self):
  assert rank==0
  return super().save()
contract.get_backend=runner.get_backend=Backend
if rank:
 def reject(*a,**k):raise AssertionError('nonzero lifecycle write')
 runner.write_json=runner.write_report=runner.atomic_write=reject
worker(directory)
""")

    def processes(command, directory, *, env):
        assert "--nproc_per_node=2" in command and "--max_restarts=0" in command
        Path(c.data.path).write_text("original source deliberately unavailable")
        children = [
            subprocess.Popen(
                [
                    sys.executable,
                    str(harness),
                    str(directory),
                    str(rank),
                    str(Path(__file__).parent),
                ],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for rank in range(2)
        ]
        for child in children:
            stdout, stderr = child.communicate(timeout=15)
            assert child.returncode == 0, stdout + stderr

    monkeypatch.setattr(distributed, "run_process", processes)
    result = run_training(c)
    assert result["status"] == "completed"
    directory = Path(c.output.directory)
    assert (directory / "finish1").exists() and (
        directory / "reload_predictions.json"
    ).exists() is False
    assert json.loads((directory / "worker_result.json").read_text())["status"] == "completed"


def test_failed_launch_never_reports_completed_and_releases_lock(tmp_path, monkeypatch):
    import decisionforge.training.distributed as distributed
    from decisionforge.training.runner import run_training

    c = make_config(tmp_path, distributed={"strategy": "ddp", "world_size": 2})

    def fail(*args, **kwargs):
        raise ValueError("worker exit code 9")

    monkeypatch.setattr(distributed, "run_process", fail)
    with pytest.raises(ValueError, match="exit code 9"):
        run_training(c)
    directory = Path(c.output.directory)
    assert json.loads((directory / "manifest.json").read_text())["status"] == "failed"
    with FileLock(str(directory / ".train.lock"), timeout=0):
        pass


def test_resume_rejects_actual_precision_change_before_prepare(tmp_path):
    from decisionforge.training.runner import run_training

    c = make_config(tmp_path)

    class Precision(FakeBackend):
        selected = "bf16"
        fail = "train"

        def environment(self):
            return {**super().environment(), "precision": self.selected, "world_size": 1}

    with pytest.raises(KeyboardInterrupt):
        run_training(c, backend_factory=Precision)
    Precision.selected = "fp16"
    Precision.fail = None
    Precision.calls = []
    with pytest.raises(ValueError, match="environment|precision"):
        run_training(c, resume=True, backend_factory=Precision)
    assert Precision.calls == ["close"]


def test_backend_close_failure_cannot_commit_completed(tmp_path):
    from decisionforge.training.runner import run_training

    c = make_config(tmp_path)

    class BrokenClose(FakeBackend):
        def close(self):
            raise RuntimeError("cleanup failed")

    with pytest.raises(ValueError, match="cleanup failed"):
        run_training(c, backend_factory=BrokenClose)
    manifest = json.loads((Path(c.output.directory) / "manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["stage"] == "exported"


def test_legacy_dependency_metadata_allows_newly_recorded_packages():
    from decisionforge.training.runner import verify_environment

    verify_environment(
        {"packages": {"torch": "2.11.0"}}, {"packages": {"torch": "2.11.0", "pyarrow": "21.0.0"}}
    )
    with pytest.raises(ValueError, match="packages"):
        verify_environment(
            {"packages": {"torch": "2.11.0"}},
            {"packages": {"torch": "different", "pyarrow": "21.0.0"}},
        )


@pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper process supervision")
@pytest.mark.parametrize("exit_code", [0, 7])
def test_fast_exit_launcher_reaps_detached_descendant_before_return(tmp_path, exit_code):
    from decisionforge.training.distributed import run_process

    pidfile = tmp_path / "detached.pid"
    launcher = """import os,subprocess,sys
from pathlib import Path
child=subprocess.Popen([sys.executable,'-c','import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)'],start_new_session=True)
Path(sys.argv[1]).write_text(str(child.pid))
os._exit(int(sys.argv[2]))
"""
    unrelated = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"])
    child_pid = None
    try:
        if exit_code:
            with pytest.raises(ValueError, match="exit.*7"):
                run_process(
                    [sys.executable, "-c", launcher, str(pidfile), str(exit_code)], tmp_path
                )
        else:
            run_process([sys.executable, "-c", launcher, str(pidfile), str(exit_code)], tmp_path)
        child_pid = int(pidfile.read_text())
        assert not Path(f"/proc/{child_pid}").exists(), (
            "owned detached child must be terminated and reaped"
        )
        assert unrelated.poll() is None, (
            "supervisor must not adopt or signal unrelated coordinator children"
        )
    finally:
        if child_pid is None and pidfile.exists():
            child_pid = int(pidfile.read_text())
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        unrelated.kill()
        unrelated.wait()
