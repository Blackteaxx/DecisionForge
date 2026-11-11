"""Single-host coordinator and internal torchrun workers; no torch imports here."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import uuid
from pathlib import Path

from decisionforge.util import write_json


def run_process(command, directory, *, env=None):
    """Run a scoped Linux subreaper; return only after its complete tree is reaped."""
    directory = Path(directory)
    log = directory / "logs" / "distributed.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    previous = signal.getsignal(signal.SIGTERM)
    previous_interrupt = signal.getsignal(signal.SIGINT)

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    process = None
    try:
        signal.signal(signal.SIGTERM, terminate)
        with log.open("a", encoding="utf-8") as stream:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "decisionforge.training.supervisor",
                    "--parent-pid",
                    str(os.getpid()),
                    "--",
                    *command,
                ],
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
            code = process.wait()
            if code:
                raise ValueError(f"Distributed worker exit code {code}; see {log}")
    finally:
        # A second cancellation must not kill the subreaper before it has drained
        # adopted workers. It owns only this launch, never other application children.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            if process is not None:
                if process.poll() is None:
                    try:
                        os.kill(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                process.wait()
        finally:
            signal.signal(signal.SIGTERM, previous)
            signal.signal(signal.SIGINT, previous_interrupt)


def launch_distributed(config, directory, resume):
    token = uuid.uuid4().hex
    result_path = directory / "worker_result.json"
    result_path.unlink(missing_ok=True)
    plan = {
        "token": token,
        "config": config.model_dump(),
        "resume": resume,
        "manifest": json.loads((directory / "manifest.json").read_text()),
        "report": json.loads((directory / "report.json").read_text()),
    }
    write_json(directory / "worker_plan.json", plan)
    env = {**os.environ, "DFORGE_TRAIN_WORKER": token}
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={config.distributed.world_size}",
        "--max_restarts=0",
        "--module",
        "decisionforge.training.distributed",
        "--run-directory",
        str(directory),
    ]
    run_process(command, directory, env=env)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("status") != "completed" or not result_path.is_file():
        raise ValueError("Distributed workers exited without completed rank-zero lifecycle")
    result = json.loads(result_path.read_text())
    if result.pop("token", None) != token:
        raise ValueError("Distributed worker result does not match this launch")
    return result


def run_peer(config, directory, prepared, manifest, resume, backend_factory=None):
    """Nonzero rank: train/save collectively, then release the distributed group."""
    from decisionforge.training.backend import get_backend, preflight
    from decisionforge.training.runner import (
        checkpoint_progress,
        effective_backend_config,
        latest_checkpoint,
        verify_environment,
    )

    backend = (backend_factory or get_backend)(
        effective_backend_config(config, manifest, resume), directory
    )
    try:
        cap = preflight(config, resume=resume, data=prepared)
        environment = backend.environment()
        if resume:
            verify_environment(manifest.get("environment", {}), environment)
        checkpoint = latest_checkpoint(directory, backend) if resume else None
        backend.prepare(prepared)
        if manifest["stage"] == "training":
            terminal, _ = checkpoint_progress(checkpoint, backend) if checkpoint else (False, None)
            if terminal:
                backend.load_final_checkpoint(checkpoint)
            else:
                backend.train(checkpoint)
                if cap.full_resume:
                    backend.save_final_checkpoint()
        else:
            backend.load_final_checkpoint(directory / manifest["final_checkpoint"])
        backend.finish_distributed()
    finally:
        backend.close()


def worker(directory):
    from decisionforge.exporter import read_training_records
    from decisionforge.training.config import TrainConfig
    from decisionforge.training.data import PreparedData
    from decisionforge.training.runner import (
        check_snapshot,
        run_lifecycle,
        verify_snapshot_identity,
    )

    directory = Path(directory).resolve()
    plan = json.loads((directory / "worker_plan.json").read_text())
    if (
        not os.environ.get("DFORGE_TRAIN_WORKER")
        or os.environ["DFORGE_TRAIN_WORKER"] != plan["token"]
    ):
        raise ValueError("Internal training worker requires the active coordinator launch")
    config = TrainConfig.model_validate(plan["config"])
    world, rank = int(os.environ.get("WORLD_SIZE", "0")), int(os.environ.get("RANK", "-1"))
    if world != config.distributed.world_size or not 0 <= rank < world:
        raise ValueError("Invalid worker world_size/rank")
    manifest, report = plan["manifest"], plan["report"]
    split = json.loads((directory / "splits.json").read_text())
    prepared = PreparedData(
        list(read_training_records(directory / "samples.jsonl")),
        split["assignments"],
        manifest["data"],
    )
    check_snapshot(prepared)
    verify_snapshot_identity(directory, manifest, config)
    if rank:
        run_peer(config, directory, prepared, manifest, plan["resume"])
    else:
        result = run_lifecycle(config, directory, prepared, manifest, report, plan["resume"])
        write_json(directory / "worker_result.json", {**result, "token": plan["token"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Internal DecisionForge training worker")
    parser.add_argument("--run-directory", required=True)
    worker(parser.parse_args().run_directory)
