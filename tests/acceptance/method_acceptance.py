"""Finite local acceptance recorder. No scheduler, model import or dependency installer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import yaml

CLI_COMMAND = [sys.executable, "-u", "-m", "decisionforge"]
ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "examples/training/data/acceptance"
TRAIN_CONFIGS = {
    "full": ROOT / "examples/training/unsloth/full-acceptance.yaml",
    "laya": ROOT / "examples/training/laya/acceptance.yaml",
    "openjev": ROOT / "examples/training/openjev/acceptance.yaml",
}
ANYJEV_CONFIG = ROOT / "examples/eval/training-free/anyjev/acceptance.yaml"
sys.path.insert(0, str(ROOT / "src"))

# A guarded child still enters the public CLI, but fails if static/score-only
# accidentally starts importing a model SDK. No CUDA library is imported here.
GUARD = """
import importlib.abc, runpy, sys
class NoSDK(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch','transformers','unsloth','laya','anyjev','openjev','openjev_rlcd','datasets','peft','bitsandbytes'}:
            raise RuntimeError('Forbidden optional SDK import: ' + fullname)
sys.meta_path.insert(0, NoSDK())
sys.argv = [sys.argv[0], *sys.argv[1:]]
runpy.run_module('decisionforge', run_name='__main__')
"""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    from decisionforge.util import atomic_write

    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def versions():
    result = {}
    for package in ("torch", "transformers", "unsloth", "unsloth_zoo", "laya", "openjev-rlcd", "anyjev", "numpy", "pyarrow"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def invoke(work, name, command, *, guarded=False):
    command = list(command)
    if command[:1] == ["train"] and "--validate-only" in command:
        command = ["train", "validate", *[item for item in command[1:] if item != "--validate-only"]]
    elif command[:1] == ["train"] and command[1:2] not in (["run"], ["validate"]):
        command = ["train", "run", *command[1:]]
    elif command[:1] == ["eval"] and "run" not in command[1:2]:
        command = ["eval", "run", *command[1:]]
    args = [sys.executable, "-u", "-c", GUARD] if guarded else list(CLI_COMMAND)
    args += list(command)
    stdout_path, stderr_path = [work / (name + suffix) for suffix in (".stdout.log", ".stderr.log")]
    scope = [
        sys.executable, str(Path(__file__).resolve().parents[2] / "src/decisionforge/training/supervisor.py"),
        "--parent-pid", str(os.getpid()), "--", *args,
    ]
    linux_scope = sys.platform == "linux"
    if not linux_scope:
        if not guarded:
            raise RuntimeError("Training acceptance requires Linux process supervision")
        scope = args
    record = {
        "command": args, "supervisor_command": scope, "returncode": None,
        "guarded": guarded, "status": "running", "started_at": datetime.now(UTC).isoformat(),
        "scope_reaped": False, "environment_overrides": {"PYTHONUNBUFFERED": "1"},
        "supervision": "linux_subreaper" if linux_scope else "direct_child",
    }
    process, cancelled, error = None, None, None
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

    def forward(signum):
        if process is not None and process.poll() is None:
            try:
                # The dedicated session contains only this launch. Detached
                # descendants remain owned by its approved Linux subreaper.
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                pass

    def cancel(signum, frame):
        nonlocal cancelled
        if cancelled is None:
            cancelled = signum
            forward(signum)

    streams = []
    try:
        for sig in previous:
            signal.signal(sig, cancel)
        # Persist the launch before allocating a child. Logs go directly to
        # unbuffered files, including while the CLI is still running.
        save(work / (name + ".command.json"), record)
        streams.append(stdout_path.open("wb", buffering=0))
        streams.append(stderr_path.open("wb", buffering=0))
        if cancelled is None:
            process = subprocess.Popen(
                scope, cwd=ROOT, stdout=streams[0], stderr=streams[1],
                env={**os.environ, "PYTHONUNBUFFERED": "1"}, start_new_session=True,
            )
            record["supervisor_pid"] = process.pid
            save(work / (name + ".command.json"), record)
            if cancelled is not None:  # Covers a signal during Popen's return.
                forward(cancelled)
            record["scope_returncode"] = process.wait()
            if process.returncode == 0 and cancelled is None:
                # Public CLI returns one JSON object, including the report path.
                record["result"] = json.loads(stdout_path.read_text())
    except BaseException as exc:
        error = exc
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        # Repeated cancellation must not interrupt the subreaper's TERM/KILL
        # escalation and wait-for-ECHILD, or erase the terminal evidence.
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        try:
            if process is not None:
                if process.poll() is None:
                    forward(cancelled or signal.SIGTERM)
                record["scope_returncode"] = process.wait()
            record["scope_reaped"] = linux_scope
            record["returncode"] = 128 + cancelled if cancelled else 1 if error else record.get("scope_returncode", 1)
            record["status"] = "interrupted" if cancelled else "failed" if record["returncode"] else "completed"
            if cancelled:
                record["cancellation_signal"] = cancelled
            record["ended_at"] = datetime.now(UTC).isoformat()
            for stream in streams:
                os.fsync(stream.fileno())
                stream.close()
            save(work / (name + ".command.json"), record)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    return record


def static(work):
    from decisionforge.eval.data import load_samples
    from decisionforge.exporter import read_training_records
    from decisionforge.training.config import load_config
    from decisionforge.training.data import prepare_data

    fixture = FIXTURES / "common-cn.v3.jsonl"
    report = {
        "gpu_acceptance": False, "optional_import_guard": True,
        "shared_cn": {"schema": "decisionforge.training.v3", "path": str(fixture), "sha256": sha(fixture)},
        "cn_consumers": {}, "rejections": {}, "full_catalogs": {},
    }
    configs = {}
    for method in ("full", "laya", "openjev"):
        config = yaml.safe_load(TRAIN_CONFIGS[method].read_text())
        config["data"]["path"] = str(fixture)
        config["output"]["directory"] = str(work / "runs" / method)
        config_path = work / (method + ".yaml")
        config_path.write_text(yaml.safe_dump(config))
        configs[method] = config_path
        result = invoke(work, method, ["train", "--config", str(config_path), "--validate-only"], guarded=True)
        if result["returncode"]:
            raise RuntimeError(f"Static {method} failed; see {work}")
        report["cn_consumers"][method] = {
            "data_path": str(fixture), "data_sha256": sha(fixture), **result,
        }
    evaluation = yaml.safe_load(ANYJEV_CONFIG.read_text())
    evaluation["data"] = {"path": str(fixture)}
    evaluation["output"] = str(work / "runs/eval")
    path = work / "eval.yaml"
    path.write_text(yaml.safe_dump(evaluation))
    result = invoke(work, "canonical_eval", ["eval", "--config", str(path), "--validate-only"], guarded=True)
    if result["returncode"]:
        raise RuntimeError(f"Static eval failed; see {work}")
    records = list(read_training_records(fixture))
    samples = load_samples(fixture)
    assert len(samples) == sum(len(r.request.questions) for r in records)
    assert all(set(s.request.model_dump()) == {"state", "questions"} for s in samples)
    report["cn_consumers"]["canonical_eval"] = {
        "data_path": str(fixture), "data_sha256": sha(fixture), "samples": len(samples), **result,
    }
    prepared = prepare_data(load_config(configs["full"]))
    roles = {}
    for assignment in prepared.assignments:
        roles.setdefault(assignment["group_id"], set()).add(assignment["role"])
    assert all(len(role) == 1 for role in roles.values())
    report["group_roles_disjoint"] = True
    for method in ("laya", "openjev"):
        result = invoke(work, method + "_resume", ["train", "--config", str(configs[method]), "--resume"], guarded=True)
        assert result["returncode"] and "resume" in (work / (method + "_resume.stderr.log")).read_text()
        report["rejections"][method + "_resume"] = result
    config = yaml.safe_load(configs["openjev"].read_text())
    config["data"]["path"] = str(FIXTURES / "common-cns.v3.jsonl")
    path = work / "openjev-score.yaml"
    path.write_text(yaml.safe_dump(config))
    result = invoke(work, "openjev_score", ["train", "--config", str(path), "--validate-only"], guarded=True)
    assert result["returncode"] and "unsupported question type score" in (work / "openjev_score.stderr.log").read_text()
    report["rejections"]["openjev_score"] = result
    for size in (27, 77, 151):
        item = samples[0].model_dump(mode="json")
        item["target"] = None
        item["request"]["questions"]["decision"] = {
            "type": "choice", "criteria": {str(i): None for i in range(size)},
        }
        data = work / f"choice-{size}.jsonl"
        data.write_text(json.dumps(item) + "\n")
        evaluation["data"]["path"] = str(data)
        path = work / f"anyjev-{size}.yaml"
        path.write_text(yaml.safe_dump(evaluation))
        name = f"anyjev_{size}"
        result = invoke(work, name, ["eval", "--config", str(path), "--validate-only"], guarded=True)
        assert result["returncode"] == 0, (work / (name + ".stderr.log")).read_text()
        report["full_catalogs"][name] = result
    assert not any((work / "runs").glob("*")), "Preflight created resource outputs"
    save(work / "static.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    preflight = sub.add_parser("static", help="Base-only shared-fixture/capability checks")
    preflight.add_argument("--work-dir", required=True)
    record = sub.add_parser("record", help="Record one public train/eval CLI invocation")
    record.add_argument("--work-dir", required=True)
    record.add_argument("command", choices=("train", "eval"))
    record.add_argument("--config", required=True)
    record.add_argument("--run-dir", required=True)
    flags = record.add_mutually_exclusive_group()
    flags.add_argument("--resume", action="store_true")
    flags.add_argument("--score-only", action="store_true")
    flags.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    work = Path(args.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=False)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True, check=True).stdout
    save(work / "source.json", {
        "commit": revision, "dirty": bool(dirty), "status_porcelain": dirty,
        "python": sys.version, "platform": platform.platform(), "executable": sys.executable,
        "at": datetime.now(UTC).isoformat(), "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "acceptance_driver_sha256": sha(__file__),
        "installed_versions": versions(),
    })
    if args.mode == "static":
        static(work)
        print(json.dumps({"status": "static_passed", "gpu_acceptance": False, "evidence": str(work)}))
        return 0
    if args.score_only and args.command != "eval":
        parser.error("--score-only is eval only")
    config_path = Path(args.config).resolve()
    (work / "input-config.yaml").write_bytes(config_path.read_bytes())
    save(work / "configuration.json", {"path": str(config_path), "sha256": sha(config_path)})
    command = [args.command, "--config", str(Path(args.config).resolve()), "--run-dir", str(Path(args.run_dir).resolve())]
    if args.command == "eval":
        command.insert(1, "run")
    for name in ("resume", "score_only", "validate_only"):
        if getattr(args, name):
            command.append("--" + name.replace("_", "-"))
    guarded = args.score_only or args.validate_only
    result = invoke(work, "cli", command, guarded=guarded)
    if not result["returncode"] and args.command == "eval" and "report" in result["result"]:
        relative = Path(result["result"]["report"])
        if relative.is_absolute() or relative.parts[0] != "reports" or ".." in relative.parts:
            raise ValueError("Unexpected public eval report path")
        report = Path(args.run_dir).resolve() / relative / "metrics.json"
        save(work / "metrics.json", json.loads(report.read_text()))
    print(json.dumps(result))
    return result["returncode"]


if __name__ == "__main__":
    raise SystemExit(main())
