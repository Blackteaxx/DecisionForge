"""Run the external Decisions API through an isolated, temporary mihomo route."""

from __future__ import annotations

import argparse
import copy
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import yaml

from decisionforge.eval import load_config
from decisionforge.util import now, write_json


def task_proxy_config(original, port, node):
    nodes = {p["name"]: p for p in original.get("proxies", [])}
    selected, active = {}, set()

    def include(name):
        if name in {"DIRECT", "REJECT"} or name in selected:
            return
        if name not in nodes or name in active:
            raise ValueError("Proxy node has an unknown, dynamic or cyclic dialer dependency")
        active.add(name)
        item = copy.deepcopy(nodes[name])
        if item.get("dialer-proxy"):
            include(item["dialer-proxy"])
        selected[name] = item
        active.remove(name)

    include(node)
    if node not in selected:
        raise ValueError("Select an explicit configured outbound node")
    return {
        "mixed-port": port,
        "bind-address": "127.0.0.1",
        "allow-lan": False,
        "mode": "rule",
        "log-level": "warning",
        "ipv6": False,
        "dns": {"enable": False},
        "tun": {"enable": False},
        "proxies": list(selected.values()),
        "rules": [f"DOMAIN-SUFFIX,openrouter.ai,{node}", "MATCH,DIRECT"],
    }


@contextmanager
def isolated_proxy(binary, original_config, node, evidence):
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    source = yaml.safe_load(original_config.read_text())
    config = task_proxy_config(source, port, node)
    # Credentials are only copied into a 0600 config in a 0700 temporary directory.
    with tempfile.TemporaryDirectory(prefix="dforge-medical-mihomo-") as directory:
        root = Path(directory)
        path = root / "config.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        path.chmod(0o600)
        with (root / "private.log").open("w") as log:
            process = subprocess.Popen(
                [str(binary), "-d", str(root), "-f", str(path)], stdout=log, stderr=log
            )
            try:
                deadline = time.monotonic() + 15
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(
                            f"Isolated mihomo exited {process.returncode}; private config/log removed, no credential output"
                        )
                    try:
                        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                            break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                "Isolated mihomo did not open its local listener"
                            ) from None
                        time.sleep(0.2)
                write_json(
                    evidence,
                    {
                        "started_at": now(),
                        "pid": process.pid,
                        "port": port,
                        "node": node,
                        "fixed_dialer_chain": [p["name"] for p in config["proxies"]],
                        "original_config": str(original_config),
                        "shared_proxy_mutated": False,
                        "credentials_archived": False,
                        "temporary_permissions": "directory0700/config0600",
                        "routing": "openrouter.ai only; remaining traffic direct",
                    },
                )
                yield f"http://127.0.0.1:{port}"
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()


def run_configs(configs, calibration_manifest, control, binary, proxy_config, node, env_file):
    selected = json.loads(calibration_manifest.read_text())
    if selected["status"] != "completed" or not selected.get("reload", {}).get("new_process"):
        raise ValueError("Formal external evaluation waits for selected calibrated/reloaded model")
    for path in configs:
        config = load_config(path)
        if (
            config.model.adapter != "jev"
            or config.model.endpoint_path != "/api/alpha/decisions"
            or config.model.base_url != "https://openrouter.ai"
        ):
            raise ValueError("External Jev config must use the actual OpenRouter Decisions API")
        if config.data.limit is not None or config.data.splits != ["test"]:
            raise ValueError("Formal external Jev requires full frozen tests")
        if Path(config.output).exists():
            raise ValueError(
                "Run already exists; inspect current state and use native compatible resume explicitly"
            )
    control.mkdir(parents=True, exist_ok=False)
    state = {
        "status": "running",
        "started_at": now(),
        "runs": [],
        "calibration_manifest": str(calibration_manifest),
    }
    write_json(control / "state.json", state)
    try:
        with isolated_proxy(binary, proxy_config, node, control / "proxy.json") as proxy:
            env = dict(
                os.environ,
                HTTP_PROXY=proxy,
                HTTPS_PROXY=proxy,
                ALL_PROXY=proxy,
                http_proxy=proxy,
                https_proxy=proxy,
                all_proxy=proxy,
            )
            for path in configs:
                folder = path.parent
                command = [
                    sys.executable,
                    "-m",
                    "decisionforge",
                    "eval",
                    "run", "--config",
                    str(path),
                    "--env-file",
                    str(env_file),
                ]
                with (
                    (folder / "stdout.log").open("w") as out,
                    (folder / "stderr.log").open("w") as err,
                ):
                    outcome = subprocess.run(command, env=env, stdout=out, stderr=err)
                state["runs"].append({"config": str(path), "returncode": outcome.returncode})
                write_json(control / "state.json", state)
                if outcome.returncode:
                    raise RuntimeError(
                        f"External Jev failed for {path}; preserve attempts for native recovery"
                    )
                config = load_config(path)
                report = max(
                    Path(config.output).glob("reports/*/metrics.json"),
                    key=lambda p: p.stat().st_mtime,
                )
                metrics = json.loads(report.read_text())
                if metrics["partial"]:
                    raise ValueError(
                        "External Jev returned a partial report; remaining work is incomplete"
                    )
                state["runs"][-1].update(
                    status="completed", report=str(report), counts=metrics["counts"]
                )
                write_json(control / "state.json", state)
        state["status"] = "completed"
    except BaseException as exc:
        state.update(status="failed", error={"kind": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        state["updated_at"] = now()
        write_json(control / "state.json", state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, nargs="+", required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--mihomo-bin", type=Path, required=True)
    parser.add_argument("--mihomo-config", type=Path, required=True)
    parser.add_argument("--proxy-node", default="bandwagon-vps")
    parser.add_argument("--env-file", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run_configs(
                args.configs,
                args.calibration_manifest,
                args.control_root,
                args.mihomo_bin,
                args.mihomo_config,
                args.proxy_node,
                args.env_file,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
