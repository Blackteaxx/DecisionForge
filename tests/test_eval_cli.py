import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml
from click import unstyle

from decisionforge.cli import main
from decisionforge.eval.data import write_jsonl
from tests.eval_fixtures import sample

ROOT = Path(__file__).resolve().parents[1]


def test_eval_help_validate_relative_paths_and_schema(tmp_path, capsys):
    config_path = tmp_path / "config.yaml"
    write_jsonl(tmp_path / "samples.jsonl", [sample()])
    config_path.write_text(yaml.safe_dump({"data": {"path": "samples.jsonl"}, "output": "run"}))
    assert main(["eval", "run", "--config", str(config_path), "--validate-only"]) == 0
    assert json.loads(capsys.readouterr().out)["samples"] == 1
    assert not (tmp_path / "run").exists()
    assert main(["schema", "export", "--directory", str(tmp_path / "schemas")]) == 0
    capsys.readouterr()
    assert (
        json.loads((tmp_path / "schemas/eval_sample.v1.json").read_text())["additionalProperties"]
        is False
    )
    assert (tmp_path / "schemas/eval_config.v1.json").exists()
    result = subprocess.run(
        ["dforge", "eval", "run", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "--score-only" in unstyle(result.stdout)


def test_standalone_offline_jevbench_preparation_from_another_cwd(tmp_path):
    raw = tmp_path / "dataset/raw/public"
    raw.mkdir(parents=True)
    (raw / "original.jsonl").write_bytes(
        (ROOT / "tests/fixtures/jevbench_oracle/original.jsonl").read_bytes()
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "data/benchmarks/jevbench/prepare.py"),
            "--convert-only",
            "--data-dir",
            str(tmp_path / "dataset"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["conversion"]["records"] == 72
    output = tmp_path / "dataset/processed/public.jsonl"
    assert len(output.read_text().splitlines()) == 72
    assert output.with_suffix(".jsonl.manifest.json").exists()
    assert (tmp_path / "dataset/processed/original.jsonl").exists()


def test_real_http_cli_round_trip_resume_and_offline_rescore(tmp_path):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            assert self.path == "/v1/systemone"
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(payload)
            question = payload["questions"]["decision"]
            kind = question["type"]
            answer = {"type": kind}
            if kind == "noul":
                answer["noul"] = 0.8
            else:
                labels = (
                    list(question["criteria"])
                    if kind == "choice"
                    else [str(i) for i in range(len(question["criteria"]))]
                )
                answer.update(confidence=0.2, probabilities={k: 1 / len(labels) for k in labels})
                if kind == "choice":
                    answer["choice"] = labels[0]
                else:
                    answer.update(
                        score=(len(labels) - 1) / 2,
                        legend={str(i): v for i, v in enumerate(question["criteria"])},
                    )
            body = json.dumps(
                {
                    "model": "local-oracle",
                    "answers": {"decision": answer},
                    "usage": {"input_tokens": 10, "output_tokens": 3},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    config_path = tmp_path / "eval.yaml"
    write_jsonl(
        tmp_path / "samples.jsonl", [sample(k, identity=k) for k in ("choice", "noul", "score")]
    )
    settings = {
        "data": {"path": "samples.jsonl"},
        "output": "run",
        "model": {"base_url": f"http://127.0.0.1:{server.server_port}", "api_key_env": None},
    }
    config_path.write_text(yaml.safe_dump(settings))
    command = ["dforge", "eval", "run", "--config", str(config_path)]
    try:
        for suffix in ([], ["--resume"]):
            result = subprocess.run(
                command + suffix, cwd=ROOT, capture_output=True, text=True, timeout=15
            )
            assert result.returncode == 0, result.stderr
        assert len(calls) == 3
        assert all(set(p) == {"state", "questions", "model"} for p in calls)
        (tmp_path / "samples.jsonl").unlink()
        settings["model"]["api_key_env"] = "NO_SUCH_EVAL_KEY"
        config_path.write_text(yaml.safe_dump(settings))
        result = subprocess.run(
            command + ["--score-only", "--run-dir", str(tmp_path / "run")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0, result.stderr
        output = json.loads(result.stdout)
        metrics = json.loads((tmp_path / "run" / output["report"] / "metrics.json").read_text())
        assert metrics["counts"]["scoring_valid"] == 3
        assert metrics["performance_source"] == "existing_attempts"
        assert len(list((tmp_path / "run/reports").iterdir())) == 3
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def test_cli_preflight_error_exit(tmp_path):
    path = tmp_path / "eval.yaml"
    path.write_text(yaml.safe_dump({"data": {"path": "missing.jsonl"}}))
    assert main(["eval", "run", "--config", str(path), "--validate-only"]) == 1


def test_cli_routes_through_proxy_loaded_from_env_file(tmp_path):
    calls = []

    class Proxy(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            calls.append(self.path)
            self.rfile.read(int(self.headers["Content-Length"]))
            body = json.dumps(
                {"model": "proxy-test", "answers": {"decision": {"type": "noul", "noul": 0.9}}}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Proxy)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    write_jsonl(tmp_path / "samples.jsonl", [sample("noul")])
    path = tmp_path / "eval.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "data": {"path": "samples.jsonl"},
                "output": "run",
                "model": {"base_url": "http://jev.invalid", "api_key_env": None},
            }
        )
    )
    env_path = tmp_path / ".env"
    env_path.write_text(f"HTTP_PROXY=http://127.0.0.1:{server.server_port}\nNO_PROXY=localhost\n")
    environment = {k: v for k, v in os.environ.items() if not k.lower().endswith("_proxy")}
    try:
        result = subprocess.run(
            ["dforge", "eval", "run", "--config", str(path), "--env-file", str(env_path)],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0, result.stderr
        assert calls == ["http://jev.invalid/v1/systemone"]
        prediction = json.loads((tmp_path / "run/predictions.jsonl").read_text())
        assert prediction["transport_ok"] is True
        assert prediction["model"] == "proxy-test"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def test_offline_cli_does_not_load_credentials(tmp_path, monkeypatch):

    def unexpected(*args, **kwargs):
        raise AssertionError("offline command read credentials")

    monkeypatch.setattr("decisionforge.typer_cli.load_dotenv", unexpected)
    path = tmp_path / "config.yaml"
    write_jsonl(tmp_path / "samples.jsonl", [sample()])
    path.write_text(yaml.safe_dump({"data": {"path": "samples.jsonl"}}))
    assert (
        main(["eval", "run", "--config", str(path), "--validate-only", "--env-file", "nonexistent"]) == 0
    )
