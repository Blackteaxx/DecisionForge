import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from decisionforge.eval.data import write_jsonl
from decisionforge.eval.models import EvalConfig, load_config
from decisionforge.eval.runner import run_evaluation
from tests.eval_fixtures import sample

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/eval/training-free"


@pytest.mark.parametrize("route", ["chat", "llm2jev"])
@pytest.mark.parametrize(
    "name", ["jevbench-public", "typed-decisions", "banking77", "clinc150-oos"]
)
def test_training_free_paired_configs_keep_dataset_and_model_identity(route, name):
    config = load_config(ROOT / f"configs/eval/{route}/{name}.yaml")
    peer = load_config(
        ROOT / f"configs/eval/{'llm2jev' if route == 'chat' else 'chat'}/{name}.yaml"
    )
    assert config.data.model_dump() == peer.data.model_dump()
    assert config.metrics == peer.metrics
    assert config.model.model == peer.model.model
    assert config.model.provenance.backbone_revision == peer.model.provenance.backbone_revision
    assert config.model.provenance.precision == peer.model.provenance.precision
    assert config.output != peer.output
    assert config.model.adapter == ("chat_api" if route == "chat" else "jev")


@pytest.fixture
def example_checkout(tmp_path):
    """Exercise the scripts with isolated canonical data, without downloading datasets."""
    root = tmp_path / "checkout"
    shutil.copytree(EXAMPLE, root / "examples/eval/training-free")
    (root / "src").symlink_to(ROOT / "src", target_is_directory=True)
    for dataset, split in [
        ("jevbench", "public"),
        ("typed_decisions", "test"),
        ("banking77", "test"),
        ("clinc150_oos", "test"),
    ]:
        items = []
        for index in range(3):
            item = sample(
                target="soft" if dataset == "typed_decisions" else "hard", identity=str(index)
            )
            item["split"] = split
            if dataset == "typed_decisions":
                item["metadata"].update(question_name="decision", gold_label="B")
            if dataset in {"banking77", "clinc150_oos"}:
                labels = ["A", "B"] + [str(i) for i in range(75 if dataset == "banking77" else 148)]
                if dataset == "clinc150_oos":
                    labels += ["oos"]
                item["request"]["questions"]["decision"]["criteria"] = {k: k for k in labels}
            items.append(item)
        write_jsonl(root / f"data/benchmarks/{dataset}/processed/{split}.jsonl", items)
    return root


def test_offline_compare_example_can_be_invoked_from_another_directory(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples/eval/training-free/scripts/compare.py"), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "--paired" in result.stdout


@pytest.mark.parametrize("route", ["chat", "llm2jev", "both"])
def test_example_run_script_preflights_every_benchmark_without_a_service(
    tmp_path, example_checkout, route
):
    result = subprocess.run(
        [
            "bash",
            str(example_checkout / "examples/eval/training-free/scripts/run.sh"),
            route,
            "--validate-only",
        ],
        cwd=tmp_path,
        env={**os.environ, "DFORGE_PYTHON": sys.executable},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    decoder, remaining, reports = json.JSONDecoder(), result.stdout.strip(), []
    while remaining:
        report, end = decoder.raw_decode(remaining)
        reports.append(report)
        remaining = remaining[end:].strip()
    assert len(reports) == (8 if route == "both" else 4)
    assert all(r["status"] == "validated" and r["samples"] == 3 for r in reports)
    assert {r["benchmark"] for r in reports} == {
        "jevbench",
        "typed_decisions",
        "banking77",
        "clinc150_oos",
    }
    assert not (example_checkout / "outputs").exists()


def test_batch_script_rejects_one_shared_output_override(tmp_path):
    result = subprocess.run(
        [
            "bash",
            str(EXAMPLE / "scripts/run.sh"),
            "both",
            "--validate-only",
            "--run-dir",
            str(tmp_path / "run"),
        ],
        cwd=tmp_path,
        env={**os.environ, "DFORGE_PYTHON": sys.executable},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "--run-dir" in result.stderr
    assert not (tmp_path / "run").exists()


@pytest.mark.asyncio
async def test_report_defaults_to_all_four_pairs_and_uses_native_intent_accuracy(
    tmp_path, example_checkout
):
    import httpx

    for route in ["chat", "llm2jev"]:
        for name in ["jevbench-public", "typed-decisions", "banking77", "clinc150-oos"]:
            config = load_config(
                example_checkout / f"examples/eval/training-free/configs/{route}/{name}.yaml"
            )

            def response(request):
                body = json.loads(request.content)
                chat = "messages" in body
                public = json.loads(body["messages"][1]["content"]) if chat else body
                labels = public["questions"]["decision"]["criteria"]
                # A conflicting argmax must not replace the native choice for intent accuracy.
                probabilities = {k: float(k == ("A" if len(labels) > 2 else "B")) for k in labels}
                answers = {
                    "decision": {
                        "type": "choice",
                        "choice": "B",
                        "confidence": 1.0,
                        "probabilities": probabilities,
                    }
                }
                payload = {"model": "qwen3-4b-instruct-2507", "answers": answers}
                if chat:
                    payload = {
                        "model": "qwen3-4b-instruct-2507",
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "stop",
                                "message": {
                                    "role": "assistant",
                                    "content": json.dumps({"answers": answers}),
                                },
                            }
                        ],
                    }
                return httpx.Response(200, json=payload)

            await run_evaluation(config, transport=httpx.MockTransport(response))

    output = tmp_path / "report"
    result = subprocess.run(
        [
            sys.executable,
            str(example_checkout / "examples/eval/training-free/scripts/compare.py"),
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    comparison = json.loads((output / "comparison.json").read_text())
    assert len(comparison["benchmarks"]) == 4
    assert all(group["paired"] for group in comparison["benchmarks"])
    for group in comparison["benchmarks"]:
        for row in group["runs"]:
            assert not Path(row["run_directory"]).is_absolute()
    assert comparison["run_directory_base"] == "comparison_file_directory"
    # Exported run links still resolve when reports and runs are moved together.
    moved = tmp_path / "moved"
    shutil.copytree(example_checkout / "outputs", moved / "checkout/outputs")
    shutil.copytree(output, moved / "report")
    from decisionforge.eval.compare import load_run

    exported = json.loads((moved / "report/comparison.json").read_text())
    for group in exported["benchmarks"]:
        for row in group["runs"]:
            _, _, _, metrics, _ = load_run(moved / "report" / row["run_directory"])
            assert metrics["counts"]["planned"] == 3
    report = (output / "report.md").read_text()
    banking_row = next(
        line for line in report.splitlines() if "BANKING77" in line and "Chat" in line
    )
    assert "100.00%" in banking_row and "3 / 3" in banking_row
    assert "LLM2Jev" in report and "Typed Decisions" in report


def test_eval_config_schema_matches_runtime_contract():
    import json

    assert (
        json.loads((ROOT / "schemas/eval_config.v1.json").read_text())
        == EvalConfig.model_json_schema()
    )
