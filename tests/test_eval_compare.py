import asyncio
import json
from pathlib import Path

import httpx
import pytest

from decisionforge.eval.data import write_jsonl
from decisionforge.eval.runner import run_evaluation
from tests.eval_fixtures import sample
from tests.test_eval_chat import chat_config, completion
from tests.test_eval_methods import LLM2JEV
from tests.test_eval_runner import config_at, response


async def paired_runs(
    tmp_path, *, settings=None, state_values=None, items=None, method_handler=None, interrupt=None
):
    configs = []
    for index, (name, chat) in enumerate([("chat", True), ("method", False)]):
        root = tmp_path / name
        root.mkdir()
        config = (
            chat_config(root, items, serving={"enable_thinking": False})
            if chat
            else config_at(root, items)
        )
        if state_values is not None:
            item = sample()
            item["request"]["state"]["enabled"] = state_values[index]
            write_jsonl(Path(config.data.path), [item])
        config.model.base_url = "http://same-service.test"
        config.model.model = "same-alias"
        from decisionforge.eval.models import ModelProvenance

        config.model.provenance = ModelProvenance.model_validate(
            {
                **LLM2JEV,
                "method": "chat_baseline" if chat else "llm2jev",
                "settings": {**LLM2JEV["settings"], **(settings[index] if settings else {})},
            }
        )
        handler = (
            (lambda req: httpx.Response(200, json=completion()))
            if chat
            else method_handler or response
        )
        task = asyncio.create_task(run_evaluation(config, transport=httpx.MockTransport(handler)))
        if not chat and interrupt is not None:
            await interrupt.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await run_evaluation(config, score_only=True)
        else:
            await task
        configs.append(config)
    return configs


@pytest.mark.asyncio
async def test_offline_paired_comparison_has_denominators_and_unknown_cost(tmp_path):
    configs = await paired_runs(tmp_path)
    import decisionforge.eval.compare as compare

    result = compare.compare_runs([c.output for c in configs], paired=True)
    assert result["benchmarks"][0]["paired"] is True
    rows = result["benchmarks"][0]["runs"]
    assert [r["method"] for r in rows] == ["chat_baseline", "llm2jev"]
    assert rows[0]["metrics"]["accuracy"]["eligible"] == 1
    assert rows[0]["metrics"]["accuracy"]["value"] == 1
    assert rows[1]["metrics"]["accuracy"]["value"] == 0
    assert rows[0]["reported_cost_usd"] is None
    assert rows[0]["sample_latency_ms"]["n"] == 1
    assert rows[0]["probability_sources"] == ["llm_self_reported"]
    markdown = compare.comparison_markdown(result)
    assert "llm2jev" in markdown
    assert "10 / 20" in markdown
    assert "Qwen/Qwen3.5-4B" in markdown
    assert result["benchmarks"][0]["metric_deltas"]["accuracy"] == -1


@pytest.mark.parametrize("change", ["target", "metric", "backbone", "thinking", "snapshot"])
@pytest.mark.asyncio
async def test_paired_comparison_refuses_incompatible_or_tampered_inputs(tmp_path, change):
    configs = await paired_runs(tmp_path)
    from pathlib import Path

    import decisionforge.eval.compare as compare

    root = Path(configs[1].output)
    report_path = next(root.glob("reports/*/metrics.json"))
    report = json.loads(report_path.read_text())
    if change in {"target", "snapshot"}:
        path = root / "samples.jsonl"
        value = json.loads(path.read_text())
        value["target"]["value"] = "A"
        path.write_text(json.dumps(value) + "\n")
    elif change == "metric":
        report["metrics"]["ece"]["parameters"]["bins"] = 20
        report_path.write_text(json.dumps(report))
    else:
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        provenance = manifest["config"]["model"]["provenance"]
        if change == "backbone":
            provenance["backbone_revision"] = "a" * 40
        else:
            provenance["settings"]["enable_thinking"] = True
        report["inference_manifest"] = manifest
        manifest_path.write_text(json.dumps(manifest))
        report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        compare.compare_runs([c.output for c in configs], paired=True)


@pytest.mark.asyncio
async def test_partial_comparison_keeps_failures_and_suppresses_improvement(tmp_path):
    sent = asyncio.Event()
    calls = []

    async def interrupted(req):
        calls.append(req)
        if len(calls) == 1:
            return response(req)
        sent.set()
        await asyncio.Event().wait()

    configs = await paired_runs(
        tmp_path,
        items=[sample(identity="one"), sample(identity="two")],
        method_handler=interrupted,
        interrupt=sent,
    )

    import decisionforge.eval.compare as compare

    result = compare.compare_runs([c.output for c in configs], paired=True)
    benchmark = result["benchmarks"][0]
    assert benchmark["paired"] is False
    assert benchmark["metric_deltas"] is None
    assert benchmark["runs"][1]["partial"] is True
    assert benchmark["runs"][1]["counts"]["completed"] == 1
    assert benchmark["runs"][1]["counts"]["incomplete"] == 1
    assert benchmark["runs"][1]["counts"]["unknown_outcomes"] == 1


@pytest.mark.parametrize(
    "settings",
    [
        [{"context_length": 32768}, {"context_length": 1024}],
        [{"disable_cuda_graph": False}, {"disable_cuda_graph": 0}],
    ],
)
@pytest.mark.asyncio
async def test_paired_comparison_rejects_all_different_serving_settings(tmp_path, settings):
    configs = await paired_runs(tmp_path, settings=settings)
    from decisionforge.eval.compare import compare_runs

    with pytest.raises(ValueError, match="serving"):
        compare_runs([c.output for c in configs], paired=True)


@pytest.mark.asyncio
async def test_paired_comparison_distinguishes_boolean_and_number_in_public_json(tmp_path):
    configs = await paired_runs(tmp_path, state_values=[True, 1])
    from decisionforge.eval.compare import compare_runs

    with pytest.raises(ValueError, match="data"):
        compare_runs([c.output for c in configs], paired=True)


@pytest.mark.parametrize(
    "change", ["partial", "attempted", "coverage", "metric_value", "metric_denominator"]
)
@pytest.mark.asyncio
async def test_comparison_rejects_inconsistent_report_counts_or_metrics(tmp_path, change):
    configs = await paired_runs(tmp_path)
    from pathlib import Path

    from decisionforge.eval.compare import compare_runs

    path = next(Path(configs[1].output).glob("reports/*/metrics.json"))
    report = json.loads(path.read_text())
    if change == "partial":
        report["counts"].update(completed=0, incomplete=1)
    elif change == "attempted":
        report["counts"]["attempted"] = 0
    elif change == "coverage":
        report["counts"]["coverage"] = 0
    elif change == "metric_value":
        report["metrics"]["accuracy"]["value"] = 123.0
    else:
        report["metrics"]["accuracy"]["used"] = 2
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="report|Metric"):
        compare_runs([c.output for c in configs], paired=True)
