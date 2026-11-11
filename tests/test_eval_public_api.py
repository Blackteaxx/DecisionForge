import json

import httpx
import pytest

from decisionforge.cli import main
from tests.test_eval_runner import config_at, response


def test_eval_list_is_offline_and_describes_registered_capabilities(capsys, monkeypatch):
    monkeypatch.setattr(httpx, "Client", lambda *a, **k: pytest.fail("listing opened HTTP"))
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: pytest.fail("listing opened HTTP"))
    assert main(["eval", "list"]) == 0
    result = json.loads(capsys.readouterr().out)
    benchmarks = {b["name"]: b for b in result["benchmarks"]}
    metrics = {m["name"]: m for m in result["metrics"]}
    assert benchmarks["banking77"]["default_metrics"] == {"macro_f1": {}, "native_accuracy": {}}
    assert benchmarks["typed_decisions"]["supported_types"] == ["choice", "noul", "score"]
    assert metrics["macro_f1"]["direction"] == "higher"


@pytest.mark.asyncio
async def test_public_registry_is_used_for_preflight_run_and_offline_scoring(tmp_path):
    from decisionforge.eval import (
        BaseBenchmark,
        EvalConfig,
        Metric,
        Registry,
        Sample,
        load_registry,
        run_evaluation,
    )

    assert isinstance(load_registry(), Registry)
    assert EvalConfig and Sample

    class MyBenchmark(BaseBenchmark):
        name = "test_plugin"
        version = "1"
        default_metrics = {"custom_valid_rate": {}}

    def valid_rate(rows, params):
        valid = sum(row["scoring_valid"] for row in rows)
        return {
            "value": valid / len(rows),
            "eligible": len(rows),
            "used": len(rows),
            "failed": len(rows) - valid,
        }

    registry = Registry()
    registry.register_metric(Metric("custom_valid_rate", "1", "higher", "ratio", valid_rate))
    registry.register_benchmark(MyBenchmark())
    config = config_at(tmp_path)
    config.benchmark = "test_plugin"
    calls = []

    def send(request):
        calls.append(request)
        return response(request)

    preflight = await run_evaluation(config, registry=registry, validate_only=True)
    assert preflight["benchmark"] == "test_plugin" and not calls
    result = await run_evaluation(config, registry=registry, transport=httpx.MockTransport(send))
    scored = await run_evaluation(config, registry=registry, score_only=True)
    assert len(calls) == 1
    for report in (result, scored):
        metrics = json.loads((tmp_path / "run" / report["report"] / "metrics.json").read_text())
        assert metrics["metrics"]["custom_valid_rate"]["value"] == 1
