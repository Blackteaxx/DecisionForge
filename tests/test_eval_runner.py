import asyncio
import json
import time

import httpx
import pytest

from decisionforge.eval.data import write_jsonl
from decisionforge.eval.models import EvalConfig
from decisionforge.eval.runner import run_evaluation
from tests.eval_fixtures import sample


def config_at(tmp_path, items=None, **execution):
    path = tmp_path / "data.jsonl"
    write_jsonl(path, items or [sample()])
    return EvalConfig(
        data={"path": str(path)},
        output=str(tmp_path / "run"),
        model={"base_url": "http://jev.test", "model": "test", "api_key_env": None},
        execution=execution,
    )


def response(request, choice="A"):
    q = json.loads(request.content)["questions"]["decision"]
    if q["type"] == "noul":
        answer = {"type": "noul", "noul": 0.8}
    elif q["type"] == "score":
        answer = {
            "type": "score",
            "score": 1.6,
            "confidence": 0.4,
            "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7},
            "legend": {str(i): v for i, v in enumerate(q["criteria"])},
        }
    else:
        answer = {
            "type": "choice",
            "choice": choice,
            "confidence": 0.6,
            "probabilities": {"A": 0.8, "B": 0.2},
        }
    return httpx.Response(
        200,
        json={
            "model": "actual-model",
            "answers": {"decision": answer},
            "usage": {"input_tokens": 5, "output_tokens": 2},
        },
    )


@pytest.mark.asyncio
async def test_payload_wrong_prediction_resume_and_rescore(tmp_path, monkeypatch):
    config = config_at(tmp_path)
    calls = []

    def handle(req):
        events = (tmp_path / "run/.eval/events.jsonl").read_text().splitlines()
        sent = json.loads(events[-1])["updates"][0]["record"]
        assert sent["status"] == "in_flight"
        assert sent["request"] == json.loads(req.content)
        calls.append(json.loads(req.content))
        return response(req)

    result = await run_evaluation(config, transport=httpx.MockTransport(handle))
    assert calls == [{"model": "test", **sample()["request"]}]
    report = json.loads((tmp_path / "run" / result["report"] / "metrics.json").read_text())
    assert report["metrics"]["accuracy"]["value"] == 0
    assert report["counts"]["scoring_valid"] == 1
    pred = json.loads((tmp_path / "run/predictions.jsonl").read_text())
    assert pred["answer"]["choice"] == "A"
    assert pred["model"] == "actual-model"
    await run_evaluation(config, resume=True, transport=httpx.MockTransport(handle))
    assert len(calls) == 1
    (tmp_path / "data.jsonl").unlink()
    monkeypatch.setenv("TYPESAFE_API_KEY", "must-not-be-read")
    config.model.api_key_env = "MISSING_FOR_SCORE_ONLY"
    rescored = await run_evaluation(config, score_only=True)
    assert rescored["report"] != result["report"]
    assert len(calls) == 1
    assert pred == json.loads((tmp_path / "run/predictions.jsonl").read_text())


@pytest.mark.asyncio
async def test_preflight_no_calls_and_changed_resume_refused(tmp_path):
    config = config_at(tmp_path)
    calls = []
    transport = httpx.MockTransport(lambda req: calls.append(req) or response(req))
    config.metrics = {"unknown": {}}
    with pytest.raises(ValueError, match="Unknown"):
        await run_evaluation(config, transport=transport)
    assert not calls
    config.metrics = {}
    await run_evaluation(config, transport=transport)
    config.model.model = "different"
    with pytest.raises(ValueError, match="incompatible"):
        await run_evaluation(config, resume=True, transport=transport)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_resume_missing_journal_refuses_to_repeat_completed_calls(tmp_path):
    config = config_at(tmp_path)
    calls = []
    transport = httpx.MockTransport(lambda req: calls.append(req) or response(req))
    await run_evaluation(config, transport=transport)
    (tmp_path / "run/.eval/events.jsonl").unlink()
    with pytest.raises(ValueError, match="journal does not exist"):
        await run_evaluation(config, resume=True, transport=transport)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_offline_torn_tail_is_unchanged_and_resume_skips_terminal_prediction(tmp_path):
    config = config_at(tmp_path)
    calls = []
    transport = httpx.MockTransport(lambda req: calls.append(req) or response(req))
    await run_evaluation(config, transport=transport)
    journal = tmp_path / "run/.eval/events.jsonl"
    committed = journal.read_bytes()
    journal.write_bytes(committed + b'{"version":')
    await run_evaluation(config, score_only=True)
    assert journal.read_bytes() == committed + b'{"version":'
    await run_evaluation(config, resume=True, transport=transport)
    assert journal.read_bytes() == committed
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_retry_after_records_each_attempt_and_usage_once(tmp_path):
    config = config_at(tmp_path, max_attempts=2, backoff=0)
    calls = []

    def handle(req):
        calls.append(req)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, json={"error": "limit"})
        return response(req)

    result = await run_evaluation(config, transport=httpx.MockTransport(handle))
    records = [json.loads(x) for x in (tmp_path / "run/attempts.jsonl").read_text().splitlines()]
    assert [r["http_status"] for r in records] == [429, 200]
    assert len({r["id"] for r in records}) == 2
    report = json.loads((tmp_path / "run" / result["report"] / "metrics.json").read_text())
    assert report["performance"]["sessions"][0]["attempts"] == 2
    assert report["performance"]["usage"]["input_tokens"] == 5
    assert report["performance"]["usage"]["missing_attempts"] == 1
    assert report["performance"]["attempt_latency_ms"]["p50_ms"] == pytest.approx(
        sum(a["attempt_latency_ms"] for a in records) / 2
    )


@pytest.mark.asyncio
async def test_terminal_failure_auth_stops_and_resume_does_not_retry(tmp_path):
    config = config_at(tmp_path, [sample(identity="one"), sample(identity="two")], max_attempts=3)
    calls = []
    transport = httpx.MockTransport(
        lambda req: calls.append(req) or httpx.Response(401, json={"error": "auth"})
    )
    result = await run_evaluation(config, transport=transport)
    assert result["status"] == "authentication_failed"
    assert len(calls) == 1
    await run_evaluation(config, resume=True, transport=httpx.MockTransport(response))
    assert len(calls) == 1
    records = [json.loads(x) for x in (tmp_path / "run/predictions.jsonl").read_text().splitlines()]
    assert len(records) == 2
    assert records[0]["transport_ok"] is False


@pytest.mark.asyncio
async def test_concurrency_and_secret_redaction(tmp_path, monkeypatch):
    config = config_at(tmp_path, [sample(identity=str(i)) for i in range(4)], concurrency=2)
    config.model.api_key_env = "EVAL_TEST_SECRET"
    secret = "secret-eval-bearer-value"
    monkeypatch.setenv("EVAL_TEST_SECRET", secret)
    active, maximum = 0, 0

    async def handle(req):
        nonlocal active, maximum
        assert req.headers["Authorization"] == f"Bearer {secret}"
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0.005)
        active -= 1
        result = response(req)
        raw = result.json()
        raw["echo"] = secret
        return httpx.Response(200, json=raw)

    await run_evaluation(config, transport=httpx.MockTransport(handle))
    assert maximum == 2
    for file in (tmp_path / "run").rglob("*"):
        if file.is_file():
            assert secret.encode() not in file.read_bytes()


@pytest.mark.asyncio
async def test_unknown_inflight_is_recorded_and_replayed(tmp_path):
    config = config_at(tmp_path)
    sent = asyncio.Event()

    async def interrupted_send(req):
        sent.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(
        run_evaluation(config, transport=httpx.MockTransport(interrupted_send))
    )
    await sent.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    calls = []
    await run_evaluation(
        config,
        resume=True,
        transport=httpx.MockTransport(lambda req: calls.append(req) or response(req)),
    )
    assert len(calls) == 1
    records = [json.loads(x) for x in (tmp_path / "run/attempts.jsonl").read_text().splitlines()]
    assert any(r["status"] == "outcome_unknown" for r in records)
    pred = json.loads((tmp_path / "run/predictions.jsonl").read_text())
    assert pred["replayed_unknown_attempts"] == 1


@pytest.mark.asyncio
async def test_protocol_failure_is_distinct_from_transport_success(tmp_path):
    config = config_at(tmp_path)
    result = await run_evaluation(
        config,
        transport=httpx.MockTransport(
            lambda req: httpx.Response(200, json={"model": "test", "answers": {}})
        ),
    )
    report = json.loads((tmp_path / "run" / result["report"] / "metrics.json").read_text())
    assert report["counts"]["protocol_failed"] == 1
    assert report["metrics"]["brier"]["value"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("status,headers", [(400, {}), (429, {"Retry-After": "120"})])
async def test_nonretryable_or_excessive_retry_after_is_terminal(tmp_path, status, headers):
    config = config_at(tmp_path, max_attempts=3, max_retry_delay=1)
    calls = []
    await run_evaluation(
        config,
        transport=httpx.MockTransport(
            lambda req: calls.append(req) or httpx.Response(status, headers=headers)
        ),
    )
    assert len(calls) == 1
    pred = json.loads((tmp_path / "run/predictions.jsonl").read_text())
    assert pred["http_status"] == status


@pytest.mark.asyncio
async def test_real_cancellation_recovers_unknown_send_and_keeps_completed_results(tmp_path):
    config = config_at(tmp_path, [sample(identity="one"), sample(identity="two")])
    sent = asyncio.Event()
    calls = []

    async def handle(req):
        calls.append(req)
        if len(calls) == 1:
            return response(req)
        sent.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(run_evaluation(config, transport=httpx.MockTransport(handle)))
    await sent.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    recovered = []
    await run_evaluation(
        config,
        resume=True,
        transport=httpx.MockTransport(lambda req: recovered.append(req) or response(req)),
    )
    assert len(recovered) == 1
    records = [json.loads(x) for x in (tmp_path / "run/attempts.jsonl").read_text().splitlines()]
    assert [r["status"] for r in records] == ["finished", "outcome_unknown", "finished"]
    predictions = [
        json.loads(x) for x in (tmp_path / "run/predictions.jsonl").read_text().splitlines()
    ]
    assert [p["replayed_unknown_attempts"] for p in predictions] == [0, 1]


@pytest.mark.asyncio
async def test_score_only_rejects_modified_sample_snapshot(tmp_path):
    config = config_at(tmp_path)
    await run_evaluation(config, transport=httpx.MockTransport(response))
    path = tmp_path / "run/samples.jsonl"
    changed = json.loads(path.read_text())
    changed["target"]["value"] = "A"
    write_jsonl(path, [changed])
    with pytest.raises(ValueError, match="snapshot"):
        await run_evaluation(config, score_only=True)


@pytest.mark.asyncio
async def test_unknown_usage_is_null_instead_of_zero(tmp_path):
    config = config_at(tmp_path)
    result = await run_evaluation(
        config,
        transport=httpx.MockTransport(lambda req: httpx.Response(500, json={"error": "no usage"})),
    )
    report = json.loads((tmp_path / "run" / result["report"] / "metrics.json").read_text())
    assert report["performance"]["usage"]["input_tokens"] is None
    assert report["performance"]["usage"]["missing_attempts"] == 1


@pytest.mark.asyncio
async def test_openrouter_endpoint_and_reported_cost(tmp_path):
    config = config_at(tmp_path)
    config.model = type(config.model).model_validate(
        {
            "base_url": "https://openrouter.ai",
            "endpoint_path": "/api/alpha/decisions",
            "model": "typesafe/jev-1.13",
            "api_key_env": None,
        }
    )

    def handle(req):
        assert str(req.url) == "https://openrouter.ai/api/alpha/decisions"
        raw = response(req).json()
        raw["usage"]["cost"] = 0.000012
        return httpx.Response(200, json=raw)

    result = await run_evaluation(config, transport=httpx.MockTransport(handle))
    report = json.loads((tmp_path / "run" / result["report"] / "metrics.json").read_text())
    assert report["performance"]["reported_cost_usd"] == 0.000012
    assert report["performance"]["price_per_1000_decisions_usd"] == 0.012


@pytest.mark.asyncio
async def test_permission_denial_distinct_and_retry_cost_includes_all_attempts(tmp_path):
    config = config_at(tmp_path)
    result = await run_evaluation(
        config,
        transport=httpx.MockTransport(
            lambda req: httpx.Response(403, json={"error": {"message": "Region unavailable"}})
        ),
    )
    assert result["status"] == "access_denied"
    config = config_at(tmp_path / "retry", max_attempts=2, backoff=0)
    calls = []

    def handle(req):
        calls.append(req)
        if len(calls) == 1:
            return httpx.Response(
                529, json={"usage": {"input_tokens": 2, "output_tokens": 0, "cost": 0.000001}}
            )
        raw = response(req).json()
        raw["usage"]["cost"] = 0.000002
        return httpx.Response(200, json=raw)

    await run_evaluation(config, transport=httpx.MockTransport(handle))
    prediction = json.loads((tmp_path / "retry/run/predictions.jsonl").read_text())
    assert prediction["cost_usd"] == pytest.approx(0.000003)


@pytest.mark.asyncio
async def test_resume_honors_remaining_retry_after_wait(tmp_path):
    config = config_at(tmp_path, max_attempts=2, backoff=0)
    calls = []
    returned = asyncio.Event()

    async def handle(req):
        calls.append(time.monotonic())
        if len(calls) == 1:
            returned.set()
            return httpx.Response(429, headers={"Retry-After": "0.2"})
        return response(req)

    transport = httpx.MockTransport(handle)
    task = asyncio.create_task(run_evaluation(config, transport=transport))
    await returned.wait()
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await run_evaluation(config, resume=True, transport=transport)
    assert len(calls) == 2
    assert calls[1] - calls[0] >= 0.19


@pytest.mark.asyncio
@pytest.mark.parametrize("unknown", [True, False])
@pytest.mark.parametrize("benchmark", ["generic", "jevbench"])
async def test_interrupted_attempts_remain_visible_to_offline_scoring(tmp_path, unknown, benchmark):
    items = [sample(identity=i) for i in ("one", "two", "unattempted")]
    config = config_at(tmp_path, items, max_attempts=2)
    config.benchmark = benchmark
    sent = asyncio.Event()
    calls = []

    async def handle(req):
        calls.append(req)
        if len(calls) == 1:
            return httpx.Response(
                200,
                json={
                    "model": "actual-model",
                    "answers": {
                        "decision": {
                            "type": "choice",
                            "choice": "B",
                            "confidence": 0.6,
                            "probabilities": {"A": 0.2, "B": 0.8},
                        }
                    },
                    "usage": {"input_tokens": 5, "output_tokens": 2},
                },
            )
        sent.set()
        if unknown:
            await asyncio.Event().wait()
        return httpx.Response(429, headers={"Retry-After": "10"})

    task = asyncio.create_task(run_evaluation(config, transport=httpx.MockTransport(handle)))
    await sent.wait()
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    before = (tmp_path / "run/predictions.jsonl").read_bytes()
    result = await run_evaluation(config, score_only=True)
    report = json.loads((tmp_path / "run" / result["report"] / "metrics.json").read_text())
    assert report["counts"]["attempted"] == 2
    assert report["counts"]["completed"] == 1
    assert report["counts"]["unattempted"] == 1
    assert report["counts"]["incomplete"] == 1
    assert report["counts"]["unknown_outcomes"] == int(unknown)
    assert report["counts"]["coverage"] == pytest.approx(2 / 3)
    assert report["partial"] is True
    assert report["metrics"]["accuracy"]["eligible"] == 2
    assert report["metrics"]["accuracy"]["value"] == 0.5
    assert report["performance"]["unknown_attempts"] == int(unknown)
    assert before == (tmp_path / "run/predictions.jsonl").read_bytes()
    assert len(calls) == 2
    if benchmark == "jevbench":
        assert report["upstream_summary"]["n_attempted"] == 1
        assert report["upstream_summary"]["accuracy"] == 1
        assert report["upstream_summary"]["complete"] is False
        assert report["upstream_summary_record_basis"] == "terminal_predictions"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind,field", [("choice", "confidence"), ("noul", "noul"), ("score", "score")]
)
async def test_extreme_native_numeric_response_is_terminal_and_never_replayed(
    tmp_path, kind, field
):
    config = config_at(tmp_path, [sample(kind)])
    calls = []

    def handle(req):
        calls.append(req)
        raw = response(req).json()
        raw["answers"]["decision"][field] = 10**400
        return httpx.Response(200, json=raw)

    transport = httpx.MockTransport(handle)
    result = await run_evaluation(config, transport=transport)
    report = json.loads((tmp_path / "run" / result["report"] / "metrics.json").read_text())
    assert report["counts"]["protocol_failed"] == 1
    await run_evaluation(config, resume=True, transport=transport)
    assert len(calls) == 1
    pred = json.loads((tmp_path / "run/predictions.jsonl").read_text())
    assert pred["raw_response"]["answers"]["decision"][field] == 10**400


@pytest.mark.asyncio
async def test_noul_distribution_provenance_is_declared(tmp_path):
    config = config_at(tmp_path, [sample("noul")])
    result = await run_evaluation(config, transport=httpx.MockTransport(response))
    pred = json.loads((tmp_path / "run/predictions.jsonl").read_text())
    assert pred["probability_source"] == "derived_from_native_noul"
    report = json.loads((tmp_path / "run" / result["report"] / "metrics.json").read_text())
    assert report["prediction_provenance"]["probability_sources"] == ["derived_from_native_noul"]


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["probabilities", "cost"])
async def test_extreme_probability_or_cost_does_not_abort_received_response(tmp_path, field):
    config = config_at(tmp_path)

    def handle(req):
        raw = response(req).json()
        if field == "probabilities":
            raw["answers"]["decision"]["probabilities"] = {"A": 10**400, "B": 0}
        else:
            raw["usage"]["cost"] = 10**400
        return httpx.Response(200, json=raw)

    result = await run_evaluation(config, transport=httpx.MockTransport(handle))
    report = json.loads((tmp_path / "run" / result["report"] / "metrics.json").read_text())
    assert report["counts"]["completed"] == 1
    assert report["counts"]["scoring_invalid"] == int(field == "probabilities")
    assert report["performance"]["reported_cost_usd"] is None
