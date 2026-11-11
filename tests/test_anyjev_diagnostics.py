"""Large catalog evidence remains exact while journal copies share immutable blobs."""

import copy
import json
from contextlib import closing
from types import SimpleNamespace

import pytest

from decisionforge.eval.models import Request
from decisionforge.training_free.anyjev.processing import response_from_decision


def question(size):
    return Request.model_validate(
        {
            "state": "test",
            "questions": {
                "decision": {"type": "choice", "criteria": {str(i): None for i in range(size)}}
            },
        }
    ).questions["decision"]


def decision(size):
    return SimpleNamespace(
        probs=[1.0 / size] * size,
        level="L0",
        confidence=1.0 / size,
        argmax='{"description":null,"key":"0"}',
        diagnostics={
            "permutations": size,
            "prior_method": "content_free",
            "prior_strength": 1.0,
            "matrix": [[i / size, -0.0, "NaN", "Infinity"] for i in range(size)],
        },
    )


def test_large_diagnostics_keep_every_value_and_share_blob_when_copied():
    from decisionforge.training_free.anyjev.diagnostics import decode_diagnostics

    source = decision(151)
    response = response_from_decision(question(151), source)
    encoded = response["anyjev"]["diagnostics"]
    assert encoded["permutations"] == 151
    assert encoded["prior_method"] == "content_free"
    assert "matrix" not in encoded
    assert decode_diagnostics(encoded) == source.diagnostics
    saved = json.loads(json.dumps(response, allow_nan=False))
    assert decode_diagnostics(saved["anyjev"]["diagnostics"]) == source.diagnostics
    cloned = copy.deepcopy(encoded)
    assert cloned["compressed"]["payload"] is encoded["compressed"]["payload"]
    assert response["answers"]["decision"]["choice"] == "0"


def test_small_diagnostics_remain_verbatim():
    source = decision(26)
    response = response_from_decision(question(26), source)
    assert response["anyjev"]["diagnostics"] == source.diagnostics
    assert "compressed" not in response["anyjev"]["diagnostics"]


def test_invalid_large_output_retains_nonfinite_raw_evidence():
    from decisionforge.training_free.anyjev.diagnostics import decode_diagnostics

    source = decision(151)
    source.probs[0] = float("nan")
    source.diagnostics["bad"] = float("inf")
    response = response_from_decision(question(151), source)
    assert "answers" not in response
    assert response["anyjev"]["probs"][0] == "NaN"
    assert decode_diagnostics(response["anyjev"]["diagnostics"])["bad"] == "Infinity"
    json.dumps(response, allow_nan=False)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 9),
        ("codec", "unknown"),
        ("uncompressed_sha256", "0" * 64),
        ("uncompressed_bytes", 1),
        ("payload", "invalid!"),
    ],
)
def test_diagnostics_reject_corrupt_codec_or_payload(field, value):
    from decisionforge.training_free.anyjev.diagnostics import (
        decode_diagnostics,
        encode_diagnostics,
    )

    encoded = encode_diagnostics(decision(151).diagnostics)
    encoded["compressed"][field] = value
    with pytest.raises(ValueError):
        decode_diagnostics(encoded)


def test_compressed_evidence_survives_read_only_journal_reload(tmp_path):
    from decisionforge.eval.store import EvalStore
    from decisionforge.training_free.anyjev.diagnostics import decode_diagnostics

    response = response_from_decision(question(151), decision(151))
    with closing(EvalStore(tmp_path)) as store:
        store.put("predictions", {"id": "sample", "raw_response": response})
    before = (tmp_path / ".eval/events.jsonl").read_bytes()
    with closing(EvalStore(tmp_path, read_only=True)) as store:
        raw = store.records("predictions")[0]["raw_response"]["anyjev"]["diagnostics"]
        assert decode_diagnostics(raw) == decision(151).diagnostics
    assert (tmp_path / ".eval/events.jsonl").read_bytes() == before


def compressed_fixture_worker(connection, options):
    connection.send(
        {
            "kind": "ready",
            "metadata": {
                "model": options["model"],
                "backbone": {"name": options["model"], "revision": options["revision"]},
                "runtime": {"precision": "torch.bfloat16"},
            },
        }
    )
    while True:
        command = connection.recv()
        if command["kind"] == "close":
            break
        public = Request.model_validate(command["request"])
        q = public.questions["decision"]
        response = response_from_decision(q, decision(len(q.criteria)))
        connection.send({"kind": "prediction", "response": response})
    connection.close()


@pytest.mark.asyncio
async def test_large_compressed_prediction_supports_public_score_only(tmp_path):
    from decisionforge.eval.adapters.anyjev import AnyJevAdapter
    from decisionforge.eval.runner import run_evaluation
    from decisionforge.training_free.anyjev.diagnostics import decode_diagnostics
    from tests.eval_fixtures import sample
    from tests.test_eval_anyjev import config_at

    item = sample()
    item["request"]["questions"]["decision"]["criteria"] = {str(i): None for i in range(151)}
    item["target"] = None
    config = config_at(tmp_path, [item])

    async def factory(model, execution, transport=None):
        return await AnyJevAdapter(
            model, execution, worker_target=compressed_fixture_worker
        ).start()

    result = await run_evaluation(config, adapter_factory=factory)
    predictions = (tmp_path / "run/predictions.jsonl").read_text()
    saved = json.loads(predictions)
    assert (
        decode_diagnostics(saved["raw_response"]["anyjev"]["diagnostics"])
        == decision(151).diagnostics
    )
    journal = (tmp_path / "run/.eval/events.jsonl").read_bytes()
    (tmp_path / "samples.jsonl").unlink()

    async def reject(*args, **kwargs):
        pytest.fail("score-only attempted model startup")

    rescored = await run_evaluation(config, score_only=True, adapter_factory=reject)
    assert rescored["status"] == "scored"
    assert rescored["report"] != result["report"]
    metrics = json.loads((tmp_path / "run" / rescored["report"] / "metrics.json").read_text())
    assert metrics["counts"]["strict_valid"] == 1
    assert (tmp_path / "run/.eval/events.jsonl").read_bytes() == journal
