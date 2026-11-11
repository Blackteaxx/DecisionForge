import copy
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from decisionforge.eval.models import EvalConfig


def test_distinct_typed_eval_and_single_worker():
    c = EvalConfig(data={"path": "unused"}, model={"adapter": "openjev_local", "path": "model"})
    assert c.model.model == "model" and c.model.precision == "bf16"
    for patch in ({"precision": "fp16"}, {"load_in_4bit": True}, {"m": 2}):
        with pytest.raises(ValidationError):
            EvalConfig(
                data={"path": "unused"},
                model={"adapter": "openjev_local", "path": "model", **patch},
            )
    with pytest.raises(ValidationError, match="concurrency"):
        EvalConfig(
            data={"path": "unused"},
            model={"adapter": "openjev_local", "path": "model"},
            execution={"concurrency": 2},
        )


def test_single_rationale_delegates_official_sampler_reader_preserves_raw():
    from decisionforge.eval.adapters.openjev_worker import predict_public

    public = {
        "state": [{"role": "user", "content": "x", "extra": [1]}],
        "questions": {"decision": {"type": "choice", "criteria": {"z": None, "a": {"text": "A"}}}},
    }
    before = copy.deepcopy(public)
    calls = []

    class Rationale:
        tok = SimpleNamespace(decode=lambda ids: "raw rationale", encode=lambda *a, **kw: [1])
        suffix = [9]

        def sample(self, model, items, m, max_new_tokens):
            assert m == 1 and max_new_tokens == 16
            assert items[0]["public_request"] == public
            assert "q" not in items[0] and "counts" not in items[0]
            calls.append("sample")
            return [[([1, 2], [3, 4], True)]]

        def forward(self, model, group, grad, k):
            assert group == [([1, 2], [3, 4], True)] and not grad and k == 2
            calls.append("forward")
            return SimpleNamespace(
                detach=lambda: SimpleNamespace(
                    cpu=lambda: SimpleNamespace(tolist=lambda: [[0.7, 0.3] + [0.0] * 8])
                )
            ), None

    raw = predict_public(object(), Rationale(), public, max_new_tokens=16)
    assert calls == ["sample", "forward"] and public == before
    assert raw["answers"]["decision"]["probabilities"] == {"z": 0.7, "a": 0.3}
    assert raw["openjev"]["rationale_ids"] == [3, 4] and raw["openjev"]["forced_marker"] is True
    assert raw["openjev"]["public_request"] == public and raw["openjev"]["appended_marker_ids"] == [
        9
    ]


@pytest.mark.parametrize(
    "values", [[0.2, 0.2], [-0.1, 1.1], [float("nan"), 0.1], [0.1], [True, 0.0]]
)
def test_malformed_probabilities_never_repaired(values):
    from decisionforge.training.backends.openjev_task import answer_from_distribution

    with pytest.raises(ValueError, match="probabilit|distribution"):
        answer_from_distribution({"type": "noul", "instructions": "flag"}, values)


def test_artifact_rejected_before_model_sdk_import(tmp_path):
    from decisionforge.eval.adapters.openjev_worker import OpenJevRuntime

    with pytest.raises(ValueError, match="OpenJev|openjev"):
        OpenJevRuntime({"path": str(tmp_path)})


def test_public_focus_rejects_private_fields_and_unsupported_types():
    from decisionforge.training.backends.openjev_task import public_item

    public = {"state": "x", "questions": {"decision": {"type": "noul", "instructions": "flag?"}}}
    assert public_item(public, "decision")["label_mapping"] == {"A": "false", "B": "true"}
    for request in (
        {**public, "q": [0.4, 0.6]},
        {**public, "questions": {"decision": {"type": "score", "criteria": ["a", "b"]}}},
    ):
        with pytest.raises(ValueError):
            public_item(request, "decision")


def test_budget_preserves_complete_public_input_or_fails():
    from decisionforge.training.backends.openjev_task import check_prompt_budget

    tok = SimpleNamespace(encode=lambda text, **kw: list(range(len(text))))
    with pytest.raises(ValueError, match="truncation"):
        check_prompt_budget(tok, "all public input", 18, 4, [1, 2])
    assert check_prompt_budget(tok, "all public input", 32, 4, [1, 2]) == 16


@pytest.mark.asyncio
async def test_eval_identity_resume_and_offline_score_only(tmp_path):
    import json

    from test_eval_local import fixture_worker
    from test_training_openjev import artifact_at

    from decisionforge.eval.adapters.openjev import OpenJevAdapter
    from decisionforge.eval.data import write_jsonl
    from decisionforge.eval.runner import run_evaluation
    from tests.eval_fixtures import sample

    model = artifact_at(tmp_path / "model")
    path = tmp_path / "samples.jsonl"
    write_jsonl(path, [sample("choice", "soft", "one")])
    c = EvalConfig(
        data={"path": str(path)},
        model={"adapter": "openjev_local", "path": str(model)},
        output=str(tmp_path / "eval"),
    )

    async def factory(model, execution, transport):
        return await OpenJevAdapter(model, execution, worker_target=fixture_worker).start()

    result = await run_evaluation(c, adapter_factory=factory)
    assert result["status"] == "completed"
    report = json.loads((tmp_path / "eval" / result["report"] / "metrics.json").read_text())
    assert report["prediction_provenance"]["actual_models"] == [str(model)]
    assert report["prediction_provenance"]["unresolved_model_predictions"] == 0
    manifest = json.loads((tmp_path / "eval/manifest.json").read_text())
    assert manifest["inference"]["openjev"]["artifact_format"] == "openjev_rlcd"
    marker = model / "openjev_artifact.json"
    data = json.loads(marker.read_text())
    data["extra_identity_change"] = True
    marker.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Resume incompatible"):
        await run_evaluation(c, resume=True, adapter_factory=factory)
    (model / "stages/reinforce/weights.pt").unlink()
    assert (await run_evaluation(c, score_only=True))["status"] == "scored"


def test_eval_static_preflight_rejects_score_and_large_choice(tmp_path):
    from test_training_openjev import artifact_at

    from decisionforge.eval.data import write_jsonl
    from decisionforge.eval.runner import preflight
    from tests.eval_fixtures import sample

    model = artifact_at(tmp_path / "model")
    path = tmp_path / "data.jsonl"
    write_jsonl(path, [sample("score", "soft", "one")])
    config = EvalConfig(
        data={"path": str(path)}, model={"adapter": "openjev_local", "path": str(model)}
    )
    with pytest.raises(ValueError, match="score"):
        preflight(config)


def test_openjev_static_imports_do_not_load_optional_sdks():
    import subprocess
    import sys

    script = "import sys; import decisionforge.training.backends.openjev; import decisionforge.training.backends.openjev_task; import decisionforge.training.backends.openjev_stage; import decisionforge.eval.adapters.openjev_worker; assert not ({'torch','transformers','datasets','openjev_rlcd'} & set(sys.modules))"
    subprocess.run([sys.executable, "-c", script], check=True)


def test_malformed_reader_output_preserves_raw_distribution_for_protocol_journal():
    from decisionforge.eval.adapter import parse_answer
    from decisionforge.eval.adapters.openjev_worker import predict_public
    from decisionforge.eval.models import Question

    values = [[0.2, 0.2] + [0.0] * 8]
    rat = SimpleNamespace(
        sample=lambda *args, **kwargs: [[([1], [2], False)]],
        forward=lambda *args, **kwargs: (
            SimpleNamespace(
                detach=lambda: SimpleNamespace(cpu=lambda: SimpleNamespace(tolist=lambda: values))
            ),
            None,
        ),
        tok=SimpleNamespace(decode=lambda ids: "raw"),
        suffix=[3],
    )
    public = {"state": "x", "questions": {"decision": {"type": "noul", "instructions": "flag?"}}}
    response = predict_public(object(), rat, public)
    assert response["openjev"]["raw_label_distributions"] == values
    assert response["openjev"]["validation_error"]
    with pytest.raises(ValueError):
        parse_answer(response, Question(type="noul", instructions="flag?"))


def test_official_sampler_cannot_silently_remove_public_padding_tokens():
    from decisionforge.training.backends.openjev_task import check_prompt_budget

    tok = SimpleNamespace(pad_token_id=7, encode=lambda *args, **kw: [1, 7, 3])
    with pytest.raises(ValueError, match="padding|removed"):
        check_prompt_budget(tok, "source contains special padding token", 128, 4, [5])


def openjev_runtime_fixture(tmp_path, *, samples=2):
    from test_training_openjev import artifact_at

    from decisionforge.eval.data import write_jsonl
    from decisionforge.training.backends.openjev import UPSTREAM_REVISION, artifact_identity
    from decisionforge.training.backends.openjev_task import PUBLIC_PROTOCOL, SAMPLING
    from tests.eval_fixtures import sample

    model = artifact_at(tmp_path / "model")
    source = tmp_path / "samples.jsonl"
    write_jsonl(source, [sample("choice", "soft", str(i)) for i in range(samples)])
    config = EvalConfig(
        data={"path": str(source)},
        model={"adapter": "openjev_local", "path": str(model)},
        output=str(tmp_path / "eval"),
    )
    # The runtime fixture matches OpenJevRuntime's ready metadata, without an SDK/GPU.
    metadata = {
        "model": str(model),
        "model_identity_source": "loaded_openjev_stage2_weights",
        "upstream_repo": "https://github.com/ZimmyGao/openjev-rlcd",
        "upstream_revision": UPSTREAM_REVISION,
        "source_verified": True,
        "train_sha256": "trainer-source",
        "source_sha256": {"train.py": "trainer-source", "estimators.py": "loss-source"},
        **artifact_identity(model),
        "sampling": {**SAMPLING, "seed": 3407, "max_new_tokens": 16},
        "generation_config": {"eos_token_id": [151645, 151643], "pad_token_id": 151643},
        "public_protocol": PUBLIC_PROTOCOL,
        "dependencies": {"torch": "2.11.0", "transformers": "5.17.0"},
        "runtime": {
            "device": "cuda:0",
            "gpu": "fixture-BF16-GPU",
            "precision": "bfloat16_parameters_and_autocast",
            "logits_precision": "float32",
            "cuda": "13.0",
            "load_seconds": 1.0,
            "effective_max_seq_length": 2048,
            "attention_implementation": "sdpa",
        },
    }
    return config, metadata


def openjev_runtime_factory(metadata, calls, *, interrupt=False, instances=None):
    import asyncio

    from test_eval_local import fixture_worker

    from decisionforge.eval.adapters.openjev import OpenJevAdapter

    class Adapter(OpenJevAdapter):
        async def send(self, payload, question):
            calls.append(copy.deepcopy(payload))
            if interrupt and len(calls) == 2:
                raise asyncio.CancelledError()
            return await super().send(payload, question)

    async def factory(model, execution, transport):
        adapter = await Adapter(model, execution, worker_target=fixture_worker).start()
        adapter.metadata = copy.deepcopy(metadata)
        if instances is not None:
            instances.append(adapter)
        return adapter

    return factory


@pytest.mark.asyncio
async def test_openjev_loaded_metadata_persisted_and_finished_paths_need_no_sdk(tmp_path):
    import json
    import sys
    from pathlib import Path

    from decisionforge.eval.runner import run_evaluation

    config, metadata = openjev_runtime_fixture(tmp_path, samples=1)
    await run_evaluation(config, adapter_factory=openjev_runtime_factory(metadata, []))
    manifest_path = Path(config.output) / "manifest.json"
    assert json.loads(manifest_path.read_text())["local_model"] == metadata

    async def no_worker(*args, **kwargs):
        pytest.fail("Completed resume or score-only started a model/SDK worker")

    sdks_before = {
        name for name in ("torch", "transformers", "openjev_rlcd") if name in sys.modules
    }
    assert (await run_evaluation(config, resume=True, adapter_factory=no_worker))[
        "status"
    ] == "completed"
    Path(config.data.path).unlink()
    (Path(config.model.path) / "stages/reinforce/weights.pt").unlink()
    assert (await run_evaluation(config, score_only=True, adapter_factory=no_worker))[
        "status"
    ] == "scored"
    assert {
        name for name in ("torch", "transformers", "openjev_rlcd") if name in sys.modules
    } == sdks_before
    assert json.loads(manifest_path.read_text())["local_model"] == metadata


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path,value",
    [
        (
            ("checkpoint", "files", "stages/reinforce/weights.pt", "sha256"),
            "different-loaded-weights",
        ),
        (("artifact_format",), "another-format"),
        (("sampling", "max_new_tokens"), 17),
        (("sampling", "seed"), 3408),
        (("generation_config", "eos_token_id"), [151645]),
        (("public_protocol", "eval_cli"), "different-public-request-protocol"),
        (("dependencies", "torch"), "different-torch"),
        (("upstream_revision",), "different-upstream"),
        (("source_sha256", "estimators.py"), "different-loss-source"),
        (("model_identity_source",), "different-loaded-source"),
        (("label_protocol",), {"protocol_hash": "different-protocol"}),
        (("trained_pool",), {"labels": ["different-trained-label"]}),
        (("readout_policy",), {"version": "changed-extension-policy"}),
        (("runtime", "device"), "cuda:1"),
        (("runtime", "gpu"), "different-GPU"),
        (("runtime", "precision"), "float32"),
        (("runtime", "logits_precision"), "bfloat16"),
        (("runtime", "cuda"), "different-cuda"),
        (("runtime", "effective_max_seq_length"), 1024),
        (("runtime", "attention_implementation"), "eager"),
    ],
)
async def test_openjev_partial_resume_rejects_changed_loaded_identity_before_prediction(
    tmp_path, path, value
):
    import asyncio
    import json
    from pathlib import Path

    from decisionforge.eval.runner import run_evaluation

    config, metadata = openjev_runtime_fixture(tmp_path)
    with pytest.raises(asyncio.CancelledError):
        await run_evaluation(
            config, adapter_factory=openjev_runtime_factory(metadata, [], interrupt=True)
        )
    predictions_path = Path(config.output) / "predictions.jsonl"
    first = predictions_path.read_bytes()
    assert len(first.splitlines()) == 1
    changed = copy.deepcopy(metadata)
    target = changed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    calls, instances = [], []
    with pytest.raises(ValueError, match="Resume incompatible.*loaded"):
        await run_evaluation(
            config,
            resume=True,
            adapter_factory=openjev_runtime_factory(changed, calls, instances=instances),
        )
    assert calls == []
    assert predictions_path.read_bytes() == first
    assert instances[0].process is None and instances[0].connection is None
    assert (
        json.loads((Path(config.output) / "manifest.json").read_text())["local_model"] == metadata
    )


@pytest.mark.asyncio
async def test_openjev_partial_resume_ignores_only_load_timing_and_keeps_first_prediction(tmp_path):
    import asyncio
    import json
    from pathlib import Path

    from decisionforge.eval.runner import run_evaluation

    config, metadata = openjev_runtime_fixture(tmp_path)
    with pytest.raises(asyncio.CancelledError):
        await run_evaluation(
            config, adapter_factory=openjev_runtime_factory(metadata, [], interrupt=True)
        )
    path = Path(config.output) / "predictions.jsonl"
    first = json.loads(path.read_text())
    reloaded = copy.deepcopy(metadata)
    reloaded["runtime"]["load_seconds"] = 99.0
    calls = []
    assert (
        await run_evaluation(
            config, resume=True, adapter_factory=openjev_runtime_factory(reloaded, calls)
        )
    )["status"] == "completed"
    predictions = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(calls) == 1 and len(predictions) == 2 and predictions[0] == first
    assert (
        json.loads((Path(config.output) / "manifest.json").read_text())["local_model"] == reloaded
    )
