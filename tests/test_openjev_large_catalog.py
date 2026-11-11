"""Large-catalog regressions: original keys and complete q survive the bridge."""

import copy
import json
from types import SimpleNamespace

import pytest


def public(k):
    return {
        "state": [{"role": "user", "content": "complete state"}],
        "questions": {
            "decision": {
                "type": "choice",
                "instructions": "select the intent",
                "criteria": {f"intent_{i}": {"text": f"intent {i}"} for i in range(k)},
            },
            "sibling": {"type": "noul", "instructions": "retain me"},
        },
    }


@pytest.mark.parametrize("k", [77, 151])
def test_complete_large_catalog_mapping_and_registered_pool(k):
    from decisionforge.training.backends.openjev_task import message, public_item, register_task

    labels = [f"L{i}" for i in range(k)]
    request = public(k)
    before = copy.deepcopy(request)
    item = public_item(request, "decision", labels=labels)
    assert len(item["label_mapping"]) == k
    assert item["label_mapping"][f"L{k - 1}"] == f"intent_{k - 1}"
    assert item["public_request"] == before and request == before
    assert "retain me" in message(item) and f"intent_{k - 1}" in message(item)
    tasks = {}
    register_task(tasks, {"train": [item]}, labels=labels)
    assert tasks["dforge_v3"]["labels"] == labels
    assert tasks["dforge_v3"]["k"](item) == k


def test_unvalidated_large_catalog_is_only_static_and_cannot_render():
    from decisionforge.training.backends.openjev_task import message, public_item

    item = public_item(public(77), "decision")
    assert item["label_mapping"] is None
    assert item["label_validation"] == "deferred_to_tokenizer"
    with pytest.raises(ValueError, match="label|token"):
        message(item)


def test_insufficient_pool_rejected_without_dropping_original_options():
    from decisionforge.training.backends.openjev_task import public_item

    with pytest.raises(ValueError, match="label|pool"):
        public_item(public(77), "decision", labels=list("ABCDEFGHIJ"))
    with pytest.raises(ValueError, match="label|unique"):
        public_item(public(11), "decision", labels=["A"] * 11)


@pytest.mark.parametrize("k", [77, 151])
def test_official_full_pool_reader_uses_effective_k_mask(k):
    from decisionforge.eval.adapters.openjev_worker import predict_public

    labels = [f"L{i}" for i in range(k + 3)]
    values = [[0.0] * (k + 3)]
    values[0][k - 1] = 1.0
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
    response = predict_public(object(), rat, public(k), labels=labels)
    assert "validation_error" not in response["openjev"]
    answer = response["answers"]["decision"]
    assert answer["choice"] == f"intent_{k - 1}"
    assert len(answer["probabilities"]) == k
    assert response["openjev"]["raw_label_distributions"] == values
    values[0][-1] = 0.1
    assert "validation_error" in predict_public(object(), rat, public(k), labels=labels)["openjev"]


def test_training_static_preflight_allows_full_151_catalog(tmp_path):
    from test_training_openjev import cn_config

    from decisionforge.training.backend import preflight
    from decisionforge.training.data import prepare_data

    config = cn_config(tmp_path)
    data = prepare_data(config)
    record = data.records[0]
    question = record.request.questions["decision"]
    question.criteria = public(151)["questions"]["decision"]["criteria"]
    preflight(config, data=data)


def protocol_for(labels):
    from decisionforge.label_tokens import label_protocol_hash

    protocol = {
        "schema_version": 1,
        "encoding_style": "space_prefixed",
        "labels": labels,
        "token_ids": list(range(len(labels))),
        "encodings": [" " + x for x in labels],
        "legacy_count": 10,
    }
    protocol["protocol_hash"] = label_protocol_hash(protocol)
    return protocol


def upgrade_artifact(root):
    marker = root / "openjev_artifact.json"
    data = json.loads(marker.read_text())
    labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [f"X{i}" for i in range(125)]
    data.update(
        schema_version="decisionforge.openjev.artifact.v2",
        mapping_version="decisionforge.openjev.mapping.v2",
        labels=labels,
        label_token_ids=list(range(151)),
        label_protocol=protocol_for(labels),
        training_max_options=151,
    )
    marker.write_text(json.dumps(data))
    return marker, data


def test_v1_and_dynamic_v2_artifacts_are_self_contained_and_validated(tmp_path):
    from test_training_openjev import artifact_at

    from decisionforge.training.backends.openjev import validate_artifact

    legacy = artifact_at(tmp_path / "legacy")
    assert validate_artifact(legacy) == legacy
    root = artifact_at(tmp_path / "large")
    upgrade_artifact(root)
    assert validate_artifact(root) == root


@pytest.mark.parametrize("mutation", ["collision", "count", "hash", "encoding", "train_count"])
def test_dynamic_artifact_rejects_broken_protocol(tmp_path, mutation):
    from test_training_openjev import artifact_at

    from decisionforge.training.backends.openjev import validate_artifact

    marker, data = upgrade_artifact(artifact_at(tmp_path / "model"))
    if mutation == "collision":
        data["label_token_ids"][-1] = data["label_token_ids"][0]
    elif mutation == "count":
        data["label_protocol"]["labels"].pop()
    elif mutation == "hash":
        data["label_protocol"]["protocol_hash"] = "stale"
    elif mutation == "encoding":
        data["label_protocol"]["encoding_style"] = "bare_then_space"
    else:
        data["training_max_options"] = 152
    marker.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="OpenJev|label|protocol"):
        validate_artifact(marker.parent)


def test_extended_readout_requires_unchanged_trained_prefix():
    from decisionforge.training.backends.openjev_task import inference_label_protocol

    class Tok:
        def encode(self, text, **kw):
            return [sum((i + 1) * ord(c) for i, c in enumerate(text))]

    tok = Tok()
    # Legacy IDs come from the actual tokenizer, with A-J strictly preserved.
    manifest = {
        "schema_version": "decisionforge.openjev.artifact.v1",
        "labels": list("ABCDEFGHIJ"),
        "label_token_ids": [tok.encode(" " + c)[0] for c in "ABCDEFGHIJ"],
    }
    protocol = inference_label_protocol(tok, manifest, 77)
    assert len(protocol["labels"]) == 77 and protocol["labels"][:10] == list("ABCDEFGHIJ")
    manifest["label_token_ids"][0] = -1
    with pytest.raises(ValueError, match="identity|prefix|token"):
        inference_label_protocol(tok, manifest, 77)


@pytest.mark.parametrize("k", [77, 151])
def test_complete_soft_v3_mapping_survives_real_data_parser(tmp_path, k):
    from test_training_data import make_config
    from test_training_openjev import cn_example

    from decisionforge.training.backends.openjev_task import map_roles
    from decisionforge.training.data import prepare_data

    rows = []
    for i, split in enumerate(("train", "validation", "test"), 1):
        row = cn_example(i, split)
        row["request"] = public(k)
        row["request"]["state"].append({"record": i})
        q = {f"intent_{j}": 0.7 if j == 0 else (0.3 if j == k - 1 else 0.0) for j in range(k)}
        row["supervision"]["answers"] = {
            "decision": {
                "type": "choice",
                "choice": "intent_0",
                "confidence": 0.7,
                "probabilities": q,
            },
            "sibling": {"type": "noul", "noul": 0.3},
        }
        rows.append(row)
    data = prepare_data(make_config(tmp_path, rows=rows, backend="openjev"))
    splits, audit = map_roles(data, labels=[f"L{i}" for i in range(k)])
    for role in ("train", "dev", "test"):
        decision, sibling = splits[role]
        assert len(decision["q"]) == len(decision["counts"]) == k
        assert decision["q"][0] == 0.7 and decision["q"][-1] == 0.3
        assert decision["option_keys"][-1] == f"intent_{k - 1}"
        assert len(decision["public_request"]["questions"]["decision"]["criteria"]) == k
        assert sibling["q"] == [0.7, 0.3]
    assert audit["train"]["mapping"][0]["source_target"][f"intent_{k - 1}"] == 0.3


@pytest.mark.parametrize("name", ["prepare_cn.py", "prepare_eval.py"])
def test_explicit_projection_keeps_full_large_catalog(tmp_path, name):
    import runpy
    from pathlib import Path

    from test_training_openjev import cn_example

    from tests.eval_fixtures import sample

    prepare = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "examples/training/openjev" / name)
    )["prepare"]
    if name == "prepare_cn.py":
        row = cn_example()
        row["request"]["questions"]["decision"]["criteria"] = public(151)["questions"]["decision"][
            "criteria"
        ]
        row["supervision"]["answers"]["decision"] = {
            "type": "choice",
            "choice": "intent_0",
            "confidence": 1.0,
            "probabilities": {f"intent_{i}": 1.0 if i == 0 else 0.0 for i in range(151)},
        }
    else:
        row = sample("choice", "soft", "large")
        row["request"]["questions"]["decision"]["criteria"] = public(151)["questions"]["decision"][
            "criteria"
        ]
        row["target"]["probabilities"] = {f"intent_{i}": 1.0 if i == 0 else 0.0 for i in range(151)}
    source, output = tmp_path / "source.jsonl", tmp_path / "out.jsonl"
    source.write_text(json.dumps(row) + "\n")
    prepare(source, output)
    converted = json.loads(output.read_text())
    assert len(converted["request"]["questions"]["decision"]["criteria"]) == 151
    assert converted["request"] == row["request"]


def test_multiletter_prompt_requires_complete_displayed_label_but_legacy_stays_exact():
    from decisionforge.training.backends.openjev_task import message, public_item

    item = public_item(public(77), "decision", labels=["A", "B"] + [f"X{i}" for i in range(75)])
    assert "complete displayed label" in message(item)
    legacy = public_item(public(2), "decision")
    assert "Answer letter to original option key" in message(legacy)
    assert "where X is the letter of the correct option." in message(legacy)


@pytest.mark.parametrize("k", [10, 77, 151])
def test_allocated_protocol_serializes_to_self_validating_v2_artifact(tmp_path, k):
    from test_training_openjev import artifact_at

    from decisionforge.label_tokens import allocate_labels
    from decisionforge.training.backends.openjev import validate_artifact
    from decisionforge.training.backends.openjev_task import inference_label_protocol

    class Tok:
        def encode(self, text, **kw):
            value = 0
            for char in text.strip():
                value = value * 26 + ord(char) - 64
            return [100 + value]

    tok = Tok()
    protocol = allocate_labels(tok, k, encoding_style="space_prefixed", legacy_count=10)
    root = artifact_at(tmp_path / "model")
    marker = root / "openjev_artifact.json"
    data = json.loads(marker.read_text())
    data.update(
        schema_version="decisionforge.openjev.artifact.v2",
        mapping_version="decisionforge.openjev.mapping.v2",
        labels=protocol["labels"],
        label_token_ids=protocol["token_ids"],
        label_protocol=protocol,
        training_max_options=k,
    )
    marker.write_text(json.dumps(data))
    assert validate_artifact(root) == root
    assert inference_label_protocol(tok, json.loads(marker.read_text()), k) == protocol


@pytest.mark.asyncio
async def test_full_151_eval_preflight_reports_deferred_tokenizer_checks(tmp_path):
    from test_training_openjev import artifact_at

    from decisionforge.eval.models import EvalConfig
    from decisionforge.eval.runner import run_evaluation
    from tests.eval_fixtures import sample

    row = sample("choice", "hard", "full")
    row["request"]["questions"]["decision"]["criteria"] = public(151)["questions"]["decision"][
        "criteria"
    ]
    row["target"]["value"] = "intent_0"
    source = tmp_path / "data.jsonl"
    source.write_text(json.dumps(row) + "\n")
    root = artifact_at(tmp_path / "legacy")
    config = EvalConfig(
        data={"path": str(source)}, model={"adapter": "openjev_local", "path": str(root)}
    )
    result = await run_evaluation(config, validate_only=True)
    assert result["samples"] == 1
    assert result["openjev"]["label_candidate_limit"] == 18278
    assert "deferred" in result["openjev"]["runtime_checks"]


def test_training_preflight_does_not_claim_tokenizer_pool_validation(tmp_path):
    from test_training_openjev import cn_config

    from decisionforge.training.runner import run_training

    result = run_training(cn_config(tmp_path), validate_only=True)
    assert result["openjev"]["label_candidate_limit"] == 18278
    assert "deferred" in result["openjev"]["runtime_checks"]


def test_request_extension_preserves_legacy_seed_and_records_only_inference_claim():
    from contextlib import nullcontext

    from decisionforge.eval.adapters.openjev_worker import OpenJevRuntime
    from decisionforge.label_tokens import allocate_labels
    from decisionforge.util import digest

    class Tok:
        pad_token_id = 0

        def encode(self, text, **kw):
            if not text.startswith(" "):
                return [1]
            value = 0
            for char in text.strip():
                value = value * 26 + ord(char) - 64
            return [100 + value]

        def decode(self, ids):
            return "raw"

    tok, tasks = Tok(), {}

    class Reader:
        suffix = [3]

        def __init__(self, tokenizer, name):
            self.tok = tokenizer
            self.labels = list(tasks[name]["labels"])

        def sample(self, model, items, m, max_new_tokens):
            return [[([1], [2], False)]]

        def forward(self, model, group, grad, k):
            values = [[1.0] + [0.0] * (len(self.labels) - 1)]
            return SimpleNamespace(
                detach=lambda: SimpleNamespace(cpu=lambda: SimpleNamespace(tolist=lambda: values))
            ), None

    trained = allocate_labels(tok, 10, encoding_style="space_prefixed", legacy_count=10)
    tasks["dforge_v3"] = {"labels": trained["labels"]}
    runtime = OpenJevRuntime.__new__(OpenJevRuntime)
    runtime.manifest = {
        "schema_version": "decisionforge.openjev.artifact.v1",
        "labels": trained["labels"],
        "label_token_ids": trained["token_ids"],
    }
    runtime.trained_protocol = trained
    runtime.rat = Reader(tok, "dforge_v3")
    runtime.readers = {10: runtime.rat}
    runtime.upstream = SimpleNamespace(
        TASKS=tasks, Rationales=Reader, prompt=lambda *a: "complete public prompt"
    )
    seeds = []
    runtime.torch = SimpleNamespace(manual_seed=seeds.append, inference_mode=nullcontext)
    runtime.seed, runtime.limit, runtime.max_new_tokens = 3407, 32768, 4
    runtime.model, runtime.metadata = object(), {"model": "legacy-model"}
    small = public(2)
    original = runtime.predict(small)
    extended = runtime.predict(public(151))
    repeated = runtime.predict(small)
    expected_seed = int(digest({"public": small, "focus": "decision", "seed": 3407})[:8], 16)
    assert seeds[0] == seeds[2] == expected_seed
    assert original == repeated
    assert original["openjev"]["inference_only_label_extension"] is False
    assert len(extended["answers"]["decision"]["probabilities"]) == 151
    assert extended["openjev"]["inference_only_label_extension"] is True
    assert extended["openjev"]["trained_pool_size"] == 10
    assert extended["openjev"]["training_max_options"] is None
    assert len(extended["openjev"]["label_protocol"]["labels"]) == 151


def test_legacy_artifact_cannot_claim_extended_protocol_as_training_pool(tmp_path):
    from test_training_openjev import artifact_at

    from decisionforge.training.backends.openjev import validate_artifact

    root = artifact_at(tmp_path / "model")
    marker = root / "openjev_artifact.json"
    manifest = json.loads(marker.read_text())
    manifest["label_protocol"] = protocol_for(
        list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [f"X{i}" for i in range(51)]
    )
    marker.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="OpenJev|protocol"):
        validate_artifact(root)


@pytest.mark.parametrize("mutation", ["v1_protocol", "v2_labels", "v2_token_ids", "unknown_schema"])
def test_direct_readout_helper_rejects_conflicting_manifest_pool(mutation):
    from decisionforge.label_tokens import allocate_labels
    from decisionforge.training.backends.openjev_task import inference_label_protocol

    class Tok:
        def encode(self, text, **kw):
            value = 0
            for char in text.strip():
                value = value * 26 + ord(char) - 64
            return [100 + value]

    tok = Tok()
    trained = allocate_labels(tok, 10, encoding_style="space_prefixed", legacy_count=10)
    manifest = {
        "schema_version": "decisionforge.openjev.artifact.v2",
        "labels": trained["labels"].copy(),
        "label_token_ids": trained["token_ids"].copy(),
        "label_protocol": trained,
    }
    if mutation == "v1_protocol":
        manifest["schema_version"] = "decisionforge.openjev.artifact.v1"
        manifest["label_protocol"] = allocate_labels(
            tok, 77, encoding_style="space_prefixed", legacy_count=10
        )
    elif mutation == "v2_labels":
        manifest["labels"][-1] = "foreign"
    elif mutation == "v2_token_ids":
        manifest["label_token_ids"][-1] = 999999
    else:
        manifest["schema_version"] = "unknown"
    with pytest.raises(ValueError, match="OpenJev|protocol|identity|schema"):
        inference_label_protocol(tok, manifest, 2)


def test_large_label_prompt_never_rewrites_nested_public_content():
    from decisionforge.training.backends.openjev_task import message, public_item

    phrase = "where X is the letter of the correct option. Answer letter to original option key"
    request = public(77)
    request["state"] = {"nested": [{"text": phrase}]}
    request["questions"]["decision"]["instructions"] = {"nested": [phrase]}
    request["questions"]["decision"]["criteria"].pop("intent_76")
    request["questions"]["decision"]["criteria"][phrase] = {"nested": [phrase]}
    request["questions"]["sibling"]["instructions"] = phrase
    before = copy.deepcopy(request)
    item = public_item(request, "decision", labels=[f"L{i}" for i in range(77)])
    rendered = message(item)
    mapping = rendered.split("\nAnswer label to original option key: ", 1)[1].split("\n\nThink", 1)[
        0
    ]
    assert json.loads(mapping) == item["label_mapping"]
    serialized = rendered.removeprefix("Public request: ").split("\n\nFocus question key:", 1)[0]
    assert json.loads(serialized) == before
    assert request == before
    assert "Answer label to original option key:" in rendered
    assert rendered.endswith("where X is the complete displayed label of the correct option.")


def test_legacy_prompt_text_is_preserved_exactly():
    from decisionforge.training.backends.openjev_task import message, public_item

    request = {
        "state": "legacy",
        "questions": {"decision": {"type": "noul", "instructions": "flag?"}},
    }
    item = public_item(request, "decision")
    assert message(item) == (
        'Public request: {"state": "legacy", "questions": {"decision": {"type": "noul", "instructions": "flag?"}}}\n\n'
        'Focus question key: "decision"\nAnswer letter to original option key: {"A": "false", "B": "true"}\n\n'
        "Think step by step, briefly (a few sentences), then finish with exactly one line of the form "
        "'Answer: X', where X is the letter of the correct option."
    )
