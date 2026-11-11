"""Multi-route decision contracts, native CPU encoding and artifact boundaries."""

import contextlib
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from decisionforge.eval.adapters.swift import SwiftAdapter
from decisionforge.eval.models import EvalConfig, Question
from decisionforge.models import TrainingExample
from decisionforge.training.backends.swift import SwiftBackend, validate_artifact
from decisionforge.training.backends.swift_routes import get_route
from decisionforge.training.backends.swift_routes.artifacts import sources, write_manifest
from decisionforge.training.backends.swift_routes.sdk import PROCESSOR_FILES, head_environment
from decisionforge.training.backends.swift_task import map_request
from decisionforge.training.config import SwiftConfig, TrainConfig

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples/training/swift"
NEW = ["clef", "kev", "jev", "startlux"]


def settings(name, **kwargs):
    return SwiftConfig(
        model_type=name,
        **({"usage": "noncommercial_research"} if name == "startlux" else {}),
        **kwargs,
    )


def config(name, **kwargs):
    return TrainConfig(
        backend="swift",
        data={"path": str(EXAMPLES / "data/text.jsonl")},
        swift=settings(name),
        **kwargs,
    )


def record(image=False):
    return TrainingExample.model_validate_json(
        (EXAMPLES / "data" / ("multimodal.jsonl" if image else "text.jsonl"))
        .read_text()
        .splitlines()[0]
    )


@pytest.mark.parametrize("name", NEW)
def test_route_defaults_pins_modes_and_native_command(name, tmp_path):
    route, cfg = get_route(name), config(name)
    assert cfg.model.name == route.model and cfg.model.revision == route.revision
    assert not cfg.model.load_in_4bit and cfg.calibration.enabled == route.calibration
    backend = SwiftBackend(cfg, tmp_path)
    assert backend.route == route
    backend.plugin = "plugin.py"
    backend.adapter = Path("adapter") if route.adapter else None
    command = backend._cmd()
    args = dict(zip(command[3::2], command[4::2]))
    assert args["--model_type"] == name and args["--template"] == route.template
    assert args["--tuner_type"] == "lora" and args["--loss_type"] == route.loss
    assert args["--decision_subsample_options"] == "0"
    assert args["--packing"] == args["--padding_free"] == "false"
    with pytest.raises(ValueError):
        config(name, model={"revision": "main"})
    with pytest.raises(ValueError):
        config(name, model={"load_in_4bit": True})
    with pytest.raises(ValueError):
        settings(name, packing=True)
    if route.merged:
        full = config(name, finetuning="full")
        assert full.lora is None
    else:
        with pytest.raises(ValueError, match="published LoRA"):
            config(name, finetuning="full")
    if not route.calibration:
        with pytest.raises(ValueError):
            config(name, calibration={"enabled": True})


@pytest.mark.parametrize("name", ["thx02", "laya", "generic_decision", "arbitrary"])
def test_unlicensed_and_out_of_scope_routes_are_rejected(name):
    with pytest.raises(ValueError):
        settings(name)


def test_noncommercial_gate_and_fixed_adapters():
    with pytest.raises(ValueError, match="noncommercial"):
        SwiftConfig(model_type="startlux")
    with pytest.raises(ValueError, match="noncommercial"):
        EvalConfig(
            data={"path": "x"},
            model={
                "adapter": "swift_local",
                "model_type": "startlux",
                "path": get_route("startlux").model,
            },
        )
    for name in NEW:
        with pytest.raises(ValueError, match="fixed published adapter"):
            settings(name, adapter="unverified", adapter_revision="main")
        with pytest.raises(ValueError):
            settings(name, decision_subsample_options=True)


@pytest.mark.parametrize("name", NEW)
def test_native_preprocessor_targets_and_reversible_cns(name):
    pytest.importorskip("swift")
    from decisionforge.training.backends.swift_routes.sdk import preprocess

    r, opts = record(), settings(name)
    rows, mappings = map_request(r.request, opts, answers=r.supervision.answers, record_id=1)
    assert len(rows) == (1 if name == "clef" else 3)
    assert [m["question_index"] for m in mappings] == ([0, 1, 2] if name == "clef" else [0, 0, 0])
    normalized = [preprocess(name, row) for row in rows]
    gold = [target for row in normalized for target in row["target_probs"]]
    for m, target in zip(mappings, gold):
        q = r.request.questions[m["question_id"]]
        result, extra = get_route(name).readout(q, target, m)
        expected = r.supervision.answers[m["question_id"]]
        if q.type == "noul":
            assert result["noul"] == expected.noul
        else:
            assert result["probabilities"] == expected.probabilities
        assert extra is None
    if name == "clef":
        assert mappings[0]["native_keys"] == ["alpha", "zeta"]
        assert gold[0] == [0.2, 0.8]
    if name == "startlux":
        assert gold[1] == [0.75, 0.25]
    else:
        assert gold[1] == [0.25, 0.75]


@pytest.mark.parametrize("name", NEW)
def test_limits_and_media_reject_before_encoding(name):
    r, opts = record(), settings(name)
    r.request.questions["pick:α"].criteria = {str(i): str(i) for i in range(opts.max_options + 1)}
    with pytest.raises(ValueError, match="option"):
        map_request(r.request, opts)
    r = record()
    r.request.questions["rating"].criteria = [f"level{i}" for i in range(11)]
    with pytest.raises(ValueError, match="levels"):
        map_request(r.request, opts)
    r = record(True)
    if name == "kev":
        with pytest.raises(ValueError, match="image count"):
            map_request(r.request, opts, root=EXAMPLES / "data")
    else:
        assert map_request(r.request, opts, root=EXAMPLES / "data")[0][0]["images"]
        with pytest.raises(ValueError, match="pixels"):
            map_request(r.request, settings(name, max_pixels=100), root=EXAMPLES / "data")
        r.request.images *= opts.max_images + 1
        with pytest.raises(ValueError, match="image count"):
            map_request(r.request, opts, root=EXAMPLES / "data")
    r = record()
    r.request.state = "<|vision_start|><|image_pad|>"
    with pytest.raises(ValueError, match="reserved"):
        map_request(r.request, opts)


def test_jev_fixed_verbalizers_and_star_none():
    r = record()
    r.request.questions["rating"].criteria = ["one", "two"]
    with pytest.raises(ValueError, match="levels"):
        map_request(r.request, settings("jev"))
    r = record()
    opts = settings("startlux", allow_none=True)
    rows, maps = map_request(r.request, opts, answers=r.supervision.answers)
    assert rows[0]["questions"]["pick:α"]["gold"] == [0.8, 0.2, 0.0]
    result, extra = get_route("startlux").readout(
        r.request.questions["pick:α"], [0.2, 0.3, 0.5], maps[0]
    )
    assert result["probabilities"] == {"zeta": 0.4, "alpha": 0.6} and extra == 0.5
    with pytest.raises(ValueError, match="All probability mass"):
        get_route("startlux").readout(r.request.questions["pick:α"], [0.0, 0.0, 1.0], maps[0])
    r.request.questions["pick:α"].criteria = {str(i): f"option{i}" for i in range(26)}
    with pytest.raises(ValueError, match="26 letter"):
        map_request(r.request, opts)


@pytest.mark.parametrize("name", NEW)
@pytest.mark.parametrize(
    "bad", [[0.2], [0.2, 0.2], [float("nan"), 1.0], [-0.1, 1.1], [True, False]]
)
def test_raw_columns_are_strict(name, bad):
    with pytest.raises(ValueError, match="Invalid"):
        get_route(name).readout(record().request.questions["pick:α"], bad)


def artifact(root, name, mode="lora"):
    root.mkdir()
    route = get_route(name)
    files = {"scoring_head.safetensors", "scoring_head.json"} | {a[2] for a in route.assets}
    files |= (
        {"adapter_model.safetensors", "adapter_config.json"}
        if mode == "lora"
        else {"config.json", "model.safetensors"}
    )
    for filename in files:
        (root / filename).write_bytes(
            b"{}" if filename.endswith(".json") else b"synthetic-unit-fixture"
        )
    lineage = {**sources(route), "usage": "noncommercial_research" if route.noncommercial else None}
    write_manifest(root, route, mode, lineage)
    return root


@pytest.mark.parametrize("name", NEW)
def test_artifact_lineage_files_head_environment_and_tampering(name, tmp_path, monkeypatch):
    root = artifact(tmp_path / name, name)
    assert validate_artifact(root, model_type=name) == root
    assert validate_artifact(get_route(name).model, model_type=name) == Path(get_route(name).model)
    with pytest.raises(ValueError):
        validate_artifact(root, model_type="omnijev")
    for _, _, _, env in get_route(name).assets:
        monkeypatch.delenv(env, raising=False)
    head_environment(get_route(name), root, root, artifact=root)
    import os

    for _, _, filename, env in get_route(name).assets:
        assert os.environ[env] == str(root / filename)
    marker = root / "dforge_swift_artifact.json"
    manifest = json.loads(marker.read_text())
    original = copy.deepcopy(manifest)
    manifest["lineage"]["base_model"]["revision"] = "main"
    marker.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="lineage"):
        validate_artifact(root)
    marker.write_text(json.dumps(original))
    (root / "scoring_head.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_artifact(root)


@pytest.mark.parametrize("name", ["clef", "startlux"])
def test_full_backbone_export_and_missing_shard(name, tmp_path):
    root = artifact(tmp_path / name, name, "full")
    assert validate_artifact(root) == root
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": "model-00001.safetensors"}})
    )
    info = json.loads((root / "dforge_swift_artifact.json").read_text())
    write_manifest(root, get_route(name), "full", info["lineage"])
    with pytest.raises(ValueError, match="shard index"):
        validate_artifact(root)


@pytest.mark.parametrize("name", NEW)
def test_runtime_joint_readout_and_eval_adapter_without_model(name):
    pytest.importorskip("swift")
    from decisionforge.training.backends.swift_routes.runtime import RouteRuntime

    r, route = record(), get_route(name)
    runtime = object.__new__(RouteRuntime)
    runtime.config, runtime.route = config(name), route
    runtime.torch = SimpleNamespace(
        cuda=SimpleNamespace(
            reset_peak_memory_stats=lambda: None,
            synchronize=lambda: None,
            max_memory_allocated=lambda: 0,
        )
    )
    runtime.temperature = lambda _: contextlib.nullcontext()
    rows, maps = map_request(r.request, runtime.config.swift, answers=r.supervision.answers)
    targets = []
    for m in maps:
        a = r.supervision.answers[m["question_id"]]
        targets.append(
            [a.noul if k == "true" else 1 - a.noul for k in m["native_keys"]]
            if m["type"] == "noul"
            else [a.probabilities[k] for k in m["native_keys"]]
        )
    responses = iter([targets] if name == "clef" else [[target] for target in targets])

    def infer(*args, **kwargs):
        return [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=[{"probs": p} for p in next(responses)])
                    )
                ]
            )
        ]

    runtime.engine = SimpleNamespace(infer=infer)
    raw = runtime.predict(r.request.model_dump(mode="json"))
    assert len(raw["swift"]["columns"]) == 3
    for qid, answer in raw["answers"].items():
        if answer["type"] == "noul":
            assert answer["noul"] == r.supervision.answers[qid].noul
        else:
            assert answer["probabilities"] == r.supervision.answers[qid].probabilities
    q = Question.model_validate(r.request.questions["pick:α"].model_dump())
    item = raw["swift"]["columns"][0]
    item["question_id"] = "decision"
    raw["answers"] = {"decision": raw["answers"]["pick:α"]}
    raw["swift"]["columns"] = [item]
    model = {"adapter": "swift_local", "model_type": name, "path": route.model}
    if route.noncommercial:
        model["usage"] = "noncommercial_research"
    cfg = EvalConfig(data={"path": "x"}, model=model)
    adapter = SwiftAdapter(cfg.model, cfg.execution)
    assert adapter.response_answer(raw, q)["probabilities"] == {"zeta": 0.8, "alpha": 0.2}
    item["native"][0]["probs"] = [0.5, 0.5]
    with pytest.raises(ValueError, match="differs"):
        adapter.response_answer(raw, q)


def test_weighted_json_is_excluded_from_metadata_allowlist():
    assert "decision_head.json" not in PROCESSOR_FILES
    assert not any(
        "*" in name or name.endswith((".pt", ".safetensors", ".bin")) for name in PROCESSOR_FILES
    )


@pytest.mark.parametrize("name", NEW)
def test_native_max_catalog_and_score_boundaries(name):
    pytest.importorskip("swift")
    from decisionforge.training.backends.swift_routes.sdk import preprocess

    r, opts = record(), settings(name)
    r.request.questions["pick:α"].criteria = {
        f"key{i}": f"distinct option {i}" for i in range(opts.max_options)
    }
    rows, mapping = map_request(r.request, opts)
    native = preprocess(name, rows[0])
    assert len(native["options"][0]) == opts.max_options
    assert len(mapping[0]["native_keys"]) == opts.max_options
    for count in [6] if name == "jev" else [2, 10]:
        r.request.questions["rating"].criteria = [f"distinct level{i}" for i in range(count)]
        assert map_request(r.request, opts)
    with pytest.raises(ValueError, match="option"):
        map_request(r.request, settings(name, max_options=2))


def test_startlux_native_none_has_one_zero_target():
    pytest.importorskip("swift")
    from decisionforge.training.backends.swift_routes.sdk import preprocess

    r = record()
    rows, _ = map_request(
        r.request, settings("startlux", allow_none=True), answers=r.supervision.answers
    )
    native = preprocess("startlux", rows[0])
    assert native["target_probs"] == [[0.8, 0.2, 0.0]]
    assert len(native["options"][0]) == 3


def test_clef_joint_row_offsets_across_records(tmp_path):
    cfg = config("clef")
    backend = SwiftBackend(cfg, tmp_path)
    r1, r2 = record(), record()
    r2.record_id = 2
    path, maps = backend._write_rows("train", [r1, r2])
    assert len(path.read_text().splitlines()) == 2
    assert [m["row_index"] for m in maps] == [0, 0, 0, 1, 1, 1]
    assert [m["item_index"] for m in maps] == list(range(6))
    assert [m["question_index"] for m in maps] == [0, 1, 2, 0, 1, 2]


@pytest.mark.parametrize("name", NEW)
def test_gpu_guard_precedes_any_weight_download(name, monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("swift")
    from decisionforge.training.backends.swift_routes import runtime

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def forbidden(*args, **kwargs):
        raise AssertionError("must not request weights before CUDA guard")

    monkeypatch.setattr(runtime, "source_roots", forbidden)
    with pytest.raises(ValueError, match="BF16 CUDA"):
        runtime.RouteRuntime(config(name), get_route(name).model)


def test_full_update_groups_and_temperature_buffers():
    torch = pytest.importorskip("torch")
    pytest.importorskip("swift")
    from decisionforge.training.backends.swift_plugin import UpdateEvidence
    from decisionforge.training.backends.swift_routes.sdk import temperature_buffers

    evidence = UpdateEvidence.__new__(UpdateEvidence)
    evidence.finetuning, evidence.steps = "full", 0
    backbone, head = torch.nn.Parameter(torch.ones(1)), torch.nn.Parameter(torch.ones(1))
    evidence.observed_parameters = {"model.weight": backbone, "scoring_head.bias": head}
    (backbone + head).sum().backward()
    evidence.stepped(torch.optim.AdamW([backbone, head]), (), {})
    assert set(evidence.first_step_gradients) == {"backbone", "head"}
    sample = SimpleNamespace(head_temperature=torch.tensor(2.0), bias=head, log_tau=backbone)
    assert list(temperature_buffers(sample)) == ["head_temperature"]


def test_real_omnijev_readout_compatibility_replay():
    import ast
    import hashlib

    from decisionforge.training.backends.swift_task import readout

    fixture = json.loads((ROOT / "tests/fixtures/swift_readout.json").read_text())
    source = (ROOT / "src/decisionforge/training/backends/swift_runtime.py").read_text()
    runtime = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == "Runtime"
    )
    methods = {
        node.name: hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
        for node in runtime.body
        if isinstance(node, ast.FunctionDef)
    }
    for name, expected in fixture["runtime_method_hashes"].items():
        assert methods[name] == expected
    assert len(fixture["readouts"]) == 2017
    for item in fixture["readouts"]:
        question = {"type": "choice", "criteria": {key: key for key in item["keys"]}}
        answer, abstain = readout(question, item["probabilities"])
        assert answer == item["answer"]
        assert abstain == item["abstain_probability"]


@pytest.mark.parametrize("name", NEW)
def test_eval_yaml_paths_and_published_source_id(name, tmp_path):
    import yaml

    from decisionforge.eval.data import load_samples
    from decisionforge.eval.models import load_config

    cfg = load_config(EXAMPLES / f"eval-{name}.yaml")
    samples = load_samples(cfg.data.path)
    assert len(samples) == 3 and {s.question.type for s in samples} == {"choice", "noul", "score"}
    value = yaml.safe_load((EXAMPLES / f"eval-{name}.yaml").read_text())
    value["model"]["path"] = get_route(name).model
    value["data"]["path"] = cfg.data.path
    p = tmp_path / "original.yaml"
    p.write_text(yaml.safe_dump(value))
    assert load_config(p).model.path == get_route(name).model


@pytest.mark.parametrize("name", NEW)
@pytest.mark.parametrize("kind", ["noul", "score"])
def test_eval_adapter_all_question_types(name, kind):
    r = record()
    qid = "truth" if kind == "noul" else "rating"
    q = Question.model_validate(r.request.questions[qid].model_dump())
    route = get_route(name)
    native = [0.7, 0.3] if kind == "noul" else [0.1, 0.2, 0.3, 0.15, 0.15, 0.1]
    answer, extra = route.readout(q, native)
    raw = {
        "answers": {"decision": answer},
        "swift": {
            "model_type": name,
            "probability_contract": route.probability_contract,
            "columns": [
                {
                    "question_id": "decision",
                    "native": [{"probs": native}],
                    "extra_choice_probability": extra,
                }
            ],
        },
    }
    model = {"adapter": "swift_local", "model_type": name, "path": route.model}
    if route.noncommercial:
        model["usage"] = "noncommercial_research"
    cfg = EvalConfig(data={"path": "unused"}, model=model)
    assert SwiftAdapter(cfg.model, cfg.execution).response_answer(raw, q) == answer


def test_source_cards_agree_with_registry_and_license_hashes():
    import hashlib

    card = json.loads((EXAMPLES / "model-sources.json").read_text())
    for name in NEW:
        route = get_route(name)
        source = card["routes"][name]
        assert (
            source["model"],
            source["revision"],
            source["adapter"],
            source["adapter_revision"],
        ) == (route.model, route.revision, route.adapter, route.adapter_revision)
        for file in source["license_evidence"]:
            assert (
                hashlib.sha256((EXAMPLES / file["file"]).read_bytes()).hexdigest() == file["sha256"]
            )
    assert not card["routes"]["thx02"]["enabled"]
