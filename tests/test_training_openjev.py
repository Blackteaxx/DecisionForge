import copy
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from test_training_data import example, make_config

from decisionforge.training.config import TrainConfig
from decisionforge.training.data import prepare_data


def cn_example(i=1, split="train"):
    row = example(i, split)
    del row["request"]["questions"]["rating"]
    del row["supervision"]["answers"]["rating"]
    return row


def cn_config(tmp_path, **kw):
    return make_config(
        tmp_path,
        rows=[cn_example(i, split) for i, split in ((1, "train"), (2, "validation"), (3, "test"))],
        backend="openjev",
        **kw,
    )


def test_openjev_defaults_roundtrip_dispatch():
    from decisionforge.training.backend import get_backend

    c = TrainConfig(backend="openjev", data={"path": "unused"})
    assert c.finetuning == "full" and c.lora is None and not c.model.load_in_4bit
    assert not c.calibration.enabled and c.training.max_steps == 400
    assert c.training.num_train_epochs is None and c.training.precision == "bf16"
    assert c.training.gradient_accumulation_steps == 1 and c.training.save_steps is None
    assert c.training.learning_rate == c.training.head_learning_rate == 2e-6
    assert c.openjev.m == 4 and TrainConfig.model_validate(c.model_dump()) == c
    assert type(get_backend(c, ".")).__name__ == "OpenJevBackend"


@pytest.mark.parametrize(
    "patch",
    [
        {"training": {"num_train_epochs": 1.0}},
        {"training": {"gradient_accumulation_steps": 2}},
        {"training": {"head_learning_rate": 0.1}},
        {"training": {"precision": "fp16"}},
        {"training": {"warmup_steps": 1}},
        {"training": {"weight_decay": 0.01}},
        {"training": {"lr_scheduler_type": "cosine"}},
        {"training": {"save_steps": 1}},
        {"training": {"logging_steps": 1}},
        {"training": {"gradient_checkpointing": False}},
        {"calibration": {"enabled": True}},
        {"finetuning": "lora"},
        {"model": {"load_in_4bit": True}},
        {"output": {"save_merged": True}},
        {"openjev": {"m": 1}},
    ],
)
def test_explicit_unsupported_settings_fail(patch):
    with pytest.raises(ValidationError):
        TrainConfig(backend="openjev", data={"path": "unused"}, **patch)


def test_soft_mapping_retains_full_public_context_and_fractional_weights(tmp_path):
    from decisionforge.training.backends.openjev_task import map_roles, message

    data = prepare_data(cn_config(tmp_path))
    splits, report = map_roles(data)
    a, b = splits["train"]
    assert a["q"] == a["counts"] == [0.7, 0.3]
    assert b["q"] == b["counts"] == [0.8, 0.2]
    assert a["option_keys"] == ["b", "a"] and b["option_keys"] == ["false", "true"]
    assert a["public_request"] == data.roles["train"][0].request.model_dump(
        mode="json", exclude_none=True
    )
    assert a["focus"] == "decision" and b["focus"] == "flag"
    assert (
        report["train"]["mapping"][0]["weight_source"]
        == "sampling_weights_derived_from_dforge_supervision"
    )
    before = copy.deepcopy(a)
    rendered = message(a)
    assert '"flag"' in rendered and '"b"' in rendered and '"text": "B"' in rendered
    assert '"q"' not in rendered and "counts" not in rendered and "supervision" not in rendered
    assert a == before
    assert len(splits["dev"]) == len(splits["test"]) == 2


def test_static_rejects_score_ddp_resume_missing_test_and_accepts_large_choice(tmp_path, monkeypatch):
    from decisionforge.training.backend import preflight
    from decisionforge.training.runner import run_training

    c = cn_config(tmp_path)
    with pytest.raises(ValueError, match="resume"):
        run_training(c, resume=True)
    assert not (tmp_path / "run").exists()
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="single|DDP"):
        preflight(c)
    monkeypatch.delenv("WORLD_SIZE")
    for row in (example(), cn_example()):
        if "rating" not in row["request"]["questions"]:
            row["request"]["questions"]["decision"]["criteria"] = {
                str(i): str(i) for i in range(11)
            }
            row["supervision"]["answers"]["decision"].update(
                choice="0",
                probabilities={str(i): 1.0 if i == 0 else 0.0 for i in range(11)},
                confidence=1.0,
            )
        bad = make_config(
            tmp_path,
            rows=[row, cn_example(2, "validation"), cn_example(3, "test")],
            backend="openjev",
        )
        if "rating" in row["request"]["questions"]:
            with pytest.raises(ValueError, match="score"):
                preflight(bad, data=prepare_data(bad))
        else:
            preflight(bad, data=prepare_data(bad))
    c = cn_config(tmp_path)
    data = prepare_data(c)
    data.assignments[-1]["role"] = "train"
    with pytest.raises(ValueError, match="test"):
        preflight(c, data=data)


def test_labels_must_be_unique_single_token():
    from decisionforge.training.backends.openjev_task import validate_labels

    tok = SimpleNamespace(encode=lambda s, **kw: [ord(s[-1])])
    assert validate_labels(tok) == list(range(65, 75))
    for fn in (lambda *a, **k: [], lambda *a, **k: [1, 2], lambda *a, **k: [1]):
        with pytest.raises(ValueError, match="token"):
            validate_labels(SimpleNamespace(encode=fn))


def test_stage_cli_is_official_two_stage_recipe_and_lineage(tmp_path):
    from decisionforge.training.backends.openjev import stage_arguments

    c = cn_config(tmp_path, training={"max_steps": 2}, openjev={"reinforce_steps": 3})
    first = stage_arguments(c, "calibrate", "/snapshot", tmp_path / "s1")
    second = stage_arguments(
        c, "reinforce", "/snapshot", tmp_path / "s2", tmp_path / "s1/weights.pt"
    )
    assert first[first.index("--diversity-weight") + 1] == "0"
    assert "--no-score-function" in first and "--no-score-function" not in second
    assert second[second.index("--init-checkpoint") + 1] == str(tmp_path / "s1/weights.pt")
    assert second[second.index("--sf-coef") + 1] == "0.3"
    assert first[first.index("--steps") + 1] == "2" and second[second.index("--steps") + 1] == "3"
    with pytest.raises(ValueError, match="stage|checkpoint"):
        stage_arguments(c, "reinforce", "/snapshot", tmp_path / "s2")


def test_invalid_init_rejected_before_output_writes(tmp_path):
    from decisionforge.training.runner import run_training

    c = cn_config(tmp_path, openjev={"init_checkpoint": str(tmp_path / "missing")})
    with pytest.raises(ValueError, match="OpenJev|openjev"):
        run_training(c)
    assert not (tmp_path / "run").exists()


def artifact_at(root):
    from decisionforge.training.backends.openjev import UPSTREAM_REVISION, inventory
    from decisionforge.training.backends.openjev_task import (
        LABELS,
        MAPPING_VERSION,
        PUBLIC_PROTOCOL,
        SAMPLING,
    )

    (root / "backbone").mkdir(parents=True)
    for name, value in (
        ("config.json", {"model_type": "qwen3"}),
        ("generation_config.json", {"eos_token_id": [1, 2]}),
        ("tokenizer_config.json", {}),
        ("tokenizer.json", {}),
    ):
        (root / "backbone" / name).write_text(json.dumps(value))
    stages = []
    for name in ("calibrate", "reinforce"):
        file = root / "stages" / name / "weights.pt"
        file.parent.mkdir(parents=True)
        file.write_bytes(name.encode())
        stages.append(
            {
                "name": name,
                "weights": file.relative_to(root).as_posix(),
                "initialization": {"kind": "backbone"}
                if name == "calibrate"
                else {"kind": "previous_stage", "sha256": stages[0]["sha256"]},
                **inventory(root, [file])[file.relative_to(root).as_posix()],
                "actual_optimizer_steps": 1,
                "parameter_updates": {"max_abs_delta": 0.01},
            }
        )
    manifest = {
        "schema_version": "decisionforge.openjev.artifact.v1",
        "artifact_format": "openjev_rlcd",
        "upstream_revision": UPSTREAM_REVISION,
        "labels": LABELS,
        "label_token_ids": list(range(10)),
        "mapping_version": MAPPING_VERSION,
        "public_protocol": PUBLIC_PROTOCOL,
        "sampling": {**SAMPLING, "max_new_tokens": 16},
        "max_seq_length": 2048,
        "stages": stages,
        "files": inventory(root),
        "full_resume": False,
    }
    (root / "openjev_artifact.json").write_text(json.dumps(manifest))
    return root


def test_artifact_checks_exact_stage_lineage_and_weight_digest(tmp_path):
    from decisionforge.training.backends.openjev import artifact_identity, validate_artifact

    root = artifact_at(tmp_path / "model")
    initial = artifact_identity(root)
    assert initial["artifact_format"] == "openjev_rlcd"
    (root / "stages/calibrate/weights.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest|identity|changed"):
        validate_artifact(root)
    root = artifact_at(tmp_path / "other")
    path = root / "openjev_artifact.json"
    data = json.loads(path.read_text())
    data["stages"][1]["initialization"]["sha256"] = "foreign"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="lineage|stage"):
        validate_artifact(root)


@pytest.mark.parametrize(
    "backend,default_name,revision",
    [
        ("laya", "convaiinnovations/laya", "7b928d828b7b0e022f929d9bd2e44165aa270148"),
        ("openjev", "Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"),
    ],
)
def test_custom_backbone_does_not_inherit_unrelated_default_revision(
    backend, default_name, revision
):
    for model in ({"name": "custom/repository"}, {"name": "/local/artifact"}):
        c = TrainConfig(backend=backend, data={"path": "unused"}, model=model)
        assert c.model.revision is None
        assert TrainConfig.model_validate(c.model_dump()) == c
    c = TrainConfig(backend=backend, data={"path": "unused"}, model={"name": default_name})
    assert c.model.revision == revision
    explicit = TrainConfig(
        backend=backend,
        data={"path": "unused"},
        model={"name": "custom/repo", "revision": "a" * 40},
    )
    assert explicit.model.revision == "a" * 40
    assert TrainConfig(data={"path": "unused"}).model.revision is None


def test_backend_runs_two_distinct_workers_and_uses_this_runs_first_weights(tmp_path, monkeypatch):
    import decisionforge.training.distributed as processes
    from decisionforge.training.backends.openjev import OpenJevBackend, inventory

    c = cn_config(tmp_path, training={"max_steps": 2}, openjev={"reinforce_steps": 3})
    backend = OpenJevBackend(c, tmp_path / "run")
    backend.backbone, backend.device, backend.limit = tmp_path / "base", "cuda:0", 2048
    backend.initialization = None
    backend.source_identity = {"actual_name": "base"}
    calls = []

    def worker(command, directory, *, env):
        plan = json.loads((directory / "plan.json").read_text())
        args = plan["argv"]
        stage = directory.name
        calls.append(stage)
        if stage == "reinforce":
            init = args[args.index("--init-checkpoint") + 1]
            assert init == str(backend.model_root / "stages/calibrate/weights.pt")
            assert (
                plan["expected_init_sha256"]
                == inventory(Path(init).parent, [Path(init)])["weights.pt"]["sha256"]
            )
        (directory / "weights.pt").write_bytes(stage.encode())
        (directory / "updates.json").write_text(
            json.dumps(
                {
                    "actual_optimizer_steps": int(args[args.index("--steps") + 1]),
                    "parameter_updates": {"max_abs_delta": 0.01},
                }
            )
        )
        (directory / "upstream.json").write_text("{}")

    from pathlib import Path

    monkeypatch.setattr(processes, "run_process", worker)
    report = backend.train()
    assert calls == ["calibrate", "reinforce"] and report["actual_optimizer_steps"] == 5
    assert report["stages"][1]["initialization"]["sha256"] == report["stages"][0]["sha256"]


@pytest.mark.parametrize("mutation", ["missing_tokens", "malformed_stages", "nonfinite_update"])
def test_artifact_rejects_missing_label_identity_and_malformed_stage_evidence(tmp_path, mutation):
    from decisionforge.training.backends.openjev import validate_artifact

    root = artifact_at(tmp_path / "model")
    marker = root / "openjev_artifact.json"
    manifest = json.loads(marker.read_text())
    if mutation == "missing_tokens":
        manifest.pop("label_token_ids", None)
    elif mutation == "malformed_stages":
        manifest["stages"] = [None, {}]
    else:
        manifest["stages"][0]["parameter_updates"]["max_abs_delta"] = -1
    marker.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="OpenJev"):
        validate_artifact(root)


def test_diagnostics_cannot_substitute_another_role(tmp_path):
    from decisionforge.training.backends.openjev import OpenJevBackend

    backend = OpenJevBackend(cn_config(tmp_path), tmp_path)
    for role in ("train", "calibration"):
        with pytest.raises(ValueError, match="validation|test"):
            backend.evaluate(role, "uncalibrated")


def test_weight_initialization_path_and_cli_budgets_roundtrip(tmp_path):
    from decisionforge.training.backends.openjev import stage_arguments
    from decisionforge.training.config import load_config

    path = tmp_path / "config.yaml"
    path.write_text(
        "backend: openjev\ndata: {path: source.jsonl}\nopenjev: {init_checkpoint: ./old-model, reinforce_steps: 3, reinforce_seed: 17}\ntraining: {max_steps: 2, learning_rate: 0.000003}\n"
    )
    c = load_config(path)
    assert c.openjev.init_checkpoint == str(tmp_path / "old-model")
    assert c.training.head_learning_rate == c.training.learning_rate == 0.000003
    args = stage_arguments(c, "reinforce", "/base", tmp_path, tmp_path / "s1")
    assert args[args.index("--seed") + 1] == "17"
    with pytest.raises(ValidationError, match="divide"):
        TrainConfig(
            backend="openjev",
            data={"path": "unused"},
            training={"max_steps": 3},
            openjev={"eval_every": 2},
        )


def test_explicit_cn_fixture_preparation_audits_every_score_exclusion(tmp_path):
    import runpy
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "examples/training/openjev/prepare_cn.py"
    prepare = runpy.run_path(str(script))["prepare"]
    source = tmp_path / "original.jsonl"
    rows = [example(1), example(2, "test")]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "cn.jsonl"
    audit = prepare(source, output)
    assert audit["excluded_questions"] == [
        {"record_id": 1, "question_id": "rating", "type": "score"},
        {"record_id": 2, "question_id": "rating", "type": "score"},
    ]
    converted = [json.loads(line) for line in output.read_text().splitlines()]
    assert [r["request"] for r in converted] == [
        cn_example(1)["request"],
        cn_example(2, "test")["request"],
    ]
    assert [r["supervision"] for r in converted] == [
        cn_example(1)["supervision"],
        cn_example(2, "test")["supervision"],
    ]
    assert source.read_text() == "".join(json.dumps(row) + "\n" for row in rows)


def test_reload_probe_cannot_complete_with_malformed_inference(tmp_path, monkeypatch):
    from decisionforge.eval.adapters import openjev_worker
    from decisionforge.training.backends.openjev import OpenJevBackend

    class Runtime:
        metadata = {"runtime": {}}

        def __init__(self, options):
            pass

        def predict(self, public, *, focus):
            return {
                "answers": {},
                "openjev": {
                    "validation_error": "illegal probability mass",
                    "raw_label_distributions": [[0.2, 0.2]],
                },
            }

        def close(self):
            pass

    monkeypatch.setattr(openjev_worker, "OpenJevRuntime", Runtime)
    config = cn_config(tmp_path)
    backend = OpenJevBackend(config, tmp_path / "run")
    backend.device = "cuda:0"
    with pytest.raises(ValueError, match="reload|probabilit"):
        backend.reload_probe(prepare_data(config))
    evidence = json.loads((tmp_path / "run/openjev_reload_probe.json").read_text())
    assert evidence["openjev"]["raw_label_distributions"] == [[0.2, 0.2]]


def test_export_requires_generation_config_for_upstream_stop_semantics(tmp_path):
    from decisionforge.training.backends.openjev import inventory, validate_artifact

    root = artifact_at(tmp_path / "model")
    (root / "backbone/generation_config.json").unlink(missing_ok=True)
    path = root / "openjev_artifact.json"
    manifest = json.loads(path.read_text())
    manifest["files"] = inventory(root)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="OpenJev"):
        validate_artifact(root)


def test_pinned_installation_checks_original_estimator_and_prompt_sources(tmp_path, monkeypatch):
    import hashlib

    from decisionforge.training.backends import openjev

    files = {"train.py": b"trainer", "estimators.py": b"loss", "tasks/__init__.py": b"registry"}
    for name, content in files.items():
        path = tmp_path / "openjev_rlcd" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    monkeypatch.setattr(
        openjev,
        "SOURCE_SHA256",
        {name: hashlib.sha256(content).hexdigest() for name, content in files.items()},
    )
    package = SimpleNamespace(
        read_text=lambda name: json.dumps({"vcs_info": {"commit_id": openjev.UPSTREAM_REVISION}}),
        locate_file=lambda name: tmp_path / name,
    )
    monkeypatch.setattr(openjev, "distribution", lambda name: package)
    assert openjev.validate_installation()["source_verified"] is True
    (tmp_path / "openjev_rlcd/estimators.py").write_bytes(b"modified loss")
    with pytest.raises(ValueError, match="estimators.py"):
        openjev.validate_installation()


@pytest.mark.parametrize("backend", ["laya", "openjev"])
@pytest.mark.parametrize("field,value", [("model", None), ("model", []), ("training", None), ("training", "invalid")])
def test_backend_defaults_preserve_strict_child_type_errors(backend, field, value):
    with pytest.raises(ValidationError):
        TrainConfig.model_validate({"backend": backend, "data": {"path": "unused"}, field: value})
