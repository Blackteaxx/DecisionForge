"""CPU-only contracts for the experimental ms-swift integration."""

import hashlib

import pytest

from decisionforge.media import MediaAsset, VideoMosaic, reject_media, resolve_request
from decisionforge.models import TrainingExample
from decisionforge.training.backend import capabilities, preflight
from decisionforge.training.backends.swift_task import map_request, readout
from decisionforge.training.config import SwiftConfig, TrainConfig


def record():
    return TrainingExample.model_validate(
        {
            "record_id": 1,
            "group_id": 1,
            "split": "train",
            "request": {
                "state": {"nested": ["retained", {"value": "public"}]},
                "questions": {
                    "custom-question": {
                        "type": "choice",
                        "instructions": "Pick",
                        "criteria": {"abstain": "real original option", "x": "other"},
                    }
                },
            },
            "supervision": {
                "answers": {
                    "custom-question": {
                        "type": "choice",
                        "choice": "x",
                        "confidence": 0.4,
                        "probabilities": {"abstain": 0.3, "x": 0.7},
                    }
                }
            },
        }
    )


def test_explicit_backend_and_pinned_defaults():
    cfg = TrainConfig.model_validate({"backend": "swift", "data": {"path": "x"}})
    assert cfg.swift.model_type == "omnijev" and cfg.lora is None
    assert not cfg.model.load_in_4bit and not cfg.training.gradient_checkpointing
    assert capabilities("swift").artifact_format == "swift_omnijev"
    with pytest.raises(ValueError, match="complete state resume"):
        preflight(cfg, resume=True)


@pytest.mark.parametrize(
    "value",
    [
        {"model_type": "laya"},
        {"model_type": "generic_decision"},
        {"decision_subsample_options": True},
        {"adapter_revision": "main"},
        {"swift_revision": "main"},
        {"max_pixels": 99999999},
    ],
)
def test_unsupported_profiles_rejected(value):
    with pytest.raises(ValueError):
        SwiftConfig.model_validate(value)


def test_reversible_choice_and_soft_supervision():
    r = record()
    rows, mapping = map_request(
        r.request, SwiftConfig(), answers=r.supervision.answers, record_id=r.record_id
    )
    assert rows[0]["questions"][0]["gold"] == [0.3, 0.7, 0.0]
    assert mapping[0]["record_id"] == 1 and mapping[0]["question_id"] == "custom-question"
    assert mapping[0]["option_keys"] == ["abstain", "x"]
    assert mapping[0]["native_keys"] == ["k0", "k1"]
    assert "nested" in rows[0]["questions"][0]["instructions"]
    answer, abstain = readout(r.request.questions["custom-question"], [0.15, 0.35, 0.5])
    assert answer["probabilities"] == pytest.approx({"abstain": 0.3, "x": 0.7})
    assert abstain == 0.5


@pytest.mark.parametrize(
    "p", [[0.2, 0.8], [-0.1, 0.8, 0.3], [float("nan"), 0.5, 0.5], [0.1, 0.1, 0.1], [0.0, 0.0, 1.0]]
)
def test_illegal_probabilities_are_not_repaired(p):
    with pytest.raises(ValueError):
        readout(record().request.questions["custom-question"], p)


def test_score_and_noul_column_order():
    answer, _ = readout({"type": "noul"}, [0.8, 0.2])
    assert answer["noul"] == 0.8
    q = {"type": "score", "criteria": ["low", "mid", "high"]}
    a, _ = readout(q, [0.2, 0.3, 0.5])
    assert a["score"] == pytest.approx(1.3)
    assert a["legend"] == {"0": "low", "1": "mid", "2": "high"}


def test_limits_fail_without_subsampling():
    r = record()
    cfg = SwiftConfig(max_options=2)
    r.request.questions["custom-question"].criteria["third"] = "third"
    with pytest.raises(ValueError, match="option count"):
        map_request(r.request, cfg)
    r = record()
    r.request.questions["custom-question"].criteria["x"] = "long " * 50
    with pytest.raises(ValueError, match="200 characters"):
        map_request(r.request, cfg)


def test_media_hashes_relocation_and_pixels(tmp_path):
    from PIL import Image

    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    p = a / "picture.png"
    Image.new("RGB", (32, 32), "red").save(p)
    ref = {"path": "picture.png", "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    r = record()
    r.request.images = [MediaAsset(**ref)]
    rows, _ = map_request(r.request, SwiftConfig(), root=a)
    assert rows[0]["images"] == [str(p)]
    (b / p.name).write_bytes(p.read_bytes())
    r.request.images = [MediaAsset(**ref)]
    resolve_request(r.request, b)
    assert r.request.images[0].path == str(b / p.name)
    with pytest.raises(ValueError, match="pixels"):
        map_request(r.request, SwiftConfig(max_pixels=100), root=b)
    (b / p.name).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA-256"):
        map_request(r.request, SwiftConfig(), root=b)


@pytest.mark.parametrize("path", ["https://a/b.png", "data:image/png;base64,YQ==", "file:///tmp/a"])
def test_unmaterialized_media_rejected(path):
    with pytest.raises(ValueError):
        MediaAsset(path=path, sha256="0" * 64)


@pytest.mark.parametrize(
    "adapter",
    ["chat_api", "jev", "llm2jev", "anyjev_local", "unsloth_local", "openjev_local", "laya_local"],
)
def test_text_adapters_cannot_drop_images(adapter):
    with pytest.raises(ValueError, match="does not support multimodal"):
        reject_media({"images": [{"path": "a", "sha256": "0" * 64}]}, adapter)


def test_plain_text_serialization_unchanged():
    r = record()
    assert set(r.request.model_dump()) == {"state", "questions"}


def test_offline_video_contract(tmp_path):
    from PIL import Image

    p = tmp_path / "mosaic.jpg"
    Image.new("RGB", (1536, 1536)).save(p)
    v = VideoMosaic(
        mosaic=MediaAsset(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()),
        timestamps=[float(i) for i in range(16)],
        duration=16.0,
        source_sha256="0" * 64,
    )
    r = record()
    r.request.videos = [v]
    rows, _ = map_request(r.request, SwiftConfig())
    assert rows[0]["video"]["n_frames"] == 16 and rows[0]["images"] == [str(p)]
    with pytest.raises(ValueError):
        VideoMosaic.model_validate(
            {
                "mosaic": v.mosaic.model_dump(),
                "timestamps": [0.0],
                "duration": 16.0,
                "source_sha256": "0" * 64,
            }
        )


def test_native_preprocessor_columns_match_contract():
    pytest.importorskip("swift")
    from swift.dataset.preprocessor.decision import OmniJevPreprocessor

    r = record()
    rows, _ = map_request(r.request, SwiftConfig(), answers=r.supervision.answers)
    row = OmniJevPreprocessor().preprocess(rows[0])[0]
    assert row["target_probs"] == [[0.3, 0.7, 0.0]]
    assert row["omnijev_column_keys"] == [["k0", "k1", "abstain"]]


def test_multiple_questions_keep_consecutive_reversible_row_ids(tmp_path):
    from decisionforge.training.backends.swift import SwiftBackend

    r = record()
    r.request.questions["second-question"] = r.request.questions["custom-question"].model_copy(
        deep=True
    )
    r.request.questions["second-question"].instructions = "Second question"
    r.supervision.answers["second-question"] = r.supervision.answers["custom-question"].model_copy(
        deep=True
    )
    config = TrainConfig(backend="swift", data={"path": "unused"})
    backend = SwiftBackend(config, tmp_path)
    path, mapping = backend._write_rows("train", [r])
    assert len(path.read_text().splitlines()) == 2
    assert [item["item_index"] for item in mapping] == [0, 1]
    assert [item["question_id"] for item in mapping] == ["custom-question", "second-question"]
    assert all(
        item["record_id"] == 1 and item["group_id"] == 1 and item["role"] == "train"
        for item in mapping
    )


def test_update_evidence_counts_real_adamw_steps_only():
    torch = pytest.importorskip("torch")
    pytest.importorskip("swift")
    from torch.optim.optimizer import register_optimizer_step_post_hook

    from decisionforge.training.backends.swift_plugin import UpdateEvidence

    evidence = UpdateEvidence.__new__(UpdateEvidence)
    evidence.steps = 0
    p = torch.nn.Parameter(torch.tensor([1.0]))
    h = torch.nn.Parameter(torch.tensor([2.0]))
    evidence.observed_parameters = {"lora_A": p, "scoring_head.weight": h}
    adam = torch.optim.AdamW([p, h], lr=0.01)
    sgd = torch.optim.SGD([p], lr=0.01)
    hook = register_optimizer_step_post_hook(evidence.stepped)
    try:
        (p.square().sum() + h.square().sum()).backward()
        adam.step()
        assert evidence.steps == 1
        assert evidence.first_step_gradients["adapter"]["tensors_with_gradient"] == 1
        sgd.step()
        assert evidence.steps == 1
        assert p.item() != 1.0
    finally:
        hook.remove()


def test_update_evidence_rejects_disconnected_adapter_at_first_step():
    torch = pytest.importorskip("torch")
    pytest.importorskip("swift")
    from decisionforge.training.backends.swift_plugin import UpdateEvidence

    evidence = UpdateEvidence.__new__(UpdateEvidence)
    evidence.steps = 0
    adapter = torch.nn.Parameter(torch.tensor([1.0]))
    head = torch.nn.Parameter(torch.tensor([2.0]))
    evidence.observed_parameters = {"lora_A": adapter, "scoring_head.weight": head}
    head.square().sum().backward()
    optimizer = torch.optim.AdamW([adapter, head], lr=0.01)
    with pytest.raises(ValueError, match="No nonzero adapter gradient at the first AdamW"):
        evidence.stepped(optimizer, (), {})


def test_native_swift_checkpoint_override_preserves_lora_gradient(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    torch = pytest.importorskip("torch")
    pytest.importorskip("swift")
    import torch.utils.checkpoint as checkpoint
    from swift.trainers.mixin import SwiftMixin

    from decisionforge.training.backends.swift import SwiftBackend

    backend = SwiftBackend(TrainConfig(backend="swift", data={"path": "unused"}), tmp_path)
    backend.plugin = "fixture.py"
    backend.adapter = tmp_path / "original-adapter"
    command = backend._cmd()
    kwargs = json.loads(command[command.index("--gradient_checkpointing_kwargs") + 1])
    assert command[command.index("--gradient_checkpointing") + 1] == "false"
    assert command[command.index("--check_model") + 1] == "false"
    assert kwargs == {"use_reentrant": False}

    # Exercise the installed Swift wrapper, which overrides call-site kwargs.
    monkeypatch.delattr(checkpoint, "_old_checkpoint", raising=False)
    monkeypatch.setattr(checkpoint, "checkpoint", checkpoint.checkpoint)
    monkeypatch.setattr(checkpoint, "_old_checkpoint", None, raising=False)
    monkeypatch.delattr(checkpoint, "_old_checkpoint")
    import transformers.modeling_utils

    monkeypatch.setattr(
        transformers.modeling_utils, "checkpoint", checkpoint.checkpoint, raising=False
    )
    SwiftMixin._fix_gradient_checkpointing(
        SimpleNamespace(args=SimpleNamespace(gradient_checkpointing_kwargs=kwargs))
    )
    adapter = torch.nn.Parameter(torch.tensor([2.0]))
    token_ids = torch.tensor([3])
    output = checkpoint.checkpoint(lambda ids: adapter * ids, token_ids, use_reentrant=False)
    output.sum().backward()
    assert adapter.grad.item() == 3.0


def test_export_retains_release_notices_and_hashes(tmp_path, monkeypatch):
    pytest.importorskip("huggingface_hub")
    import json

    import huggingface_hub

    from decisionforge.training.backends.swift import SwiftBackend, validate_artifact
    from decisionforge.training.backends.swift_task import (
        ADAPTER_REVISION,
        BASE_REVISION,
        SWIFT_REVISION,
    )

    config = TrainConfig(backend="swift", data={"path": "unused"})
    backend = SwiftBackend(config, tmp_path / "run")
    backend.checkpoint = tmp_path / "native-final"
    backend.checkpoint.mkdir()
    for name in (
        "adapter_config.json",
        "adapter_model.safetensors",
        "scoring_head.safetensors",
        "scoring_head.json",
        "head.pt",
        "ord.pt",
        "head_meta.json",
        "new_tok_emb.pt",
    ):
        (backend.checkpoint / name).write_bytes(
            b"{}" if name.endswith(".json") else b"unit-fixture"
        )
    (backend.checkpoint / "optimizer.pt").write_bytes(
        b"optimizer state excluded from weights export"
    )
    (backend.checkpoint / "swift_lineage.json").write_text(
        json.dumps(
            {
                "base_model": {"name": config.model.name, "revision": BASE_REVISION},
                "adapter": {"name": "tinnel123/OmniJev", "revision": ADAPTER_REVISION},
                "ms_swift": SWIFT_REVISION,
            }
        )
    )
    backend.adapter = tmp_path / "original-adapter"
    backend.adapter.mkdir()
    (backend.adapter / "LICENSE").write_text("Original adapter license fixture")
    base = tmp_path / "base"
    base.mkdir()
    (base / "LICENSE").write_text("Original backbone license fixture")

    def cached_base(name, *, revision, local_files_only):
        assert name == config.model.name and revision == BASE_REVISION and local_files_only
        return str(base)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", cached_base)
    backend.save()
    assert (backend.exported / "LICENSE").read_text() == "Original adapter license fixture"
    assert (backend.exported / "QWEN_LICENSE").read_text() == "Original backbone license fixture"
    assert "decision head were updated" in (backend.exported / "NOTICE").read_text()
    assert not (backend.exported / "optimizer.pt").exists()
    assert validate_artifact(backend.exported) == backend.exported
    (backend.exported / "LICENSE").write_text("tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_artifact(backend.exported)


def test_native_checkpoint_save_carries_lineage(tmp_path, monkeypatch):
    pytest.importorskip("swift")
    import json
    from types import SimpleNamespace

    from decisionforge.training.backends.swift_plugin import UpdateEvidence

    source = tmp_path / "lineage.json"
    lineage = {"base_model": {"name": "fixture"}, "adapter": {"revision": "fixture"}}
    source.write_text(json.dumps(lineage))
    checkpoint = tmp_path / "checkpoint-10"
    checkpoint.mkdir()
    for name in ("adapter_model.safetensors", "scoring_head.safetensors", "scoring_head.json"):
        (checkpoint / name).write_bytes(b"native-file-fixture")
    monkeypatch.setenv("DFORGE_SWIFT_LINEAGE", str(source))
    callback = UpdateEvidence.__new__(UpdateEvidence)
    callback.on_save(
        SimpleNamespace(output_dir=str(tmp_path)),
        SimpleNamespace(global_step=10, last_model_checkpoint=str(checkpoint)),
        None,
    )
    assert json.loads((checkpoint / "swift_lineage.json").read_text()) == lineage
    (checkpoint / "scoring_head.json").unlink()
    with pytest.raises(ValueError, match="missing scoring_head.json"):
        callback.on_save(
            SimpleNamespace(output_dir=str(tmp_path)),
            SimpleNamespace(global_step=10, last_model_checkpoint=str(checkpoint)),
            None,
        )
