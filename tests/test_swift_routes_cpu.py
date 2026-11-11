"""Opt-in exact pinned SDK/processor integration checks, always without model weights.

Set DFORGE_SWIFT_METADATA_CACHE to a populated metadata-only cache. Ordinary
base installs still run the pure contract tests in test_swift_routes.py.
"""

import copy
import os
from pathlib import Path

import pytest

from decisionforge.models import TrainingExample
from decisionforge.training.backends.swift_routes import get_route
from decisionforge.training.backends.swift_routes.sdk import (
    PROCESSOR_FILES,
    metadata_snapshot,
    preprocess,
)
from decisionforge.training.backends.swift_runtime import cpu_template, encode_check
from decisionforge.training.backends.swift_task import map_request
from decisionforge.training.config import load_config

EXAMPLES = Path(__file__).resolve().parents[1] / "examples/training/swift"
pytestmark = pytest.mark.skipif(
    not os.environ.get("DFORGE_SWIFT_METADATA_CACHE"),
    reason="explicit metadata-only cache required for SDK integration",
)


@pytest.fixture(autouse=True)
def forbid_model_weights(monkeypatch):
    torch = pytest.importorskip("torch")
    import safetensors.torch
    from transformers import PreTrainedModel

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU template validation must never load model weights")

    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(safetensors.torch, "load_file", forbidden)
    monkeypatch.setattr(PreTrainedModel, "from_pretrained", forbidden)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


@pytest.mark.parametrize("name", ["clef", "kev", "jev", "startlux"])
def test_real_cns_templates_images_collator_and_no_weight_load(name):
    cfg = load_config(EXAMPLES / f"{name}.yaml")
    route = get_route(name)
    root, files = metadata_snapshot(route)
    assert set(files) <= set(PROCESSOR_FILES) and "decision_head.json" not in files
    assert not list(root.glob("*.safetensors"))
    records = [
        TrainingExample.model_validate_json(line)
        for line in Path(cfg.data.path).read_text().splitlines()
    ]
    rows = [
        row
        for r in records
        for row in map_request(r.request, cfg.swift, root=Path(cfg.data.path).parent)[0]
    ]
    report = encode_check(cfg, rows)
    assert report["questions"] == 21 and report["rows"] == (7 if name == "clef" else 21)
    template = cpu_template(cfg)
    encoded = [template.encode(preprocess(name, row)) for row in rows[: 1 if name == "clef" else 3]]
    # Call the real Transformers RoPE implementation with config only. Swift's
    # default collator otherwise constructs a dummy model via ambiguous auto-
    # detection; no model instance or parameters are needed for these positions.
    from types import MethodType, SimpleNamespace

    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    rope = SimpleNamespace(config=AutoConfig.from_pretrained(root, local_files_only=True))
    rope.get_rope_index = MethodType(Qwen3_5Model.get_rope_index, rope)
    rope.get_vision_position_ids = MethodType(Qwen3_5Model.get_vision_position_ids, rope)
    template.model = rope
    collated = template.data_collator(copy.deepcopy(encoded))
    assert collated["input_ids"].shape[0] == len(encoded)
    if name != "kev":
        assert collated["pixel_values"].numel() > 0
    if name == "clef":
        item = encoded[0]
        meta = item["decision_meta"]
        questions = [
            template.tokenizer.decode(item["input_ids"][a:b]) for a, b in meta["question_spans"]
        ]
        assert len(questions) == 3 and "Which color" in questions[0]
        options = [
            [template.tokenizer.decode(item["input_ids"][a:b]) for a, b in spans]
            for spans in meta["option_spans"]
        ]
        assert "blue" in options[0][0] and "red" in options[0][1]
        plain = dict(rows[0], images=[])
        text_encoded = template.encode(preprocess(name, plain))
        text_meta = text_encoded["decision_meta"]
        assert questions == [
            template.tokenizer.decode(text_encoded["input_ids"][a:b])
            for a, b in text_meta["question_spans"]
        ]
    # A processor may round dimensions upward. A raw-size check alone is insufficient.
    if name != "kev":
        template.max_pixels = 100
        with pytest.raises(ValueError, match="grid exceeds"):
            template.encode(preprocess(name, rows[0]))


@pytest.mark.parametrize("name", ["clef", "kev", "jev", "startlux"])
def test_no_native_silent_truncation(name):
    cfg = load_config(EXAMPLES / f"{name}.yaml")
    template = cpu_template(cfg)
    template.max_length = 128
    record = TrainingExample.model_validate_json(
        (EXAMPLES / "data/text.jsonl").read_text().splitlines()[0]
    )
    record.request.state = "A deliberately long evidence passage. " * 200
    rows, _ = map_request(record.request, cfg.swift)
    with pytest.raises(Exception, match="[Mm]ax|exceed|long|length"):
        template.encode(preprocess(name, rows[0]))
    assert template.max_length == 128
