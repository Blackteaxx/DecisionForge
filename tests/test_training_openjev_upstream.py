"""Optional pinned-source CPU checks, explicitly not CUDA acceptance."""

import json
import sys

import pytest


@pytest.mark.parametrize("k", [2, 77, 151])
def test_original_main_two_stages_real_adamw_updates_and_official_reader(tmp_path, monkeypatch, k):
    torch = pytest.importorskip("torch")
    upstream = pytest.importorskip("openjev_rlcd.train")
    transformers = pytest.importorskip("transformers")
    from decisionforge.training.backends.openjev import stage_arguments
    from decisionforge.training.backends.openjev_stage import run_stage
    from decisionforge.training.backends.openjev_task import public_item
    from decisionforge.training.config import TrainConfig

    class Tokenizer:
        eos_token_id, pad_token_id = 2, 0

        def encode(self, text, **kw):
            if text.startswith(" "):
                value = 0
                for char in text.strip():
                    value = value * 26 + ord(char) - 64
                return [9 + value]
            return [1, 3]

        def decode(self, tokens):
            return "raw rationale Answer:"

        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"]

    tok = Tokenizer()
    qwen_config = transformers.Qwen3Config(
        vocab_size=2048,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
        head_dim=8,
        tie_word_embeddings=False,
    )
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: tok)

    def load_model(*args, **kw):
        return transformers.Qwen3ForCausalLM(qwen_config).to(kw["dtype"])

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", load_model)
    monkeypatch.setattr(upstream, "DEVICE", "cpu")
    # Replace only expensive autoregressive sampling, not the official reader,
    # estimator/loss/gradients/model/AdamW/train-loop/evaluation/checkpoint loading.
    monkeypatch.setattr(
        upstream.Rationales,
        "sample",
        lambda self, model, items, m, max_new_tokens: [
            [([1, 3], [4, 5], True) for _ in range(m)] for item in items
        ],
    )
    public = {
        "state": "case",
        "questions": {"decision": {"type": "choice", "criteria": {f"intent_{i}": str(i) for i in range(k)}}},
    }
    from decisionforge.label_tokens import allocate_labels
    protocol = allocate_labels(tok, max(10, k), encoding_style="space_prefixed", legacy_count=10)
    values = [0.8] + [0.2 / (k - 1)] * (k - 1)
    item = {**public_item(public, "decision", labels=protocol["labels"]), "q": values, "counts": values}
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({role: [item] for role in ("train", "dev", "test")}))
    c = TrainConfig(
        backend="openjev",
        data={"path": "unused"},
        training={"max_steps": 2, "batch_size": 1, "learning_rate": 0.01},
        openjev={"m": 2, "max_new_tokens": 4},
    )
    previous = None
    import decisionforge.training.backends.openjev as backend

    for stage in ("calibrate", "reinforce"):
        root = tmp_path / stage
        root.mkdir()
        args = stage_arguments(c, stage, "/toy", root, previous)
        plan = {
            "argv": args,
            "splits": str(splits),
            "evidence": str(root / "updates.json"),
            "max_seq_length": 128,
            "label_protocol": protocol,
            "expected_init_sha256": backend.inventory(previous.parent, [previous])[previous.name][
                "sha256"
            ]
            if previous
            else None,
        }
        argv_before = sys.argv
        run_stage(plan)
        assert sys.argv is argv_before
        evidence = json.loads((root / "updates.json").read_text())
        assert evidence["actual_optimizer_steps"] == 2 and evidence["optimizer_state_steps"] == [2]
        assert evidence["parameter_updates"]["max_abs_delta"] > 0
        previous = root / "weights.pt"
        assert torch.load(previous, map_location="cpu", weights_only=True)
    # Actual exported-state reconstruction API used by OpenJevRuntime, on CPU.
    reloaded = transformers.AutoModelForCausalLM.from_config(
        qwen_config, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    reloaded.load_state_dict(
        torch.load(previous, map_location="cpu", weights_only=True), strict=True
    )
    reader = upstream.Rationales(tok, "dforge_v3")
    u, _ = reader.forward(reloaded, [([1, 3], [4, 5], True)], grad=False, k=k)
    assert u.shape == (1, max(10, k)) and torch.allclose(u.sum(-1), torch.ones(1), atol=1e-6)
    assert not torch.cuda.is_initialized()
