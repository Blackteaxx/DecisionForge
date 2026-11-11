"""Optional source-pinned CPU algorithm/API checks; these are not GPU acceptance."""

import contextlib
import json
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("world_size", [1, 2])
def test_official_loop_updates_encoder_head_and_counts_real_steps(
    tmp_path, monkeypatch, world_size
):
    torch = pytest.importorskip("torch")
    api = pytest.importorskip("laya.train")
    from decisionforge.training.backends.laya import LayaBackend
    from decisionforge.training.config import TrainConfig

    class Encoder(torch.nn.Linear):
        def __init__(self):
            super().__init__(2, 2)
            self.config = SimpleNamespace(
                save_pretrained=lambda path: (
                    path_mkdir(path),
                    (tmp_path / "checkpoint_latest/encoder/config.json").write_text("{}"),
                )
            )

    def path_mkdir(path):
        from pathlib import Path

        Path(path).mkdir(parents=True, exist_ok=True)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder, self.head = Encoder(), torch.nn.Linear(2, 2)

        def forward(self, ids, attention, positions, mask, qtype, **kwargs):
            return self.head(self.encoder(ids.float())), None

    def collate(chunks, pad):
        items = chunks[0]
        return {
            "input_ids": torch.tensor([[1.0, 0.5]] * len(items)),
            "attention_mask": torch.ones(len(items), 2),
            "marker_pos": torch.zeros(len(items), 2, dtype=torch.long),
            "marker_mask": torch.ones(len(items), 2, dtype=torch.bool),
            "qtype": torch.zeros(len(items), dtype=torch.long),
            "target": torch.tensor([[0.8, 0.2]] * len(items)),
        }

    # Only tokenization/collation and large model construction are replaced. The actual
    # pinned loop, loss/rewards, AdamW, partial accumulation and weight saving execute.
    monkeypatch.setattr(api, "encode_item", lambda tok, item, *args: item)
    monkeypatch.setattr(api, "collate_items", collate)
    events = []
    if world_size == 2:

        class DDP(torch.nn.Module):
            def __init__(self, model, *, device_ids, find_unused_parameters):
                super().__init__()
                assert device_ids == [0] and find_unused_parameters
                self.module = model

            def forward(self, *args):
                return self.module(*args)

        monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", DDP)
        monkeypatch.setattr(torch.distributed, "barrier", lambda: events.append("barrier"))
        monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor: tensor.mul_(2))
        monkeypatch.setattr(
            torch.distributed,
            "all_gather_object",
            lambda output, item: output.__setitem__(slice(None), [item, item]),
        )
        monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: contextlib.nullcontext())
        scaler = torch.amp.GradScaler
        monkeypatch.setattr(
            torch.amp, "GradScaler", lambda *args, **kwargs: scaler("cpu", enabled=False)
        )
    saver = api.save_checkpoint

    def save(*args):
        events.append("save")
        return saver(*args)

    monkeypatch.setattr(api, "save_checkpoint", save)
    c = TrainConfig(
        backend="laya",
        data={"path": "unused"},
        distributed={"strategy": "ddp" if world_size == 2 else "single", "world_size": world_size},
        training={
            "num_train_epochs": 2.0,
            "batch_size": 2,
            "gradient_accumulation_steps": 2,
            "gradient_checkpointing": False,
            "learning_rate": 0.01,
            "head_learning_rate": 0.02,
        },
    )
    b = LayaBackend(c, tmp_path)
    b.torch, b.api, b.device = torch, api, torch.device("cpu")
    b.common = SimpleNamespace(collate_items=collate)
    b.model = Model()
    b.tok = SimpleNamespace(pad_token_id=0, save_pretrained=lambda path: path_mkdir(path))
    b.cfg = {"temperature": [1, 1, 1]}
    b.train_items = [{"q": {"t": "choice"}, "k": 2, "target": [0.8, 0.2]}] * 5
    b.partition = {"retained_items": 5}
    b.before = {
        group: {
            n: p.detach().flatten()[:256].float().clone()
            for n, p in b.model.named_parameters()
            if n.startswith("encoder.") == enc
        }
        for group, enc in (("encoder", True), ("head", False))
    }
    result = b.train()
    assert result["optimizer_updates"] == 4
    assert result["actual_optimizer_steps"] == 4
    assert result["scheduler_T_max"] == (4 if world_size == 1 else 2)
    assert result["parameter_updates"]["encoder"]["max_abs_delta"] > 0
    assert result["parameter_updates"]["head"]["max_abs_delta"] > 0
    assert (tmp_path / "checkpoint_latest/model.safetensors").is_file()
    assert (
        json.loads((tmp_path / "checkpoint_latest/checkpoint_meta.json").read_text())["epoch"] == 2
    )
    assert events == (["save", "save"] if world_size == 1 else ["barrier", "save", "barrier"] * 2)


def test_upstream_structured_fields_survive_and_head_loss_is_detected():
    common = pytest.importorskip("laya.common")
    api = pytest.importorskip("laya.train")
    from decisionforge.training.backends.laya import input_loss

    state = [{"role": "user", "content": "read", "extra": {"routing": "keep this"}}]
    assert json.loads(common.serialize_state(state)) == state
    question = {
        "type": "choice",
        "instructions": {"task": "judge", "extra": ["preserve"]},
        "criteria": {
            "A": {"desc": "short", "extra": "included"},
            "B": ["other", {"field": "also kept"}],
        },
    }
    internal = api.to_internal("decision", question)
    assert json.loads(internal["ins"]) == question["instructions"]
    options = common.render_options(internal)
    assert json.loads(options[0].split(": ", 1)[1]) == question["criteria"]["A"]
    assert json.loads(options[1].split(": ", 1)[1]) == question["criteria"]["B"]

    class Tokenizer:
        mask_token, mask_token_id, cls_token_id, sep_token_id = "[MASK]", -1, -2, -3

        def __call__(self, text, **kwargs):
            ids = list(range(len(text.split())))
            return {"input_ids": ids[: kwargs["max_length"]] if kwargs.get("truncation") else ids}

    assert input_loss(Tokenizer(), state, "decision", question, api, common, 512, 192) is None
    question["criteria"]["A"]["extra"] = " ".join(["word"] * 100)
    loss = input_loss(Tokenizer(), state, "decision", question, api, common, 512, 192)
    assert loss["option_tokens_dropped"]["A"] > 0
    question["instructions"] = " ".join(["instruction"] * 500)
    assert (
        input_loss(Tokenizer(), state, "decision", question, api, common, 512, 192)[
            "instruction_tokens_dropped"
        ]
        > 0
    )
