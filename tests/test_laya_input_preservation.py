"""Optional pinned-upstream/tokenizer checks; no model weights or CUDA allocation."""

import copy
import json
import os
from importlib.metadata import distribution
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def upstream():
    api = pytest.importorskip("laya.train")
    common = pytest.importorskip("laya.common")
    transformers = pytest.importorskip("transformers")
    from decisionforge.training.backends.laya import UPSTREAM_REVISION

    direct = json.loads(distribution("laya").read_text("direct_url.json"))
    assert direct["vcs_info"]["commit_id"] == UPSTREAM_REVISION
    path = os.environ.get("DFORGE_LAYA_TOKENIZER")
    if not path:
        pytest.skip("set DFORGE_LAYA_TOKENIZER to the cached official Laya tokenizer directory")
    tok = transformers.AutoTokenizer.from_pretrained(path, local_files_only=True)
    return tok, api, common


def public_case(case, mask):
    public = {
        "state": {"text": "ordinary evidence"},
        "questions": {
            "decision": {
                "type": "choice",
                "instructions": {"task": "judge"},
                "criteria": {"B": {"text": "bee"}, "A": ["aye"]},
            }
        },
    }
    q = public["questions"]["decision"]
    if case == "state_value":
        public["state"] = [{"text": ["literal " + mask]}]
    elif case == "state_key":
        public["state"] = {mask: "evidence"}
    elif case == "instructions":
        q["instructions"] = "judge " + mask
    elif case == "instruction_structure":
        q["instructions"] = {mask: ["judge"]}
    elif case == "choice_key":
        q["criteria"] = {"odd:" + mask: None, "A": "aye"}
    elif case == "choice_description":
        q["criteria"]["B"] = {"text": [mask]}
    elif case == "choice_structure_key":
        q["criteria"]["B"] = {mask: "bee"}
    elif case == "score":
        q.update(type="score", criteria=[{"level": [mask]}, "high"])
    elif case in ("noul_false", "noul_true"):
        q.update(type="noul", criteria={case[5:]: {"text": mask}})
    return public


CASES = [
    ("state_value", "state"),
    ("state_key", "state"),
    ("instructions", "instructions"),
    ("instruction_structure", "instructions"),
    ("choice_key", "criteria"),
    ("choice_description", "criteria"),
    ("choice_structure_key", "criteria"),
    ("score", "criteria"),
    ("noul_false", "criteria"),
    ("noul_true", "criteria"),
]


@pytest.mark.parametrize("case,field", CASES)
def test_actual_tokenizer_rejects_mask_before_upstream_can_erase_it(upstream, case, field):
    from decisionforge.training.backends.laya import input_loss

    tok, api, common = upstream
    public = public_case(case, tok.mask_token)
    q = public["questions"]["decision"]
    before = copy.deepcopy(public)
    # Characterize the surprising upstream sanitization with the actual cached tokenizer.
    sanitized = public_case(case, " ")
    actual = common.build_sequence(tok, public["state"], api.to_internal("decision", q))
    clean = common.build_sequence(
        tok, sanitized["state"], api.to_internal("decision", sanitized["questions"]["decision"])
    )
    assert actual == clean
    with pytest.raises(ValueError, match=rf"question=decision.*field=.*{field}.*mask_token"):
        input_loss(tok, public["state"], "decision", q, api, common, 512, 192)
    assert public == before


@pytest.mark.parametrize("case,field", CASES)
@pytest.mark.parametrize("policy", ["error", "allow"])
def test_training_rejects_reserved_text_even_when_truncation_allowed(upstream, case, field, policy):
    from test_training_data import example

    from decisionforge.models import TrainingExample
    from decisionforge.training.backends.laya import LayaBackend, encode_roles
    from decisionforge.training.config import TrainConfig
    from decisionforge.training.data import PreparedData

    tok, api, common = upstream
    raw = example()
    public = public_case(case, tok.mask_token)
    q = public["questions"]["decision"]
    raw["request"]["state"] = public["state"]
    raw["request"]["questions"] = {"decision": q}
    if q["type"] == "choice":
        keys = list(q["criteria"])
        answer = {
            "type": "choice",
            "choice": keys[0],
            "confidence": 0.4,
            "probabilities": dict(zip(keys, [0.7, 0.3])),
        }
    elif q["type"] == "score":
        answer = {
            "type": "score",
            "score": 0.4,
            "confidence": 0.4,
            "probabilities": {"0": 0.6, "1": 0.4},
            "legend": {str(i): value for i, value in enumerate(q["criteria"])},
        }
    else:
        answer = {"type": "noul", "noul": 0.2}
    # Reuse validated answer confidence values rather than invent supervision.
    if q["type"] in ("choice", "score"):
        source = raw["supervision"]["answers"]["decision" if q["type"] == "choice" else "rating"]
        answer["confidence"] = source["confidence"]
    raw["supervision"]["answers"] = {"decision": answer}
    record = TrainingExample.model_validate(raw)
    before = record.model_dump()
    data = PreparedData([record], [{"record_id": 1, "role": "train"}], {})
    config = TrainConfig(
        backend="laya",
        data={"path": "unused", "on_truncated": policy},
        calibration={"enabled": False},
    )
    backend = LayaBackend(config, ".")
    backend.tok, backend.api, backend.common = tok, api, common
    with pytest.raises(
        ValueError, match=rf"record_id=1.*question=decision.*field=.*{field}.*mask_token"
    ):
        encode_roles(data, tok, api, config, backend._truncation)
    assert record.model_dump() == before


@pytest.mark.parametrize("case,field", CASES)
def test_eval_rejects_reserved_text_before_model_call_and_preserves_raw(upstream, case, field):
    from decisionforge.eval.adapters.laya_worker import LayaRuntime

    tok, _api, _common = upstream
    runtime = LayaRuntime.__new__(LayaRuntime)
    runtime.agent = SimpleNamespace(tok=tok, cfg={}, cpu_fallback_count=0)
    public = public_case(case, tok.mask_token)
    before = copy.deepcopy(public)
    with pytest.raises(ValueError, match=rf"question=decision.*field=.*{field}.*mask_token"):
        runtime.predict(public)
    assert public == before


def test_ordinary_fixture_encoding_and_soft_targets_unchanged_with_metadata_only_mask(upstream):
    from decisionforge.training.backends.laya import LayaBackend, encode_roles
    from decisionforge.training.config import load_config
    from decisionforge.training.data import prepare_data, to_unsloth_row

    tok, api, common = upstream
    config = load_config("examples/training/laya/ddp.yaml")
    data = prepare_data(config)
    # Question identifiers and role/source bookkeeping are not rendered model text.
    for record in data.records:
        record.request.questions["bookkeeping" + tok.mask_token] = record.request.questions.pop(
            next(iter(record.request.questions))
        )
        record.supervision.answers["bookkeeping" + tok.mask_token] = record.supervision.answers.pop(
            next(iter(record.supervision.answers))
        )
    backend = LayaBackend(config, ".")
    backend.tok, backend.api, backend.common = tok, api, common
    before = [record.model_dump() for record in data.records]
    items, report = encode_roles(data, tok, api, config, backend._truncation)
    for role, records in data.roles.items():
        expected, skipped = api.items_from_rows(
            tok,
            [to_unsloth_row(r) for r in records],
            max_len=config.model.max_seq_length,
            head_max_len=config.laya.head_max_length,
        )
        assert not skipped
        assert items[role] == expected
        assert report[role]["truncated"] == []
    assert [record.model_dump() for record in data.records] == before
    import torch

    assert not torch.cuda.is_initialized()
