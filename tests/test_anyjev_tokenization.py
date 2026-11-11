"""Request caches may remove repeated encoding, never change prompts or batch tensors."""

from types import SimpleNamespace

import pytest


class Tokenizer:
    is_fast = True
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self):
        self.model_input_names = list(type(self).model_input_names)
        self.encoded = []
        self.batches = []

    def encode(self, text, **kwargs):
        self.encoded.append((text, kwargs))
        return [ord(char) for char in text]

    def __call__(self, texts, **kwargs):
        self.batches.append((texts, kwargs))
        return {"original": texts}

    def pad(self, values, **kwargs):
        rows = values["input_ids"]
        size = max(map(len, rows))
        return {
            "input_ids": [[0] * (size - len(row)) + row for row in rows],
            "attention_mask": [[0] * (size - len(row)) + [1] * len(row) for row in rows],
        }


def test_same_prompt_reuses_encode_for_lengths_and_preserves_batch_order():
    from decisionforge.training_free.anyjev.tokenization import RequestTokenizer

    tokenizer = Tokenizer()
    cached = RequestTokenizer(tokenizer)
    original = cached.encode("ab", add_special_tokens=False)
    original.append(999)
    assert cached.encode("ab", add_special_tokens=False) == [97, 98]
    assert cached.encode("c", add_special_tokens=False) == [99]
    batch = cached(["c", "ab"], return_tensors="pt", padding=True, add_special_tokens=False)
    assert batch == {"input_ids": [[0, 99], [97, 98]], "attention_mask": [[0, 1], [1, 1]]}
    assert tokenizer.encoded == [
        ("ab", {"add_special_tokens": False}),
        ("c", {"add_special_tokens": False}),
    ]
    assert tokenizer.batches == []


@pytest.mark.parametrize("extra", [{"truncation": True}, {"return_token_type_ids": True}])
def test_unknown_batch_kwargs_fall_back_to_original_tokenizer(extra):
    from decisionforge.training_free.anyjev.tokenization import RequestTokenizer

    tokenizer = Tokenizer()
    cached = RequestTokenizer(tokenizer)
    cached.encode("ab", add_special_tokens=False)
    assert cached(["ab"], return_tensors="pt", padding=True, add_special_tokens=False, **extra) == {
        "original": ["ab"]
    }
    assert len(tokenizer.batches) == 1


def test_unknown_extra_model_fields_disable_cache():
    from decisionforge.training_free.anyjev.tokenization import RequestTokenizer

    tokenizer = Tokenizer()
    tokenizer.model_input_names.append("token_type_ids")
    cached = RequestTokenizer(tokenizer)
    cached.encode("ab", add_special_tokens=False)
    cached.encode("ab", add_special_tokens=False)
    assert len(tokenizer.encoded) == 2
    assert cached(["ab"], return_tensors="pt", padding=True, add_special_tokens=False) == {
        "original": ["ab"]
    }


@pytest.mark.parametrize("max_entries,max_tokens", [(1, 100), (10, 2)])
def test_cache_capacity_is_bounded_and_missing_batches_fall_back(max_entries, max_tokens):
    from decisionforge.training_free.anyjev.tokenization import RequestTokenizer

    tokenizer = Tokenizer()
    cached = RequestTokenizer(tokenizer, max_entries=max_entries, max_tokens=max_tokens)
    cached.encode("ab", add_special_tokens=False)
    cached.encode("c", add_special_tokens=False)
    cached.encode("ab", add_special_tokens=False)
    cached.encode("c", add_special_tokens=False)
    assert [text for text, _ in tokenizer.encoded] == ["ab", "c", "c"]
    assert cached(["c"], return_tensors="pt", padding=True, add_special_tokens=False) == {
        "original": ["c"]
    }


@pytest.mark.parametrize("enabled", [True, False])
def test_tokenizer_scope_restores_objects_and_clears_cache_after_exception(enabled):
    from decisionforge.training_free.anyjev.tokenization import cached_tokenization

    tokenizer = Tokenizer()
    upstream = SimpleNamespace(tokenizer=tokenizer)
    bounded = SimpleNamespace(tokenizer=tokenizer)
    with pytest.raises(RuntimeError):
        with cached_tokenization(upstream, bounded, enabled=enabled) as cached:
            assert upstream.tokenizer is bounded.tokenizer
            assert (upstream.tokenizer is tokenizer) is (not enabled)
            upstream.tokenizer.encode("ab", add_special_tokens=False)
            raise RuntimeError("fixture backend failed")
    assert upstream.tokenizer is bounded.tokenizer is tokenizer
    if enabled:
        assert cached.encode("ab", add_special_tokens=False) == [97, 98]
        assert len(tokenizer.encoded) == 2
