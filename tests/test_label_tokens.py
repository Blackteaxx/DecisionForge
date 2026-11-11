"""Label protocol tests catch truncation, tokenizer collisions and encoding drift."""

import copy
import json
import string

import pytest


class AlphabetTokenizer:
    all_special_ids = [0]

    def encode(self, text, add_special_tokens=False):
        # AA deliberately needs two tokens, AB collides with A; both must be skipped.
        text = text.removeprefix(" ")
        if text == "AA":
            return [1, 1]
        if text == "AB":
            return [1]
        if not text or any(c not in string.ascii_uppercase for c in text):
            return [0, 0]
        value = 0
        for char in text:
            value = value * 26 + ord(char) - ord("A") + 1
        return [value]


@pytest.mark.parametrize("size", [26, 77, 151])
@pytest.mark.parametrize("style", ["space_prefixed", "bare_then_space"])
def test_complete_unique_single_token_catalog_and_stable_prefix(size, style):
    from decisionforge.label_tokens import allocate_labels, validate_label_protocol

    protocol = allocate_labels(AlphabetTokenizer(), size, encoding_style=style)
    assert len(protocol["labels"]) == len(set(protocol["token_ids"])) == size
    assert protocol["labels"][:26] == list(string.ascii_uppercase)
    assert protocol["token_ids"][:26] == list(range(1, 27))
    if size > 26:
        assert protocol["labels"][26] == "AC"
    assert protocol["encodings"][0] == (" A" if style == "space_prefixed" else "A")
    assert (
        validate_label_protocol(AlphabetTokenizer(), json.loads(json.dumps(protocol))) == protocol
    )
    assert (
        allocate_labels(AlphabetTokenizer(), size + 1, encoding_style=style)["labels"][:size]
        == protocol["labels"]
    )


def test_openjev_legacy_ten_and_tokenizer_revalidation():
    from decisionforge.label_tokens import allocate_labels, validate_label_protocol

    protocol = allocate_labels(AlphabetTokenizer(), 151, legacy_count=10)
    assert protocol["labels"][:10] == list("ABCDEFGHIJ")
    assert protocol["legacy_count"] == 10
    corrupted = copy.deepcopy(protocol)
    corrupted["token_ids"][-1] = 999
    with pytest.raises(ValueError):
        validate_label_protocol(AlphabetTokenizer(), corrupted)

    class Drifted(AlphabetTokenizer):
        def encode(self, text, **kwargs):
            return [999] if text.strip() == "A" else super().encode(text, **kwargs)

    with pytest.raises(ValueError):
        validate_label_protocol(Drifted(), protocol)


@pytest.mark.parametrize("broken", ["multitoken", "collision", "special"])
def test_legacy_labels_cannot_be_silently_replaced(broken):
    from decisionforge.label_tokens import allocate_labels

    class Broken(AlphabetTokenizer):
        def encode(self, text, **kwargs):
            if text.strip() == "B":
                return {"multitoken": [1, 2], "collision": [1], "special": [0]}[broken]
            return super().encode(text, **kwargs)

    with pytest.raises(ValueError, match="legacy"):
        allocate_labels(Broken(), 151)


def test_insufficient_tokenizer_fails_without_returning_short_catalog():
    from decisionforge.label_tokens import allocate_labels

    class Tiny(AlphabetTokenizer):
        def encode(self, text, **kwargs):
            if len(text.strip()) > 1:
                return [1, 2]
            return super().encode(text, **kwargs)

    with pytest.raises(ValueError, match="151"):
        allocate_labels(Tiny(), 151)


def test_shared_label_implementation_is_part_of_inference_resume_identity():
    from decisionforge.eval.source_identity import inference_sources, source_versions

    versions = source_versions()
    assert "label_tokens.py" in versions
    assert inference_sources(versions)["label_tokens.py"] == versions["label_tokens.py"]
