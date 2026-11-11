"""Optional actual-upstream integration; run in the pinned AnyJev environment."""

import json

import pytest

np = pytest.importorskip("numpy")
anyjev = pytest.importorskip("anyjev")
from anyjev.backends.fake import FakeBackend  # noqa: E402

from decisionforge.eval.adapters.anyjev_worker import AnyJevRuntime  # noqa: E402
from decisionforge.eval.models import Request  # noqa: E402
from decisionforge.training_free.anyjev import SETTINGS  # noqa: E402
from tests.eval_fixtures import sample  # noqa: E402


def runtime_at():
    runtime = object.__new__(AnyJevRuntime)
    runtime.backend = FakeBackend(
        lambda state, option: (
            2.0
            if state == "real" and (option == "Yes" or '"key":"B"' in option or option == '"high"')
            else 0.0
        ),
        position_bias=[2.0, 0.0, -1.0],
    )
    runtime.backend.shared_fallbacks = 0
    runtime.decider = anyjev.Decider(runtime.backend, **SETTINGS)
    runtime.question_type = anyjev.Question
    runtime.metadata = {"model": "fixture/base"}
    return runtime


@pytest.mark.parametrize("kind,permutations", [("choice", 2), ("noul", 2), ("score", 1)])
def test_actual_upstream_l0_readout_cache_and_restart(kind, permutations):
    runtime = runtime_at()
    public = Request.model_validate(sample(kind)["request"]).model_dump()
    public["state"] = "real"
    cold = runtime.predict(public)
    # A different request changes running diagnostics, never the content-free readout.
    for index in range(9):
        runtime.predict({**public, "state": f"unrelated {index}"})
    warm = runtime.predict(public)
    restarted = runtime_at().predict(public)
    assert cold["answers"] == warm["answers"] == restarted["answers"]
    raw = cold["anyjev"]
    assert raw["level"] == "L0"
    assert raw["diagnostics"]["permutations"] == permutations
    assert raw["diagnostics"]["prior_method"] == "content_free"
    assert raw["diagnostics"]["prior_strength"] == 1.0
    assert raw["diagnostics"]["batch_prior"] is None
    assert warm["anyjev"]["diagnostics"]["batch_prior"] is not None
    assert raw["workload"]["backend_calls"] == 2
    assert warm["anyjev"]["workload"]["backend_calls"] == 1
    assert raw["workload"]["flat_prompts"] == permutations * 4
    assert warm["anyjev"]["workload"]["flat_prompts"] == permutations
    assert cold.get("usage") is None
    answer = cold["answers"]["decision"]
    if kind == "noul":
        assert answer["noul"] > 0.5
        assert answer["noul"] == raw["probs"][0]
    elif kind == "choice":
        assert answer["choice"] == "B"
    else:
        assert answer["score"] > 1.0
        assert answer["legend"] == {"0": "low", "1": "middle", "2": "high"}
    json.dumps(cold, allow_nan=False)


@pytest.mark.parametrize("kind,size", [("choice", 26), ("score", 2), ("score", 10)])
def test_supported_boundaries_reach_upstream_without_option_loss(kind, size):
    public = sample(kind)["request"]
    public["questions"]["decision"]["criteria"] = (
        {f"original {i}": None for i in range(size)} if kind == "choice" else ["same"] * size
    )
    response = runtime_at().predict(public)
    assert len(response["answers"]["decision"]["probabilities"]) == size
    assert len(response["anyjev"]["probs"]) == size
    if kind == "score":
        assert response["answers"]["decision"]["legend"] == {str(i): "same" for i in range(size)}
        assert response["anyjev"]["diagnostics"]["permutations"] == 1


@pytest.mark.parametrize("size", [77, 151])
def test_large_catalog_official_l0_all_rotations_and_inverse_mapping(size):
    import math
    import re

    from anyjev.backends.fake import FakeTokenizer

    from decisionforge.training_free.anyjev.bridge import make_decider

    class CatalogTokenizer(FakeTokenizer):
        def encode(self, text, add_special_tokens=False):
            if re.fullmatch(r" ?[A-Z]{1,3}", text):
                key = text.strip()
                if key not in self._vocab:
                    self._vocab[key] = 1000 + len(self._vocab)
                return [self._vocab[key]]
            return super().encode(text, add_special_tokens)

    class CatalogBackend(FakeBackend):
        def _parse(self, prompt):
            state, _, _ = super()._parse(prompt)
            lines = [re.match(r"^([A-Z]+)\. (.*)$", line) for line in prompt.splitlines()]
            lines = [line for line in lines if line]
            return state, [line[1] for line in lines], [line[2] for line in lines]

    runtime = runtime_at()
    runtime.backend = CatalogBackend(
        lambda state, option: 2.0 if f'"key":"intent {size - 1}"' in option else 0.0,
        position_bias=[(i % 5) * 0.3 for i in range(size)],
    )
    runtime.backend.tokenizer = CatalogTokenizer()
    runtime.backend.shared_fallbacks = 0
    runtime.decider = make_decider(runtime.backend, **SETTINGS)
    public = sample()["request"]
    public["state"] = "real"
    public["questions"]["decision"]["criteria"] = {f"intent {i}": None for i in range(size)}
    response = runtime.predict(public)
    answer = response["answers"]["decision"]
    assert answer["choice"] == f"intent {size - 1}"
    assert list(answer["probabilities"]) == [f"intent {i}" for i in range(size)]
    np.testing.assert_allclose(
        list(answer["probabilities"].values()),
        [1 / (math.exp(2) + size - 1)] * (size - 1) + [math.exp(2) / (math.exp(2) + size - 1)],
        rtol=1e-12,
        atol=1e-14,
    )
    raw = response["anyjev"]
    assert raw["diagnostics"]["permutations"] == size
    from decisionforge.training_free.anyjev.diagnostics import decode_diagnostics

    assert len(decode_diagnostics(raw["diagnostics"])["perms"]) == size
    assert raw["workload"]["flat_prompts"] == size * 4
    assert len(raw["label_protocol"]["labels"]) == size
    assert len(raw["probs"]) == size
    warm = runtime.predict(public)
    assert warm["answers"] == response["answers"]
    assert warm["anyjev"]["workload"]["flat_prompts"] == size


@pytest.mark.parametrize("kind", ["choice", "score", "noul"])
def test_small_catalog_bridge_preserves_exact_official_result(kind):
    from decisionforge.training_free.anyjev.bridge import make_decider

    original = runtime_at()
    extended = runtime_at()
    extended.decider = make_decider(extended.backend, **SETTINGS)
    public = sample(kind)["request"]
    public["state"] = "real"
    expected = original.predict(public)
    actual = extended.predict(public)
    assert expected == actual
