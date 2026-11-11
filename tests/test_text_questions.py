"""Offline routing/grounding contracts; doubles do not measure model semantic quality."""

import pytest
from conftest import FixtureProvider
from test_pipeline import run, source

from decisionforge.util import BudgetExhausted

MATERIAL = (
    "Two reserve units each supply 2 MW. The dispatch rule is to request their combined "
    "capacity when both are available. Both units are available today."
)
QUESTION = "Both 2 MW reserve units are available. What total capacity should dispatch request?"


class NarrativeProvider(FixtureProvider):
    def __init__(self, modes=("choice",), evidence=MATERIAL, abstain=False, reject_source=False):
        super().__init__()
        self.modes, self.evidence = modes, evidence
        self.abstain, self.reject_source = abstain, reject_source

    async def call(self, role, payload, schema, record, stage):
        if role == "text_assessor":
            self.calls.append((role, payload, stage))
            return schema(
                is_narrative=True,
                opportunities=[
                    {
                        "mode": mode,
                        "focus": "Select the dispatch capacity",
                        "evidence": [self.evidence],
                        "reason": "The stated rule and unit capacities determine a unique decision.",
                    }
                    for mode in self.modes
                ],
                reason="Grounded dispatch decisions" if self.modes else "No decision evidence",
            )
        if role == "qa_constructor":
            self.calls.append((role, payload, stage))
            return schema(
                qa=None if self.abstain else {"question": QUESTION, "answer": "4"},
                reason="Insufficient support" if self.abstain else "Two units times 2 MW = 4 MW.",
            )
        if role == "checker" and self.reject_source:
            self.calls.append((role, payload, stage))
            return schema(
                passed=False,
                reason="Constructed answer is not supported by the original material",
                failed_fields=["answer"],
                repair_instructions=["Review original evidence"],
                modules=["source"],
            )
        return await super().call(role, payload, schema, record, stage)


@pytest.mark.parametrize("mode", ["choice", "noul", "score", "auto"])
async def test_narrative_builds_grounded_qa_for_requested_mode(config, mode):
    source(config, MATERIAL, ".txt")
    config.task.modes = [mode]
    config.task.domain_profile = "power_grid"
    selected = "choice" if mode == "auto" else mode
    result, files, provider = await run(config, NarrativeProvider(modes=[selected]))
    assert result["accepted"] == 1
    assert [c[0] for c in provider.calls] == [
        "extractor",
        "text_assessor",
        "qa_constructor",
        selected,
        "checker",
    ]
    assessment = provider.calls[1][1]
    assert assessment["requested_modes"] == [mode]
    assert assessment["domain"] == "power_grid"
    assert assessment["target_option_count"] == config.choice.target_option_count
    assert assessment["difficulty"] == config.task.difficulty
    record = files["accepted"][0]
    assert record["request"]["questions"]["q1"]["type"] == selected
    assert record["request"]["state"] == MATERIAL
    normalized = files["meta"]["records"]["1"]["normalized"]
    assert normalized["question"] == QUESTION and QUESTION not in MATERIAL
    assert normalized["text_origin"]["evidence"] == [MATERIAL]
    checker = provider.calls[-1][1]
    assert checker["original_input"]["text"] == MATERIAL
    assert checker["source"]["text_origin"] == normalized["text_origin"]
    assert checker["requested_modes"] == [mode]
    assert "text_origin" not in record


async def test_multiple_modes_do_not_cross_expand_constructed_questions(config):
    source(config, MATERIAL, ".txt")
    config.task.modes = ["choice", "score"]
    result, files, _ = await run(config, NarrativeProvider(modes=["choice", "score"]))
    assert result["accepted"] == 2
    assert result["modes"] == {"choice": 1, "score": 1}
    assert len({r["group_id"] for r in files["accepted"]}) == 1


@pytest.mark.parametrize("enabled", [False, True])
async def test_no_worthwhile_question_or_disabled_fallback(config, enabled):
    source(config, "Electricity is important. People have many opinions about it.", ".txt")
    config.input.generate_qa_from_text = enabled
    result, files, provider = await run(config, NarrativeProvider(modes=[]))
    assert result["accepted"] == 0 and result["input_failures"] == 1
    assert files["accepted"] == files["fail"] == []
    assert [c[0] for c in provider.calls] == (
        ["extractor", "text_assessor"] if enabled else ["extractor"]
    )
    if enabled:
        assert "No decision evidence" in files["meta"]["records"]["1"]["reason"]


async def test_unanswered_question_does_not_become_narrative(config):
    source(config, "Which relay should be replaced?", ".txt")

    class NotNarrative(NarrativeProvider):
        async def call(self, role, payload, schema, record, stage):
            if role == "text_assessor":
                self.calls.append((role, payload, stage))
                return schema(is_narrative=False, opportunities=[], reason="Unanswered question")
            return await super().call(role, payload, schema, record, stage)

    result, _, provider = await run(config, NotNarrative())
    assert result["input_failures"] == 1
    assert "qa_constructor" not in [c[0] for c in provider.calls]


@pytest.mark.parametrize(
    "modes,evidence,limit,reason",
    [
        (["score"], MATERIAL, 5, "unrequested mode"),
        (["choice"], "Invented dispatch rule", 5, "exact nonempty source spans"),
        (["choice"], " ", 5, "exact nonempty source spans"),
        (["choice", "choice"], MATERIAL, 1, "max_generated_questions"),
    ],
)
async def test_invalid_assessment_fails_before_construction(config, modes, evidence, limit, reason):
    source(config, MATERIAL, ".txt")
    config.input.max_generated_questions = limit
    result, files, provider = await run(config, NarrativeProvider(modes, evidence))
    assert result["input_failures"] == 1
    assert reason in files["meta"]["records"]["1"]["reason"]
    assert [c[0] for c in provider.calls] == ["extractor", "text_assessor"]


async def test_constructor_can_abstain(config):
    source(config, MATERIAL, ".txt")
    result, files, provider = await run(config, NarrativeProvider(abstain=True))
    assert result["input_failures"] == 1 and files["fail"] == []
    assert "Insufficient support" in files["meta"]["records"]["1"]["reason"]
    assert [c[0] for c in provider.calls] == ["extractor", "text_assessor", "qa_constructor"]


async def test_checker_rejects_unsupported_constructed_qa_without_rewriting_it(config):
    source(config, MATERIAL, ".txt")
    config.verification.max_semantic_repairs = 2
    result, files, provider = await run(config, NarrativeProvider(reject_source=True))
    assert result["accepted"] == 0 and result["failed"] == 1
    assert files["meta"]["records"]["1"]["source"]["text_origin"]["evidence"] == [MATERIAL]
    assert [c[0] for c in provider.calls].count("checker") == 1
    assert [c[0] for c in provider.calls].count("qa_constructor") == 1


async def test_evidence_is_retained_when_context_extraction_is_disabled(config):
    source(config, MATERIAL, ".txt")
    config.input.extract_context = False
    result, files, _ = await run(config, NarrativeProvider())
    assert result["accepted"] == 1
    assert files["accepted"][0]["request"]["state"] == ""
    assert files["meta"]["records"]["1"]["normalized"]["text_origin"]["evidence"] == [MATERIAL]


async def test_json_narrative_does_not_enable_txt_construction(config):
    source(config, {"text": MATERIAL})
    result, _, provider = await run(config, NarrativeProvider())
    assert result["input_failures"] == 1
    assert [c[0] for c in provider.calls] == ["extractor"]


async def test_construction_budget_interruption_can_resume(config):
    source(config, MATERIAL, ".txt")

    class Interrupted(NarrativeProvider):
        async def call(self, role, *args):
            if role == "qa_constructor":
                raise BudgetExhausted("injected construction interruption")
            return await super().call(role, *args)

    result, _, _ = await run(config, Interrupted())
    assert result["status"] == "budget_exhausted"
    result, files, _ = await run(config, NarrativeProvider())
    assert result["accepted"] == 1
    result, again, provider = await run(config, NarrativeProvider())
    assert result["accepted"] == 1 and again["accepted"] == files["accepted"]
    assert provider.calls == []


def test_text_roles_use_configured_models(config):
    config.llm.analyzer_model = "analyzer"
    config.llm.generator_model = "generator"
    assert config.llm.model_for("text_assessor") == "analyzer"
    assert config.llm.model_for("qa_constructor") == "generator"
