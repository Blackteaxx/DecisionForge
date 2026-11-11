import asyncio
import json
from pathlib import Path

import pytest
from conftest import FixtureProvider

from decisionforge.config import Config
from decisionforge.exporter import validate_file
from decisionforge.generation.contracts import CheckReport
from decisionforge.generation.intake import normalize
from decisionforge.pipeline import Pipeline
from decisionforge.storage import Store


def source(config, value, suffix=".json"):
    path = Path(config.input.path).with_suffix(suffix)
    path.write_text(json.dumps(value) if suffix == ".json" else value, encoding="utf-8")
    config.input.path, config.input.type = str(path), "auto"


async def run(config, provider=None):
    provider = provider or FixtureProvider()
    directory = Path(config.output.directory)
    store = Store(directory)
    try:
        result = await Pipeline(config, store, provider).run()
    finally:
        store.close()
    files = {
        name: json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
        for name in ("accepted", "fail", "meta")
    }
    return result, files, provider


@pytest.mark.parametrize(
    "mode,extra",
    [
        ("choice", {"options": ["4", "5"]}),
        ("noul", {"statement": "Two plus two is four.", "noul_label": 1}),
        ("score", {"ranking_items": ["5", "4"], "score_index": 1}),
    ],
)
async def test_complete_input_only_calls_checker(config, mode, extra):
    config.task.modes = [mode]
    source(config, {"question": "What is 2+2?", "answer": "4", **extra})
    result, files, provider = await run(config)
    assert result["accepted"] == 1
    assert [role for role, _, _ in provider.calls] == ["checker"]
    assert files["accepted"][0]["request"]["state"] == ""
    assert set(files["accepted"][0]) == {
        "schema_version",
        "record_id",
        "group_id",
        "split",
        "request",
        "supervision",
    }
    assert validate_file(Path(config.output.directory) / "accepted.json") == 1


@pytest.mark.parametrize("mode", ["choice", "noul", "score"])
async def test_missing_components_and_coefficients(config, mode, caplog):
    caplog.set_level("INFO", logger="decisionforge.pipeline")
    source(config, {"question": "What is 2+2?", "answer": "4"})
    config.task.modes = [mode]
    config.task.difficulty = 0.9
    config.context.generate_if_missing = True
    config.context.relevance = 0.8
    result, files, provider = await run(config)
    assert result["accepted"] == 1
    assert [r for r, _, _ in provider.calls] == ["context", mode, "checker"]
    assert provider.calls[1][1]["difficulty"] == 0.9
    assert provider.calls[0][1]["relevance"] == 0.8
    assert files["accepted"][0]["request"]["state"] != "What is 2+2?"
    messages = [record.getMessage() for record in caplog.records]
    assert any("当前：背景补全" in message and "后续：" in message for message in messages)
    assert any("当前：Checker 检查（第 1 轮）" in message for message in messages)
    assert any("记录处理完成 1/1" in message for message in messages)
    assert messages[-1].endswith("当前：本次流程结束 | 后续：无")


async def test_noul_generates_true_and_false(config):
    config.task.modes, config.noul.labels = ["noul"], [1, 0]
    result, files, _ = await run(config)
    assert result["accepted"] == 2
    assert {r["supervision"]["answers"]["q1"]["noul"] for r in files["accepted"]} == {0, 1}


async def test_feedback_routes_only_failed_module(config, caplog):
    caplog.set_level("INFO", logger="decisionforge.pipeline")
    config.verification.max_semantic_repairs = 2

    class Repair(FixtureProvider):
        async def call(self, role, payload, schema, record, stage):
            self.reject = role == "checker" and stage.endswith(":0")
            return await super().call(role, payload, schema, record, stage)

    result, files, provider = await run(config, Repair())
    assert result["accepted"] == 1
    assert [r for r, _, _ in provider.calls] == ["choice", "checker", "choice", "checker"]
    repair = provider.calls[2][1]
    assert repair["feedback"]["failed_fields"] == ["criteria"]
    assert repair["current_jev"]["request"]["questions"]
    assert repair["source"]["answer"] == "4"
    assert len(files["meta"]["records"]["1"]["checks"]) == 2
    assert "第 1 次修复：Choice 选项生成" in caplog.text
    assert "Checker 检查（第 2 轮）" in caplog.text


async def test_fail_export_contains_jev_only_and_reasons_live_in_meta(config):
    config.verification.max_semantic_repairs = 1
    result, files, provider = await run(config, FixtureProvider(bad_selector=True))
    assert result["accepted"] == 0 and result["failed"] == 1
    assert len([c for c in provider.calls if c[0] == "checker"]) == 2
    assert "reason" not in files["fail"][0]
    assert files["meta"]["records"]["1"]["reason"] == "fixture rejection"
    assert validate_file(Path(config.output.directory) / "fail.json") == 1
    assert not (Path(config.output.directory) / "failures.json").exists()


async def test_missing_required_source_is_not_invented(config):
    source(config, [{"question": "No answer?"}, {"question": "Good?", "answer": "yes"}])
    result, files, _ = await run(config)
    assert result["input_failures"] == 1 and result["accepted"] == 1
    assert files["fail"] == []
    assert any(v["output"] is None for v in files["meta"]["records"].values())


@pytest.mark.parametrize("invent", [False, True])
async def test_txt_exact_extraction_and_no_invention(config, invent):
    source(config, "Question: What is 2+2?\nAnswer: 4\nContext: Add two and two.", ".txt")

    class Extract(FixtureProvider):
        async def call(self, role, payload, schema, record, stage):
            if role == "extractor":
                return schema(
                    records=[
                        {
                            "question": "What is 2+2?",
                            "answer": "5" if invent else "4",
                            "context": "Add two and two.",
                        }
                    ],
                    reason="Source spans",
                )
            return await super().call(role, payload, schema, record, stage)

    result, _, _ = await run(config, Extract())
    assert result["accepted"] == (0 if invent else 1)
    assert result["input_failures"] == int(invent)


def test_nested_mapping_and_optional_context(config):
    row = normalize({"data": {"query": "Q?", "response": "A"}}, config)
    assert row.question == "Q?" and row.answer == "A" and row.context == ""
    row = normalize(
        {"x": "Q", "y": "B", "opts": {"A": "no", "B": "yes"}},
        config,
        {"question": "x", "answer": "y", "options": "opts", "answer_encoding": "key"},
    )
    assert row.answer == "yes"
    config.input.extract_context = False
    assert normalize({"question": "Q", "answer": "A", "context": "omit"}, config).context == ""


async def test_context_equal_question_rejected_without_permission(config):
    source(
        config,
        {
            "question": "What is 2+2?",
            "answer": "4",
            "context": "What is 2+2?",
            "options": ["4", "5"],
        },
    )
    result, files, provider = await run(config)
    assert result["accepted"] == 0 and files["fail"] == []
    assert not provider.calls
    assert "context" in files["meta"]["records"]["1"]["reason"]


async def test_bounded_concurrency_and_resume(config, caplog):
    caplog.set_level("INFO", logger="decisionforge.pipeline")
    source(config, [{"question": f"Question {i}?", "answer": "4"} for i in range(12)])
    config.execution.concurrency, config.execution.llm_concurrency = 4, 2

    class Slow(FixtureProvider):
        active = 0
        peak = 0

        async def call(self, *args):
            self.active += 1
            self.peak = max(self.peak, self.active)
            try:
                await asyncio.sleep(0.01)
                return await super().call(*args)
            finally:
                self.active -= 1

    result, files, provider = await run(config, Slow())
    assert result["accepted"] == 12 and provider.peak == 2
    assert "记录处理完成 12/12" in caplog.text
    caplog.clear()
    again, saved, second = await run(config)
    assert again["accepted"] == 12 and saved["accepted"] == files["accepted"]
    assert second.calls == []
    assert "复用标准化检查点（12 条）" in caplog.text
    assert "已完成 12/12 条" in caplog.text
    assert "当前：Choice 选项生成" not in caplog.text


async def test_budget_checkpoint_resumes(config, caplog):
    caplog.set_level("INFO", logger="decisionforge.pipeline")
    result, _, _ = await run(config, FixtureProvider(fail_role="checker"))
    assert result["status"] == "budget_exhausted"
    assert "预算耗尽，生成暂停" in caplog.text
    assert "已暂停并导出已有结果" in caplog.text
    assert "本次流程结束" not in caplog.text
    result, _, _ = await run(config)
    assert result["accepted"] == 1


def test_pdf_input_is_supported_and_retired_levels_are_rejected():
    assert Config(input={"path": "data.pdf", "type": "pdf"}).input.type == "pdf"
    with pytest.raises(ValueError):
        Config(input={"path": "data.csv", "type": "csv"})
    with pytest.raises(ValueError):
        Config(input={"path": "data.json"}, harness={"level": "level1"})


def test_checker_cannot_pass_with_unresolved_issues():
    with pytest.raises(ValueError):
        CheckReport(
            passed=True, reason="bad", failed_fields=["context"], modules=[], repair_instructions=[]
        )


@pytest.mark.parametrize("mode", ["choice", "noul", "score"])
async def test_new_run_eval_contract(config, mode):
    config.task.modes = [mode]
    result, files, _ = await run(config)
    assert result["accepted"] == 1
    from decisionforge.eval.data import load_samples

    # The native evaluation reader consumes the unchanged v3 training contract.
    samples = list(load_samples(Path(config.output.directory) / "accepted.json"))
    assert samples


async def test_json_unknown_keys_extract_only_existing_spans(config):
    source(config, {"unusual_q": "What is 2+2?", "unusual_a": "4", "type": {"format": "qa"}})

    class Extract(FixtureProvider):
        async def call(self, role, payload, schema, record, stage):
            if role == "extractor":
                return schema(
                    records=[{"question": "What is 2+2?", "answer": "4", "context": ""}],
                    reason="existing",
                )
            return await super().call(role, payload, schema, record, stage)

    result, _, _ = await run(config, Extract())
    assert result["accepted"] == 1


async def test_structural_failure_repairs_context_then_still_generates_choices(config):
    from decisionforge.util import ProviderError

    source(config, {"question": "What is 2+2?", "answer": "4"})
    config.context.generate_if_missing = True
    config.verification.max_semantic_repairs = 1

    class BrokenContext(FixtureProvider):
        failed = False

        async def call(self, role, payload, schema, record, stage):
            if role == "context" and not self.failed:
                self.failed = True
                raise ProviderError("Invalid context output")
            return await super().call(role, payload, schema, record, stage)

    result, _, provider = await run(config, BrokenContext())
    assert result["accepted"] == 1
    assert [c[0] for c in provider.calls] == ["context", "choice", "checker"]


async def test_single_score_item_is_preserved(config):
    config.task.modes = ["score"]
    source(config, {"question": "What is 2+2?", "answer": "4", "ranking_items": ["4"]})
    result, files, _ = await run(config)
    assert result["accepted"] == 1
    assert "4" in files["accepted"][0]["request"]["questions"]["q1"]["criteria"]


@pytest.mark.parametrize("encoding,index", [("index0", 0), ("index1", 1)])
def test_explicit_index_encoding(config, encoding, index):
    row = normalize(
        {"question": "Q?", "answer": index, "options": ["right", "wrong"]},
        config,
        {"question": "question", "answer": "answer", "answer_encoding": encoding},
    )
    assert row.answer == "right"


def test_explicit_text_answer_does_not_silently_decode_as_key(config):
    row = normalize(
        {"question": "Q?", "answer": "A", "options": {"A": "Alpha", "B": "Beta"}},
        config,
        {"question": "question", "answer": "answer", "answer_encoding": "text"},
    )
    assert row.answer == "A"


async def test_source_question_is_not_rewritten_by_legacy_export_cleanup(config):
    question = "What does the record_id column represent?"
    source(
        config,
        {
            "question": question,
            "answer": "A row identifier",
            "options": ["A row identifier", "A password"],
        },
    )
    result, files, _ = await run(config)
    assert result["accepted"] == 1
    assert files["accepted"][0]["request"]["questions"]["q1"]["instructions"] == question


@pytest.mark.parametrize(
    "mode,extra",
    [
        ("choice", {"options": ["4", "5"]}),
        ("noul", {"statement": "Two plus two is four.", "noul_label": 1}),
        ("score", {"ranking_items": ["5", "4"], "score_index": 1}),
    ],
)
async def test_txt_preserves_complete_mode_fields_without_generation(config, mode, extra):
    item = {"question": "What is 2+2?", "answer": "4", "context": "", **extra}
    source(config, json.dumps(item), ".txt")
    config.task.modes = [mode]

    class Extract(FixtureProvider):
        async def call(self, role, payload, schema, record, stage):
            if role == "extractor":
                return schema(records=[item], reason="Explicit source components")
            return await super().call(role, payload, schema, record, stage)

    result, _, provider = await run(config, Extract())
    assert result["accepted"] == 1
    assert [c[0] for c in provider.calls] == ["checker"]


def test_option_key_a_is_not_misidentified_as_reference_answer(config):
    row = normalize(
        {"prompt": "Q?", "correct_answer": "right", "choices": {"A": "wrong", "B": "right"}}, config
    )
    assert row.answer == "right"
    from decisionforge.util import NeedsReview

    with pytest.raises(NeedsReview, match="Missing required source fields: answer"):
        normalize({"question": "Q?", "options": {"A": "wrong", "B": "right"}}, config)


def test_choice_gt_is_shuffled_with_reproducible_letter_labels(config):
    from decisionforge.generation.components import compile_record, initial_draft
    from decisionforge.models import TrainingExample

    original = normalize(
        {"question": "What is 2+2?", "answer": "4", "options": ["4", "5", "6", "7"]}, config
    )
    draft = initial_draft(original, "choice", None)
    labels = set()
    for i in range(64):
        args = (original, draft, "choice", f"source_{i}", f"child_{i}", config)
        record = compile_record(*args)
        assert compile_record(*args) == record
        example = TrainingExample.from_record(record, record_id=i + 1, group_id=i + 1)
        criteria = example.request.questions["q1"].criteria
        answer = example.supervision.answers["q1"]
        assert list(criteria) == ["A", "B", "C", "D"]
        assert criteria[answer.choice] == "4"
        assert answer.probabilities == {key: float(key == answer.choice) for key in criteria}
        labels.add(answer.choice)
    assert labels == {"A", "B", "C", "D"}
    assert draft["options"] == original.options == ["4", "5", "6", "7"]


@pytest.mark.parametrize("count,last_label", [(27, "AA"), (255, "IU")])
def test_choice_letter_labels_preserve_large_option_sets(config, count, last_label):
    from decisionforge.generation.components import compile_record, initial_draft

    options = [f"Answer {i}" for i in range(count)]
    original = normalize(
        {"question": "Which answer?", "answer": options[0], "options": options}, config
    )
    record = compile_record(
        original, initial_draft(original, "choice", None), "choice", "source", "child", config
    )
    criteria = record.request.questions["q1"].criteria
    assert len(criteria) == count
    assert list(criteria)[25:27] == ["Z", "AA"]
    assert list(criteria)[-1] == last_label
    assert set(criteria.values()) == set(options)
    assert criteria[record.supervision["q1"].value] == original.answer
