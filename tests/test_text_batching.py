"""Offline tests of model planning, exact slices, sequential build and shared recovery."""

import json
from pathlib import Path

import pytest
import yaml
from conftest import FixtureProvider
from test_pipeline import run, source

from decisionforge.cli import inspect_input, main, signature
from decisionforge.config import Pricing, load_config
from decisionforge.generation.contracts import TextBatchPlan, TextWindowPlan
from decisionforge.generation.text_batching import (
    batch_char_limit,
    context_path,
    load_batch_context,
    planning_payload,
    planning_windows,
    prepare_text_batches,
    sha256,
    source_units,
    validate_batches,
    validate_plan,
)
from decisionforge.pipeline import Pipeline
from decisionforge.storage import Store
from decisionforge.util import BudgetExhausted, NeedsReview, json_text

NORTH = "North has two available reserve units each supplying 2 MW; dispatch requires both units to be requested together。"
SOUTH = "South has two available reserve units each supplying 2 MW; dispatch requires both units to be requested together。"
MATERIAL = NORTH + SOUTH


def setup_text(config, text=MATERIAL):
    source(config, text, ".txt")
    config.input.text_batching.threshold_chars = 180
    # Existing order/budget tests explicitly exercise the supported serial mode.
    config.execution.text_batch_concurrency = 1


class BatchProvider(FixtureProvider):
    def __init__(self, *, split=True, invalid=None, interrupt_south=False, store=None, config=None):
        super().__init__()
        self.split, self.invalid, self.interrupt_south = split, invalid, interrupt_south
        self.store, self.config = store, config

    async def call(self, role, payload, schema, record, stage):
        if self.store is not None:
            self.store.reserve(record, stage, 0.01, self.config.execution.budget_limit_usd, 24)
        if role == "extractor" and payload["source"].startswith("South") and self.interrupt_south:
            raise BudgetExhausted("interrupt at second batch")
        if role == "text_batch_planner":
            self.calls.append((role, payload, stage))
            if not self.split:
                return schema(
                    should_split=False, reason="Keep this coherent material whole", batches=[]
                )
            boundary = next(u["id"] for u in payload["units"] if u["end"] == len(NORTH))
            parts = [
                {
                    "end_unit": endpoint,
                    "questions": [{"mode": "choice", "count": 1}],
                    "reason": "Independent station decision",
                }
                for endpoint in (boundary, payload["units"][-1]["id"])
            ]
            if self.invalid == "coverage":
                parts[-1]["end_unit"] -= 1
            elif self.invalid == "order":
                parts[1]["end_unit"] = boundary
            elif self.invalid == "mode":
                parts[0]["questions"][0]["mode"] = "score"
            elif self.invalid == "count":
                parts[0]["questions"][0]["count"] = 50
            elif self.invalid == "duplicate":
                parts[0]["questions"] *= 2
            elif self.invalid == "per_mode":
                for part in parts:
                    part["questions"].append({"mode": "noul", "count": 1})
            return schema(should_split=True, reason="Two independent decisions", batches=parts)
        if role == "text_assessor":
            self.calls.append((role, payload, stage))
            opportunities = [
                {
                    "mode": "choice",
                    "focus": "Reserve dispatch",
                    "evidence": [payload["source"]],
                    "reason": "Explicit rule",
                }
            ]
            if self.invalid in {"assessment_count", "per_mode"}:
                opportunities *= 2
            return schema(is_narrative=True, opportunities=opportunities, reason="Sufficient facts")
        if role == "qa_constructor":
            self.calls.append((role, payload, stage))
            station = "South" if payload["source"].startswith("South") else "North"
            return schema(
                qa={"question": f"What combined reserve should {station} dispatch?", "answer": "4"},
                reason="Two times two",
            )
        return await super().call(role, payload, schema, record, stage)


async def test_split_is_lossless_sequential_and_preserves_settings(config, caplog):
    caplog.set_level("INFO", logger="decisionforge.pipeline")
    setup_text(config)
    unrelated = Path(config.input.path).parent / "batches" / "user_notes.txt"
    unrelated.parent.mkdir()
    unrelated.write_text("Do not include or replace this unrelated file", encoding="utf-8")
    config.task.difficulty = 0.8
    result, files, provider = await run(config, BatchProvider())
    assert result["accepted"] == 2
    assert "当前：长文本分批规划" in caplog.text
    assert "[批次 1/2] 当前：2/4 问答标准化" in caplog.text
    assert "当前：4/4 结果导出" in caplog.text
    for stage in ("原文问答提取", "材料出题评估", "基础问答构造"):
        assert f"当前：{stage}" in caplog.text
    assert [c[0] for c in provider.calls] == ["text_batch_planner"] + [
        "extractor",
        "text_assessor",
        "qa_constructor",
        "choice",
        "checker",
    ] * 2
    plan = files["meta"]["text_batching"]
    assert plan["batch_count"] == 2
    chunks = [Path(item["input_path"]).read_bytes().decode("utf-8") for item in plan["batches"]]
    assert "".join(chunks) == MATERIAL
    assert all("\n" not in chunk for chunk in chunks)
    assert unrelated.read_text(encoding="utf-8").startswith("Do not include")
    for item in plan["batches"]:
        settings = yaml.safe_load(Path(item["config_path"]).read_text(encoding="utf-8"))
        expected = config.model_dump(mode="json")
        expected["input"]["path"] = item["input_path"]
        assert settings == expected
        assert load_config(item["config_path"]).output.directory == config.output.directory
    assert len({row["group_id"] for row in files["accepted"]}) == 1
    assert [c[1]["max_questions"] for c in provider.calls if c[0] == "text_assessor"] == [1, 1]
    assert all(c[1]["difficulty"] == 0.8 for c in provider.calls if c[0] == "choice")
    assert (
        files["meta"]["records"]["1"]["source"]["text_batch"]["original_source"]
        == config.input.path
    )


async def test_no_split_decision_is_reused(config):
    setup_text(config)
    result, files, first = await run(config, BatchProvider(split=False))
    assert result["accepted"] == 1
    assert files["meta"]["text_batching"]["should_split"] is False
    assert not (Path(config.input.path).parent / "batches").exists()
    again, _, provider = await run(config, BatchProvider())
    assert again["accepted"] == 1 and not provider.calls
    assert [c[0] for c in first.calls].count("text_batch_planner") == 1


@pytest.mark.parametrize("disabled", [False, True])
async def test_short_or_disabled_input_does_not_plan(config, disabled):
    setup_text(config, NORTH if not disabled else MATERIAL)
    config.input.text_batching.enabled = not disabled
    result, _, provider = await run(config, BatchProvider())
    assert result["accepted"] == 1
    assert "text_batch_planner" not in [c[0] for c in provider.calls]


@pytest.mark.parametrize("invalid", ["coverage", "order", "mode", "count", "duplicate"])
async def test_invalid_plan_does_not_write_batches_or_generate(config, invalid):
    setup_text(config)
    provider = BatchProvider(invalid=invalid)
    with pytest.raises(NeedsReview):
        await run(config, provider)
    assert [c[0] for c in provider.calls] == ["text_batch_planner"]
    assert not (Path(config.input.path).parent / "batches").exists()


async def test_assessment_cannot_exceed_batch_quota(config):
    setup_text(config)
    result, _, provider = await run(config, BatchProvider(invalid="assessment_count"))
    assert result["accepted"] == 0 and result["input_failures"] == 2
    assert "qa_constructor" not in [c[0] for c in provider.calls]


async def test_per_mode_quota_cannot_be_reallocated_by_assessor(config):
    setup_text(config)
    config.task.modes = ["choice", "noul"]
    result, files, provider = await run(config, BatchProvider(invalid="per_mode"))
    assert result["input_failures"] == 2
    assert all("per-mode" in entry["reason"] for entry in files["meta"]["records"].values())
    assert "qa_constructor" not in [c[0] for c in provider.calls]


async def test_planner_respects_call_budget_even_with_fifty_question_setting(config):
    setup_text(config)
    config.input.max_generated_questions = 50
    provider = BatchProvider(invalid="count")
    with pytest.raises(NeedsReview, match="question/call limit"):
        await run(config, provider)
    assert provider.calls[0][1]["max_questions_per_batch"] == 22


@pytest.mark.parametrize("cached", [False, True])
async def test_oversized_batch_is_truncated_and_resumes(config, cached, caplog):
    setup_text(config)
    config.input.text_batching.threshold_chars = 100
    config.input.type = "txt"
    units = source_units(MATERIAL, 25)
    provider = BatchProvider()
    store = Store(Path(config.output.directory))
    if cached:
        plan = await provider.call(
            "text_batch_planner",
            planning_payload(MATERIAL, units, config),
            TextBatchPlan,
            "fixture",
            "text_batch_plan",
        )
        store.put(
            "text_batch_decision",
            {
                "source_sha256": sha256(MATERIAL),
                "planning_view": "full",
                "plan": plan.model_dump(),
            },
        )
        provider.calls.clear()
    try:
        result = await prepare_text_batches(config, store, provider.call)
        assert result["batch_count"] == 2
        chunks = [Path(b["input_path"]).read_bytes().decode() for b in result["batches"]]
        assert chunks == [NORTH[:100], SOUTH[:100]]
        assert [b["char_start"] for b in result["batches"]] == [0, len(NORTH)]
        assert [b["char_end"] for b in result["batches"]] == [100, len(NORTH) + 100]
        for batch in result["batches"]:
            child = load_config(batch["config_path"])
            assert load_batch_context(child).text_sha256 == batch["text_sha256"]
            assert "Code truncated batch" in batch["reason"]
        assert Path(config.input.path).read_text() == MATERIAL
        assert all(b["questions"] == [{"mode": "choice", "count": 1}] for b in result["batches"])
        assert "truncated" in caplog.text
        before = list(provider.calls)
        assert len(before) == (0 if cached else 1)
        assert await prepare_text_batches(config, store, provider.call) == result
        assert provider.calls == before
    finally:
        store.close()


def test_truncation_uses_unicode_characters_and_effective_limit(config):
    config.input.text_batching.threshold_chars = 1000
    config.llm.max_input_chars = 5000
    text = "甲🙂" * 1000
    units = source_units(text, batch_char_limit(config) // 4)
    plan = TextWindowPlan(
        reason="fixture",
        batches=[
            {"end_unit": len(units), "questions": [], "reason": "fixture"},
        ],
    )
    spans = validate_batches(plan.batches, text, units, config)
    assert spans == [(0, batch_char_limit(config))]
    assert text[slice(*spans[0])] == text[:299]
    assert plan.batches[0].end_unit == len(units)


def test_truncation_keeps_short_batches_and_original_starts(config):
    config.input.text_batching.threshold_chars = 100
    fragments = ["甲" * 100, "乙" * 135, "丙" * 70]
    text = "".join(fragments)
    units = [
        {"id": 1, "start": 0, "end": 100},
        {"id": 2, "start": 100, "end": 235},
        {"id": 3, "start": 235, "end": 305},
    ]
    plan = TextBatchPlan(
        should_split=True,
        reason="fixture",
        batches=[{"end_unit": i, "questions": [], "reason": "fixture"} for i in (1, 2, 3)],
    )
    spans = validate_plan(plan, text, units, config)
    assert spans == [(0, 100), (100, 200), (235, 305)]
    assert [text[start:end] for start, end in spans] == [f[:100] for f in fragments]


async def test_truncation_does_not_add_batches(config):
    setup_text(config)
    config.input.text_batching.threshold_chars = 100
    config.input.text_batching.max_batches = 2
    _, files, _ = await run(config, BatchProvider())
    assert files["meta"]["text_batching"]["batch_count"] == 2


async def test_budget_interruption_keeps_completed_batch_and_resumes(config):
    setup_text(config)
    first, _, _ = await run(config, BatchProvider(interrupt_south=True))
    assert first["status"] == "budget_exhausted" and first["accepted"] == 1
    result, files, second = await run(config, BatchProvider())
    assert result["accepted"] == 2 and result["status"] == "complete"
    assert [c[0] for c in second.calls] == [
        "extractor",
        "text_assessor",
        "qa_constructor",
        "choice",
        "checker",
    ]
    assert files["meta"]["active_text_batch"] is None


async def test_shared_dollar_budget_includes_planning_and_all_batches(config):
    setup_text(config)
    config.execution.budget_limit_usd = 0.065
    config.llm.pricing = {config.llm.model: Pricing(input_per_million=1, output_per_million=1)}
    directory = Path(config.output.directory)
    store = Store(directory)
    try:
        provider = BatchProvider(store=store, config=config)
        result = await Pipeline(config, store, provider).run()
        assert result["status"] == "budget_exhausted" and result["accepted"] == 1
        assert store.spent() == pytest.approx(0.06)
        config.execution.budget_limit_usd = 0.2
        provider = BatchProvider(store=store, config=config)
        result = await Pipeline(config, store, provider).run()
        assert result["accepted"] == 2
        assert store.spent() == pytest.approx(0.11)
        assert "text_batch_planner" not in [c[0] for c in provider.calls]
    finally:
        store.close()


async def test_generated_yaml_replays_without_recursive_planning_and_checks_hash(config):
    setup_text(config)
    _, files, _ = await run(config, BatchProvider())
    item = files["meta"]["text_batching"]["batches"][0]
    child = load_config(item["config_path"])
    child.output.directory = str(Path(config.output.directory).parent / "replay")
    child.input.text_batching.threshold_chars = 100
    result, _, provider = await run(child, BatchProvider())
    assert result["accepted"] == 1
    assert "text_batch_planner" not in [c[0] for c in provider.calls]
    previous = signature(child)
    metadata = context_path(Path(child.input.path))
    content = json.loads(metadata.read_bytes())
    content["reason"] = "Updated metadata"
    metadata.write_text(json_text(content), encoding="utf-8")
    assert signature(child) != previous
    Path(child.input.path).write_text("modified", encoding="utf-8")
    with pytest.raises(NeedsReview, match="modified"):
        load_batch_context(child)


async def test_resume_refuses_overwriting_edited_artifact(config):
    setup_text(config)
    _, files, _ = await run(config, BatchProvider())
    item = files["meta"]["text_batching"]["batches"][0]
    path = Path(item["config_path"])
    path.write_text("user edited content", encoding="utf-8")
    with pytest.raises(NeedsReview, match="overwrite"):
        await run(config, BatchProvider())
    assert path.read_text(encoding="utf-8") == "user edited content"


def test_units_preserve_unicode_crlf_and_no_punctuation():
    text = "甲\r\n乙。" + "🙂" * 170 + "\t\n"
    units = source_units(text, 25)
    assert "".join(u["text"] for u in units) == text
    assert all(text[u["start"] : u["end"]] == u["text"] for u in units)


def test_oversized_planner_uses_explicit_previews_for_every_unit(config):
    config.input.text_batching.threshold_chars = 2000
    config.llm.max_input_chars = 10000
    text = "A" * 20000
    units = source_units(text, 500)
    payload = planning_payload(text, units, config)
    assert payload["view"] == "head_tail_previews"
    assert len(payload["units"]) == len(units)
    assert payload["units"][-1]["end"] == len(text)
    assert all(u["omitted_middle_characters"] > 0 for u in payload["units"])
    assert len(json_text({"output_language": config.project.language, **payload})) <= 10000
    with pytest.raises(NeedsReview, match="declined"):
        validate_plan(
            TextBatchPlan(should_split=False, reason="No", batches=[]), text, units, config
        )


def test_inspect_only_reports_threshold_without_creating_files(config):
    setup_text(config)
    result = inspect_input(config)
    assert result["records"] == 1
    assert result["text_batching"]["requires_model_assessment"] is True
    assert not (Path(config.input.path).parent / "batches").exists()
    config.input.text_batching.threshold_chars = len(MATERIAL)
    assert inspect_input(config)["text_batching"]["requires_model_assessment"] is False


async def test_json_is_never_sent_to_batch_planner(config):
    store = Store(Path(config.output.directory))
    try:
        assert await prepare_text_batches(config, store, BatchProvider().call) is None
    finally:
        store.close()


def window_response(payload, schema):
    batches, start = [], 0
    previous = None
    for unit in payload["units"]:
        if unit["end"] - start > payload["max_batch_chars"]:
            batches.append(
                {
                    "end_unit": previous["id"],
                    "questions": [{"mode": "choice", "count": 1}],
                    "reason": "Fixture boundary",
                }
            )
            start = previous["end"]
        previous = unit
    batches.append(
        {
            "end_unit": previous["id"],
            "questions": [{"mode": "choice", "count": 1}],
            "reason": "Fixture final boundary",
        }
    )
    return schema(reason="Offline model-plan fixture", batches=batches)


async def test_windowed_planning_is_lossless_and_resumes_partial_plan(config):
    text = (NORTH + "\n\n" + SOUTH + "\n\n") * 45
    setup_text(config, text)
    config.input.text_batching.max_batches = 256
    config.input.type = "txt"
    store = Store(Path(config.output.directory))
    calls = []

    async def interrupted(role, payload, schema, record, stage):
        calls.append(record)
        if len(calls) == 2:
            raise BudgetExhausted("fixture interruption")
        assert payload["planning_window"]
        return window_response(payload, schema)

    async def resumed(role, payload, schema, record, stage):
        calls.append(record)
        assert (
            len(json_text({"output_language": config.project.language, **payload}))
            <= config.llm.max_input_chars
        )
        return window_response(payload, schema)

    try:
        with pytest.raises(BudgetExhausted):
            await prepare_text_batches(config, store, interrupted)
        assert not (Path(config.input.path).parent / "batches").exists()
        result = await prepare_text_batches(config, store, resumed)
        assert result["planning_view"] == "windowed"
        assert calls.count(calls[0]) == 1  # Completed window reused.
        assert calls.count(calls[1]) == 2  # Interrupted window retried.
        assert "".join(Path(b["input_path"]).read_text() for b in result["batches"]) == text
        assert all(0 < b["char_end"] - b["char_start"] <= 180 for b in result["batches"])
        before = list(calls)
        assert await prepare_text_batches(config, store, resumed) == result
        assert calls == before
    finally:
        store.close()


@pytest.mark.parametrize("cached", [False, True])
async def test_oversized_window_plan_is_truncated_including_old_checkpoints(config, cached):
    text = MATERIAL * 40
    setup_text(config, text)
    config.input.text_batching.max_batches = 256
    config.input.type = "txt"
    units = source_units(text, 45)
    windows = list(planning_windows(text, units, config))
    calls = []

    def oversized(payload):
        return {
            "reason": "Fixture oversized window",
            "batches": [
                {
                    "end_unit": payload["units"][-1]["id"],
                    "questions": [{"mode": "choice", "count": 1}],
                    "reason": "Keep window together",
                }
            ],
        }

    async def planner(role, payload, schema, record, stage):
        calls.append(record)
        return schema.model_validate(oversized(payload))

    store = Store(Path(config.output.directory))
    try:
        if cached:
            first, fragment, _, payload = windows[0]
            store.put(
                f"text-plan-window:{sha256(text)[:24]}:1",
                {
                    "text_sha256": sha256(fragment),
                    "first_unit": first,
                    "plan": oversized(payload),
                },
            )
        result = await prepare_text_batches(config, store, planner)
        assert len(calls) == len(windows) - int(cached)
        chunks = [Path(b["input_path"]).read_bytes().decode() for b in result["batches"]]
        assert chunks == [fragment[:180] for _, fragment, _, _ in windows]
        assert result["batch_count"] == len(windows)
        assert [b["char_start"] for b in result["batches"]] == [
            units[first]["start"] for first, _, _, _ in windows
        ]
        before = list(calls)
        assert await prepare_text_batches(config, store, planner) == result
        assert calls == before
    finally:
        store.close()


@pytest.mark.parametrize("fault", ["coverage", "quota"])
async def test_invalid_window_plan_is_saved_and_rejected_without_materializing(config, fault):
    setup_text(config, MATERIAL * 40)
    config.input.text_batching.max_batches = 256
    config.input.type = "txt"
    store = Store(Path(config.output.directory))
    calls = []

    async def invalid(role, payload, schema, record, stage):
        calls.append(record)
        plan = window_response(payload, schema)
        if fault == "coverage":
            plan.batches[-1].end_unit -= 1
        elif fault == "quota":
            plan.batches[0].questions[0].count = 50
        return plan

    try:
        for _ in range(2):
            with pytest.raises(NeedsReview):
                await prepare_text_batches(config, store, invalid)
        assert len(calls) == 1
        assert not (Path(config.input.path).parent / "batches").exists()
    finally:
        store.close()


def test_window_requests_bound_dense_indexes_and_preserve_unicode(config):
    config.input.text_batching.threshold_chars = 1000
    config.llm.max_input_chars = 5000
    text = "甲。🙂!\r\n" * 1000
    units = source_units(text, 250)
    windows = list(planning_windows(text, units, config))
    assert len(windows) > 1
    assert "".join(fragment for _, fragment, _, _ in windows) == text
    for _, fragment, local, payload in windows:
        assert len(json_text({"output_language": config.project.language, **payload})) <= 5000
        assert "".join(unit["text"] for unit in local) == fragment
        assert local[0]["id"] == 1 and local[0]["start"] == 0


async def test_impossible_total_batch_capacity_fails_before_api(config):
    setup_text(config, MATERIAL * 40)
    config.input.text_batching.max_batches = 2
    config.input.type = "txt"
    store = Store(Path(config.output.directory))

    async def never(*args):
        pytest.fail("Do not spend tokens when total batch capacity is insufficient")

    try:
        with pytest.raises(NeedsReview, match="cannot fit"):
            await prepare_text_batches(config, store, never)
    finally:
        store.close()


def test_planner_uses_analyzer_model(config):
    config.llm.analyzer_model = "planner-model"
    assert config.llm.model_for("text_batch_planner") == "planner-model"


def test_cli_build_keeps_overrides_and_resolves_yaml_paths(config, monkeypatch, capsys):
    setup_text(config)
    config_path = Path(config.input.path).parent / "run.yaml"
    data = config.model_dump(mode="json")
    data["input"]["path"] = Path(config.input.path).name
    data["output"]["directory"] = "cli-output"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    monkeypatch.setattr("decisionforge.cli.OpenAIProvider", lambda cfg, store: BatchProvider())
    assert (
        main(
            [
                "data",
                "build",
                "--config",
                str(config_path),
                "--difficulty",
                "0.9",
                "--concurrency",
                "2",
                "--text-batch-concurrency",
                "2",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["accepted"] == 2
    meta = json.loads((Path(result["run_directory"]) / "meta.json").read_text(encoding="utf-8"))
    for batch in meta["text_batching"]["batches"]:
        child = load_config(batch["config_path"])
        assert child.task.difficulty == 0.9 and child.execution.concurrency == 2
        assert child.execution.text_batch_concurrency == 2
        assert child.output.directory == str(config_path.parent / "cli-output")
        assert child.input.max_generated_questions == config.input.max_generated_questions
