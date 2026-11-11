"""Exercise actual overlapping coroutines, shared limits, cancellation and recovery."""

import asyncio
from collections import Counter
from pathlib import Path

import pytest
from conftest import FixtureProvider
from pydantic import ValidationError
from test_pipeline import source
from test_text_batching import NORTH, SOUTH, setup_text

from decisionforge.cli import signature
from decisionforge.config import ExecutionConfig, Pricing
from decisionforge.exporter import validate_file
from decisionforge.pipeline import Pipeline
from decisionforge.storage import Store
from decisionforge.util import BudgetExhausted


class ConcurrentProvider(FixtureProvider):
    def __init__(self, store, config, *, interrupt=False, fatal=False):
        super().__init__()
        self.store, self.config = store, config
        self.interrupt, self.fatal = interrupt, fatal
        self.active = self.peak = self.batch_peak = 0
        self.constructing, self.construct_peak = Counter(), Counter()
        self.cancelled = 0
        self.south_started = asyncio.Event()
        self.south_finished = asyncio.Event()
        self.actual_calls = []

    async def call(self, role, payload, schema, record, stage):
        # Model-stage checkpoints let the test verify recovery within a partially built TXT.
        saved = self.store.step(record, stage)
        if saved is not None:
            return schema.model_validate(saved)
        self.actual_calls.append((record, stage))
        self.active += 1
        self.peak = max(self.peak, self.active)
        if role == "qa_constructor":
            self.constructing[record] += 1
            self.construct_peak[record] = max(
                self.construct_peak[record], self.constructing[record]
            )
        try:
            if self.config.execution.budget_limit_usd is not None:
                self.store.reserve(record, stage, 0.01, self.config.execution.budget_limit_usd, 100)
            await asyncio.sleep(0.005)
            self.batch_peak = max(self.batch_peak, len(self.store.get("active_text_batches") or []))
            if role == "text_batch_planner":
                result = schema(
                    should_split=True,
                    reason="Independent stations",
                    batches=[
                        {
                            "end_unit": u["id"],
                            "questions": [{"mode": "choice", "count": 3}],
                            "reason": "Station case",
                        }
                        for u in payload["units"]
                        if u["text"].endswith("。")
                    ],
                )
            elif role == "text_assessor":
                result = schema(
                    is_narrative=True,
                    reason="Grounded",
                    opportunities=[
                        {
                            "mode": "choice",
                            "focus": f"case {i}",
                            "evidence": [payload["source"]],
                            "reason": "Rule",
                        }
                        for i in range(3)
                    ],
                )
            elif role == "qa_constructor":
                station = payload["source"].split()[0]
                if station == "South" and stage == "text_3:construct_qa" and self.interrupt:
                    self.south_started.set()
                    try:
                        await asyncio.Future()
                    finally:
                        self.south_finished.set()
                if station == "North" and stage == "text_3:construct_qa" and self.interrupt:
                    await asyncio.wait_for(self.south_started.wait(), timeout=2)
                    if self.fatal:
                        raise RuntimeError("injected batch crash")
                    raise BudgetExhausted("injected shared budget interruption")
                # Reverse completion order without changing the resulting question order.
                if stage.startswith("text_1:"):
                    await asyncio.sleep(0.01)
                result = schema(
                    qa={
                        "question": f"{station} {payload['opportunity']['focus']}: capacity?",
                        "answer": "4",
                    },
                    reason="Two times two",
                )
            else:
                result = await super().call(role, payload, schema, record, stage)
            self.store.save_step(record, stage, result.model_dump())
            return result
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        finally:
            self.active -= 1
            if role == "qa_constructor":
                self.constructing[record] -= 1


@pytest.mark.parametrize(
    "batch_workers,qa_workers,llm_limit", [(2, 2, 4), (2, 2, 1), (1, 2, 8), (2, 1, 8)]
)
async def test_batches_and_questions_overlap_with_global_limit(
    config, batch_workers, qa_workers, llm_limit
):
    material = NORTH + SOUTH + NORTH.replace("North", "East") + NORTH.replace("North", "West")
    setup_text(config, material)
    config.execution.text_batch_concurrency = batch_workers
    config.execution.concurrency = qa_workers
    config.execution.llm_concurrency = llm_limit
    config.output.validation_fraction = 0.25
    directory = Path(config.output.directory)
    store = Store(directory)
    try:
        provider = ConcurrentProvider(store, config)
        result = await asyncio.wait_for(Pipeline(config, store, provider).run(), timeout=10)
        assert result["accepted"] == 12
        assert result["splits"] == {"train": 9, "validation": 3}
        assert provider.batch_peak == batch_workers
        assert provider.peak == min(batch_workers * qa_workers, llm_limit)
        assert max(provider.construct_peak.values()) <= qa_workers
        if llm_limit >= batch_workers * qa_workers:
            assert set(provider.construct_peak.values()) == {qa_workers}
        assert provider.active == 0
        records = store.get("normalized_records")
        assert [r["origin"]["text_batch"]["batch_index"] for r in records] == [1] * 3 + [2] * 3 + [
            3
        ] * 3 + [4] * 3
        for offset in range(0, 12, 3):
            assert [
                r["record"]["question"].split("case ")[1][0] for r in records[offset : offset + 3]
            ] == ["0", "1", "2"]
        assert store.get("source_total") == 12
        assert store.get("active_text_batches") == []
        assert store.get("active_text_batch") is None
        assert validate_file(directory / "accepted.json") == 12
        before = (directory / "accepted.json").read_bytes()
        resumed = ConcurrentProvider(store, config)
        await Pipeline(config, store, resumed).run()
        assert resumed.actual_calls == []
        assert (directory / "accepted.json").read_bytes() == before
    finally:
        store.close()


@pytest.mark.parametrize("fatal", [False, True])
async def test_failure_joins_other_batches_and_resume_reuses_constructor_checkpoints(config, fatal):
    setup_text(config)
    config.execution.text_batch_concurrency = 2
    config.execution.concurrency = 2
    config.execution.llm_concurrency = 4
    store = Store(Path(config.output.directory))
    try:
        provider = ConcurrentProvider(store, config, interrupt=True, fatal=fatal)
        pipeline = Pipeline(config, store, provider)
        if fatal:
            with pytest.raises(RuntimeError, match="injected batch crash"):
                await asyncio.wait_for(pipeline.run(), timeout=5)
        else:
            result = await asyncio.wait_for(pipeline.run(), timeout=5)
            assert result["status"] == "budget_exhausted"
        assert provider.active == 0
        assert provider.cancelled >= 1 and provider.south_finished.is_set()
        assert store.get("active_text_batches") == []
        saved = {(r, s) for r, s in provider.actual_calls if store.step(r, s) is not None}
        # The second direction finishes before the deliberately slower first direction.
        assert any(s == "text_2:construct_qa" for _, s in saved)
        resumed = ConcurrentProvider(store, config)
        result = await Pipeline(config, store, resumed).run()
        assert result["accepted"] == 6 and result["status"] == "complete"
        assert not saved.intersection(resumed.actual_calls)
        assert store.get("source_total") == 6
        assert len(store.get("normalized_records")) == 6
    finally:
        store.close()


async def test_parallel_batches_share_one_dollar_budget(config):
    setup_text(config)
    config.execution.text_batch_concurrency = 2
    config.execution.budget_limit_usd = 0.075
    config.llm.pricing = {config.llm.model: Pricing(input_per_million=1, output_per_million=1)}
    store = Store(Path(config.output.directory))
    try:
        provider = ConcurrentProvider(store, config)
        result = await Pipeline(config, store, provider).run()
        assert result["status"] == "budget_exhausted"
        assert 0.06 <= store.spent() <= 0.075
        assert provider.active == 0
        config.execution.budget_limit_usd = 1.0
        result = await Pipeline(config, store, ConcurrentProvider(store, config)).run()
        assert result["accepted"] == 6
        assert store.spent() < 1.0
    finally:
        store.close()


async def test_single_unsplit_txt_constructs_questions_in_parallel(config):
    source(config, NORTH, ".txt")
    config.input.text_batching.enabled = False
    config.execution.concurrency = 3
    store = Store(Path(config.output.directory))
    try:
        provider = ConcurrentProvider(store, config)
        result = await Pipeline(config, store, provider).run()
        assert result["accepted"] == 3
        assert list(provider.construct_peak.values()) == [3]
    finally:
        store.close()


async def test_completed_batch_survives_other_batch_failure(config):
    setup_text(config)
    config.execution.text_batch_concurrency = 2
    directory = Path(config.output.directory)
    store = Store(directory)

    class LateFailure(ConcurrentProvider):
        async def call(self, role, payload, schema, record, stage):
            if role == "extractor" and payload["source"].startswith("South"):
                while not store.db.execute(
                    "SELECT 1 FROM events WHERE stage='text_batch_complete'"
                ).fetchone():
                    await asyncio.sleep(0.005)
                raise BudgetExhausted("interrupt after another batch completed")
            return await super().call(role, payload, schema, record, stage)

    try:
        result = await asyncio.wait_for(
            Pipeline(config, store, LateFailure(store, config)).run(), 5
        )
        assert result["status"] == "budget_exhausted" and result["accepted"] == 3
        assert store.get("source_total") == 3
        assert len(store.get("normalized_records")) == 3
        assert validate_file(directory / "accepted.json") == 3
        resumed = ConcurrentProvider(store, config)
        result = await Pipeline(config, store, resumed).run()
        assert result["accepted"] == 6
        # Only the unfinished TXT needs extraction, assessment and construction.
        assert sum(stage == "extract" for _, stage in resumed.actual_calls) == 1
        assert store.get("source_total") == 6
    finally:
        store.close()


async def test_external_cancellation_joins_all_batch_api_calls(config):
    setup_text(config)
    config.execution.text_batch_concurrency = 2
    store = Store(Path(config.output.directory))
    blocked = set()
    both_started = asyncio.Event()

    class Blocked(ConcurrentProvider):
        async def call(self, role, payload, schema, record, stage):
            if role == "extractor":
                blocked.add(record)
                if len(blocked) == 2:
                    both_started.set()
                try:
                    await asyncio.Future()
                finally:
                    blocked.remove(record)
            return await super().call(role, payload, schema, record, stage)

    task = asyncio.create_task(Pipeline(config, store, Blocked(store, config)).run())
    try:
        await asyncio.wait_for(both_started.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not blocked
        assert store.get("active_text_batches") == []
        assert store.get("run_status") == "interrupted"
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        store.close()


def test_batch_concurrency_config_is_operational(config):
    assert ExecutionConfig().text_batch_concurrency == 4
    for value in (0, 65):
        with pytest.raises(ValidationError):
            ExecutionConfig(text_batch_concurrency=value)
    before = signature(config)
    config.execution.text_batch_concurrency = 1
    assert signature(config) == before
