"""Bounded concurrent normalization -> generation -> check/repair -> export."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

from decisionforge.adapters import input_kind, read_records
from decisionforge.config import load_config
from decisionforge.exporter import export_run
from decisionforge.generation.components import (
    compile_record,
    generate,
    initial_draft,
    missing_modules,
)
from decisionforge.generation.contracts import CheckReport, StandardRecord
from decisionforge.generation.intake import extract
from decisionforge.generation.text_batching import load_batch_context, prepare_text_batches
from decisionforge.models import MappingSpec, TrainingExample
from decisionforge.util import BudgetExhausted, PipelineError, digest, map_bounded

logger = logging.getLogger(__name__)
BATCH_EXPORT_INTERVAL_SECONDS = 60.0
MODULE_NAMES = {
    "context": "背景补全",
    "choice": "Choice 选项生成",
    "noul": "Noul 陈述生成",
    "score": "Score 排序生成",
}


class Pipeline:
    def __init__(self, config, store, provider, *, checkpoint_namespace=""):
        self.config, self.store, self.provider = config, store, provider
        if config.input.type == "auto":
            config.input.type = input_kind(Path(config.input.path))
        self.batch_context = load_batch_context(config)
        self.progress_scope = (
            f"批次 {self.batch_context.batch_index}" if self.batch_context else "整体"
        )
        self.normalized_key = "normalized_records" + checkpoint_namespace
        self.stop = asyncio.Event()
        self.llm_slots = asyncio.Semaphore(config.execution.llm_concurrency)
        self.mapping = (
            json.loads(Path(config.input.mapping_spec).read_text(encoding="utf-8-sig"))
            if config.input.mapping_spec
            else None
        )
        if self.mapping is not None and not isinstance(self.mapping, dict):
            raise ValueError("mapping_spec must be a JSON object of field paths")
        if self.mapping is not None:
            MappingSpec.model_validate(self.mapping)

    def progress(self, current, pending="无", *, record=None):
        scope = self.progress_scope
        if record is not None:
            scope += f" / 记录 {record}"
        logger.info("[进度][%s] 当前：%s | 后续：%s", scope, current, pending)

    async def call(self, role, payload, schema, record, stage):
        async with self.llm_slots:
            if self.stop.is_set():
                raise BudgetExhausted("Run stopped; resume pending stages")
            intake_stages = {
                "text_batch_planner": "长文本分批规划",
                "extractor": "原文问答提取",
                "text_assessor": "材料出题评估",
                "qa_constructor": "基础问答构造",
            }
            if role in intake_stages:
                self.progress(
                    intake_stages[role], "完成标准化 → 按需生成 → 检查/修复 → 导出", record=record
                )
            try:
                return await self.provider.call(role, payload, schema, record, stage)
            except BudgetExhausted:
                self.stop.set()
                raise

    async def map_bounded(self, function, items, *, concurrency=None):
        return await map_bounded(
            function,
            items,
            self.config.execution.concurrency if concurrency is None else concurrency,
            stop=self.stop,
        )

    def failure(self, identifier, reason, source, jev=None, history=None, draft=None):
        value = {
            "record_id": identifier,
            "status": "review",
            "reason": str(reason),
            "source": source,
            "jev": jev,
            "checks": history or [],
            "draft": draft,
        }
        self.store.finish(identifier, "review", value)
        self.store.event(identifier, "review", value)

    async def prepare(self):
        saved = self.store.get(self.normalized_key)
        if saved is not None:
            self.progress(f"复用标准化检查点（{len(saved)} 条）", "按需生成 → 检查/修复 → 导出")
            return saved

        async def one(raw):
            if self.store.done(raw.source_record_id):
                return []
            try:
                question_plan = (
                    [item.model_dump() for item in self.batch_context.questions]
                    if self.batch_context is not None
                    else None
                )
                records = await extract(raw, self.config, self.call, self.mapping, question_plan)
                entries = []
                for i, source in enumerate(records):
                    identifier = (
                        raw.source_record_id
                        if len(records) == 1
                        else f"{raw.source_record_id}_{i + 1}"
                    )
                    origin = {**raw.model_dump(), "source_record_id": identifier}
                    if self.batch_context is not None:
                        source.group_id = self.batch_context.group_id
                        origin["text_batch"] = self.batch_context.model_dump()
                    if source.text_origin is not None:
                        origin["input_record_id"] = raw.source_record_id
                    entries.append(
                        {"id": identifier, "record": source.model_dump(), "origin": origin}
                    )
                    self.store.put("qa:" + identifier, {**source.model_dump(), **origin})
                    self.store.event(identifier, "normalize", source.model_dump())
                return entries
            except BudgetExhausted:
                raise
            except (PipelineError, ValueError) as exc:
                self.failure(raw.source_record_id, exc, raw.model_dump())
                return []

        raw = list(
            read_records(
                Path(self.config.input.path), self.config.input.type, self.config.input.records_path
            )
        )
        batches = await self.map_bounded(one, raw)
        entries = [entry for batch in batches for entry in batch]
        for entry in entries:
            source = StandardRecord.model_validate(entry["record"])
            self.store.register_group(source.group_id or entry["id"], source.split)
        if self.normalized_key == "normalized_records":
            self.store.put(
                "source_total",
                len(entries)
                + sum(1 for _, status, _ in self.store.outcomes() if status == "review"),
            )
        self.store.put(self.normalized_key, entries)
        return entries

    async def generate_records(self, records=None):
        if records is None:
            self.progress("2/4 问答标准化", "按需生成 → 检查/修复 → 导出")
            records = await self.prepare()
        total = len(records)
        pending = [entry for entry in records if not self.store.done(entry["id"])]
        completed = total - len(pending)
        self.progress(
            f"3/4 生成与检查（已完成 {completed}/{total} 条）", "按需补全 → 检查/修复 → 导出"
        )

        async def one(entry):
            nonlocal completed
            if self.store.done(entry["id"]):
                return
            await self.process(entry)
            completed += 1
            self.progress(
                f"记录处理完成 {completed}/{total}",
                f"剩余 {total - completed} 条 → 导出",
                record=entry["id"],
            )

        await self.map_bounded(one, pending)
        return records

    async def run_inputs(self):
        self.progress("1/4 输入准备（按需分批）", "问答标准化 → 生成与检查 → 导出")
        batching = (
            await prepare_text_batches(self.config, self.store, self.call)
            if self.batch_context is None
            else None
        )
        if not batching or not batching["should_split"]:
            await self.generate_records()
            return
        batches = batching["batches"]
        namespaces = {
            batch["batch_index"]: ":" + digest(batch["input_path"])[:24] for batch in batches
        }
        # Include already prepared batches on resume, even before a worker reaches them.
        batch_records = {}
        for index, namespace in namespaces.items():
            saved = self.store.get("normalized_records" + namespace)
            if saved is not None:
                # [] is a completed preparation with no questions, not an absent checkpoint.
                batch_records[index] = saved
        completed = {
            index
            for index, records in batch_records.items()
            if all(self.store.done(entry["id"]) for entry in records)
        }
        pending = [batch for batch in batches if batch["batch_index"] not in completed]
        active = set()
        record_count = sum(len(records) for records in batch_records.values())
        last_export = time.monotonic()

        def publish_state(*, records_changed=False):
            nonlocal record_count
            values = {}
            if records_changed:
                records = [
                    record for index in sorted(batch_records) for record in batch_records[index]
                ]
                record_count = len(records)
                values["normalized_records"] = records
            values["source_total"] = record_count + self.store.input_failure_count()
            values["active_text_batches"] = sorted(active)
            # Retain the single-batch field for consumers of older metadata.
            values["active_text_batch"] = next(iter(active)) if len(active) == 1 else None
            self.store.put_many(values)

        async def one(batch):
            nonlocal last_export
            # Reuse the exact build implementation, rather than shelling out and losing the
            # shared run budget, export IDs, cancellation signal and checkpoint ownership.
            config = load_config(batch["config_path"])
            child = Pipeline(
                config,
                self.store,
                self.provider,
                checkpoint_namespace=namespaces[batch["batch_index"]],
            )
            child.stop, child.llm_slots = self.stop, self.llm_slots
            child.progress_scope = f"批次 {batch['batch_index']}/{len(batching['batches'])}"
            index = batch["batch_index"]
            active.add(index)
            publish_state()
            try:
                child.progress("2/4 问答标准化", "按需生成 → 检查/修复 → 导出")
                records = await child.prepare()
                changed = batch_records.get(index) != records
                batch_records[index] = records
                publish_state(records_changed=changed)
                await child.generate_records(records)
                completed.add(index)
                self.store.event(
                    "text-batch:" + str(index),
                    "text_batch_complete",
                    {"input_path": batch["input_path"], "normalized_questions": len(records)},
                )
            finally:
                active.discard(index)
                publish_state()
            # Export runs synchronously on this event loop, so concurrent batches cannot
            # interleave writes to the shared output files or export ID allocation.
            # Each export rebuilds the entire run. Checkpoints already persist every outcome;
            # refresh public files periodically, and always on completion/interruption in run().
            if time.monotonic() - last_export >= BATCH_EXPORT_INTERVAL_SECONDS:
                child.progress(
                    "批次结果导出", f"剩余 {len(batches) - len(completed)} 个批次 → 最终导出"
                )
                export_run(self.store, Path(self.config.output.directory))
                last_export = time.monotonic()

        self.progress(
            f"复用已完成批次 {len(completed)}/{len(batches)}",
            f"处理剩余 {len(pending)} 个批次 → 最终导出",
        )
        publish_state(records_changed=True)
        await self.map_bounded(
            one, pending, concurrency=self.config.execution.text_batch_concurrency
        )

    async def process_child(self, entry, mode, label, name):
        source_id = entry["id"]
        identifier = source_id + ":" + name
        if self.store.done(identifier):
            return
        source = StandardRecord.model_validate(entry["record"])
        draft = initial_draft(source, mode, label)
        modules = missing_modules(source, draft, mode, self.config)
        history, current_jev, last_record = [], None, None
        feedback = None
        for attempt in range(self.config.verification.max_semantic_repairs + 1):
            try:
                modules = list(
                    dict.fromkeys(modules + missing_modules(source, draft, mode, self.config))
                )
                for index, module in enumerate(modules):
                    pending = [MODULE_NAMES.get(item, item) for item in modules[index + 1 :]]
                    self.progress(
                        (f"第 {attempt} 次修复：" if attempt else "")
                        + MODULE_NAMES.get(module, module),
                        " → ".join(
                            [*pending, "本地校验", "Checker 检查（未通过时按需修复）", "导出"]
                        ),
                        record=identifier,
                    )
                    await generate(
                        module,
                        source,
                        draft,
                        self.config,
                        self.call,
                        source_id,
                        f"{name}:{module}:{attempt}",
                        feedback,
                        current_jev,
                    )
                self.progress("本地编译与校验", "Checker 检查 → 按需修复 → 导出", record=identifier)
                record = compile_record(source, draft, mode, source_id, identifier, self.config)
                example = TrainingExample.from_record(record, record_id=1, group_id=1)
                current_jev = {
                    "request": example.request.model_dump(),
                    "supervision": example.supervision.model_dump(),
                }
                last_record = record.model_dump()
                self.progress(
                    f"Checker 检查（第 {attempt + 1} 轮）", "按需修复 → 导出", record=identifier
                )
                report = await self.call(
                    "checker",
                    {
                        "source": source.model_dump(),
                        "original_input": entry["origin"]["payload"],
                        "current_jev": current_jev,
                        "difficulty": self.config.task.difficulty,
                        "relevance": self.config.context.relevance,
                        "requested_modes": self.config.task.modes,
                        "domain": self.config.task.domain_profile,
                    },
                    CheckReport,
                    source_id,
                    f"{name}:check:{attempt}",
                )
                feedback = report.model_dump()
                history.append(feedback)
                self.store.event(identifier, "check", feedback)
                if report.passed:
                    self.store.put(
                        "generation_meta:" + identifier,
                        {
                            "source": entry["origin"],
                            "normalized": source.model_dump(),
                            "checks": history,
                        },
                    )
                    self.store.finish(identifier, "accepted", last_record)
                    self.store.event(identifier, "accepted", {"record_id": identifier})
                    return
                modules = list(dict.fromkeys(report.modules))
                if "source" in modules or any(m not in {"context", mode} for m in modules):
                    break
                if "context" in modules and not self.config.context.generate_if_missing:
                    break
            except BudgetExhausted:
                raise
            except (PipelineError, ValueError) as exc:
                # Structural failures feed the same bounded repair loop; preserve last valid JEV.
                module = "context" if "context" in str(exc).lower() else mode
                feedback = {
                    "passed": False,
                    "reason": str(exc),
                    "failed_fields": [module],
                    "repair_instructions": [str(exc)],
                    "modules": [module],
                }
                history.append(feedback)
                modules = [module]
            if attempt < self.config.verification.max_semantic_repairs:
                self.store.event(identifier, "repair", feedback)
        self.failure(
            identifier,
            feedback["reason"],
            {**entry["origin"], **source.model_dump(), "source_record_id": source_id},
            last_record,
            history,
            draft,
        )

    async def process(self, entry):
        identifier = entry["id"]
        if self.store.done(identifier):
            return
        started = time.perf_counter()
        source = StandardRecord.model_validate(entry["record"])
        modes = self.config.task.modes
        if source.text_origin is not None:
            # Narrative opportunities are assessed separately for each requested mode.
            if modes != ["auto"] and source.mode not in modes:
                raise ValueError("Constructed text QA has an unrequested mode")
            modes = [source.mode]
        elif modes == ["auto"]:
            modes = [
                source.mode
                or (
                    "score"
                    if source.ranking_items
                    else "noul"
                    if source.statement is not None
                    else "choice"
                )
            ]
        children = []
        for mode in modes:
            labels = (
                (
                    [source.noul_label]
                    if source.statement and source.noul_label is not None
                    else self.config.noul.labels
                )
                if mode == "noul"
                else [None]
            )
            for label in labels:
                name = mode if label is None else f"noul_{label}"
                children.append(name)
                self.store.event(
                    identifier,
                    "declare",
                    {
                        "source_record_id": identifier,
                        "child_record_id": identifier + ":" + name,
                        "question_id": name,
                        "mode": mode,
                    },
                )
                await self.process_child(entry, mode, label, name)
        self.store.finish(identifier, "complete", {"children": children})
        self.store.event(
            identifier, "record_timing", {"elapsed_seconds": time.perf_counter() - started}
        )

    async def run(self):
        started = time.perf_counter()
        self.store.put("effective_config", self.config.model_dump())
        try:
            await self.run_inputs()
            status = "complete"
        except BudgetExhausted:
            status = "budget_exhausted"
            self.progress("预算耗尽，生成暂停", "导出已有结果；恢复后继续未完成任务")
        except BaseException:
            self.store.put("run_status", "interrupted")
            self.progress("运行中断，保存已有结果", "恢复后继续未完成任务")
            export_run(self.store, Path(self.config.output.directory))
            raise
        self.store.put("run_status", status)
        self.store.put(
            "run_metrics",
            {
                "elapsed_seconds": time.perf_counter() - started,
                "qa_concurrency": self.config.execution.concurrency,
                "text_batch_concurrency": self.config.execution.text_batch_concurrency,
                "llm_concurrency": self.config.execution.llm_concurrency,
            },
        )
        self.progress("4/4 结果导出", "恢复后继续未完成任务" if status != "complete" else "无")
        result = {
            "status": status,
            **export_run(
                self.store,
                Path(self.config.output.directory),
                self.config.output.export_requests,
                self.config.output.include_audit,
            ),
        }
        self.progress(
            "本次流程结束" if status == "complete" else "已暂停并导出已有结果",
            "无" if status == "complete" else "恢复后继续未完成任务",
        )
        return result
