"""Offline dataset conversion; references never enter a model request."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from decisionforge.eval.data import parse_json, write_jsonl
from decisionforge.eval.jevbench import JEVBENCH_PIN
from decisionforge.eval.metrics import distribution
from decisionforge.eval.models import Sample
from decisionforge.util import write_json


def read_rows(path):
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ValueError('Parquet conversion requires pip install -e ".[eval-data]"') from exc
        return pq.read_table(path).to_pylist()
    return [
        parse_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def source_info(directory):
    manifest = directory / "source.json"
    if not manifest.exists():
        return {}
    source = parse_json(manifest.read_text(encoding="utf-8"))
    for item in source["files"]:
        path = directory / item["name"]
        if (
            path.parent != directory
            or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]
        ):
            raise ValueError("Downloaded source hash mismatch")
    return source


def finish_conversion(samples, source, output, converter, source_version, extra=None):
    output = Path(output).resolve()
    if output in source:
        raise ValueError("Conversion output must differ from source")
    if not samples or len({s.id for s in samples}) != len(samples):
        raise ValueError("Empty source or duplicate source id")
    manifest = {
        "converter": converter,
        "converter_version": "1",
        "source_version": source_version,
        "source_files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source},
        "records": len(samples),
        "output": str(output),
        **(extra or {}),
    }
    write_jsonl(output, (s.model_dump(mode="json") for s in samples))
    write_json(output.with_suffix(output.suffix + ".manifest.json"), manifest)
    return manifest


def decode(value):
    return parse_json(value) if isinstance(value, str) else value


def convert_typed_decisions(source, output, source_version=None):
    source = Path(source).resolve()
    info = source_info(source) if source.is_dir() else {}
    if info and info.get("dataset") != "typed_decisions":
        raise ValueError("Source bundle belongs to another dataset")
    path = source / f"{info.get('split', 'test')}.parquet" if source.is_dir() else source
    version = source_version or info.get("revision", "unspecified")
    samples = []
    for index, raw in enumerate(read_rows(path), 1):
        try:
            state, questions, gold = (decode(raw[k]) for k in ("state", "questions", "gold"))
            if set(questions) != set(gold) or not questions or raw["n_questions"] != len(questions):
                raise ValueError("Question/gold sets or counts differ")
            for name, question in questions.items():
                identity = f"{raw['id']}::{name}"
                # First validate the public question independently of its reference.
                base = {
                    "schema_version": "decisionforge.eval.v1",
                    "id": identity,
                    "group_id": raw["id"],
                    "split": raw["split"],
                    "subset": raw["workflow"],
                    "request": {"state": state, "questions": {"decision": question}},
                    "target": None,
                    "metadata": {},
                }
                public = Sample.model_validate(base)
                reference = gold[name]
                if reference["type"] != public.question.type:
                    raise ValueError("Reference type differs from question")
                q = distribution(reference["probabilities"], public.question.labels, 1e-5)
                total = sum(q.values())
                label = reference["label"]
                if label not in q or q[label] + 1e-6 < max(q.values()):
                    raise ValueError("Reference modal label differs from probability distribution")
                base.update(
                    target={"kind": "soft", "probabilities": {k: v / total for k, v in q.items()}},
                    metadata={
                        "dataset": "typed_decisions",
                        "source_version": version,
                        "question_name": name,
                        "gold_label": label,
                        "source_gold": reference,
                        "source_factors": decode(raw.get("factors", {})),
                        "label_agreement": decode(raw.get("label_agreement", {})),
                        "target_source": "teacher_mean_of_three_temperature_0.7",
                        "request_protocol": "single_question",
                        "reference_normalization": {
                            "original_sum": total,
                            "rule": "six_decimal_rounding_tolerance_1e-5",
                        },
                    },
                )
                samples.append(Sample.model_validate(base))
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"{path}: row {index}: {exc}") from exc
    return finish_conversion(
        samples,
        [path],
        output,
        "typed_decisions",
        version,
        {
            "cases": len({s.group_id for s in samples}),
            "request_protocol": "single_question",
            "source": info,
        },
    )


def convert_intents(dataset, source, output, source_version=None):
    directory = Path(source).resolve()
    info = source_info(directory)
    if info and info.get("dataset") != dataset:
        raise ValueError("Source bundle belongs to another dataset")
    split, version = (
        info.get("split", "test"),
        source_version or info.get("revision", "unspecified"),
    )
    labels_path = directory / "labels.json"
    labels = parse_json(labels_path.read_text(encoding="utf-8"))
    expected_count = {"banking77": 77, "clinc150_oos": 151}[dataset]
    if (
        not isinstance(labels, list)
        or len(labels) != expected_count
        or any(not isinstance(k, str) or not k for k in labels)
        or len(set(labels)) != len(labels)
    ):
        raise ValueError("Invalid fixed intent label catalog")
    if dataset == "clinc150_oos" and labels.count("oos") != 1:
        raise ValueError("CLINC150+OOS requires exactly one oos label")
    if dataset == "banking77":
        path = directory / f"{split}.csv"
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
    else:
        path = directory / f"{split}.parquet"
        rows = read_rows(path)
    criteria = {
        k: "None of the supported intents applies; this request is out of scope."
        if dataset == "clinc150_oos" and k == "oos"
        else k.replace("_", " ")
        for k in labels
    }
    instructions = (
        "Classify this banking customer message into exactly one of the listed intents."
        if dataset == "banking77"
        else "Classify this user request into exactly one listed intent. Choose oos only "
        "when none of the supported intents applies."
    )
    samples = []
    for index, raw in enumerate(rows):
        label = raw["category"] if dataset == "banking77" else raw["intent"]
        if dataset == "clinc150_oos":
            if type(label) is not int or not 0 <= label < len(labels):
                raise ValueError(f"Invalid intent index at row {index}")
            label = labels[label]
        if label not in labels:
            raise ValueError(f"Unknown intent label at row {index}")
        samples.append(
            Sample.model_validate(
                {
                    "schema_version": "decisionforge.eval.v1",
                    "id": f"{dataset}:{split}:{index}",
                    "group_id": None,
                    "split": split,
                    "subset": "oos" if label == "oos" and dataset == "clinc150_oos" else "in_scope",
                    "request": {
                        "state": raw["text"],
                        "questions": {
                            "decision": {
                                "type": "choice",
                                "instructions": instructions,
                                "criteria": criteria,
                            }
                        },
                    },
                    "target": {"kind": "hard", "value": label},
                    "metadata": {
                        "dataset": dataset,
                        "source_version": version,
                        "source_row": index,
                        "target_source": "original_dataset_annotation",
                        "label_description_rule": "original_label_replace_underscores_v1",
                        "evaluation_mode": "zero_shot_all_labels",
                    },
                }
            )
        )
    return finish_conversion(
        samples, [path, labels_path], output, dataset, version, {"source": info}
    )


def convert_jevbench(source, output, source_version=JEVBENCH_PIN):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output:
        raise ValueError("Conversion output must differ from source")
    info = source_info(source) if source.is_dir() else {}
    if info and info.get("dataset") != "jevbench":
        raise ValueError("Source bundle belongs to another dataset")
    paths = (
        [
            source / f"{tier}.jsonl"
            for tier in ("original", "easy", "hard")
            if (source / f"{tier}.jsonl").is_file()
        ]
        if source.is_dir()
        else [source]
    )
    if not paths:
        raise ValueError("No JevBench JSONL sources found")
    if output in {p.resolve() for p in paths}:
        raise ValueError("Conversion output must differ from source")
    samples, seen = [], set()
    lines = (
        (path, number, line)
        for path in paths
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
    )
    for path, number, line in lines:
        if not line.strip():
            continue
        try:
            raw = parse_json(line)
            kind, expected, labels = raw["question"]["type"], raw.get("expected"), raw["labels"]
            value = expected
            if kind == "noul":
                if labels != ["no", "yes"] or expected not in (None, "no", "yes"):
                    raise ValueError("JevBench Noul requires exact no/yes labels")
                value = expected == "yes"
            elif kind == "choice":
                if set(labels) != set(raw["question"]["criteria"]):
                    raise ValueError("Choice labels disagree with criteria")
            elif kind == "score":
                if labels != [str(i) for i in range(len(raw["question"]["criteria"]))]:
                    raise ValueError("Score labels disagree with rubric indices")
            provenance = raw.get("provenance", {})
            sample = Sample.model_validate(
                {
                    "schema_version": "decisionforge.eval.v1",
                    "id": raw["id"],
                    "group_id": raw.get("group"),
                    "split": raw["split"],
                    "subset": raw["family"],
                    "request": {"state": raw["state"], "questions": {"decision": raw["question"]}},
                    "target": None if expected is None else {"kind": "hard", "value": value},
                    "metadata": {
                        "family": raw["family"],
                        "labels": labels,
                        "provenance": provenance,
                        "exclude_reason": provenance.get("exclude_reason"),
                        "source_version": source_version,
                        "target_source": provenance.get("label_basis"),
                        "tier": path.stem,
                    },
                }
            )
            if sample.id in seen:
                raise ValueError("duplicate source id")
            seen.add(sample.id)
            samples.append(sample)
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"{path}: line {number}: {exc}") from exc
    if not samples:
        raise ValueError("Empty JevBench source")
    manifest = {
        "converter": "jevbench",
        "converter_version": "1",
        "source_version": source_version,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()
        if source.is_file()
        else None,
        "source_files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "source": info,
        "records": len(samples),
        "output": str(output),
    }
    write_jsonl(output, (s.model_dump(mode="json") for s in samples))
    write_json(output.with_suffix(output.suffix + ".manifest.json"), manifest)
    return manifest
