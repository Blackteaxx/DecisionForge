"""Full-data preparation keeps roles and audits explicit method projections."""

import hashlib
import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_freezing_roles_preserves_original_test_and_cn_group_assignments(tmp_path):
    script = ROOT / "examples/training/prepare_full.py"
    assert script.is_file(), "The shared full-data role preparation CLI is missing"
    freeze = runpy.run_path(str(script))["freeze"]
    project = runpy.run_path(str(ROOT / "examples/training/openjev/prepare_cn.py"))["prepare"]
    rows = [json.loads(line) for line in (ROOT / "examples/training/data/typed-decisions-sample.jsonl").read_text().splitlines()[:22]]
    for index, row in enumerate(rows):
        row["split"] = "train" if index < 20 else "test"
    source = tmp_path / "original.jsonl"
    original = "".join(json.dumps(row) + "\n" for row in rows)
    source.write_text(original)
    output = tmp_path / "frozen.jsonl"
    audit = freeze(source, output)
    frozen = [json.loads(line) for line in output.read_text().splitlines()]
    roles = {row["record_id"]: row["split"] for row in frozen}
    assert {identity for identity, role in roles.items() if role == "validation"} == {17, 19}
    assert {identity for identity, role in roles.items() if role == "calibration"} == {1, 20}
    assert {identity for identity, role in roles.items() if role == "test"} == {21, 22}
    assert audit["source_sha256"] == hashlib.sha256(original.encode()).hexdigest()
    assert audit["report"]["roles"]["train"]["groups"] == 16
    assert [{**row, "split": original_row["split"]} for row, original_row in zip(frozen, rows, strict=True)] == rows
    cn = tmp_path / "cn.jsonl"
    project(output, cn)
    projected = [json.loads(line) for line in cn.read_text().splitlines()]
    assert [(r["record_id"], r["group_id"], r["split"]) for r in projected] == [
        (r["record_id"], r["group_id"], r["split"]) for r in frozen
    ]
    assert source.read_text() == original
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        freeze(source, output)
    assert output.read_bytes() == before


def test_openjev_eval_projection_preserves_retained_samples_and_audits_score(tmp_path):
    script = ROOT / "examples/training/openjev/prepare_eval.py"
    assert script.is_file(), "The explicit OpenJev benchmark projection CLI is missing"
    project = runpy.run_path(str(script))["prepare"]
    source = ROOT / "examples/eval/training-free/anyjev/fixture.jsonl"
    output = tmp_path / "cn.jsonl"
    audit = project(source, output)
    from decisionforge.eval.data import load_samples

    original = load_samples(source)
    retained = load_samples(output)
    assert retained == [sample for sample in original if sample.question.type != "score"]
    assert audit["selected"] == 2
    assert audit["excluded"] == [
        {"id": sample.id, "group_id": sample.group_id, "type": "score"}
        for sample in original if sample.question.type == "score"
    ]
    assert json.loads(output.with_suffix(".source.json").read_text()) == audit
    with pytest.raises(FileExistsError):
        project(source, output)


def test_openjev_eval_projection_retains_full_large_catalog(tmp_path):
    script = ROOT / "examples/training/openjev/prepare_eval.py"
    assert script.is_file(), "The explicit OpenJev benchmark projection CLI is missing"
    project = runpy.run_path(str(script))["prepare"]
    from decisionforge.eval.data import load_samples

    sample = load_samples(ROOT / "examples/eval/training-free/anyjev/fixture.jsonl")[0].model_dump(mode="json")
    key = next(iter(sample["request"]["questions"]))
    sample["request"]["questions"][key]["criteria"] = {str(i): str(i) for i in range(77)}
    sample["target"] = None
    source, output = tmp_path / "banking.jsonl", tmp_path / "cn.jsonl"
    source.write_text(json.dumps(sample) + "\n")
    audit = project(source, output)
    converted = load_samples(output)[0].model_dump(mode="json")
    assert converted == sample
    assert len(converted["request"]["questions"][key]["criteria"]) == 77
    assert audit["selected"] == 1 and audit["excluded"] == []
    assert json.loads(output.with_suffix(".source.json").read_text()) == audit
