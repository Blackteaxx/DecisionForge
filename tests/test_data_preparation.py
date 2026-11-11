import json
import subprocess
import sys
from pathlib import Path

import pytest

from data.benchmarks._shared import prepare
from tests.test_eval_benchmark_extensions import typed_row

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("dataset", prepare.SPLITS)
def test_script_help_defaults_are_relative_to_script_from_another_cwd(dataset, tmp_path):
    script = ROOT / "data/benchmarks" / dataset / "prepare.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert str(script.parent).replace(" ", "") in "".join(result.stdout.split())
    assert "--convert-only" in result.stdout


@pytest.mark.parametrize("dataset", prepare.SPLITS)
def test_offline_preparation_never_fetches_and_keeps_raw_data(dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(
        prepare, "fetch_dataset", lambda *a, **k: pytest.fail("offline preparation fetched")
    )
    split = prepare.SPLITS[dataset][0]
    raw = tmp_path / "raw" / split
    raw.mkdir(parents=True)
    if dataset == "jevbench":
        (raw / "original.jsonl").write_bytes(
            (ROOT / "tests/fixtures/jevbench_oracle/original.jsonl").read_bytes()
        )
        expected = 72
    else:
        if dataset == "banking77":
            labels = [f"intent_{i}" for i in range(77)]
            (raw / "labels.json").write_text(json.dumps(labels))
            (raw / "test.csv").write_text("text,category\nmessage,intent_0\n")
            expected = 1
        else:
            pa = pytest.importorskip("pyarrow")
            pq = pytest.importorskip("pyarrow.parquet")
            if dataset == "typed_decisions":
                row = typed_row()
                row["factors"] = json.dumps(row["factors"])
                row["label_agreement"] = "{}"
                table = pa.Table.from_pylist([row])
                expected = 2
            else:
                (raw / "labels.json").write_text(
                    json.dumps(["oos"] + [f"intent_{i}" for i in range(150)])
                )
                table = pa.table({"text": ["outside scope"], "intent": [0]})
                expected = 1
            pq.write_table(table, raw / "test.parquet")
    before = {p.name: p.read_bytes() for p in raw.iterdir()}
    result = prepare.prepare_dataset(dataset, tmp_path, split, convert_only=True)
    assert result["conversion"]["records"] == expected
    assert Path(result["processed"]) == tmp_path / "processed" / (split + ".jsonl")
    assert {p.name: p.read_bytes() for p in raw.iterdir()} == before
