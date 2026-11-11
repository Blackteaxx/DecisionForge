import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest
from PIL import Image


def prepare_module():
    path = Path(__file__).parents[1] / "data/benchmarks/scienceqa_img/prepare.py"
    spec = importlib.util.spec_from_file_location("scienceqa_prepare_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_at(root):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    module = prepare_module()
    (root / "raw").mkdir()
    source = {"source_revision": module.REVISION, "files": {}}
    for split in module.FILES:
        image = io.BytesIO()
        Image.new("RGB", (1200, 1000), "blue").save(image, format="PNG")
        rows = [
            {
                "image": {"bytes": image.getvalue(), "path": None},
                "choices": ["first", "second"],
                "answer": 1,
                "question": f"{split}: question",
                "hint": "Verbatim public context",
                "lecture": "PRIVATE LECTURE",
                "solution": "PRIVATE SOLUTION",
                "subject": "science",
            },
            {
                "image": None,
                "choices": ["one", "two"],
                "answer": 0,
                "question": "text-only",
                "hint": "",
                "lecture": "secret",
                "solution": "secret",
                "subject": "science",
            },
        ]
        path = root / "raw" / f"{split}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)
        source["files"][split] = {
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (root / "raw/source.json").write_text(json.dumps(source))
    return module


def test_full_img_split_preserved_and_private_explanations_excluded(tmp_path):
    from decisionforge.eval.data import load_samples
    from decisionforge.exporter import read_training_records

    module = source_at(tmp_path)
    assert module.convert(tmp_path) == {"train": 1, "validation": 1, "test": 1}
    rows = load_samples(tmp_path / "processed/test.jsonl")
    assert rows[0].request.state == "Verbatim public context"
    assert rows[0].target.value == "1"
    public = rows[0].request.model_dump_json()
    assert "PRIVATE" not in public and "solution" not in public
    image = Image.open(rows[0].request.images[0].path)
    assert image.width * image.height <= module.MAX_PIXELS
    manifest = json.loads((tmp_path / "processed/test.jsonl.manifest.json").read_text())
    assert manifest["records"] == 1 and manifest["excluded_text_only"] == 1
    assert manifest["images"][0]["original_size"] == [1200, 1000]
    training = list(read_training_records(tmp_path / "processed/training.jsonl"))
    assert len(training) == 3 and len({r.record_id for r in training}) == 3
    assert next(r for r in training if r.split == "test").request.questions[
        "decision"
    ].criteria == {"0": "first", "1": "second"}


def test_convert_only_rejects_changed_source_bytes(tmp_path):
    module = source_at(tmp_path)
    path = tmp_path / "raw/test.parquet"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        module.convert(tmp_path, splits=["test"])
