import json
import re
from collections import Counter
from pathlib import Path

import pytest
from conftest import FixtureProvider

from decisionforge.adapters import read_records
from decisionforge.cli import inspect_input
from decisionforge.config import load_config
from decisionforge.exporter import validate_file
from decisionforge.generation.intake import normalize
from decisionforge.pipeline import Pipeline
from decisionforge.storage import Store

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "tests/fixtures/power_grid"


def text_cases(text):
    """Parse this fixture's explicit labels; production TXT intake still uses an LLM."""
    records = []
    for block in re.split(r"(?:\r?\n){2}", text.strip()):
        record = {"context": "", "options": []}
        for line in block.splitlines():
            if line.startswith("- "):
                record["options"].append(line[2:])
            elif ": " in line:
                key, value = line.split(": ", 1)
                record[key.lower()] = value
        records.append(record)
    return records


class PowerGridProvider(FixtureProvider):
    async def call(self, role, payload, schema, record, stage):
        if role == "extractor":
            self.calls.append((role, payload, stage))
            return schema.model_validate(
                {"records": text_cases(payload["source"]), "reason": "Existing labeled cases"}
            )
        if role == "context":
            self.calls.append((role, payload, stage))
            return schema.model_validate(
                {
                    "context": "The reference value for this training case is "
                    + payload["source"]["answer"]
                    + "."
                }
            )
        return await super().call(role, payload, schema, record, stage)


class PowerGridNarrativeProvider(FixtureProvider):
    """Known semantic outputs for testing the actual narrative fixture/config wiring."""

    async def call(self, role, payload, schema, record, stage):
        if role in {"extractor", "text_assessor", "qa_constructor"}:
            self.calls.append((role, payload, stage))
        if role == "extractor":
            return schema(records=[], reason="Narrative without existing QA")
        if role == "text_assessor":
            blocks = re.split(r"(?:\r?\n){2}", payload["source"].strip())[1:]
            return schema(
                is_narrative=True,
                opportunities=[
                    {
                        "mode": "choice",
                        "focus": str(i),
                        "evidence": [block],
                        "reason": "Explicit fictional decision rule and observations",
                    }
                    for i, block in enumerate(blocks)
                ],
                reason="Three grounded decisions" if blocks else "No actionable decision evidence",
            )
        if role == "qa_constructor":
            index = int(payload["opportunity"]["focus"])
            return schema(
                qa={
                    "question": [
                        "With a 12 MW deficit and available 8/6 MW reserves, which minimum unit set covers it?",
                        "With 24 MW demand and 20 MW supply, which smallest single 3/4/6 MW block covers the deficit?",
                        "Under the stated 80/90 degree policy, which action applies to T7 at 86 degrees with no exceptions?",
                    ][index],
                    "answer": [
                        "Activate both units",
                        "Remove the 4 MW Pumping block",
                        "Repeat the measurement this shift",
                    ][index],
                },
                reason="Decision follows the cited rule and observed values",
            )
        return await super().call(role, payload, schema, record, stage)


@pytest.mark.parametrize("name,count", [("narrative", 3), ("narrative_unusable", 0)])
async def test_power_grid_narrative_fixtures(name, count, monkeypatch, tmp_path):
    from pipeline_helpers import fixture_pipeline

    monkeypatch.setenv("OPENAI_MODEL", "fixture")
    config = load_config(DIRECTORY / f"{name}.yaml")
    config.output.directory = str(tmp_path / name)
    assert inspect_input(config)["records"] == 1
    provider = PowerGridNarrativeProvider()
    store = Store(Path(config.output.directory))
    try:
        result = await fixture_pipeline(config, store, provider).run()
        assert result["accepted"] == count
        assert result["input_failures"] == (0 if count else 1)
        assert validate_file(Path(config.output.directory) / "accepted.json") == count
        calls = Counter(role for role, _, _ in provider.calls)
        assert calls["extractor"] == calls["text_assessor"] == 1
        assert calls["qa_constructor"] == calls["choice"] == calls["checker"] == count
        assert calls["context"] == 0
        if count:
            records = json.loads((Path(config.output.directory) / "accepted.json").read_text())
            assert len({row["group_id"] for row in records}) == 1
            assert all(row["request"]["questions"]["q1"]["type"] == "choice" for row in records)
    finally:
        store.close()
def test_power_grid_formats_preserve_all_ten_cases(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "fixture")
    assert {p.name for p in DIRECTORY.iterdir() if p.suffix in {".json", ".txt"}} == {
        "standard.json",
        "messy_keys.json",
        "source.txt",
        "narrative.txt",
        "narrative_unusable.txt",
    }
    config = load_config(DIRECTORY / "standard.yaml")
    standard = [normalize(r.payload, config) for r in read_records(DIRECTORY / "standard.json")]
    messy = [normalize(r.payload, config) for r in read_records(DIRECTORY / "messy_keys.json")]
    assert len(standard) == len(messy) == 10
    assert messy == standard
    extracted = text_cases((DIRECTORY / "source.txt").read_text(encoding="utf-8"))
    assert len(extracted) == 10
    for row, text_row in zip(standard, extracted, strict=True):
        assert {key: row.model_dump()[key] for key in text_row} == text_row
    coverage = Counter((bool(r.context), len(r.options) == 4) for r in standard)
    assert coverage == {(True, True): 3, (True, False): 2, (False, True): 2, (False, False): 3}
    for path in DIRECTORY.iterdir():
        if path.suffix in {".json", ".txt"}:
            assert path.read_text(encoding="utf-8").isascii()


@pytest.mark.parametrize("name", ["standard", "messy_keys", "txt"])
@pytest.mark.parametrize("generate_context", [True, False])
async def test_power_grid_examples_route_missing_components(
    name, generate_context, tmp_path, monkeypatch
):
    """Offline integration contract; the model double does not assess semantic quality."""
    monkeypatch.setenv("OPENAI_MODEL", "fixture")
    config = load_config(DIRECTORY / f"{name}.yaml")
    assert inspect_input(config)["records"] == (1 if name == "txt" else 10)
    config.context.generate_if_missing = generate_context
    config.output.directory = str(tmp_path / name)
    source = json.loads((DIRECTORY / "standard.json").read_text(encoding="utf-8"))
    gold = {r["question"]: r for r in source}
    provider = PowerGridProvider()
    store = Store(Path(config.output.directory))
    try:
        result = await Pipeline(config, store, provider).run()
        assert result["accepted"] == 10 and result["review"] == 0, result
        calls = Counter(role for role, _, _ in provider.calls)
        assert calls["extractor"] == (1 if name == "txt" else 0)
        assert calls["context"] == (5 if generate_context else 0)
        assert calls["choice"] == 5
        assert calls["checker"] == 10
        path = Path(config.output.directory) / "accepted.json"
        assert validate_file(path) == 10
        accepted = json.loads(path.read_text(encoding="utf-8"))
        assert len({row["request"]["questions"]["q1"]["instructions"] for row in accepted}) == 10
        for row in accepted:
            request = row["request"]
            original = gold[request["questions"]["q1"]["instructions"]]
            choice = row["supervision"]["answers"]["q1"]["choice"]
            criteria = request["questions"]["q1"]["criteria"]
            assert set(criteria) == {"A", "B", "C", "D"}
            assert criteria[choice] == original["answer"]
            assert row["supervision"]["answers"]["q1"]["probabilities"] == {
                key: float(key == choice) for key in criteria
            }
            if len(original.get("options", [])) == 4:
                assert set(criteria.values()) == set(original["options"])
            if original["context"] or not generate_context:
                assert request["state"] == original["context"]
            else:
                assert request["state"] and request["state"] != original["question"]
        assert (
            json.loads((Path(config.output.directory) / "fail.json").read_text(encoding="utf-8"))
            == []
        )
    finally:
        store.close()
