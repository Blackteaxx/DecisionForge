import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from test_pipeline import run
from test_text_batching import BatchProvider, setup_text

from decisionforge.config import OutputConfig, load_config
from decisionforge.exporter import export_chat_sft, export_run, validate_file
from decisionforge.generation.components import compile_record, initial_draft
from decisionforge.generation.intake import normalize
from decisionforge.storage import Store
from decisionforge.training.config import TrainConfig
from decisionforge.training.data import prepare_data


def add_record(store, config, index, *, split="train", group="document", accepted=True):
    source = normalize(
        {
            "question": f"Case {index}: what is 2+2?",
            "answer": "4",
            "options": ["4", "5"],
            "split": split,
            "group_id": group,
        },
        config,
    )
    identifier = f"case-{index}:choice"
    record = compile_record(
        source, initial_draft(source, "choice", None), "choice", f"case-{index}", identifier, config
    ).model_dump(mode="json")
    store.finish(
        identifier,
        "accepted" if accepted else "review",
        record if accepted else {"jev": record, "reason": "rejected"},
    )


@pytest.mark.parametrize("fraction", [-0.1, 1.0, float("nan"), float("inf"), True, "0.2"])
def test_invalid_fraction(fraction):
    with pytest.raises(ValidationError):
        OutputConfig(validation_fraction=fraction)


@pytest.mark.parametrize("seed", [-1, 0.5, True])
def test_invalid_seed(seed):
    with pytest.raises(ValidationError):
        OutputConfig(validation_seed=seed)


def test_yaml_split_configuration(tmp_path):
    path = tmp_path / "generation.yaml"
    path.write_text(
        "input: {path: source.txt}\noutput: {validation_fraction: 0.2, validation_seed: 7}\n"
    )
    config = load_config(path)
    assert config.output.validation_fraction == 0.2
    assert config.output.validation_seed == 7


def test_export_holdout_preserves_labels_existing_splits_and_failed_records(config):
    directory = Path(config.output.directory)
    store = Store(directory)
    try:
        for i in range(10):
            add_record(store, config, i)
        for i, split in enumerate(("validation", "test", "calibration"), 10):
            add_record(store, config, i, split=split, group=split)
        add_record(store, config, 13, accepted=False)
        original_outcomes = list(store.outcomes())
        store.put("effective_config", config.model_dump())
        export_run(store, directory)
        original = json.loads((directory / "accepted.json").read_text())
        original_failed = (directory / "fail.json").read_bytes()

        config.output.validation_fraction = 0.3
        store.put("effective_config", config.model_dump())
        summary = export_run(store, directory, export_requests=True)
        assert summary["splits"] == {"train": 7, "validation": 4, "test": 1, "calibration": 1}
        assert summary["validation_split"]["eligible_records"] == 10
        assert summary["validation_split"]["validation_records"] == 3
        assert (directory / "fail.json").read_bytes() == original_failed
        assert list(store.outcomes()) == original_outcomes
        assert validate_file(directory / "accepted.json") == 13
        rows = json.loads((directory / "accepted.json").read_text())
        for before, after in zip(original, rows):
            assert before["record_id"] == after["record_id"]
            assert before["request"] == after["request"]
            assert before["supervision"] == after["supervision"]
            if before["split"] != "train":
                assert before == after
        targets = [
            json.loads(line) for line in (directory / "targets.jsonl").read_text().splitlines()
        ]
        assert [r["split"] for r in targets] == [r["split"] for r in rows]
        meta = json.loads((directory / "meta.json").read_text())
        for row in rows:
            assignment = meta["records"][str(row["record_id"])]["split_assignment"]
            assert assignment["split"] == row["split"]
            assert assignment["group_id"] == row["group_id"]
        c = TrainConfig(
            data={"path": str(directory / "accepted.json")}, calibration={"enabled": False}
        )
        assert prepare_data(c).report["roles"]["train"]["records"] == 7
        export_chat_sft(directory)
        assert len((directory / "chat_sft.jsonl").read_text().splitlines()) == 7

        # Repeated exports must not take another fraction from the remaining training set.
        export_run(store, directory)
        assert json.loads((directory / "accepted.json").read_text()) == rows
        # New original groups must not collide with previously allocated validation groups.
        add_record(store, config, 14, split="test", group="new-document")
        export_run(store, directory)
        assert validate_file(directory / "accepted.json") == 14
        # Opting out restores the source grouping and source splits.
        config.output.validation_fraction = 0
        store.put("effective_config", config.model_dump())
        summary = export_run(store, directory)
        assert "validation_split" not in summary
        restored = json.loads((directory / "accepted.json").read_text())
        assert all(r in restored for r in original)
    finally:
        store.close()


@pytest.mark.parametrize(
    "total,fraction,expected",
    [
        (0, 0.2, 0),
        (1, 0.2, 0),
        (2, 0.01, 1),
        (64, 0.2, 12),
        (3, 0.99, 2),
    ],
)
def test_rounding_and_small_exports(config, total, fraction, expected):
    directory = Path(config.output.directory)
    store = Store(directory)
    try:
        config.output.validation_fraction = fraction
        store.put("effective_config", config.model_dump())
        for i in range(total):
            add_record(store, config, i)
        summary = export_run(store, directory)
        assert summary["splits"].get("validation", 0) == expected
        assert summary["splits"].get("train", 0) == total - expected
        assert summary["validation_split"]["status"] == (
            "assigned" if total >= 2 else "insufficient_records"
        )
        assert validate_file(directory / "accepted.json") == total
    finally:
        store.close()


async def test_batched_txt_holdout_resume_and_training(config):
    setup_text(config)
    config.output.validation_fraction = 0.2
    first, files, _ = await run(config, BatchProvider(interrupt_south=True))
    assert first["status"] == "budget_exhausted"
    assert first["validation_split"]["status"] == "insufficient_records"
    result, files, _ = await run(config, BatchProvider())
    assert result["splits"] == {"train": 1, "validation": 1}
    assert len({r["group_id"] for r in files["accepted"]}) == 2
    assert (
        len(
            {
                r["split_assignment"]["original_group_id"]
                for r in files["meta"]["records"].values()
                if r["output"] == "accepted.json"
            }
        )
        == 1
    )
    c = TrainConfig(
        data={"path": str(Path(config.output.directory) / "accepted.json")},
        calibration={"enabled": False},
    )
    prepared = prepare_data(c)
    assert len(prepared.roles["train"]) == len(prepared.roles["validation"]) == 1
    _, repeated, provider = await run(config, BatchProvider())
    assert repeated["accepted"] == files["accepted"]
    assert not provider.calls


def test_seed_reproducible_and_export_order_independent(config, tmp_path):
    def selection(directory, seed, incremental):
        config.output.validation_fraction = 0.3
        config.output.validation_seed = seed
        store = Store(directory)
        try:
            store.put("effective_config", config.model_dump())
            for i in reversed(range(30)):
                add_record(store, config, i)
                if incremental:
                    export_run(store, directory)
            export_run(store, directory)
            return {
                r["request"]["questions"]["q1"]["instructions"]
                for r in json.loads((directory / "accepted.json").read_text())
                if r["split"] == "validation"
            }
        finally:
            store.close()

    first = selection(tmp_path / "first", 42, False)
    assert first == selection(tmp_path / "incremental", 42, True)
    assert first != selection(tmp_path / "other_seed", 43, False)
