import json
from pathlib import Path

import httpx
import pytest
import yaml

from data.benchmarks._shared.converters import (
    convert_intents,
    convert_jevbench,
    convert_typed_decisions,
)
from data.benchmarks._shared.prepare import main as prepare_main
from data.benchmarks._shared.sources import CLINC_PIN, fetch_dataset
from decisionforge.cli import main
from decisionforge.eval.benchmarks import GenericBenchmark
from decisionforge.eval.data import load_samples, write_jsonl
from decisionforge.eval.models import Sample
from decisionforge.eval.registry import load_registry
from tests.eval_fixtures import prediction, sample


def test_macro_f1_uses_native_choice_and_counts_failed_attempts():
    items = [sample(identity=str(i)) for i in range(4)]
    for item, label in zip(items, ["A", "A", "B", "B"], strict=True):
        item["target"]["value"] = label
    outcomes = [prediction(str(i)) for i in range(4)]
    for outcome, label in zip(outcomes, ["A", "B", "B", None], strict=True):
        if label is None:
            outcome.update(transport_ok=False, protocol_ok=False, answer=None)
        else:
            outcome["answer"]["choice"] = label
    # Classification must remain usable even with a malformed probability sum.
    outcomes[0]["answer"]["probabilities"] = {"A": 0.6, "B": 0.6}
    result = GenericBenchmark().evaluate(
        [Sample.model_validate(s) for s in items],
        outcomes,
        load_registry(),
        {"macro_f1": {}, "native_accuracy": {}},
    )["metrics"]
    assert result["macro_f1"]["value"] == pytest.approx((2 / 3 + 1 / 2) / 2)
    assert result["macro_f1"]["failed"] == 1
    assert result["macro_f1"]["per_class"]["B"]["fn"] == 1
    assert result["native_accuracy"]["value"] == 0.5


def test_macro_f1_keeps_absent_classes_and_handles_no_attempts():
    s = sample()
    s["target"]["value"] = "B"
    result = GenericBenchmark().evaluate(
        [Sample.model_validate(s)], [prediction()], load_registry(), {"macro_f1": {}}
    )["metrics"]["macro_f1"]
    assert result["value"] == 0.5
    assert result["per_class"]["A"]["f1"] == 0
    result = GenericBenchmark().evaluate(
        [Sample.model_validate(s)], [], load_registry(), {"macro_f1": {}}
    )
    assert result["metrics"]["macro_f1"]["value"] is None
    assert result["partial"] is True


def test_soft_kl_and_modal_gold_accuracy_have_separate_meanings():
    s = sample(target="soft")
    s["metadata"]["gold_label"] = "B"
    result = GenericBenchmark().evaluate(
        [Sample.model_validate(s)],
        [prediction(probabilities={"A": 0.2, "B": 0.8})],
        load_registry(),
        {"gold_accuracy": {}, "soft_kl": {}, "gold_ece": {}},
    )["metrics"]
    assert result["soft_kl"]["value"] == pytest.approx(0)
    assert result["gold_accuracy"]["value"] == 1
    assert result["gold_ece"]["value"] == pytest.approx(0.2)
    with pytest.raises(ValueError):
        load_registry().metric("soft_kl").parameters({"epsilon": 0})


def typed_row():
    return {
        "id": "case-1",
        "workflow": "workflow",
        "split": "test",
        "n_questions": 2,
        "state": json.dumps({"text": "visible state"}),
        "questions": json.dumps(
            {
                "intent": {
                    "type": "choice",
                    "instructions": "Choose",
                    "criteria": {"A": "a", "B": "b"},
                },
                "urgent": {"type": "noul", "instructions": "Urgent?"},
            }
        ),
        "gold": json.dumps(
            {
                "intent": {
                    "type": "choice",
                    "label": "B",
                    "probabilities": {"A": 0.2, "B": 0.799999},
                },
                "urgent": {
                    "type": "noul",
                    "label": "true",
                    "probabilities": {"false": 0.3, "true": 0.7},
                },
            }
        ),
        "factors": {"secret_factor": "never model input"},
        "label_agreement": {},
    }


def test_typed_converter_preserves_soft_gold_and_public_request_only(tmp_path):
    source = tmp_path / "typed.jsonl"
    write_jsonl(source, [typed_row()])
    output = tmp_path / "converted.jsonl"
    manifest = convert_typed_decisions(source, output, source_version="pinned")
    items = load_samples(output)
    assert manifest["records"] == 2
    assert items[0].target.kind == "soft"
    assert items[0].metadata["reference_normalization"]["original_sum"] == pytest.approx(0.999999)
    assert items[0].metadata["source_gold"]["probabilities"]["B"] == 0.799999
    assert items[1].id == "case-1::urgent"
    assert items[0].group_id == items[1].group_id == "case-1"
    assert items[0].metadata["gold_label"] == "B"
    assert items[0].request.state == {"text": "visible state"}
    assert set(items[0].request.questions) == {"decision"}
    assert "gold" not in items[0].request.model_dump()
    assert "secret_factor" not in json.dumps(items[0].request.model_dump())


def test_typed_converter_rejects_missing_or_invalid_gold_before_writing(tmp_path):
    source = tmp_path / "typed.jsonl"
    row = typed_row()
    gold = json.loads(row["gold"])
    gold["intent"]["probabilities"]["A"] = 0.5
    row["gold"] = json.dumps(gold)
    write_jsonl(source, [row])
    output = tmp_path / "converted.jsonl"
    with pytest.raises(ValueError, match="probability|Probability"):
        convert_typed_decisions(source, output)
    assert not output.exists()


def test_intent_catalog_is_fixed_and_does_not_come_from_test_labels(tmp_path):
    labels = ["Refund_not_showing_up", "reverted_card_payment?"] + [
        f"intent_{i}" for i in range(75)
    ]
    (tmp_path / "labels.json").write_text(json.dumps(labels))
    (tmp_path / "test.csv").write_text("text,category\nRefund missing,Refund_not_showing_up\n")
    output = tmp_path / "converted.jsonl"
    convert_intents("banking77", tmp_path, output)
    s = load_samples(output)[0]
    assert len(s.question.labels) == 77
    assert s.target.value == "Refund_not_showing_up"
    assert list(s.question.criteria) == labels
    assert s.question.criteria["reverted_card_payment?"] == "reverted card payment?"
    assert s.request.state == "Refund missing"


def test_clinc_oos_mapping_and_parquet_conversion(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    pa = pytest.importorskip("pyarrow")
    labels = ["oos"] + [f"intent_{i}" for i in range(150)]
    (tmp_path / "labels.json").write_text(json.dumps(labels))
    pq.write_table(
        pa.table({"text": ["unsupported request", "known request"], "intent": [0, 1]}),
        tmp_path / "test.parquet",
    )
    output = tmp_path / "converted.jsonl"
    convert_intents("clinc150_oos", tmp_path, output)
    samples = load_samples(output)
    assert [s.target.value for s in samples] == ["oos", "intent_0"]
    assert len(samples[0].question.criteria) == 151
    assert samples[0].subset == "oos"
    assert samples[1].subset == "in_scope"
    assert "None of" in samples[0].question.criteria["oos"]


def test_extension_benchmarks_are_registered():
    registry = load_registry()
    for name in ("banking77", "clinc150_oos", "typed_decisions"):
        assert registry.benchmark(name).name == name


@pytest.mark.parametrize("name", ["banking77", "clinc150_oos"])
def test_intent_headline_f1_cannot_override_fixed_classes(name):
    registry = load_registry()
    with pytest.raises(ValueError, match="complete fixed label catalog"):
        registry.benchmark(name).validate(registry, {"macro_f1": {"labels": ["intent_0"]}})


def test_all_jevbench_tiers_and_cross_file_duplicate_rejection(tmp_path):
    fixture = Path(__file__).parent / "fixtures/jevbench_oracle/original.jsonl"
    original = json.loads(fixture.read_text().splitlines()[0])
    for tier in ("original", "easy", "hard"):
        write_jsonl(tmp_path / f"{tier}.jsonl", [{**original, "id": tier}])
    output = tmp_path / "converted.jsonl"
    assert convert_jevbench(tmp_path, output)["records"] == 3
    assert [s.metadata["tier"] for s in load_samples(output)] == ["original", "easy", "hard"]
    write_jsonl(tmp_path / "hard.jsonl", [{**original, "id": "easy"}])
    with pytest.raises(ValueError, match="duplicate"):
        convert_jevbench(tmp_path, tmp_path / "failed.jsonl")
    assert not (tmp_path / "failed.jsonl").exists()


def test_jevbench_directory_conversion_cannot_overwrite_source(tmp_path):
    fixture = Path(__file__).parent / "fixtures/jevbench_oracle/original.jsonl"
    source = tmp_path / "original.jsonl"
    source.write_bytes(fixture.read_bytes())
    before = source.read_bytes()
    with pytest.raises(ValueError, match="differ from source"):
        convert_jevbench(tmp_path, source)
    assert source.read_bytes() == before


def test_pinned_fetch_clinc_mapping_verified_cache_and_tamper_rejection(tmp_path):
    labels = [f"intent_{i}" for i in range(151)]
    labels[42] = "oos"
    card = {
        "dataset_info": [
            {
                "config_name": "plus",
                "features": [
                    {
                        "name": "intent",
                        "dtype": {
                            "class_label": {
                                "names": {str(i): name for i, name in enumerate(labels)}
                            }
                        },
                    }
                ],
            }
        ]
    }
    calls = []

    def handler(request):
        calls.append(str(request.url))
        assert CLINC_PIN in str(request.url)
        content = (
            "---\n" + yaml.safe_dump(card) + "---\nCard"
            if request.url.path.endswith("README.md")
            else "parquet placeholder"
        )
        return httpx.Response(200, text=content)

    transport = httpx.MockTransport(handler)
    source = tmp_path / "source"
    manifest = fetch_dataset("clinc150_oos", source, transport=transport)
    assert json.loads((source / "labels.json").read_text())[42] == "oos"
    assert len(calls) == 2
    assert fetch_dataset("clinc150_oos", source, transport=transport) == manifest
    assert len(calls) == 2
    with pytest.raises(ValueError, match="incompatible"):
        fetch_dataset("clinc150_oos", source, "train", transport)
    (source / "labels.json").write_text("[]")
    with pytest.raises(ValueError, match="hash mismatch"):
        fetch_dataset("clinc150_oos", source, transport=transport)
    assert len(calls) == 2


def test_source_bundle_identity_rejected_before_conversion(tmp_path):
    (tmp_path / "source.json").write_text(json.dumps({"dataset": "other", "files": []}))
    for convert in (
        lambda: convert_intents("banking77", tmp_path, tmp_path / "out.jsonl"),
        lambda: convert_typed_decisions(tmp_path, tmp_path / "out.jsonl"),
    ):
        with pytest.raises(ValueError, match="another dataset"):
            convert()


def test_clinc_full_label_macro_f1_and_oos_breakdown():
    labels = ["oos"] + [f"intent_{i}" for i in range(150)]
    samples, predictions = [], []
    for i, (gold, selected) in enumerate(
        (("intent_0", "intent_0"), ("oos", "oos"), ("oos", "intent_0"))
    ):
        item = sample(identity=str(i))
        item["request"]["questions"]["decision"]["criteria"] = dict.fromkeys(labels, "description")
        item["target"]["value"] = gold
        samples.append(Sample.model_validate(item))
        outcome = prediction(str(i), probabilities=dict.fromkeys(labels, 1 / 151))
        outcome["answer"]["choice"] = selected
        predictions.append(outcome)
    registry = load_registry()
    benchmark = registry.benchmark("clinc150_oos")
    benchmark.validate_samples(samples)
    result = benchmark.evaluate(samples, predictions, registry, {})
    assert result["metrics"]["macro_f1"]["value"] == pytest.approx((4 / 3) / 151)
    assert result["oos"]["recall"] == 0.5
    assert result["in_scope_macro_f1"]["value"] == pytest.approx((2 / 3) / 150)
    assert result["decision_rule"] == "native_choice_for_classification"


def test_extension_cli_and_label_preflight_without_credentials(tmp_path, capsys):
    source = tmp_path / "raw/test"
    source.mkdir(parents=True)
    (source / "labels.json").write_text(json.dumps([f"intent_{i}" for i in range(77)]))
    (source / "test.csv").write_text("text,category\nrequest,intent_0\n")
    output = tmp_path / "processed/test.jsonl"
    assert prepare_main("banking77", tmp_path, ["--convert-only"]) == 0
    capsys.readouterr()
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "benchmark": "banking77",
                "data": {"path": str(output)},
                "output": str(tmp_path / "run"),
            }
        )
    )
    assert main(["eval", "run", "--config", str(config), "--validate-only"]) == 0
    assert json.loads(capsys.readouterr().out)["samples"] == 1
    assert not (tmp_path / "run").exists()
    config.write_text(yaml.safe_dump({"benchmark": "clinc150_oos", "data": {"path": str(output)}}))
    assert main(["eval", "run", "--config", str(config), "--validate-only"]) == 1
