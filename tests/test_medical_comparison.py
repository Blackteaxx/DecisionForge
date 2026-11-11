"""A completed comparison must contain the full authorized method/dataset matrix."""

import pytest

METHODS = ("trained", "jev", "llm2jev", "anyjev", "chat")
DATASETS = {"care": 135, "synth": 869, "reddit": 1502}


def cells():
    return [
        {
            "method": m,
            "dataset": d,
            "status": "completed",
            "sample_sha256": "fixed-" + d,
            "counts": {"planned": n, "completed": n, "unattempted": 0, "incomplete": 0},
        }
        for m in METHODS
        for d, n in DATASETS.items()
    ]


def test_comparison_requires_every_complete_method_dataset_cell():
    from examples.domain.medical_triage.scripts.v1.report import validate_matrix

    assert (
        len(
            validate_matrix(
                cells(),
                {d: {"count": n, "sample_sha256": "fixed-" + d} for d, n in DATASETS.items()},
            )
        )
        == 15
    )
    with pytest.raises(ValueError, match="missing|matrix"):
        validate_matrix(
            cells()[:-1],
            {d: {"count": n, "sample_sha256": "fixed-" + d} for d, n in DATASETS.items()},
        )


@pytest.mark.parametrize(
    "change",
    [
        {"status": "running"},
        {"sample_sha256": "different"},
        {"counts": {"planned": 135, "completed": 134, "unattempted": 1, "incomplete": 0}},
    ],
)
def test_comparison_refuses_partial_or_different_inputs(change):
    from examples.domain.medical_triage.scripts.v1.report import validate_matrix

    values = cells()
    values[0].update(change)
    with pytest.raises(ValueError, match="complete|frozen|planned"):
        validate_matrix(
            values, {d: {"count": n, "sample_sha256": "fixed-" + d} for d, n in DATASETS.items()}
        )


def test_one_or_two_patient_components_cannot_yield_independent_pair_intervals():
    from examples.domain.medical_triage.scripts.v1.report import group_interval

    for count in (1, 2):
        result = group_interval([(str(i % count), i % 2 == 0) for i in range(100)])
        assert result["interval"] is None
        assert result["groups"] == count
        assert result["reason"]
    result = group_interval([(str(i), True) for i in range(30)])
    assert result["interval"] == [1.0, 1.0]


def test_unknown_usage_and_cost_are_preserved_as_unknown():
    from examples.domain.medical_triage.scripts.v1.report import performance_view

    result = performance_view(
        {
            "usage": {"input_tokens": None, "output_tokens": None, "missing_attempts": 135},
            "reported_cost_usd": None,
            "price_per_1000_decisions_usd": None,
            "missing_cost_attempts": 135,
        }
    )
    assert result["usage"]["input_tokens"] is None
    assert result["reported_cost_usd"] is None
    assert result["price_per_1000_decisions_usd"] is None


def test_trained_comparison_accepts_recorded_selected_file_identity():
    from examples.domain.medical_triage.scripts.v1.report import verify_selected_artifact

    files = {"adapter_model.safetensors": {"size": 10, "sha256": "original"}}
    expected = {"path": "/selected/model", "files": files}
    manifest = {
        "config": {"model": {"path": "/selected/model"}},
        "inference": {"local_checkpoint": {"checkpoint": expected}},
    }
    verify_selected_artifact(manifest, expected)


@pytest.mark.parametrize(
    "filename",
    ["adapter_model.safetensors", "joint_head.safetensors", "unsloth_decision_config.json"],
)
def test_same_model_path_with_different_recorded_weights_or_calibration_is_rejected(filename):
    from examples.domain.medical_triage.scripts.v1.report import verify_selected_artifact

    expected = {"path": "/selected/model", "files": {filename: {"size": 10, "sha256": "selected"}}}
    recorded = {"path": "/selected/model", "files": {filename: {"size": 10, "sha256": "replaced"}}}
    manifest = {
        "config": {"model": {"path": "/selected/model"}},
        "inference": {"local_checkpoint": {"checkpoint": recorded}},
    }
    with pytest.raises(ValueError, match="identity|selected|artifact"):
        verify_selected_artifact(manifest, expected)


def test_trained_comparison_refuses_missing_recorded_checkpoint_identity():
    from examples.domain.medical_triage.scripts.v1.report import verify_selected_artifact

    manifest = {"config": {"model": {"path": "/selected/model"}}, "inference": {}}
    with pytest.raises(ValueError, match="identity|selected|artifact"):
        verify_selected_artifact(
            manifest,
            {
                "path": "/selected/model",
                "files": {"joint_head.safetensors": {"size": 10, "sha256": "selected"}},
            },
        )
