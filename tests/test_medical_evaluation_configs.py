"""Generate the full comparison without truncating tests or sharing output folders."""

from decisionforge.eval import load_config


def test_full_matrix_uses_distinct_outputs_same_frozen_inputs_and_native_adapters(tmp_path):
    from examples.domain.medical_triage.scripts.v1.experiment import write_eval_configs

    task, store = tmp_path / "task", tmp_path / "store"
    paths = write_eval_configs(task, store, store / "model", "/frozen/Qwen3.5-9B")
    configs = [load_config(path) for path in paths]
    assert len(configs) == len({c.output for c in configs}) == 15
    assert {c.model.adapter for c in configs} == {
        "unsloth_local",
        "jev",
        "anyjev_local",
        "chat_api",
    }
    assert all(c.data.limit is None and c.data.splits == ["test"] for c in configs)
    assert len({c.data.path for c in configs}) == 3
    assert len({c.model.path for c in configs if c.model.adapter == "unsloth_local"}) == 1
    assert write_eval_configs(task, store, store / "model", "/frozen/Qwen3.5-9B") == paths


def test_qwen35_anyjev_configs_disable_unsupported_shared_prefix_cache(tmp_path):
    from examples.domain.medical_triage.scripts.v1.experiment import write_eval_configs

    configs = [
        load_config(path)
        for path in write_eval_configs(
            tmp_path / "task", tmp_path / "store", tmp_path / "model", "/frozen/Qwen3.5-9B"
        )
    ]
    assert all(
        config.model.shared_prefix is False
        for config in configs
        if config.model.adapter == "anyjev_local"
    )
