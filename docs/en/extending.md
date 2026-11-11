# Extending and development

[中文](../zh/extending.md) · [README](../../README.en.md)

Extend data conversion, model transport and scoring separately instead of copying the runner for each task. Development setup:

```bash
uv sync --locked --group dev
uv run ruff check
uv run pytest
```

## Convert data

Convert sources to [JEV](jev-format.md) or evaluation JSONL, preserving source, group, split and hashes. Evaluation records contain id, group_id, split, subset, request, target and metadata. Targets may be hard, soft or scalar. Never mix labels into request. Validate using `eval run --validate-only` before calling models.

## Benchmarks and metrics

```python
from decisionforge.eval import BaseBenchmark, Metric, Registry, load_registry
```

Benchmarks own data views, compatible metrics and aggregation; metrics compute scores. Built-ins live in `src/decisionforge/benchmarks/<name>/` with a `register(registry)` hook. Task-specific implementations can use Python entry points.

See the runnable [extension example](../../examples/eval/plugin/). Its entry point group is `decisionforge.eval.benchmarks` / `decisionforge.eval.metrics`; follow the example for benchmark and metric selection. Install extensions in the user environment without private local paths. Use unique registration names. Version scoring changes and preserve source identity.

## Generation and training backends

`generation/contracts.py` defines input, generation and checker responses. `intake.py` handles inputs, `components.py` compiles JEV, `pipeline.py` manages concurrency, repair and budgets, and `storage.py` manages checkpoints. `llm.py` owns model transport and caching. New modules must preserve native format, bounded retries, source isolation and resumable state.

Training backends live in `training/backends/`. Keep data, calibration, saving and evaluation reload consistent with versioned artifact contracts. Import optional SDKs lazily and name the required extra in errors. Worker argparse is an internal process protocol; public commands use Typer. Do not add parallel public entry points.

## Acceptance

`tests/acceptance/` contains manual model/GPU acceptance drivers. Normal pytest uses offline or mocked checks by default. Agree on resources and budget before paid or GPU work; preserve configuration, model/data identity and real artifacts. Reports must distinguish static validation, actual execution and unverified behavior.

See the [plugin README](../../extensions/dsh-decisionforge/README.md) and [Demo README](../../demo/README.md) for their development workflows.
