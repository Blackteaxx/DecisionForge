# Data generation

[中文](../zh/data-generation.md) · [README](../../README.en.md)

Prepare checkable JEV data from JSON, TXT or PDF. Paths in YAML resolve relative to that file; CLI path overrides resolve from the current directory.

```bash
uv run dforge data inspect --config configs/offline.yaml
uv run dforge data build --config configs/default.yaml --difficulty 0.5 --modes choice
uv run dforge data resume --run-dir outputs/run_001
```

## Inputs and configuration

`input.type` supports `auto/json/txt/pdf`. For JSON, `records_path` selects an array and `mapping_spec` maps fields. TXT preserves source evidence; a model assesses whether questions can be constructed. Material without sufficient evidence is not reliable supervision. PDF parsing uses the MinerU HTTP API to produce reusable TXT.

```bash
uv sync --extra pdf
uv run dforge data parse-pdf --config configs/pdf-mineru.yaml --input /path/to/paper.pdf
# Repeat --input for multiple files
uv run dforge data parse-pdf --config configs/pdf-mineru.yaml --input a.pdf --input b.pdf
```

MinerU uses `MINERU_API_KEY`, configurable through `mineru.api_key_env`. Cache files live under `.dforge/pdf/`; manifests record sources, parse status and TXT paths. `input.text_batching` controls long TXT: build asks a model to plan lossless boundaries above the threshold, while inspect only reports whether assessment is needed.

## Decision types and context

`task.modes` supports `auto/choice/noul/score`. `auto` preserves or infers source modes. Repeat `--modes` to select several types. `--difficulty` controls distractor difficulty (0–1), not temperature.

```bash
uv run dforge data build --config configs/default.yaml --modes choice --modes score --no-extract-context --generate-context --context-relevance 0.7
```

`--extract-context/--no-extract-context` controls source context extraction; `--generate-context/--no-generate-context` controls filling missing context. Relevance ranges from 0 to 1. Clarify label and context meaning before mapping JSON; do not infer semantics from field names alone.

## Concurrency, checks and recovery

`--concurrency` limits question workers, `--text-batch-concurrency` limits simultaneous TXT batches, and `--llm-concurrency` limits model requests. Transport retries and semantic repair are bounded separately by `execution.max_transport_attempts` and `verification.max_semantic_repairs`.

Records pass structural validation and a semantic checker before targeted repairs. Budget exhaustion or model failure preserves checkpoints and exits with code 2. Resume after resolving the cause. `--budget-usd` sets a new **total budget**, not an increment, and requires configured model pricing.

```bash
uv run dforge data resume --run-dir outputs/run_001 --budget-usd 10
uv run dforge data export --run-dir outputs/run_001 --format audit
```

Run directories have exclusive locks. Do not export concurrently with generation or start two owners. Changed inputs/configuration create versioned directories. Do not delete active locks or edit checkpoints manually.

## Outputs

Export `canonical`, `jsonl`, `requests`, `jev-api`, `chat-sft` or `audit` as needed. Train on accepted records. Preserve provenance, groups and splits, separating training, validation, calibration and final tests before generation or rewriting. See [JEV format](jev-format.md).
