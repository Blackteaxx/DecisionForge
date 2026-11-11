# Evaluation and comparison

[中文](../zh/evaluation.md) · [README](../../README.en.md)

Evaluation separates data, model adapters, benchmarks and metrics. Compare methods using identical inputs, splits, metrics and denominators.

```bash
uv run dforge eval list
uv run dforge eval run --config configs/eval/demo.yaml --validate-only
uv run dforge eval run --config configs/eval/demo.yaml
uv run dforge eval rescore --config configs/eval/demo.yaml --run-dir outputs/eval/demo
```

`--validate-only` and `rescore` are offline and do not load model credentials. Rescoring requires saved samples and predictions; it does not rerun inference. Replace the example output with the actual returned directory. `--resume` continues a compatible checkpoint. Online evaluation accepts `--env-file` for credentials.

## Inputs and methods

[Example JSONL](../../examples/eval/samples.jsonl) uses `decisionforge.eval.v1` with request, target, split, subset and metadata. Generated training JSON can also be read directly. Labels are only for scoring, never model input. Parquet data requires `uv sync --extra eval-data`.

| Method | Environment and configuration |
| --- | --- |
| Jev HTTP | Base dependencies; `configs/eval/demo.yaml`, with `TYPESAFE_API_KEY` |
| Chat baseline | Base dependencies; `configs/eval/chat/`, with its provider credentials |
| Local Unsloth artifacts | `train-unsloth`; `configs/eval/local-trained-decision.yaml` |
| LLM2Jev | `eval-llm2jev`; `configs/eval/llm2jev/` |
| AnyJev L0 | `eval-anyjev`; `examples/eval/training-free/anyjev/` |
| Swift / Laya / OpenJev artifacts | Corresponding training extra and versioned local artifacts |

For HTTP models, `api_key_env` selects credentials, while `base_url` and `endpoint_path` select the service. Method provenance and serving backend are distinct. Training-free methods still require models, compute and installation.

## Metrics and reports

Choice supports 2–255 unique options, Noul represents binary propositions, and Score supports 2–10 ordered levels. Hard, soft and scalar targets need compatible metrics; accuracy is not appropriate for every target. Full distributions support calibration, log loss and Brier-style metrics. Query exact registered names using `eval list`.

Reports preserve inference identity, data/configuration hashes, samples, per-record predictions and aggregates. Report invalid outputs, failure rate, latency and cost alongside the main metric. Check model revisions, precision, quantization, prompts, candidate sets, source overlap and checkpoint selection when comparing. Do not mix full-test denominators with smoke subsets.

See [Extending](extending.md) for conversion and custom scoring.
