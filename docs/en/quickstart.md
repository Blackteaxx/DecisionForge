# Quick start

[中文](../zh/quickstart.md) · [README](../../README.en.md)

Install uv, then run these commands from the repository root. Python >=3.11 is required; the pinned environment uses 3.12 and some training extras require <3.13.

```bash
git clone https://github.com/Oscar-dzy/Autojev-data.git
cd Autojev-data
uv sync
uv run dforge data inspect --config configs/offline.yaml
```

The final command reads sample data offline. It does not load `.env`, call a model or require an API key. Output includes record counts, field shapes and requested decision types.

## Generate and export

Copy `.env.example` to `.env`. Configure `OPENAI_API_KEY`, `OPENAI_MODEL` and, if needed, `OPENAI_BASE_URL` for generation.

```bash
uv run dforge data build --config configs/default.yaml
uv run dforge data export --run-dir outputs/run_001 --format jsonl
uv run dforge data validate --input outputs/run_001/accepted.json
```

Generation calls models and incurs provider charges. Use the returned `run_directory`; changed input or configuration may produce a versioned directory. `accepted.json` contains records that passed checks, `fail.json` contains failed records, and `meta.json` maps outputs to sources.

## Optional environments

| Extra | Purpose |
| --- | --- |
| `pdf` | MinerU HTTP PDF parsing; the client is in base dependencies |
| `eval-data` | Parquet benchmarks |
| `eval-anyjev` / `eval-llm2jev` | Respective local training-free evaluation |
| `train-unsloth` | Unsloth LoRA |
| `train-unsloth-full` | Full finetuning with a pinned Git revision |
| `train-swift` / `train-laya` / `train-openjev` | Respective training backends |

```bash
uv sync --extra eval-data
uv run dforge doctor
uv run dforge --help
```

Training groups conflict. Use separate checkouts or separate `UV_PROJECT_ENVIRONMENT` paths. Syncing a group reconciles that environment to the selected dependencies; do not combine every backend. GPU routes primarily target Linux. Windows Unsloth retains its CUDA index; macOS supports base data workflows and HTTP evaluation.

## Next steps

[Data generation](data-generation.md) · [Training](training.md) · [Evaluation](evaluation.md) · [Agent](agent.md) · [FAQ](faq.md)

`scripts/run_workflow.py` can combine stages after explicit authorization for the complete workflow. It has no interactive approval gates. Validate stages separately before combining them.
