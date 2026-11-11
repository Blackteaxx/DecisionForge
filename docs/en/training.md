# Training

[中文](../zh/training.md) · [README](../../README.en.md)

Install one backend, run CPU validation, then train on supported GPU infrastructure. Validation does not prove that GPU devices, models or kernels can load.

```bash
uv sync --extra train-unsloth
uv run dforge train validate --config examples/training/unsloth/qwen35-4b-smoke.yaml
uv run dforge train run --config examples/training/unsloth/qwen35-4b-smoke.yaml
uv run dforge train run --config examples/training/unsloth/qwen35-4b-smoke.yaml --resume
```

## Choose a backend

| Extra / backend | Method and example |
| --- | --- |
| `train-unsloth` / `unsloth` | LoRA; [20-step example](../../examples/training/unsloth/qwen35-4b-smoke.yaml) |
| `train-unsloth-full` / `unsloth` | Full finetuning; [single GPU](../../examples/training/unsloth/full-single.yaml), [DDP](../../examples/training/unsloth/full-ddp.yaml) |
| `train-swift` / `swift` | ms-swift; [configurations](../../examples/training/swift/) |
| `train-laya` / `laya` | Laya RLCD; [configuration](../../examples/training/laya/acceptance.yaml) |
| `train-openjev` / `openjev` | Two-stage OpenJev RLCD; [configuration](../../examples/training/openjev/acceptance.yaml) |

Groups conflict; use separate environments. OpenJev uses tested Arrow 21.0.0. `uv.lock` is the version source; do not independently upgrade torch, transformers or backend SDKs with pip.

## Data and configuration

Training consumes supervised JEV records with hard targets or complete soft distributions. `data.split_map` maps existing splits; `data.auto_holdout` holds out validation and calibration by group. Do not randomly split rewrites from the same source across sets. Keep final evaluation separate from training, tuning and calibration.

Keep `data.on_skipped` and `on_truncated` as `error` unless explicitly accepting another policy. Check actual processed counts in reports. `model.name` accepts a model ID or local directory; `model.revision` pins provenance. Local paths resolve relative to YAML. Configuration validation enforces model, backend, precision and export capabilities.

LoRA supports optional 4-bit loading; full finetuning disables quantization. Set `finetuning: full` for full finetuning and follow the relevant distributed template. Internal workers manage multi-GPU execution. macOS is outside GPU training acceptance.

## Artifacts and recovery

Use the returned run directory. Preserve resolved config, manifests, logs, weights, decision heads, tokenizer, calibration and export reports. Select checkpoints on validation, calibrate on calibration data and evaluate on the final test set. Reload artifacts in a fresh process; training loss alone is not evidence of task performance.

`--run-dir` overrides output; `--resume` resumes only a compatible task. Do not reuse checkpoint directories for different models or datasets. After Ctrl+C, check supervisor/worker exit and manifests to ensure no background job retains GPU resources.

See [Evaluation](evaluation.md) for local artifacts and HTTP services, and [Extending](extending.md) for backend boundaries.
