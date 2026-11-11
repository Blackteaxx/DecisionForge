# FAQ

[中文](../zh/faq.md) · [README](../../README.en.md)

## Why do offline commands need no key?

`data inspect`, `data validate`, `data export`, schema, doctor, training validation, evaluation validation and rescoring make no model calls. Use `configs/offline.yaml` to avoid unset model variables. Configurations containing `${OPENAI_MODEL}` still need that value to resolve.

## What if the provider does not support strict output?

Choose supported `llm.api: responses/chat_completions`, `structured_output: strict/json/prompt` and `chat_token_limit_param`. Compatible services may not support every combination. Validate a small input before scaling. Do not hide parameter errors with unbounded retries.

## What do exit codes mean?

0 means success, 1 a configuration/file/execution error, 2 a CLI usage error or saved budget/model pause, and 130 a user interruption. Inspect JSON status and artifacts, not just the last log line.

## Why does a GPU extra fail to install?

Backends require specific Python versions, platforms and CUDA. Run `dforge doctor`, use a separate 3.12 environment and install the matching extra. Training groups conflict; do not use `--all-extras`. OpenJev pins Arrow 21.0.0. The macOS base environment does not validate GPU training.

## Why do configuration and run paths differ?

YAML paths resolve relative to the configuration; CLI overrides resolve from the current directory. Changed input/configuration identity creates a versioned directory. Use the returned run_directory downstream. Locks protect checkpoints; do not remove active locks.

## Can I use the former Web UI or deep_search configuration?

Both features were removed. Old `input.type: deep_search` yields an explicit error. Agent tools handle source research; JSON/TXT inputs and MinerU PDF parsing remain. Use the dsh plugin for a UI; see [Agent](agent.md).

## Do checks prove domain effectiveness?

Passing the checker means records meet configured checks, not that domain facts or clinical validity are proven. Verify labels, source independence, test isolation and metric definitions. Report failures, limits and negative results, preserving traceable evidence.

## Why is Pages not accessible yet?

The repository is private. The workflow is ready; verify deployment after public visibility and Pages activation. Non-default branch deployments require matching environment rules. Open `demo/index.html` or an offline bundle locally for now.
