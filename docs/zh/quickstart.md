# 快速开始

[English](../en/quickstart.md) · [README](../../README.md)

安装 uv 后，在仓库根目录执行。要求 Python >=3.11；锁定环境使用 3.12，部分训练组要求 <3.13。

```bash
git clone https://github.com/Blackteaxx/DecisionForge.git
cd DecisionForge
uv sync
uv run dforge data inspect --config configs/offline.yaml
```

最后一条命令只读示例数据，不加载 `.env`、不调用模型、不需要 API key。输出包含记录数、字段结构与请求题型。

## 生成与导出

复制 `.env.example` 为 `.env`，配置生成服务的 `OPENAI_API_KEY`、`OPENAI_MODEL` 和可选 `OPENAI_BASE_URL`。

```bash
uv run dforge data build --config configs/default.yaml
uv run dforge data export --run-dir outputs/run_001 --format jsonl
uv run dforge data validate --input outputs/run_001/accepted.json
```

生成会调用模型并产生费用。使用 CLI 返回的 `run_directory`；输入或配置变化时实际目录可能带版本后缀。`accepted.json` 是检查通过的数据，`fail.json` 是未通过的数据，`meta.json` 保存来源与输出对应关系。

## 按需安装

| Extra | 用途 |
| --- | --- |
| `pdf` | MinerU HTTP PDF 解析，基础环境已包含客户端 |
| `eval-data` | Parquet benchmark |
| `eval-anyjev` / `eval-llm2jev` | 对应本地免训练评测 |
| `train-unsloth` | Unsloth LoRA |
| `train-unsloth-full` | 固定 Git 版本的 Unsloth 全参训练 |
| `train-swift` / `train-laya` / `train-openjev` | 各训练后端 |

```bash
uv sync --extra eval-data
uv run dforge doctor
uv run dforge --help
```

训练组互斥，应放在不同 checkout 或使用不同 `UV_PROJECT_ENVIRONMENT`。安装一个组会同步该环境到当前选择；不要把所有后端叠在同一环境。GPU 路线以 Linux 为主，Windows 的 Unsloth 使用保留的 CUDA 索引；macOS 用于基础数据与 HTTP 评测。

## 后续操作

[数据生成](data-generation.md) · [训练](training.md) · [评测](evaluation.md) · [Agent](agent.md) · [常见问题](faq.md)

`scripts/run_workflow.py` 可在明确授权全流程执行后串联阶段；它不包含交互确认。通常先分别验证各阶段，再组合。
