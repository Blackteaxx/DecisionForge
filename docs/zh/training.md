# 模型训练

[English](../en/training.md) · [README](../../README.md)

训练后端按需安装，先做 CPU 静态预检，再在支持的 GPU 环境启动。预检不证明 GPU、模型或 kernel 能加载。

```bash
uv sync --extra train-unsloth
uv run dforge train validate --config examples/training/unsloth/qwen35-4b-smoke.yaml
uv run dforge train run --config examples/training/unsloth/qwen35-4b-smoke.yaml
uv run dforge train run --config examples/training/unsloth/qwen35-4b-smoke.yaml --resume
```

## 选择后端

| Extra / backend | 方法与示例 |
| --- | --- |
| `train-unsloth` / `unsloth` | LoRA；[20 步示例](../../examples/training/unsloth/qwen35-4b-smoke.yaml) |
| `train-unsloth-full` / `unsloth` | 全参；[单卡](../../examples/training/unsloth/full-single.yaml)、[DDP](../../examples/training/unsloth/full-ddp.yaml) |
| `train-swift` / `swift` | ms-swift；[配置目录](../../examples/training/swift/) |
| `train-laya` / `laya` | Laya RLCD；[配置](../../examples/training/laya/acceptance.yaml) |
| `train-openjev` / `openjev` | OpenJev 两阶段 RLCD；[配置](../../examples/training/openjev/acceptance.yaml) |

各组互斥，使用独立环境。OpenJev 使用实测的 Arrow 21.0.0。完整固定版本以 `uv.lock` 为准，不单独 pip 升级 torch、transformers 或训练 SDK。

## 数据与配置

训练输入为 JEV 监督记录，支持硬标签与完整软分布。`data.split_map` 映射已有划分；`data.auto_holdout` 按 group 留出验证和校准。不要按单条改写样本随机拆分同一来源。最终评测集与训练、调参和校准隔离。

`data.on_skipped` 和 `on_truncated` 应保持 `error`，除非明确接受丢弃策略；检查训练报告中的实际处理计数。`model.name` 可为模型 ID 或本地目录，`model.revision` 固定来源；本地路径相对 YAML。模型结构、训练后端、精度和导出格式的能力限制由配置校验给出。

LoRA 可选择 4-bit 加载；全参训练禁用量化。全参使用 `finetuning: full`，分布式设置参照对应模板；多卡入口由内部 worker 管理，不直接面向用户。macOS 不在 GPU 训练验收范围内。

## 产物和恢复

使用实际返回的运行目录，保存 resolved config、manifest、日志、模型权重、决策头、tokenizer、校准和导出报告。按验证集选择 checkpoint，校准数据只校准，不用于最终测试。新进程重载产物后再评测，不能仅以训练 loss 声称有效。

`--run-dir` 覆盖输出目录；`--resume` 仅恢复兼容的原任务。不要将不同模型或数据的目录当作同一 checkpoint。`Ctrl+C` 后核查 supervisor/worker 的退出与 manifest，避免后台进程继续占用 GPU。

[评测](evaluation.md)说明本地产物与 HTTP 服务的选择；[扩展](extending.md)说明新增 backend 的边界。
