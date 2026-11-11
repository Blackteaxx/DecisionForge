# Unsloth 本地模型评测

在训练模型保存完成后，将 `qwen35-4b.yaml` 的 `model.path` 指向训练产物 `model/` 或 `merged/`，然后直接执行现有 eval。无需启动推理服务。

```bash
# CPU 可执行：校验随附测试数据和本地模型目录，不加载模型
bash examples/eval/unsloth/run.sh --validate-only

# Linux 单卡 CUDA 环境，使用 uv sync --extra train-unsloth 中的依赖
bash examples/eval/unsloth/run.sh

# 中断恢复；已保存预测不会再次推理
bash examples/eval/unsloth/run.sh --resume

# 离线重评分：只使用原样本和预测，不需要模型目录或 GPU/Unsloth
bash examples/eval/unsloth/run.sh --score-only
```

默认模型路径对接 `examples/training/unsloth/qwen35-4b-smoke.yaml` 的输出：`outputs/train/qwen35-4b-smoke/model`。模型不存在时预检会明确失败，请先训练或修改路径。默认结果目录为 `outputs/eval/qwen35-4b-local/`。

配置说明：

- `adapter: unsloth_local` 选择本地加载；仅接受已保存的 Clef 决策 checkpoint，不创建随机决策头。
- `path` 相对于 YAML 所在目录。可以改为 `../../../outputs/train/qwen35-4b-smoke/merged`，或自己的绝对路径。
- `load_in_4bit`、`precision` 控制加载。`max_seq_length` 默认 16384；超过长度的请求记失败，不丢弃输入文本。
- 本地推理使用 `concurrency: 1` / `max_attempts: 1`。`load_timeout` 限制模型初始化时间；`execution.timeout` 限制单个推理请求。超时/中断会关闭 worker。
- 每次运行加载模型一次，使用保存的校准参数。预测只接收公开 state/questions；标签和 metadata 留在评测进程中。

随附数据是 [固定 Typed Decisions 原 test 集的 20 案例 / 100 个问题](SOURCE.md)，覆盖四个 workflow 和 Choice/Noul/Score。每次一个独立 decision 问题，评分使用既有 `typed_decisions` benchmark；模型训练时五问题联合输入的协议差异会继续标在报告里。

完整公开测试集准备后，使用：

```bash
bash examples/training/prepare_data.sh
uv run dforge eval run --config configs/eval/local-trained-decision.yaml --validate-only
uv run dforge eval run --config configs/eval/local-trained-decision.yaml
```

`predictions.jsonl` 保存完整校准概率；不采用官方 predict 的四位小数展示值。预测概率检查键覆盖和数值边界，不要求总和为 1，也不自动归一化；评分规则见 [评测参考](../../../docs/zh/evaluation.md)。`summary.md`、metrics.json 的路径由 CLI 返回。延迟、吞吐、token 数及模型/GPU版本会记录；GPU费用不估算，也不当作零成本。
