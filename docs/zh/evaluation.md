# 评测与比较

[English](../en/evaluation.md) · [README](../../README.md)

评测分为数据、模型适配器、benchmark 和 metric。保持同一输入、划分、指标与分母，才能解释方法差异。

```bash
uv run dforge eval list
uv run dforge eval run --config configs/eval/demo.yaml --validate-only
uv run dforge eval run --config configs/eval/demo.yaml
uv run dforge eval rescore --config configs/eval/demo.yaml --run-dir outputs/eval/demo
```

`--validate-only` 与 `rescore` 均离线，不加载模型凭据。重评分要求已有样本与预测，不重新推理。使用运行时返回的目录覆盖上例目录；`--resume` 继续兼容检查点。在线运行可通过 `--env-file` 指定凭据文件。

## 输入与方法

[示例 JSONL](../../examples/eval/samples.jsonl)使用 `decisionforge.eval.v1`，含 request、target、split、subset 和 metadata。也可直接读取生成的训练 JSON；标签只供评分，不能进模型输入。Parquet 数据需要 `uv sync --extra eval-data`。

| 方法 | 环境与配置 |
| --- | --- |
| Jev HTTP | 基础依赖；`configs/eval/demo.yaml`，配置 `TYPESAFE_API_KEY` |
| Chat baseline | 基础依赖；`configs/eval/chat/`，配置相应模型服务 |
| Unsloth 本地产物 | `train-unsloth`；`configs/eval/local-trained-decision.yaml` |
| LLM2Jev | `eval-llm2jev`；`configs/eval/llm2jev/` |
| AnyJev L0 | `eval-anyjev`；`examples/eval/training-free/anyjev/` |
| Swift / Laya / OpenJev 产物 | 对应训练组与版本化本地产物 |

HTTP model 的 `api_key_env` 决定凭据变量，`base_url` 和 `endpoint_path` 决定服务；方法来源与 serving backend 不应混为一谈。免训练方法依然需要模型、算力和安装，不代表无成本。

## 指标与报告

Choice 允许 2～255 个唯一选项；Noul 判断二元命题；Score 允许 2～10 个有序等级。硬标签、软分布与标量目标各有对应指标；不能对不兼容目标强套 accuracy。完整概率分布可计算校准、对数损失或 Brier 等指标，具体名称通过 `eval list` 查询。

报告保存推理身份、数据/配置哈希、样本、逐条预测与聚合结果。无效输出、失败率、延迟和费用应与主指标一起报告。比较时核查模型版本、精度、量化、提示、候选集、来源重叠和 checkpoint 选择规则。完整测试与 smoke 子集的分母不可混用。

已有数据转换与自定义评分见 [扩展](extending.md)。
