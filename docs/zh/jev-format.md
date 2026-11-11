# JEV 数据格式

[English](../en/jev-format.md) · [README](../../README.md)

本轮品牌迁移将 schema_version 命名空间统一为 `decisionforge.*`；旧数据使用前需迁移该标识。JEV 字段结构、题型和评分语义不变。

accepted.json 与 fail.json 都是 decisionforge.training.v3 数组。每条保留 schema_version、record_id、group_id、split、request、supervision；不携带生成过程 metadata。

```json
{
  "schema_version": "decisionforge.training.v3",
  "record_id": 1,
  "group_id": 1,
  "split": "train",
  "request": {
    "state": "France is a European country.",
    "questions": {
      "q1": {"type": "choice", "instructions": "What is the capital of France?",
             "criteria": {"A": "Rome", "B": "Paris"}}
    }
  },
  "supervision": {"answers": {
    "q1": {"type": "choice", "choice": "B", "confidence": 1.0,
           "probabilities": {"A": 0.0, "B": 1.0}}
  }}
}
```

Choice criteria 是 2～255 个唯一选项；Noul 可省略 criteria，监督为 type+noul（0 或 1）；Score criteria 是 2～10 个由低到高的有序项，监督包含 score、legend、probabilities 和 confidence。

新生成的 Choice 选项键使用 `A、B、C…`，超过 26 项后使用 `AA、AB…`。选项内容先按 `execution.seed` 和样本标识随机打乱，再分配字母键；GT 的 `choice` 和 `probabilities` 同步映射，正确答案不会固定放在某个字母位置。同一样本和种子可复现相同顺序，恢复运行不会再次打乱。

生成器使用硬标签构造 one-hot 概率，confidence=1 表示该训练目标的确定性表示，不是模型实际预测置信度。最终标签必须经过 checker 检查，失败文件中的标签不受认可。

空上下文使用空字符串。问题在 instructions 中，Noul 将原问题和陈述放在 instructions 对象中。状态不能用问题原文充当背景。

`meta.json.records` 的键是公开 record_id，包含输出文件名和来源。没有合法 JEV 的失败记录 output=null，因此其 ID 不会出现在 accepted.json / fail.json 中。

评测入口可直接读取该 JSON，适配为 eval.v1 decision 输入。也可使用 `dforge data export --format requests`、`jev-api`、`jsonl` 或 `chat-sft` 导出专用格式。
