# Typed Decisions

固定来源：[LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/tree/e039ebffcc280174dd354227424fb2b249f191de)，revision `e039ebffcc280174dd354227424fb2b249f191de`。许可证：Apache-2.0。

all/test 400 案例、2,000 问题；all/train 1,200 案例。

从项目根目录运行：

```bash
python data/benchmarks/typed_decisions/prepare.py
python data/benchmarks/typed_decisions/prepare.py --convert-only
```

原始文件保存在 raw/<split>/，规范 JSONL 与转换 manifest 在 processed/。--fetch-only 只下载；--data-dir 指定缓存目录。可用划分见 prepare.py --help。数据不纳入 Git，脚本不读取模型凭据、不调用模型。

转换规则：每个问题独立转换为 questions.decision，group_id 保留案例。gold 是单教师三次采样平均软分布，原 gold/factors/agreement 只在 target/metadata；六位小数参考舍入在 1e-5 内显式归一化。协议与上游每次五题不同，不能声称榜单复现。

目录规范见 [data/README.md](../../README.md)，执行评测见 [使用指南](../../../docs/zh/evaluation.md)。
