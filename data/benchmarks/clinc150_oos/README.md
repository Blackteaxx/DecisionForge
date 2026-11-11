# CLINC150+OOS

固定来源：[clinc/clinc_oos](https://huggingface.co/datasets/clinc/clinc_oos/tree/155b9c710419136e17307b80d0a13e68cd46b4ec)，revision `155b9c710419136e17307b80d0a13e68cd46b4ec`。许可证：源数据卡未声明。

plus/test 5,500 条：4,500 域内、1,000 OOS；支持 train/validation/test。

从项目根目录运行：

```bash
python data/benchmarks/clinc150_oos/prepare.py
python data/benchmarks/clinc150_oos/prepare.py --convert-only
```

原始文件保存在 raw/<split>/，规范 JSONL 与转换 manifest 在 processed/。--fetch-only 只下载；--data-dir 指定缓存目录。可用划分见 prepare.py --help。数据不纳入 Git，脚本不读取模型凭据、不调用模型。

转换规则：从固定 README 解析完整 151 类映射，oos 索引 42；不混合重复 test 配置。零样本 Choice 分类，主指标固定 151 类 macro-F1，另报域内 150 类及 OOS 分项。

目录规范见 [data/README.md](../../README.md)，执行评测见 [使用指南](../../../docs/zh/evaluation.md)。
