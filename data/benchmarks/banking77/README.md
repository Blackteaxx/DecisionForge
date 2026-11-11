# BANKING77

固定来源：[PolyAI-LDN/task-specific-datasets](https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b)，revision `57ec275d8078af65b7731c2a98be812d844a6d6b`。许可证：CC-BY-4.0。

test 3,080 条、77 类；train 10,003 条。

从项目根目录运行：

```bash
python data/benchmarks/banking77/prepare.py
python data/benchmarks/banking77/prepare.py --convert-only
```

原始文件保存在 raw/<split>/，规范 JSONL 与转换 manifest 在 processed/。--fetch-only 只下载；--data-dir 指定缓存目录。可用划分见 prepare.py --help。数据不纳入 Git，脚本不读取模型凭据、不调用模型。

转换规则：保留原 categories.json 的完整标签及 CSV category；标签说明替换下划线。零样本 Choice 分类，不输入训练例或目标。主指标固定 77 类 macro-F1。

目录规范见 [data/README.md](../../README.md)，执行评测见 [使用指南](../../../docs/zh/evaluation.md)。
