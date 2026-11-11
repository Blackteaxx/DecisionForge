# JevBench

固定来源：[fstandhartinger/jevbench](https://github.com/fstandhartinger/jevbench/tree/b6b8fff7e345b98c060ad26c13308860ddc67004)，revision `b6b8fff7e345b98c060ad26c13308860ddc67004`。许可证：MIT。

public 231 条：original 72、easy 48、hard 111。

从项目根目录运行：

```bash
python data/benchmarks/jevbench/prepare.py
python data/benchmarks/jevbench/prepare.py --convert-only
```

原始文件保存在 raw/<split>/，规范 JSONL 与转换 manifest 在 processed/。--fetch-only 只下载；--data-dir 指定缓存目录。可用划分见 prepare.py --help。数据不纳入 Git，脚本不读取模型凭据、不调用模型。

转换规则：脚本输出 public.jsonl 和 original/easy/hard.jsonl；保留 provenance、family、group、tier。Noul no/yes 显式映射为 false/true；固定评分与原上游 oracle 对齐。

目录规范见 [data/README.md](../../README.md)，执行评测见 [使用指南](../../../docs/zh/evaluation.md)。
