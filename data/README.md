# 数据目录

这里保存数据准备脚本及本地数据缓存；评测结果保存在项目根目录的 `outputs/eval/`。

生成流程导出的 `accepted.json` 可直接作为训练数据，参见 [训练指南](../docs/zh/training.md)。训练运行在 `data/` 子目录导出按实际角色划分的 JSONL，可直接用于评测；验证和测试集合不回流训练。配套小样本位于 [examples/training](../examples/training/README.md)。

```text
data/
  benchmarks/
    _shared/                   # 下载、转换工具，仅供仓库脚本使用
    <benchmark>/
      README.md                # 来源、固定版本、转换规则
      prepare.py               # 下载和转换入口
      raw/<split>/             # 原始文件及 source.json
      processed/<split>.jsonl  # 规范样本
      processed/<split>.jsonl.manifest.json
  local/                       # 本地训练与临时数据，不纳入 Git
```

`raw/` 和 `processed/` 不纳入 Git；`README.md`、`prepare.py` 与共享工具纳入版本管理。原始数据通过固定 revision 下载，`source.json` 记录 URL、SHA-256、划分及许可证。规范 JSONL 使用库的 `decisionforge.eval.v1` 契约，转换 manifest 记录原文件哈希、规则版本、输出记录数及参考标签来源。

从项目根目录执行：

```bash
python data/benchmarks/banking77/prepare.py
python data/benchmarks/clinc150_oos/prepare.py
python data/benchmarks/typed_decisions/prepare.py
python data/benchmarks/jevbench/prepare.py
```

默认先下载再转换；已下载的源数据先校验哈希，不重复下载。`--fetch-only` 只准备原始数据；`--convert-only` 离线转换；`--split train` 选择支持的划分；`--data-dir /path/to/cache` 指定该 benchmark 的缓存目录。默认目录以脚本位置为准，从其他 cwd 执行也不会把数据写到当前目录。使用自定义缓存时，同时修改评测配置的 `data.path`。

JevBench 默认准备 public，并输出 public/original/easy/hard 四份规范 JSONL。其他 benchmark 默认输出 test.jsonl。数据准备脚本不调用模型，不读取密钥文件；下载代理通过进程的 `HTTPS_PROXY` 设置。Parquet 转换需要 `pip install -e ".[eval-data]"`。

库只加载已准备的规范数据。准备脚本和 `_shared/` 不随库的 wheel 分发，添加数据集时在这里编写转换脚本；评分与 benchmark/metric 注册放在库或独立扩展包中。完整流程见 [使用指南](../docs/zh/evaluation.md)，扩展方式见 [开发指南](../docs/zh/extending.md)。
