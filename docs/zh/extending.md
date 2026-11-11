# 扩展与开发

[English](../en/extending.md) · [README](../../README.md)

数据转换、模型传输和评分分别扩展，避免为一个任务复制整套 runner。开发环境：

```bash
uv sync --locked --group dev
uv run ruff check
uv run pytest
```

## 转换数据

将来源转换为 [JEV](jev-format.md) 或评测 JSONL，保留 source、group、split 和哈希。评测记录有 id、group_id、split、subset、request、target、metadata；target 可为 hard、soft 或 scalar。金标不能混入 request。先用 `eval run --validate-only` 校验，再调用模型。

## benchmark 与 metric

```python
from decisionforge.eval import BaseBenchmark, Metric, Registry, load_registry
```

benchmark 负责数据视图、兼容指标与聚合；metric 只计算评分。内置实现放在 `src/decisionforge/benchmarks/<name>/`，通过 `register(registry)` hook 注册；任务专用实现可使用 Python entry point。

参考可运行的 [扩展示例](../../examples/eval/plugin/)。包的 entry point group 为 `decisionforge.eval.benchmarks` / `decisionforge.eval.metrics`，配置如何选择 benchmark 与 metric 以示例为准。扩展在用户环境安装，不能依赖作者本地路径。注册名称需要唯一，修改评分口径时增加版本并保留来源身份。

## 生成模块与训练后端

`generation/contracts.py` 定义输入、生成和 checker 响应，`intake.py` 处理输入，`components.py` 编译 JEV，`pipeline.py` 管理并发、修复与预算，`storage.py` 管理检查点。模型传输与缓存集中在 `llm.py`。新模块需维护原生格式、有限重试、来源隔离和可恢复状态。

训练后端位于 `training/backends/`，数据、校准、保存与评测重载应保持一致的版本契约。可选 SDK 延迟导入，缺失时提示具体 extra。worker 的 argparse 是内部进程协议，公共命令使用 Typer；不要新增平行命令入口。

## 验收

`tests/acceptance/` 保存需要模型/GPU 的手动验收驱动；普通 pytest 默认使用离线或 mock 测试。运行付费或 GPU 任务前确定资源和预算，记录配置、模型/数据身份与实际产物。验证报告必须区分静态检查、真实运行与未验证部分。

插件开发见 [插件 README](../../extensions/dsh-decisionforge/README.md)，Demo 构建见 [Demo README](../../demo/README.md)。
