# 数据生成

[English](../en/data-generation.md) · [README](../../README.md)

从 JSON、TXT 或 PDF 准备可检查的 JEV 数据。输入文件路径按 YAML 所在目录解析；CLI 覆盖路径按当前目录解析。

```bash
uv run dforge data inspect --config configs/offline.yaml
uv run dforge data build --config configs/default.yaml --difficulty 0.5 --modes choice
uv run dforge data resume --run-dir outputs/run_001
```

## 输入与配置

`input.type` 支持 `auto/json/txt/pdf`。JSON 可用 `records_path` 指定数组位置、`mapping_spec` 映射字段。TXT 默认保留原文证据，由模型评估能否构造问答；没有足够证据的材料不能当作可靠标签。PDF 通过 MinerU HTTP API 转为可复用 TXT。

```bash
uv sync --extra pdf
uv run dforge data parse-pdf --config configs/pdf-mineru.yaml --input /path/to/paper.pdf
# 多文件使用重复的 --input 参数
uv run dforge data parse-pdf --config configs/pdf-mineru.yaml --input a.pdf --input b.pdf
```

MinerU 凭据为 `MINERU_API_KEY`，可在配置中更改 `mineru.api_key_env`。解析缓存位于 `.dforge/pdf/`，清单记录来源、解析状态和 TXT 路径。长 TXT 由 `input.text_batching` 控制：达到阈值后，build 调用模型规划边界并进行无损切片；inspect 只报告是否需要评估，不调用模型。

## 题型与上下文

`task.modes` 为 `auto/choice/noul/score`。`auto` 保留来源模式或根据字段推断。重复 `--modes` 选择多个题型。`--difficulty` 是干扰项难度（0～1），不是模型温度。

```bash
uv run dforge data build --config configs/default.yaml --modes choice --modes score --no-extract-context --generate-context --context-relevance 0.7
```

`--extract-context/--no-extract-context` 控制提取已有背景；`--generate-context/--no-generate-context` 控制缺失时补充背景；相关度范围为 0～1。JSON 问答的标签与上下文需先明确含义，不能靠字段名猜测。

## 并发、检查和恢复

`--concurrency` 限制问答 worker，`--text-batch-concurrency` 限制同时处理的 TXT 批次，`--llm-concurrency` 限制模型请求。传输重试与语义修复分别受 `execution.max_transport_attempts` 和 `verification.max_semantic_repairs` 限制。

每条数据经过结构检查与语义 checker，再按失败原因定向修复。超出预算或模型失败时保留检查点，CLI 退出码为 2；可修复原因后恢复。`--budget-usd` 是新的**总预算**，需要配置模型价格，不是追加预算。

```bash
uv run dforge data resume --run-dir outputs/run_001 --budget-usd 10
uv run dforge data export --run-dir outputs/run_001 --format audit
```

同一运行目录有独占锁，正在生成时不能并发导出或再次启动。输入/配置变化会创建版本目录，避免把不同来源混进已有结果。不要手工删除活动锁或改写检查点。

## 输出

`canonical`、`jsonl`、`requests`、`jev-api`、`chat-sft`、`audit` 可按用途导出。训练使用检查通过的数据；保留来源、group 和 split，在生成与改写之前隔离训练、验证、校准和最终测试。格式详见 [JEV](jev-format.md)。
