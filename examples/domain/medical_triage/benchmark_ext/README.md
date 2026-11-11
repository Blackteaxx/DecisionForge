# Medical triage benchmark extension

独立 Python 扩展包，通过标准 `decisionforge.eval.benchmarks` / `decisionforge.eval.metrics` entry points 注册 `medical_decisions` v1 与四个指标。评分代码均在此目录，不修改内置 benchmark。

## 安装

在 DecisionForge 仓库根目录执行：

```bash
python -m pip install --no-deps -e examples/domain/medical_triage/benchmark_ext
uv run dforge eval list
```

常规评测直接安装本目录。历史 v1 的选模与报告还需要任务所属实体副本和 `IDENTITY.json`，复制与记录命令见 [历史复现](../docs/reproduce-v1.md)。每个环境只安装一份，不重复注册。

## 数据转换

[prepare_benchmarks.py](../scripts/prepare_benchmarks.py) 完整转换 CARE Public Test1 135 条、PMR Synth 869 对、Reddit 1502 对。固定 URL、revision、哈希与许可见 [来源锁定](../configs/sources.lock.json)。患者消息和 EHR 一起换序，金标同步反转；来源标签、解释与难度不发送给模型。共享消息/EHR 以连通组聚类。

## 评分契约

- `medical_accuracy`：原生 `answer.choice`，分母为全部计划的有标签样本，失败、缺失和未尝试计错。
- `medical_macro_f1`：完整类别目录，zero-division=0；缺失预测贡献真类 false negative。
- `medical_brier`：完整有效概率向量的未缩放平方误差和，报告概率覆盖率。
- `medical_ece`：默认 10 个固定 bin，以原生 Choice 的概率作为置信度，报告覆盖率。

概率键必须与候选目录完全一致，值有限且在 [0,1]，总和容差 1e-6；不重新归一化。概率 argmax 不替换原生 Choice。Choice 有效但概率无效时，决策仍评分，另报概率与 strict-valid 失败。减少数据不能缩小固定分母。指标版本均为 1。

本包依赖 DecisionForge 公共 BaseBenchmark/Metric 接口及共享分布检查。完整方法和限制见 [方法与消耗](../docs/methods.md)。
