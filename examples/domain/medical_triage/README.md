# 医疗分诊与患者消息优先级

本目录集中交付实际冻结训练数据、来源与清洗、测试集转换和 benchmark 扩展、实验结果及方法说明。正式报告采用 **v1：Qwen3.5-9B BF16 LoRA，独立验证选出的 step200，595 条独立校准数据**。

| 方法 | CARE135 | Synth869 | Reddit1502 |
| --- | ---: | ---: | ---: |
| v1 自训模型 | **70.37%** | **67.66%** | 79.23% |
| 外部 Jev | 60.74% | 65.59% | **80.36%** |
| 同底座 LLM2Jev | 57.78% | 63.18% | 76.10% |
| 同底座 AnyJev | 54.81% | 65.36% | 74.70% |
| 同底座 Chat | 55.56% | 64.56% | 73.70% |

15 个完整评测单元、12,530 条正式预测全部 strict-valid。表中为观测 accuracy，概率质量另报。第二轮新增困难数据未改善 Reddit，保留为补充负结果。

## 阅读与运行入口

- [v1 实际训练数据](data/README.md)：完整6065条冻结输入、原验证/校准记录、两条预览样例，以及逐条来源和证据。
- [资料准备与仓库原生使用方式](docs/usage.md)：把 NLM 摘要转换为 TXT，用现有 `inspect/build/resume/validate` 完成出题与检查。
- [v1 结果](results/v1.md)与[结构化结果](results/v1.json)。
- [方法与消耗](docs/methods.md)：来源、清洗、训练选模、五种方法，以及 API/GPU/coding agent 的时间与用量。
- [benchmark 扩展](benchmark_ext/README.md)：独立安装包、三套完整测试的转换与评分。
- [v1 历史复现](docs/reproduce-v1.md)与[后续负结果](docs/followup.md)。
- [冻结来源](configs/sources.lock.json)、[模型和数据身份](results/identities.json)。

```text
medical_triage/
  README.md
  data/              v1 冻结数据、来源与数据说明
  configs/           原生配置、来源锁定与历史训练模板
  scripts/           来源清洗与测试格式转换
    v1/              仅供历史 v1 及其后续实验复现的专用实现
  benchmark_ext/     独立测试扩展
  docs/              使用、历史复现、方法与限制
  results/           正式结果、补充结果和用量摘要
```

## v1 与原生流程的关系

v1 当时使用了专用脚本调用教师模型构造虚拟患者、问题和完整选项，再交给仓库原生 `build` 编译、检查和导出。它**没有使用原生 TXT 的材料评估和问答构造阶段**，因此这些成绩不能作为通用材料出题能力的完整验证。

后续常规使用从 [原生配置](configs/native-generation.yaml) 开始。该流程只依据材料已有事实出题，不保证生成 v1 的固定四类行动、双患者结构和样本配额；它尚未重新生成数据或训练模型，也没有对应的新成绩。历史代码集中放在 `scripts/v1/`，主使用入口不调用它。

## 交付范围与限制

本目录包含 v1 实际训练数据、对应来源摘要、源码、配置、结果和方法说明；训练和推理依赖 DecisionForge 核心、固定模型版本及 Python 依赖，公开测试原文件按来源清单另行获取。没有提交模型、checkpoint 或公开测试逐题预测。结果 JSON 中的历史逻辑路径只作审计标记。

v1 的教师与 checker 均为 deepseek-flash，没有独立临床人工裁定；训练标签为合成监督。公开测试许可 CC BY-NC 4.0，来源与患者字面隔离不证明语义或预训练完全独立。LLM2Jev 的顺序偏差、AnyJev 失败恢复和未知费用均保留。Synth/Reddit 只有2/1个患者连通分量，不宣称独立pair显著性。

中间 checkpoint 和优化器状态已按要求清理，保留 v1 最终模型、冻结数据和原始预测；[清理记录](results/storage-cleanup.json)说明历史恢复的限制。运行产物写入自选工作目录，不随 Git 提交。
