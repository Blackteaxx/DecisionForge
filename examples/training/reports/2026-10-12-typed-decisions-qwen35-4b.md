# Typed Decisions 全量训练与全部 benchmark 评测

完整流程示例入口：[run_full.sh](../unsloth/run_full.sh)，配置和使用方式见 [README](../README.md)。下文是已经执行的正式实验记录；[原始训练报告 JSON](2026-10-12-typed-decisions-qwen35-4b.training.json) 和 [指标摘要](2026-10-12-typed-decisions-qwen35-4b.metrics.json) 也归档在本目录。

实验开始：2026-10-12T05:47:38.419023+00:00；完成：2026-10-12T06:36:00.858132+00:00（UTC）。
推理/训练代码固定为 `9fb9125c032e38ad1ad5fc21fa67558f78e03270`；本报告由保存的训练、预测和评分产物生成。

## 训练设置与完整性

原 train 1,200 案例 / 6,000 问题全部进入数据划分；按用户选择保留独立验证和校准集。960 案例 / 4,800 问题用于梯度训练，验证 120 / 600，校准 120 / 600。原 test 400 / 2,000 只用于诊断和评测，未训练或拟合温度。划分按 group，seed 3407。

Qwen3.5-4B + LoRA rank/alpha 16/16，4-bit 加载，bf16，梯度检查点；batch 2 × gradient accumulation 4 = 8；学习率 2e-4，head 学习率 1e-4，warmup 10，cosine，weight decay 0.01。训练 2 epochs、240 个优化 step，选择最终 checkpoint，不按测试结果挑选。

上下文上限 16,384；全部数据编码后无 skipped / truncated。每 50 step 记录训练日志并运行独立 validation，首步也记录；每 100 step 和训练终点保存完整 Trainer checkpoint。官方校准仅使用 calibration 集，保存 adapter、decision head、校准和 merged 产物，并完成重载检查。

最终平均训练 loss：**0.869234**；训练耗时（含监控/训练 checkpoint）：**613.4 秒**。LoRA / head 参数更新检查：`{'lora': True, 'head': True}`。

## 训练监控与校准诊断

| Step | Epoch | 窗口 loss | Validation 官方 accuracy | Validation 官方 ECE |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0083 | 1.235331 | 0.290000 | 0.058474 |
| 50 | 0.4167 | 1.005802 | 0.693333 | 0.084952 |
| 100 | 0.8333 | 0.873189 | 0.793333 | 0.203069 |
| 150 | 1.2500 | 0.828904 | 0.800000 | 0.200022 |
| 200 | 1.6667 | 0.804568 | 0.820000 | 0.196511 |

| 原生五题联合诊断 | 问题数 | 官方 accuracy | 官方 ECE | 官方 loss |
| --- | ---: | ---: | ---: | ---: |
| validation / uncalibrated | 600 | 0.818333 | 0.190964 | 0.829014 |
| test / uncalibrated | 2000 | 0.796500 | 0.158403 | 0.843246 |
| calibration / uncalibrated | 600 | 0.806667 | 0.156552 | 0.789007 |
| calibration / calibrated | 600 | 0.806667 | 0.036698 | 1.048056 |
| test / calibrated | 2000 | 0.796500 | 0.025000 | 1.164471 |

这里的官方诊断使用原生五题联合输入及官方参考处理，与下面逐题 benchmark 指标分别标记。温度只在独立 calibration 集拟合；test 校准后 accuracy 未变、ECE 下降，官方 loss 上升，需结合软分布指标理解这一目标差异。

## 全部正式 benchmark

以下均直接加载本次 `training/model/`，单 GPU / 单并发，每次一个独立 decision 问题；保持官方 benchmark 数据转换和评分规则。全量四个数据集共 10,811 题。demo/generic 是框架样例检查，单独列出。

| Benchmark | 样本 | 有效 / 完成 | 主指标 | 数值 |
| --- | ---: | ---: | --- | ---: |
| typed_decisions | 2000 | 2000 / 2000 | gold_accuracy ↑ | 0.789500 |
| jevbench | 231 | 231 / 231 | accuracy ↑ | 0.740260 |
| banking77 | 3080 | 3080 / 3080 | macro_f1 ↑ | 0.653296 |
| clinc150_oos | 5500 | 5500 / 5500 | macro_f1 ↑ | 0.770114 |

### typed_decisions

| 指标 | 数值 | 使用 / 适用样本 |
| --- | ---: | ---: |
| gold_accuracy (higher) | 0.789500 | 2000 / 2000 |
| soft_accuracy (higher) | 0.610063 | 2000 / 2000 |
| soft_brier (lower) | 0.127335 | 2000 / 2000 |
| soft_kl (lower) | 0.395294 | 2000 / 2000 |
| gold_ece (lower) | 0.024398 | 2000 / 2000 |
| soft_ece (lower) | 0.202957 | 2000 / 2000 |
| mae (lower) | 0.283351 | 800 / 800 |

| 题型 | 样本 | gold_accuracy | soft_brier |
| --- | ---: | ---: | ---: |
| choice | 600 | 0.758333 | 0.133504 |
| noul | 600 | 0.848333 | 0.081659 |
| score | 800 | 0.768750 | 0.156966 |

| Workflow | 样本 | gold_accuracy | soft_brier |
| --- | ---: | ---: | ---: |
| agent_trace_observability | 500 | 0.706000 | 0.166035 |
| customer_service | 500 | 0.836000 | 0.116718 |
| invoice_processing | 500 | 0.836000 | 0.087779 |
| security_incidents | 500 | 0.780000 | 0.138810 |

覆盖率 100.00%；调用失败 0，协议失败 0，概率无效 0，未完成 0。
请求延迟 p50 / p95：64.11 / 122.85 ms；会话有效 QPS：11.540。不含模型加载时间，首请求包含 kernel warmup。
固定数据 revision：`e039ebffcc280174dd354227424fb2b249f191de`。
详细数值已归档在本目录的 [指标摘要](2026-10-12-typed-decisions-qwen35-4b.metrics.json)。

### jevbench

| 指标 | 数值 | 使用 / 适用样本 |
| --- | ---: | ---: |
| accuracy (higher) | 0.740260 | 231 / 231 |
| brier (lower) | 0.399121 | 231 / 231 |
| ece (lower) | 0.146712 | 231 / 231 |
| mae (lower) | 0.309469 | 18 / 18 |
| group_agreement (higher) | 0.972222 | 36 / 36 |

| 难度 | 样本 | accuracy | brier |
| --- | ---: | ---: | ---: |
| easy | 48 | 1.000000 | 0.000054 |
| hard | 111 | 0.504505 | 0.734631 |
| original | 72 | 0.930556 | 0.147922 |

覆盖率 100.00%；调用失败 0，协议失败 0，概率无效 0，未完成 0。
请求延迟 p50 / p95：120.10 / 191.86 ms；会话有效 QPS：7.796。不含模型加载时间，首请求包含 kernel warmup。
固定数据 revision：`b6b8fff7e345b98c060ad26c13308860ddc67004`。
详细数值已归档在本目录的 [指标摘要](2026-10-12-typed-decisions-qwen35-4b.metrics.json)。

### banking77

| 指标 | 数值 | 使用 / 适用样本 |
| --- | ---: | ---: |
| macro_f1 (higher) | 0.653296 | 3080 / 3080 |
| native_accuracy (higher) | 0.668182 | 3080 / 3080 |

覆盖率 100.00%；调用失败 0，协议失败 0，概率无效 0，未完成 0。
请求延迟 p50 / p95：80.80 / 83.81 ms；会话有效 QPS：11.546。不含模型加载时间，首请求包含 kernel warmup。
固定数据 revision：`57ec275d8078af65b7731c2a98be812d844a6d6b`。
详细数值已归档在本目录的 [指标摘要](2026-10-12-typed-decisions-qwen35-4b.metrics.json)。

### clinc150_oos

| 指标 | 数值 | 使用 / 适用样本 |
| --- | ---: | ---: |
| macro_f1 (higher) | 0.770114 | 5500 / 5500 |
| native_accuracy (higher) | 0.758182 | 5500 / 5500 |

OOS precision / recall / F1：0.633180 / 0.687000 / 0.658993；support：1000。
域内 150 类 macro-F1：0.770855。

覆盖率 100.00%；调用失败 0，协议失败 0，概率无效 0，未完成 0。
请求延迟 p50 / p95：124.14 / 126.79 ms；会话有效 QPS：7.678。不含模型加载时间，首请求包含 kernel warmup。
固定数据 revision：`155b9c710419136e17307b80d0a13e68cd46b4ec`。
详细数值已归档在本目录的 [指标摘要](2026-10-12-typed-decisions-qwen35-4b.metrics.json)。

## 结果解读与口径

Typed Decisions 的 gold 是教师三次采样的模态标签，软分布是三次采样平均；一致率衡量教师标签一致性。训练与官方诊断采用每案例五题联合上下文，benchmark 采用逐题独立上下文；不可将这份结果直接视为上游榜单复现。四个数据集的 label/rubric、指标和难度不同，未将它们的准确率合成一个总分。BANKING77 / CLINC151 的类别数大于 Typed Decisions 训练中的选项数，结果同时反映跨域和大类别空间迁移。

没有未训练底座的对照实验，不能据此量化训练带来的因果提升；此前 3-step 冒烟模型仅验证流程。本报告未调整 benchmark 概率容差、未对输出概率做修补或重归一化。

本次 Typed Decisions 单题 gold_accuracy 为 78.95%，此前 3-step 冒烟模型为 41.85%；soft Brier 从 0.248135 降到 0.127335。两次的数据规模、LoRA rank、训练步数和 calibration 集大小均不同，这是流程进展对照，不能当作单一参数消融实验。

本次 Typed Decisions Gold ECE 为 0.024398，Soft ECE 为 0.202957。保存的官方温度拟合目标是 hard/modal teacher 标签，因此 Gold ECE 较小不代表对教师平均软分布的校准也好；Soft ECE 比此前冒烟模型的 0.055021 更高。报告同时保留两种参考口径，不只报告表现较好的校准指标。

JevBench hard 的 accuracy 为 50.45%，低于 original 的 93.06% 和 easy 的 100%。CLINC 的 OOS precision 为 63.32%、recall 为 68.70%；1,000 个 OOS 样本中漏掉 313 个，另外 398 个域内样本被预测为 OOS。这些分项是当前模型的具体弱项，整体准确率不能替代这些指标。

## 环境与运行问题

NVIDIA H200 单卡，Python 3.12.3、Torch 2.11.0+cu129、CUDA 12.9、Unsloth/Zoo 2026.10.3、Transformers 5.17.0、PEFT 0.21.2、Accelerate 1.13.0、BitsAndBytes 0.50.2、Safetensors 0.8.0、NumPy 1.26.4。实际环境完整记录见训练 manifest。未估算 GPU 成本。

首轮编译路径在首个 optimizer step 的 backward 遇到 custom-op stride 断言；相同 batch 对照复现失败，通过官方 `UNSLOTH_COMPILE_DISABLE=1` eager 路径完成。正式运行采用该开关，未改变 batch、梯度累积、学习率或数据划分；没有吞掉异常继续训练。失败产物和 compiled/eager probe 日志保留供复核。

merged 导出初次因主存储 100 GiB 配额不足中断，已将目标迁移到已挂载的 空间充足的持久目录，从 calibrated 阶段恢复导出和重载，不重复优化或拟合校准。主实验目录的 training/merged 是该保存目录的符号链接。

正式训练在单卡 H200 完成，导出从最终 checkpoint 恢复；全部评测结束后 GPU 资源已释放。

归档中的机器专属路径已改成示例相对路径，测量数值未修改。依赖版本与数据 revision 保留用于复现。

## 框架样例检查

- demo：9 条，严格有效 9；仅为随附示例的协议检查，不纳入正式 benchmark 总数。
- generic：9 条，严格有效 9；仅为随附示例的协议检查，不纳入正式 benchmark 总数。

## 产物与复现

本示例随仓库提供：

- [训练配置](../unsloth/qwen35-4b-full.yaml) 与 [评测配置](../unsloth/eval)。
- [完整流程脚本](../unsloth/run_full.sh) 与 [使用说明](../README.md)。
- [训练报告 JSON](2026-10-12-typed-decisions-qwen35-4b.training.json) 与 [指标摘要](2026-10-12-typed-decisions-qwen35-4b.metrics.json)。

在仓库根目录运行：

```bash
bash examples/training/unsloth/run_full.sh
```

新的训练产物保存到 `outputs/train/qwen35-4b-full/`，逐题预测及评测报告保存到 `outputs/eval/qwen35-4b-full/`。历史模型权重和逐题预测体积较大，不随示例提交；本目录保留其真实训练与指标记录。

保存的评测结果可离线重评分：

```bash
bash examples/training/unsloth/run_full.sh --score-only
```
