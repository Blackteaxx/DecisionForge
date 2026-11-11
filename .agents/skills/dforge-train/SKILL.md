---
name: dforge-train
description: Use when a user wants to configure, launch, monitor, finish, diagnose execution failures in, or resume DecisionForge native decision-model training on user-specified compute resources.
---

# 训练配置与监管

执行前读取 [任务记录](../shared/task-record.md)、[执行规则](../shared/execution.md)、[交互与进度面板](../shared/interaction.md)、[训练选项](../shared/training-options.md) 与 [训练指南](../../../docs/zh/training.md)。用户已有可用训练数据时直接进入本阶段。

## Stage 0：确定训练路线与交付终点

先读请求、实际数据和已确认配置，核查本次训练已被用户选择（特别是只有场景的可选训练分支）。按共享规则展示并确认本次训练方案；缺少会改变训练的决定时在同轮给出选择，工具不可用时文本提问。不能把“给出方案”或总体路线已批准当成阶段接受。下述 Stage 0–5 是训练内部步骤，已确认的同一训练方案不逐步骤重问。

| 需要明确 | 可以呈现给用户的选择 | 不应混淆 |
| --- | --- | --- |
| 本次终点 | 确认启动后交接／跟进到训练产物与报告 | 进程启动不等于模型完成 |
| 原生训练路线 | Unsloth LoRA／Unsloth full／Laya RLCD／OpenJev RLCD | 用 `backend`、`finetuning` 选择；DPO、GRPO 无对应配置；没有 `training.method` |
| 起始配置 | 沿用用户已确认 YAML／选择最接近的项目示例再展示调整项 | smoke、完整数据训练是运行规模，不是不同训练算法 |
| 模型与资源 | 指定基础模型或已有 Clef、revision、主机/环境/GPU | 不把当前机器视作自动授权，不自动租算力 |

只问仍缺失的关键项。一轮可合并终点、路线/模型和资源问题；常规超参数按已授权默认值整理成配置表。用户问不支持的方法时，说明原生入口的边界并提出可选方案，不生成无法解析的 YAML。

## 进度面板

独立调用时使用以下步骤；串联任务沿用入口总面板。仅在状态/分支/阻塞变化或用户问进度时显示。

```text
训练进度
[x] 0. 明确路线、资源与终点
[~] 1. 检查数据、模型与环境
[ ] 2. 准备配置并离线预检
[ ] 3. 启动训练
[ ] 4. 跟进终态并核查产物
[ ] 5. 归档与交付
```

示例状态必须替换为真实状态。仅启动请求将第 4 步标 `[-]`，第 5 步为“启动交接”；交付时明确 `request fulfilled / run running / model pending`。不要为刷新面板额外等待训练结束。

## Stage 1：检查数据、模型与环境

1. 读取已有决定、数据与实际运行记录，明确 `launch` 或 `complete` 终点、模型/已知 revision、训练配置、用户指定的资源和设备。资源形式不限，但实际后端要求 NVIDIA CUDA；Windows 原生支持 Unsloth 单卡 LoRA，其余方法使用 Linux；Unsloth full/Laya 支持单机 DDP，OpenJev 仅单卡。未指定资源时先准备方案，不擅自选本机或租算力。
2. 核查输入契约、标签、group 与 split，确认验证/校准数据实际可用且不会泄漏。依据目标推荐配置；影响实验含义且未获授权的选择先明确。已有确认不重复问。
3. 核对模型访问和完整性、依赖既定版本、CUDA/实际 GPU 与可用空间。不要仅凭模型目录存在就判断完整；也不要仅凭 import 成功判断 CUDA 训练可用。记录用户指定的 GPU 与启动环境。

## Stage 2：展示配置并离线预检

按 [训练选项表](../shared/training-options.md) 选择示例，展示本次配置和重要偏离理由。关键值至少包括：方法、模型/revision、数据/split、资源、量化/精度、最大长度、适用时的 LoRA rank/alpha/dropout、epochs 或 max_steps、batch/累积、后端实际学习率、distributed/world_size；OpenJev 分别记录 calibrate/reinforce 步数、校准、是否 merged 导出和输出路径。

展示后必须实际等待用户接受或修改该训练方案；已确认的同一具体 YAML 可引用原决定，不逐参数再问。示例默认值必须先确定为本次实际方案，不能因合法或资源已知就自动启动。方案还需说明交付终点、成本估算或未知项；用户修改参数后重新展示并确认。`num_train_epochs` / `max_steps` 恰好一个非 null；4-bit 加载不是计算精度，merged 导出不是 Full 训练。

将 `configuration.md`、实际 `config.yaml`、stdout/stderr 放入 `training/<run-id>/`；程序输出指向独立空 `run/`。重写搬移配置后的数据、本地模型及输出路径，保存来源示例和已知模型版本。不要复制示例后沿用其输出目录。

在指定环境用 `train --validate-only` 检查数据及划分。该检查不加载模型、不初始化 GPU，不能替代 Stage 1 的环境检查。失败时保留原错误，按故障边界处理。

```bash
uv run dforge train run --config outputs/tasks/grid/training/train-001/config.yaml --validate-only
uv run dforge train run --config outputs/tasks/grid/training/train-001/config.yaml
```

示例中的实际配置必须指向任务目录输出。也可核对 CLI 后使用 `--run-dir` 覆盖；远程路径按远程环境解析，保存实际生效配置与命令。

## Stage 3：启动并核查真实运行

先核对当前训练方案的接受记录及配置身份，再检查已有作业，避免重复启动。没有阶段接受记录时停在方案确认，不执行启动命令。新启动后保存资源、PID/作业 ID、实际目录、配置与 stdout/stderr；核对进程命令、启动日志和 manifest。仅有目录或 PID 数字不算正常启动依据。

`launch` 请求在正常启动得到确认后交接：请求可 fulfilled，stage/run 仍 running，尚未产生的模型为 pending。若发现进程已在任务目录外运行，记录真实位置为 external，取回小型配置/日志摘要；不要为了调整归档路径重启活跃作业，或把外部位置假写成任务内。

## Stage 4：按请求跟进训练终态

`complete` 请求继续检查阶段、步数、loss、NaN/异常、长时间无进展和资源状态。按进展调整检查间隔；有监控能力时继续，无法跨会话监管时按共享规则保留未交付请求并交接。不能仅凭 loss 下降声称模型效果合格。

| 读取的实际证据 | 应报告的内容 | 下一步判断 |
| --- | --- | --- |
| `manifest.json` 的 `status` / `stage` | training、trained、calibrated、exported、completed 的实际阶段 | 完成优化步数后可能仍在校准或导出，不能提前标完成 |
| `logs/training.jsonl` 的 `step`、已有 loss/学习率、`validation`、GPU 内存记录 | 最近时间、真实步数、最新可用 loss；尚无指标则说明仍在加载或未产生日志 | 同阶段数值变化只简报，不重画面板；有限 loss 不证明效果达标 |
| 进程/作业状态、日志更新时间、stderr | 活跃、已退出、异常或暂未核查 | 后台 wrapper 的 exit 0/完成通知不能代替真实训练终态；长时间无进展先核查再归因 |

到终态后核对产物，不凭日志最后一行或退出码宣布成功。遇等待输入、中断或失败，面板用 `[!]` 并写明原因，保存独立的 run 事实状态。

## 故障与恢复

- 环境/执行故障可在用户指定环境及项目版本约束内修复，保存证据与操作。
- OOM 时不自动降 batch、截断序列、换模型或改精度。提出具体调整及实验影响；用户已授权的策略可直接执行，记录依据。提高梯度累积也不自动保证新配置与原实验等价。
- 接续先核查真实终态和配置身份。Unsloth 仅在存在完整兼容 Trainer checkpoint 时按原配置 `train --config ... --run-dir ... --resume`。模型 adapter 不等于完整训练 checkpoint；首步失败不假定已有 checkpoint。
- Laya/OpenJev 拒绝完整状态 `--resume`；`init_checkpoint` 初始化建立新运行并记录来源，不恢复优化器。
- 已 completed 的训练不得无条件再次 resume；核查现有模型与报告，按原请求交付或进入已请求评测的方案确认。新实验用新目录；不能把缺 checkpoint 后从头训练写成恢复。

## Stage 5：完成或启动交接

完成请求核查实际 manifest 为 completed、所需模型保存与重载验证、`report.json`、`summary.md`、`logs/training.jsonl`、实际配置及划分数据。只要求 adapter 时不凭空要求额外 merged 导出；按已确认配置检查产物。

任务索引记录真实 run 状态、请求交付、效果目标与证据。基础模型和远程大模型可为 external，保留位置/已知版本/获取方式；未生成模型为 pending。取回小型实际配置、报告和日志摘要，更新 README 的模型入口与失败/待决定项。

训练正常结束但目标未达标要分别报告；训练结束不自动开始未请求的评测、重标或新一轮训练。
