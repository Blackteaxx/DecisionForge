# 在仓库中使用 Agent Skills

让 agent 从用户目标出发，使用 DecisionForge 完成数据研究与准备、训练、评测或结果分析。各阶段可以单独使用，也可以按请求串联；每次任务有一个目录，集中保存记录、配置、报告和产物入口。

## 开始使用

完整的需求澄清、数据分支、阶段确认和交付边界见 [Agent Skills 端到端工作流](../../docs/zh/agent.md)。

带 DecisionForge Logo 和 skills 使用指南的浏览器入口见 [Agent 界面与插件启动教程](../../docs/zh/agent.md)，包括固定版本、模型设置、workspace 选择和真实预检。

DeepSeek Harness 和 Codex 从仓库 `.agents/skills/` 发现这五个 skills；Claude Code 从 `.claude/skills/` 发现指向同一份源码的相对目录软链接。共享规范也通过同一兼容入口可读，无需复制两套 SOP。

在仓库目录中启动 agent，使用自然语言描述任务，或显式选择 skill：

| Harness | 显式调用 | 发现位置 |
| --- | --- | --- |
| DeepSeek Harness Web UI | `/dforge`、`/dforge-eval` 等 | 当前所选 workspace 的 `.agents/skills/<name>/SKILL.md` |
| Codex | `$dforge`、`$dforge-data` 等；也可在 `/skills` 中选择 | `.agents/skills/<name>/SKILL.md` |
| Claude Code | `/dforge`、`/dforge-data` 等 | `.claude/skills/<name>/SKILL.md`，软链接到 `.agents/skills/<name>` |

例如在 Codex 中输入：

> $dforge 帮我寻找适合电网故障判断的数据集和原始材料，比较来源并给建议，结果放到 `outputs/tasks/grid-research/`。

在 DeepSeek Harness 或 Claude Code 中使用相同请求，将开头换为 `/dforge`。是否按自然语言自动选用取决于描述匹配；显式调用可直接指定入口。若当前会话尚未显示新 skills，重新启动该会话。

其他 harness 或需要排查时，仍可明确读取入口文件：

> 请读取 `.agents/skills/dforge/SKILL.md`。帮我寻找适合电网故障判断的数据集和原始材料，比较来源并给建议，结果放到 `outputs/tasks/grid-research/`。

也可以直接进入一个阶段：

> 请读取 `.agents/skills/dforge-eval/SKILL.md`。使用我提供的模型、评测配置和资源完成评测与对比，任务目录是 `outputs/tasks/model-comparison/`。模型和配置位置如下……

五个文件均为独立 skill；本包不要求全局安装。Git 检出应保留 `.claude/skills/` 的目录软链接，否则可先使用明确读取文件的方式。本次验证方式和能力范围见 [验证记录](validation/results.md)。

目录依据：[Codex 的本地 skills 发现规则](https://learn.chatgpt.com/docs/build-skills#where-codex-loads-local-skills)、[Claude Code 的项目 skills 与软链接规则](https://code.claude.com/docs/en/skills#choose-where-skills-load)。这里使用的是带点的 `.agents/`，不是普通 `agents/`。

| 入口文件 | 适用任务 |
| --- | --- |
| [dforge](dforge/SKILL.md) | 不确定阶段、组合任务、继续已有任务 |
| [dforge-data](dforge-data/SKILL.md) | 找数据或原始材料、整理数据、生成并检查 JEV 数据 |
| [dforge-train](dforge-train/SKILL.md) | 确定训练配置、启动、监管、完成或恢复训练 |
| [dforge-eval](dforge-eval/SKILL.md) | 评测已有模型、对比结果、恢复评测或离线重评分 |
| [dforge-analyze](dforge-analyze/SKILL.md) | 分析数据、训练和评测问题，给出下一步建议 |

阶段 skill 会读取包内共享规则和项目文档。只需要部分阶段时直接描述，不必运行其他阶段。仓库程序的安装与模型服务配置仍按 [项目说明](../../README.md) 和对应阶段文档准备。

## 描述目标与执行终点

以下请求有不同交付终点：

- “找数据，给建议”：交付来源研究和推荐；“准备好 JEV 数据”：继续获取、整理、生成和检查。
- “启动训练，别等结束”：确认启动并交接；“完成训练”：跟进终态并检查模型与报告。
- “评测现有模型”：直接进入评测；“比较现有评测报告”：可复用已有预测与报告，不必重新推理。
- “分析问题，给下一步建议”：交付有证据的报告，由用户选择后续行动。

提供已知目标、数据/配置/模型位置、资源、设备和已有约束即可；agent 应先检查可读信息，再补问影响执行的缺失决定。资源形式不限，由用户指定并检查其与实际后端是否兼容。训练平台、设备和方法能力以 [训练选项](shared/training-options.md) 与实际后端为准；当前生成程序读取 JSON/TXT，其他材料先在数据准备阶段转换。

## 交互式选择与进度面板

五个入口使用同一 [阶段确认规则](shared/interaction.md)：先核查必要事实，展示本阶段目标、策略与关键参数，请用户接受或修改后才执行；无适用提问工具时用文本问题。总体路线确认不代替训练、评测或分析等具体方案的确认，同一阶段已确认且未变化的方案不逐命令重问。数据 skill 内的评测集整理、资料搜寻与训练数据整理分别确认；训练/评测内部预检、监控和归档无需拆成多轮审批。用户修改方案后重新展示并确认，未回复保持 waiting_input。

执行时展示紧凑的进度面板，例如：

```text
DecisionForge 进度
[x] 0. 需求澄清与整体路线确认
[x] 1. 整理并固定评测集
[x] 2. 搜寻资料
[x] 3. 整理训练数据
[!] 4. 完成训练（等待接受方案）
[ ] 5. 评测与比较
[ ] 6. 性能报告与 insight
```

仅在首次确定流程、状态/分支/阻塞变化或用户询问进度时显示；同阶段的 loss、步数和样本数量更新只简报。仅启动请求显示“训练启动与交接已完成”，另注明实际 run 仍在运行。面板不增加未请求的阶段，也不改变任务状态模型。

## 能选择哪些训练与评测方案

| 任务 | 当前可选择的内容 | 详细说明 |
| --- | --- | --- |
| 原生决策训练 | 选择当前后端/微调方法、模型、数据/划分、资源、训练量、适用的量化/精度与超参数 | [训练选项](shared/training-options.md) |
| 已有模型评测 | 已有原生服务或相应版本化本地产物，按实际 adapter 选择 | [评测选项](shared/evaluation-options.md) |
| Training-free 评测 | Chat baseline、LLM2Jev、AnyJev 等已实现路线；选择 benchmark、split、范围及推理配置 | [方法与配置](shared/evaluation-options.md) |
| 已有结果复用 | 离线重评分或比较，保留样本快照和可比性条件 | [评测 skill](dforge-eval/SKILL.md) |

方法和参数以当前实现与选项文档为准，不从其他框架照搬开关；`save_merged` 也不是全参数训练。LLM2Jev 使用 `jev` adapter 加方法声明，并要求真实 LLM2Jev 服务，不是 `adapter: llm2jev`。agent 会展示本次关键配置和改动理由，详细表保存为阶段目录的 `configuration.md`。

## 查看与继续任务

优先使用用户指定目录，默认在 `outputs/tasks/` 下创建唯一任务目录。研究和分析任务也会建立目录，未使用的阶段不创建空结果。

```text
outputs/tasks/my-task/
  README.md
  task.json
  data/research/
  data/inputs/
  data/prepared/
  data/runs/<run-id>/
  training/<run-id>/
  evaluation/<run-id>/
  analysis/<report-id>/
```

先打开任务的 README，查看当前进展、结果和待决定事项。`task.json` 保存交接所需的结构化索引。不同运行保留独立目录和配置；生成、训练、评测内部的 manifest 与检查点完整保留。

训练和评测的 `training/<run-id>/`、`evaluation/<run-id>/` 保存配置及命令日志，程序产物写到各自的 `run/` 子目录。首次启动时 `run/` 必须为空；不能先把配置、PID 或重定向日志写进程序输出目录。

> 请读取 `.agents/skills/dforge/SKILL.md`，继续 `outputs/tasks/my-task/`，先核实已有作业当前状态。

基础模型、既有数据和远程大产物可保留在外部。目录入口会列明位置、已知版本、获取方式及 `external/pending` 标记，小型配置、报告和日志摘要集中保存在任务目录。含外部引用时，整个目录尚不是独立备份。TXT 自动分批的工作副本和派生批次也归入所属任务，保留用户原数据。

## 运行中与已完成

只要求启动时，用户请求可以已经交付，训练或评测仍显示运行中。要求完成时，agent 必须检查终态和实际产物；效果是否达标另行说明。

会话存续期间使用当前 harness 的工具监管。会话结束后是否能自动检查取决于 harness 实际能力和已建立的检查机制；没有该能力时保存未完成状态与接续方式。单纯后台启动不表示持续监管已经安排。

环境与执行故障可在指定环境和既定版本约束内处理。改模型、改数据含义、改训练/评测方案、扩大资源/预算或修改源码时，agent 先给具体方案，已有明确授权则沿用。分析报告会区分观测、已确认原因和待验证假设，不自动启动下一轮。

## 共享规范与模板

- [任务记录规范](shared/task-record.md)：目录、状态、产物和接续规则。
- [自包含交付目录](shared/delivery.md)：可分享的来源与脚本、benchmark 扩展、README、结果和消耗细节；不提交 `outputs/`。
- [执行规则](shared/execution.md)：资源、配置、监管和故障处理边界。
- [交互规则](shared/interaction.md)：Stage 0 提问工具、回答记录、配置展示和进度面板。
- [训练选项](shared/training-options.md)、[评测选项](shared/evaluation-options.md)：当前支持的方法、重要字段、示例选择与限制。
- [任务索引模板](shared/templates/task.json)、[启动任务示例](shared/templates/task-example.json) 与 [修改方案后待确认示例](shared/templates/task-confirmation-example.json)。
- [任务 README](shared/templates/README.md)、[来源研究报告](shared/templates/research.md)、[分析报告](shared/templates/analysis.md) 模板。

复制模板后填写实际信息；示例中的路径、进程和检查时间是演示值。凭据通过环境变量或用户已有方式提供，不保存到任务报告。

## 维护与验证

修改 skill 时，先用 [行为场景](validation/scenarios.md) 观察无 skill 的行为，再在新 agent 中读取对应 skill 应用同样场景；记录实际选择及未覆盖范围。场景应用不等于真实联网研究、收费 API、GPU 或跨会话监控验收。

在已安装基础依赖的 Python 环境中，从仓库根目录执行：

```bash
python .agents/skills/validation/check_package.py
python .agents/skills/validation/check_package.py --offline-cli
```

第一条检查五个入口、frontmatter、本地链接与 JSON 示例引用；第二条还在临时任务目录中检查搬移配置后的路径，实际运行现有 inspect、validate 和 train/eval validate-only。它保留源配置，不执行生成、推理或 GPU 训练。

命令与参数以当前项目 CLI 和 [项目文档入口](../../README.md) 为准。无需为更新 SOP 改写运行时；必须改程序才能满足需求的缺口，先提出具体变更。
