# 任务目录与记录

所有 DecisionForge skills 开始执行前读取本规范和 [执行规则](execution.md)。记录用于交接；底层 manifest、实际配置、日志和产物才是运行事实依据。

## 创建和接续

优先使用用户指定的任务根目录；未指定时，在仓库 `outputs/tasks/` 下选择唯一名称。新任务不要静默使用已有目录。接续请求先读取该目录的 `README.md`、`task.json`，核查实际资源、进程与产物，确认后再执行。一次只有一个 agent 写索引；若已有会话负责该任务，先交接。

从 [task.json 模板](templates/task.json) 和 [任务入口模板](templates/README.md) 创建记录。初始化时填写任务标识、目标、时间、仓库路径与实际 revision、请求的阶段和终点；未知值用 `null` 或空数组，不能编造。见 [启动任务示例](templates/task-example.json)；示例值是演示数据，不能作为实际运行证据。

按 [交互规则](interaction.md) 将原生问答或文本回答写入 `decisions`（选择、依据、确认时间），授权引用对应决定；待回答问题保留在 `pending_decisions`，回答后移除已解决项并保留决定历史。每个业务阶段的 `configuration.md` 记录目标、输入、策略/关键配置、资源/成本、输出、方案版本、来源及偏离理由，并作为 artifact 登记；研究、评测集整理和分析同样需要。面板仅是这些事实的展示，不新增运行状态枚举。

```text
任务根目录/
  README.md                  # 状态、结果入口、外部文件与继续方式
  task.json                  # 结构化索引
  data/research/             # 来源比较、研究报告
  data/inputs/               # 获取的原始材料或必要的工作副本
  data/prepared/             # 整理后的 JSON/TXT、映射与转换记录
  data/runs/<run-id>/        # 生成配置、日志、实际运行产物
  training/<run-id>/         # 训练配置、日志、实际运行产物
  evaluation/<run-id>/       # 评测配置、日志、实际运行产物及比较报告
  analysis/<report-id>/      # 分析报告及离线计算结果
```

按需创建目录。底层运行目录保留原布局，包括 `.dforge/`、训练 checkpoint 和评测日志。每次新的实验使用新 run ID 和目录，索引累积运行历史。不能移动一部分内部文件后声称运行仍可恢复。

训练、评测的阶段目录保存 `config.yaml`、`stdout.log`、`stderr.log` 等控制文件；实际程序输出使用独立的 `run/` 子目录，首次启动前该子目录必须为空或尚不存在。比如 `training/train-001/config.yaml` 的输出指向 `training/train-001/run/`，manifest、模型与训练器保存的配置都位于 `run/` 内。不要为了收集日志预先占用程序要求为空的新运行目录。见 [执行规则](execution.md)。

## 确认范围、方案版本与等待

保持 `decisionforge.agent-task.v1` 及 `data/train/eval/analyze` 四种阶段键；不为需求澄清、路线确认或数据子步骤新增运行状态/阶段枚举。沿用模板字段：决定使用 `id/decision/basis/confirmed_at`，阶段使用 `status/run_ids/artifact_ids/blocked_on`；等待状态由待决定项表达，不另造 `accepted` 标志、`recorded_at` 确认时间或 `blocked_by` 阶段字段。

- 入口的需求澄清与总体路线方案写入任务 README；在 `decisions.decision` 明写确认范围（如“整体路线”），`basis` 保留实际用户回答及所展示方案的版本/快照。路线决定不能作为执行训练/评测的唯一授权依据。
- 具体方案分别放在 `data/eval-set/configuration.md`、`data/research/configuration.md`、`data/prepared/configuration.md` 或实际生成目录、`training/<run-id>/configuration.md`、`evaluation/target/configuration.md`、`evaluation/<run-id>/configuration.md`、`analysis/<report-id>/configuration.md`，只创建实际需要的目录。入口/资料工作完成不等于 `data/prepare` 已交付。
- 在已接受的 `decisions` 的 `decision` 和 `basis` 中写明业务步骤、最终接受的内容、方案版本与 artifact ID；`authorizations.decision_refs` 只引用实际接受当前动作的决定。草案/修改待确认时只写 `events` 和 `pending_decisions`，不追加 `decisions`；用户接受后才追加决定。旧版本保留为快照或差异记录，不能覆写后仍沿用旧确认。
- `pending_decisions` 使用具体问题、选项、原因、影响的 `stage` 与方案 artifact 引用；可附业务步骤说明。同一 `data` 阶段的不同业务步骤通过这些记录区分，不把一次资料搜寻确认应用到训练数据整理。
- 无数据路线还没有选择训练时，训练分支保留在待决定项，暂不添加 `request.stages` 的 train 项和 `stages.train`。确定选择后才更新路线；选择不训练则明确记录跳过理由，不能标为训练完成。
- 等待当前方案接受时，`request.status` 和当前 `stages.<stage>.status` 为 `waiting_input`；无实际启动就不创建 run。若等待发生在上次运行失败后，原 run 仍为 `failed`，不能随阶段等待而改写历史。入口未确定阶段时只在请求和待决定项记录等待。
- 接受当前版本后移除对应待答项，记录确认时间与用户回答，按实际进展恢复 request pending / stage preparing；只有真实启动后才记录 running，只有核查产物后才 completed。

见 [方案修改后等待确认的示例](templates/task-confirmation-example.json)；这是合成记录，不能作为实际运行证据。接续必须核对配置版本、原授权范围和作业事实；缺少确认记录或仍待回答时恢复提问，不能把“继续”自动当作接受新方案。

## 索引字段

`schema_version` 固定为 `decisionforge.agent-task.v1`。时间使用带时区的 ISO 8601。任务内路径以任务根目录为基准；CLI 参数中的路径则按各命令规则处理。

| 字段 | 内容 |
| --- | --- |
| `task_id`, `goal`, `created_at`, `updated_at` | 任务身份、用户目标及时间 |
| `request.stages` | 有序 `{stage, endpoint}` 列表；仅登记请求过的阶段 |
| `request.status`, `request.fulfilled_at` | 请求是否交付及时间，与后台运行状态分开 |
| `outcome` | 用户效果目标、验收标准及效果判断 |
| `repository` | 实际仓库路径和 revision，远程代码版本在对应 run 记录 |
| `decisions`, `authorizations` | 已确认选择、授权行动及依据；沿用有效授权 |
| `constraints` | 指定资源引用、已给预算及其他限制；`null` 表示未指定 |
| `resources` | 唯一 ID、资源类型、可定位地址、访问方式、已知版本和检查时间 |
| `stages` | 按 `data/train/eval/analyze` 索引的状态、run ID、artifact ID 与阻塞事项 |
| `runs` | 唯一 ID、阶段、实际资源与 revision、实际配置、状态、运行目录、进程或作业标识、最近检查时间和证据 |
| `artifacts` | 唯一 ID、阶段、种类、归档状态、位置、资源引用、版本、获取方式与检查时间 |
| `pending_decisions` | 待用户选择的具体问题、方案、原因和影响阶段 |
| `events` | 阶段转换、执行故障、修复、恢复与结果的有据记录 |

`stages` 中未请求的阶段直接省略。引用的 run、artifact 和 resource ID 必须存在。记录配置和产物时使用实际返回位置，特别是生成命令可能返回带版本后缀的 `run_directory`。

## 状态与完成含义

阶段与 run 使用以下状态：`pending`（尚未开始）、`preparing`（准备或预检）、`running`（有当前运行依据）、`waiting_input`（等待决定或必要输入）、`interrupted`（中断待恢复）、`failed`（有明确失败依据）、`completed`（已核查该阶段终态与交付）。每条状态附证据和最近检查时间；旧索引不自动转成当前事实。

请求状态为 `pending`、`fulfilled`、`waiting_input`、`interrupted`、`failed`。完成全部请求的交付后才设 `fulfilled`；部分完成保留剩余事项。用户只要求启动时，可在进程或作业存活且启动日志正常后设请求 `fulfilled`，同时 run 和阶段仍为 `running`。仅启动不表示模型已经生成。

终点取值：数据为 `research` 或 `prepare`；训练、评测为 `launch` 或 `complete`；分析为 `recommend`。入口按用户请求写明，不用一个含糊的“done”覆盖不同终点。

效果判断 `outcome.status` 使用 `unknown`、`met`、`not_met`、`not_applicable`。训练完成但未达到准确率目标，可同时是请求已交付、run `completed`、效果 `not_met`。没有效果阈值时报告观测值，不自设阈值宣称达标。

## 归档与外部引用

任务运行记录与可分享交付分开保存。用户要求分享、整理或提交时，按 [交付目录](delivery.md) 准备不依赖私人任务路径的集中入口；`outputs/` 保持为本地工作产物，不加入 Git 提交。

产物的 `storage` 为：

- `archived`：实际文件在任务目录内，`path` 是任务内相对路径。读取时核查文件存在；指向外部的软链接不算归档。
- `external`：文件在任务目录外或远程。`path` 写完整路径/URI，提供 `resource_id`、已知版本、最近检查时间和 `retrieval` 获取方式。
- `pending`：期望产物尚未产生或尚未取得。不能把待获取文件当作已验证证据。

基础模型、用户原有数据和远程大产物可用外部引用。远程实际配置、小型运行记录、日志摘要和报告应取回任务目录；取回失败如实保留 pending 项和剩余工作，不假装已归档。README 列出 external/pending 项；含外部引用的目录不能称为完整独立备份。

TXT 自动分批会在解析后的输入文件同级写 `batches/`。先把需要处理的 TXT 物化为任务目录中的工作副本再调用命令，避免改动原始目录。不要用指向原文件的软链接代替副本。对远程任务也先准备任务所属的远程工作目录，再登记远程批次与大产物引用。

## 更新与交接

在启动、状态变化、异常、恢复、阶段结束或交接时更新索引和 README；保留运行历史和关键事件。使用临时文件再替换索引，避免半写入 JSON；这只是文件写入规范，不需要新的状态服务。

README 的当前进展保留最近一次核实的请求终点与 run 状态。聊天面板遵守只在状态/范围/阻塞变化或用户询问时显示的规则；后台仅有 loss、步数更新不重复整张面板。新会话接续先重新核查事实，再恢复展示。

恢复时验证作业是否还在运行、manifest 是否匹配、文件是否可读。已完成的运行按项目规则复用产物；不能无条件再次 `resume`。目录索引、进程状态或证据不一致时记录差异，再选择监控、兼容恢复或提出修复方案。不得因为旧状态为 running 而启动第二个作业。
