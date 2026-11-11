---
name: dforge-eval
description: Use when a user needs DecisionForge evaluation of existing local or served models, benchmark configuration, comparison with other models, evaluation monitoring or recovery, or offline rescoring of saved predictions.
---

# 模型评测与比较

执行前读取 [任务记录](../shared/task-record.md)、[执行规则](../shared/execution.md)、[交互与进度面板](../shared/interaction.md)、[评测选项](../shared/evaluation-options.md) 与 [评测指南](../../../docs/zh/evaluation.md)。已有模型可直接评测，不先生成数据或训练。

## Stage 0：选择评测路线与范围

先读请求、已有模型/服务、配置和报告。默认先展示当前业务阶段方案，再实际等待接受；未确定的方法、benchmark、样本范围、资源或终点在同轮提问。已有同一具体方案的接受记录可沿用，总体流程确认不能替代评测配置确认。工具不可用时文本提问。下述 Stage 0–5 是评测内部步骤，不逐命令增加确认门。

只有场景且选择不训练时，先展示候选已有模型/免训练方法、选择依据、资源要求、成本与限制，取得路线选择后再准备评测方案。选择评测对象和确认实际评测配置是两个业务阶段；已有明确对象可直接预填，不要求重复选择。

| 路线 | 实际配置选择 | 适用目的 |
| --- | --- | --- |
| 已有原生决策服务 | `adapter: jev` | 评测现有服务，无需本次训练 |
| 已训练本地决策模型 | `adapter: unsloth_local` | 读取保存的 Clef LoRA/full/merged，单卡 CUDA；不能直接评测任意原始 HF 权重 |
| Laya/OpenJev 本地产物 | `adapter: laya_local` / `openjev_local` | 使用各自版本化产物；OpenJev 支持 Noul 与完整多类别 Choice，仍不支持 Score；v1 可读，新训练保存 v2 标签池 |
| Training-free AnyJev L0 | `adapter: anyjev_local` | 固定普通 HF 底座与 L0；支持完整 77／151 类 Choice，运行时逐个验证互异单 token 标签；不训练 |
| Training-free Chat baseline | `adapter: chat_api`、`provenance.method: chat_baseline` | zero-shot 生成 JEV JSON 与自报概率 |
| Training-free LLM2Jev | `adapter: jev`、`provenance.method: llm2jev` | 使用 LLM2Jev 服务从候选 yes/no logits 构造概率；不是新 adapter |
| 已有报告/预测 | `compare.py` 或 `--score-only` | 离线比较/重评分，无新增推理 |

如果用户只说“试一下 training-free”，问 Chat、LLM2Jev、AnyJev 还是适用的方法比较；已经点名两者就直接沿用。接着确定评测已有服务还是需要在指定资源部署、选择哪些 benchmark、全量还是已授权的 smoke，以及启动后交接还是完成报告。不要默认启动所有公开 benchmark，也不要为了两路线比较先训练模型。

## 进度面板

独立调用时用下面的步骤，串联任务沿用入口总面板。标记/分支/阻塞变化或用户问进度才重画；已处理样本数变化可一句简报。

```text
评测进度
[x] 0. 明确方法、样本与资源
[~] 1. 核对数据、模型身份与比较条件
[ ] 2. 准备配置并离线预检
[ ] 3. 执行评测或重评分
[ ] 4. 核查覆盖并生成比较
[ ] 5. 归档与交付
```

根据请求调整示例：仅比较已有报告时跳过新推理步骤；仅启动时将后续完成核查标 `[-]` 并交接 run running；单模型没有比较要求时第 4 步只核查报告。未执行步骤不标完成。

## Stage 1：明确输入与比较条件

1. 读取目标、现有配置、模型与报告，确定 `launch` 或 `complete` 终点、待评/对比模型、数据版本和 split、benchmark、指标、推理设置及用户指定资源。资源形式按 adapter 检查，不把 HTTP 评测强制当成本地 GPU 任务。
2. 检查实际数据契约、模型/endpoint 身份、已知版本、样本范围、评分口径、失败分母与目标是否匹配。先用 `eval-list` 查看已有注册项，不发明 benchmark 或参数。
3. 比较前区分系统比较与受控方法比较。不同模型可以做有说明的系统比较；数据、split、评分不同的分数不能直接作优劣证据。先登记差异与缺失条件，不通过重标、替换样本或改指标制造公平结论。
4. 已有完整评测目录可引用或保留布局取回；无需复制远程大模型来满足归档。只有比较已有报告时，先展示证据范围、比较方法、产物与限制并确认，再进入 Stage 4；离线重评分按 Stage 3 的快照要求准备，不要求原模型在线。

## Stage 2：展示配置并离线预检

按 [评测选项表](../shared/evaluation-options.md) 展示方法/adapter、实际模型与 endpoint/路径、benchmark/version、split/subset/limit、指标、推理设置、资源和输出。只列当前 adapter 支持的选项：Chat 的 temperature、max_tokens、response_format、thinking；本地的量化/精度/长度/加载超时；HTTP 的服务与方法 provenance。共同执行项为并发、请求超时和有限重试，本地强制 concurrency=1、max_attempts=1。

展示本次交付终点、成本估算/未知项及完整方案后，实际请用户接受或修改；即使字段完整也不能凭总体授权开始推理。已确认且未变化的同一具体评测方案可引用原决定，不重复问；参数修改后重新确认。把选定示例、关键偏离和实际值保存为 `configuration.md`。各模型/方法的 `config.yaml` 与 stdout/stderr 放在独立的 `evaluation/<run-id>/`，输出指向各自空 `run/`；搬移后重写数据、本地模型和输出路径。

AnyJev 使用 [固定 L0 示例](../../../examples/eval/training-free/anyjev/eval.yaml) 或完整目录的 [BANKING77](../../../examples/eval/training-free/anyjev/banking77.yaml)／[CLINC150+OOS](../../../examples/eval/training-free/anyjev/clinc150.yaml) 配置；小目录保留旧路径，大目录仍执行完整 K 次轮换。标签池、完整 prompt 上下文和显存需在运行时核实，不能裁剪候选来跑通。其他 training-free 配置入口为 [Chat](../../../configs/eval/chat) 与 [LLM2Jev](../../../configs/eval/llm2jev)，二者均有 JevBench、Typed Decisions、BANKING77、CLINC150+OOS 样例。核对真实服务、底座 revision、依赖/精度及 thinking，不能把历史示例的 provenance 原样当成本次事实。任务内执行使用复制后的 YAML；不要直接调用固定读取示例配置/输出的 `run.sh both`。

```bash
uv run dforge eval run --config outputs/tasks/compare/evaluation/eval-001/config.yaml --validate-only
# 仅在新实测已获授权、预检通过且服务/资源可用后执行：
uv run dforge eval run --config outputs/tasks/compare/evaluation/eval-001/config.yaml
```

离线预检检查数据/配置和相应文件布局，不证明模型已成功加载或服务可调用。保存实际返回的目录、状态、报告位置、样本覆盖和失败信息；stdout/stderr 同属任务。启动或持续监管按用户终点执行，完成前核对报告而非仅看退出码。

## Stage 3：执行或复用

新实测先核对阶段确认及实际配置身份，再核查已有作业，避免重复启动；执行上面的实际 YAML。多方法对比逐个登记 run，不能合用输出目录。新环境可建议先小样本 smoke；只有已授权才增加此调用，smoke 与全量使用不同配置/输出，不更改 limit 后直接 resume。

`launch` 按正常启动证据交接。`complete` 继续核查进程、日志、已处理/失败覆盖和真实终态；同阶段样本数变化不重复面板。一个方法失败时保留错误及未完成部分，不能将已有部分报告写成整个比较已完成。

| 操作 | 使用条件 |
| --- | --- |
| `eval ... --resume --run-dir 实际目录` | 核查既有运行、推理身份与恢复兼容性；不重复启动仍活跃作业 |
| `eval ... --score-only --run-dir 实际目录` | 已有完整样本/预测/manifest 等快照；离线重算分数，不产生新预测 |
| 新模型推理 | 用户已请求该评测且资源/口径明确；现有预测不足时不能把重评分当成替代 |

具体配置与 [恢复/评分规则](../../../docs/zh/evaluation.md) 保持一致。对空目录不能使用 score-only；不要移动部分日志破坏已有运行布局。新的评分或比较报告写入任务所属输出目录；若要从外部运行重评分，先将其完整必要运行记录准备到任务内或任务所属远程工作目录，保留来源关系，避免修改用户原评测目录。

## Stage 4：核查覆盖并生成对比报告

复用现有比较入口，显式给出实际运行/报告路径和任务内输出，避免默认写到示例输出位置：

```bash
python examples/eval/training-free/scripts/compare.py "$RUN_A" "$RUN_B" \
  --output outputs/tasks/compare/evaluation/comparison-001
```

默认是系统比较。只有符合 [成对比较条件](../../../examples/eval/training-free/README.md) 的相同冻结底座/服务方法比较才加 `--paired`；它还校验样本、评分和声明的服务身份。工具通过也不能代替独立核实服务实际权重，默认系统比较更不会证明数据与模型差异已受控。

对已核实的受控 Chat/LLM2Jev 比较，显式运行：

```bash
python examples/eval/training-free/scripts/compare.py "$RUN_CHAT" "$RUN_LLM2JEV" \
  --paired --output outputs/tasks/compare/evaluation/comparison-001
```

当前配对实现要求一致样本/target 与评分身份、冻结底座与服务声明、SGLang 及明确 thinking 设置；固定 LLM2Jev 路线关闭 thinking。输出差值为第二路线减第一路线，partial 不产出完整配对增益。详细限制见 [评测选项](../shared/evaluation-options.md)。

保留比较校验失败的原因，不关闭 paired 来伪造相同口径增益。不兼容结果可分别报告并标记限制，不能把 test v1 的 0.82 与 validation v2 的 0.91 写成“提升 0.09”。如需补充同口径评测，先核对是否已在当前用户请求/资源授权内；只要求比较已有报告时把新推理列为建议。

## Stage 5：交付

读取实际 `reports/` 中所选报告、manifest、预测与失败覆盖，记录 configured/actual model、指标与分母、数据身份、延迟和有依据的费用；未知费用保持未知。部分报告、预测失败与模型效果不足分别说明。

README/task.json 链接实际配置、报告、比较、外部模型和未完成项。启动请求与完成请求分开核查；完成运行并不保证满足效果目标。只要求比较已有报告且因口径不一致无法排名时，可以交付明确的限制说明，不假称已获得公平排名；用户要求同口径实测而尚未完成时仍保留未交付状态。深入原因分析使用 [分析 skill](../dforge-analyze/SKILL.md)。端到端路线包含性能报告与 insight 时进入报告分析方案确认；仅评测请求不自动扩展为新实验或深入诊断。
