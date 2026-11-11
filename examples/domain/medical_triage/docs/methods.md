# 数据、训练、评测与消耗细节

本页描述已经执行的实验。主结果采用 v1，第二轮仅为负结果补充；历史实现和数据身份见 [模型与数据身份](../results/identities.json)。v1 使用专用教师构造完整患者案例，再调用原生 build 的检查与导出；没有使用原生 TXT 材料评估和问答构造。后续 [原生使用方式](usage.md) 尚未运行出新数据或新成绩。

## 训练数据从哪里来

唯一原始教学资料为 MedlinePlus / National Library of Medicine 的 2026-10-27 health-topic XML。只用英文 `full-summary`，不抓取链接所指的第三方正文。每条保留主题ID、标题、URL、摘要SHA256与署名 `Source: MedlinePlus, National Library of Medicine.`。NLM自有摘要按其公共领域说明使用，链接中的第三方资料不因此变成可再分发内容。

[prepare_sources.py](../scripts/prepare_sources.py) 解析HTML实体和段落、清理多余空格，按急症、就医、自护等词检索候选主题；这只是候选筛选，是否支持具体行动仍由构造和checker检查。[sources.lock.json](../configs/sources.lock.json) 记录原始XML哈希、496个起始来源及每轮来源ID。构造患者之前，以 `SHA256(seed:family_id)`、seed3407确定约80/10/10的train/validation/calibration角色，之后不改角色。

[generate.py](../scripts/v1/generate.py) 使用boyue `deepseek-flash`：先审核来源规则，再构造显式虚构患者及有原文连续片段支持的监督，覆盖四类分诊行动和两位患者的优先级比较。来源规则、证据和标签放在私有审计字段，公开state只有患者可见信息。三套公开测试帖子、答案和错题不作为教师输入。教师one-hot标签是监督编码，不是临床真实概率。

每轮固定最多8个case/source；来源数496/309/200/80，原轮号1/2/3/4。请求、响应、usage、错误和未知结果保存在SQLite journal。构造层检查引用、患者字段、固定候选目录与角色；原生 `build` 再做结构和语义checker。教师和checker均是deepseek-flash，不能称为独立临床裁定。

## 清洗、隔离与最终数量

[assemble.py](../scripts/v1/assemble.py) 只汇总终态完整的checker输出，核对原答案内容、固定候选语义、原公开state与角色；不会为凑数量重标或静默改变题目。隔离冲突标签和跨角色来源/患者复用，合并可追溯lineage。

[audit.py](../scripts/v1/audit.py) 核查来源角色、患者消息、EHR与完整病史。规范化精确匹配至少4词；至少12词的文本使用5词shingle、Jaccard≥0.8做近重复。共享system prompt不算患者复用，多轮对话不把脱离上下文的孤立短答当患者身份。对公开测试的重叠审计不读取target。字面审计不能证明语义或底座预训练完全独立。

四轮checker分别接受2299/1996/1253/540，共6088条；review37条保留。全局隔离23条后6065条准入，309来源家族、269患者/来源连通组：

| 角色 | 记录数 |
| --- | ---: |
| train | 5024 |
| validation | 446 |
| calibration | 595 |

5024条训练含2650条分诊、2374条消息优先级。[冻结数据](../data/README.md)已随案例提供，包含原训练输入、验证/校准视图、逐条来源证据和309个来源摘要；training.json按原字节保留，SHA256见 [身份清单](../results/identities.json)。`validate`只验证格式；事实质量仍受来源、教师和checker局限约束。

## 训练、选模与导出

底座Qwen3.5-9B固定revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`，单H200，BF16非4bit，Unsloth LoRA rank16/alpha16/dropout0；batch2、累积8，LoRA学习率2e-4、head学习率1e-4，cosine、warmup10、seed3407、长度8192，两轮628次更新。LoRA与决策头更新探针均为true。

七个已提交checkpoint在同一446条独立validation上实际推理，按两任务等权accuracy、Brier、较早step依次排序，选中step200。595条calibration仅拟合原生温度，模型优化器更新为0。导出为LoRA+决策头而非merged底座，全部文件约436.7MB，需同时具备原底座。

新进程4条非测试探针Choice一致、概率最大差0.0007≤预定0.002；环境迁移后再次加载相对原导出探针差0。权重和数据身份已摘要保存，最终模型未随Git提交。

## 测试资料和 benchmark 扩展

[prepare_benchmarks.py](../scripts/prepare_benchmarks.py) 从冻结原文件完整转换，原始URL/revision/许可/文件和转换哈希见 [sources.lock.json](../configs/sources.lock.json)：

| 测试集 | 输入与金标 | 完整数量 | 患者连通组 |
| --- | --- | ---: | ---: |
| CARE Public Test1 | 官方对话前缀、四类gold_label；只发送input_messages | 135前缀 | 62 |
| PMP Synth | INBOX_A435+INBOX_B434；消息和EHR一起换序 | 869对 | 2 |
| PMR Reddit Test Pairs | 两条消息；chosen/rejected、level、comments、difficulty仅作审计 | 1502对 | 1 |

三套公开测试许可CC BY-NC 4.0。位置置换seed3407，金标与整名患者一起反转，保留全部正式行和ID；共享消息/EHR用连通分量分组。Reddit level相同的行保留官方偏好，不另造平局或删样本。

[benchmark_ext](../benchmark_ext/README.md) 是独立实体Python包，标准entry points注册`medical_decisions` v1及四项指标。所有方法使用相同转换样本、target和扩展身份。

Accuracy按原生Choice评分，分母固定全部计划有标签样本，失败/未尝试计错；macro-F1固定完整类别。Brier是未缩放多类概率平方误差和；ECE默认10bin，以原生Choice概率为置信度。概率须有限、完整、总和容差1e-6，不归一化；argmax不替换Choice，概率失败单列覆盖。CARE按62组、seed3407做2000次组bootstrap；Synth/Reddit的2/1组不支持独立患者区间。

## 做了哪些对比

| 方法 | 模型与执行方式 | 主要设置 |
| --- | --- | --- |
| trained | 自训Qwen3.5-9B，原生Unsloth决策接口 | step200、595校准、BF16 |
| Jev | OpenRouter原生Decisions API | 实际typesafe/jev-1.14-20261014；单列系统条件 |
| LLM2Jev | 同revision未训练9B，官方Transformers后端 | 上游f2bdad6，batch4，关闭thinking |
| AnyJev | 同revision未训练9B，原版HF后端 | 上游35311c5，L0/content_free、全K轮换、logmean、shared_prefix=false |
| Chat | 同revision未训练9B，独立普通HF前端 | BF16，temperature0，关闭thinking，max_tokens2048，text |

共5方法×3测试=15格、12530条正式预测，全strict-valid。外部Jev并发4，本地和其他HTTP路线并发1；服务、并发和协议不同，吞吐不是严格硬件公平基准。完整质量、延迟、token与覆盖表见 [v1 结果](../results/v1.md) 和 [v1 结构化结果](../results/v1.json)。

LLM2Jev CARE/Synth曾提前于最终选模启动，原始结果保留，训练和选模不使用其分数；本轮不宣称严格盲测。AnyJev旧CARE运行21成功、1缓存错误、113未尝试，另建完整prompt新运行，旧21条选择和概率完全相同，失败开销继续单列。

## 第二轮：固定量的数据消融

新增自然叙事优先级资料仍来自NLM，12个开发来源不进正式新增训练，公开帖子/错题/标签不进教师。四轮新增接受643/648/630/644，共2565条。control复用原训练轨迹，hard25/hard50分别替换594/1187条旧优先级训练；均5024train，共用654validation和861calibration，保持答案字母×优先患者位置联合计数。

三条件共21个checkpoint共同验证，先冻结hard25/500为处理组；control重新选628，hard50没有被选入追加公开测试。新两模型校准重载后追加6格，复用原15格，合计17542条strict-valid。hard25的Reddit78.16%低于v1的79.23%和control的79.69%，Synth70.08%高于v1的67.66%。新合成验证仍接近98%天花板。

正式报告继续采用用户选定的v1；不将control重新选模/校准后的79.69%冒称v1。数据长度、叙事风格和比较事实同时变化，不能分解各因素因果。详见 [OPTIMIZATION.md](followup.md)。

## GPU、API与实验时间

| 项目 | 已记录消耗 | 计数边界 |
| --- | --- | --- |
| v1正式训练 | 1528.43秒，约25.47分钟 | 原生训练窗口，不含选模/校准/加载/排队 |
| hard25正式训练 | 2088.19秒，约34.80分钟 | 新训练628更新 |
| hard50正式训练 | 1917.04秒，约31.95分钟 | 新训练628更新 |
| 三次正式训练合计 | 5533.66秒，约1.54 GPU小时 | 不是全部GPU作业占用时间 |
| v1显存 | PyTorch分配峰值20.515GiB | 日志观测，不是整卡或全流程峰值 |
| v1四轮教师+checker | 输入16,047,591 / 输出10,370,532 token | 有usage的请求；另有缺失及未知结果 |
| v1教师/原生checker尝试 | 1109 / 6189次 | 仅四轮正式构造，不含全部开发 |
| v1三套外部Jev | 实报USD0.103175982 | 正式2506题，不含探针 |
| 第二轮开发Jev审查 | 实报USD0.000258552 | 8条开发样本，不是临床人工审查 |
| 第二轮教师/checker | 1483 / 2649次尝试 | 含开发与四轮正式构造 |
| 第二轮已报告输入/输出 | 10,983,494 / 11,466,040 token | 成功且有usage的响应，不含未知结果 |

GPU/H200及boyue费率未核验，完整货币费用为未知，不能把已知外部Jev小计当总费用。API尝试耗时可并行，不能相加冒称墙钟时间。分轮证据见 [v1用量](../results/data-usage-v1.json) 和 [第二轮用量](../results/data-usage-optimization.json)。

任务事件按香港时间：10月28日13:29初始化、18:37首次v1交付；19:11继续优化、10月29日00:48第二轮归档；10:19选择v1主报告。两个阶段跨度约5小时8分与5小时37分，包含编码、数据服务、测试、GPU等待和交互，不等于纯编码时间或GPU运行时间。两次H200作业均已停止。

## Coding agent 的耗时与消耗

统计范围为实现、运行、优化、报告选择及其三次直接审查；排除之前issue需求起草和本次文件整理。原始会话不提交，汇总见 [agent-usage.json](../results/agent-usage.json)；一次性会话统计工具不再随案例维护。按每个会话去重response_id后累加`token_usage_record.usage`，不反复累加累计计数。

| 角色 | 模型 | 输入token | 其中缓存输入 | 输出token |
| --- | --- | ---: | ---: | ---: |
| main | gpt-6.1-sol | 223,240,148 | 220,372,736 | 964,358 |
| design_review | gpt-6.1-sol | 189,534 | 155,264 | 1,383 |
| implementation_review | gpt-6-astra | 1,103,861 | 1,027,328 | 4,365 |
| optimization_review | gpt-6-astra | 645,602 | 589,056 | 4,153 |
| 合计 | 上述四个会话 | 225,179,145 | 222,144,384 | 974,259 |

非缓存输入3,034,761；输入+输出226,153,404。缓存输入已包含在输入中，reasoning输出564,576已包含在输出中，均不能再加一次。这些是多轮重复上下文的累计处理量，不是一次上下文长度，也不是代码行数。

主会话事件覆盖10月28日13:26:55至10月29日11:03:37，约21小时36分，包含用户间隔和等待；保留下来的6个已完成主线程turn记录累计约10小时17分30秒，不保证覆盖全部早期执行。三次审查记录各约31/229/232秒，可与主流程重叠，不能相加作为项目墙钟。

主会话最后累计token计数为222,676,896，与去重逐响应之和224,204,506不一致；两种原始口径均保留在JSON，未将较小累计值冒充唯一精确账单。三次审查的两种口径一致。缺少账户账单、核验单价和折扣，coding agent货币费用未知；不按公开单价推算实际支付金额。

## 代码验证、故障与存储清理

v1历史核心测试945passed/48skipped/3failed，第二轮960passed/48skipped/3failed；三项为私人电网输入两项和training-free both预检30秒超时，相关文件相对base未改。审查修复了成功响应持久化后缓存提交中断、同路径换权重无法识别和第二轮保护过滤后补数调度三个问题。详见 [执行说明](execution.md)。这些是历史验证；目录整理后的检查范围见 PR 的验证记录。

按用户要求清理已完成实验的中间权重和优化器状态，保留v1最终导出、冻结资料、日志、用量和原始预测；清理收据及恢复影响见 [storage-cleanup.json](../results/storage-cleanup.json)。清理后不再声称全部历史checkpoint可恢复。外部大文件和本机环境未纳入Git；本目录阅读、链接和脚本发现不依赖它们。
