# 评测：路线、training-free 方法与重要配置

用于 `dforge-eval` 的选择与解释。以 [配置模型](../../../src/decisionforge/eval/models.py)、`uv run dforge eval list`、[评测参考](../../../docs/zh/evaluation.md) 和 [training-free 示例](../../../examples/eval/training-free/README.md) 为准。

## 先区分模型入口与评测方法

**OpenJev、Laya、AnyJev 和 Unsloth full 均为实验性方案。** Laya 默认官方 ModernBERT，OpenJev 默认两阶段；说明大目录坍缩、完整轮换成本和 C/N 比较范围，详见[实验性方法及已知问题](../../../docs/zh/training.md)。训练完成和 strict-valid 不代表质量优于基线。

| 用户要做的事 | 实际 adapter / 方法 | 起始配置 | 所需资源与含义 |
| --- | --- | --- | --- |
| 评测已有原生 JEV HTTP 服务 | `model.adapter: jev`；原生方法为 `native` | [demo](../../../configs/eval/demo.yaml) 或目标 benchmark 的配置 | 已指定服务与凭据；服务可能已经训练过，不能称作 training-free 方法 |
| 评测已训练的本地决策模型 | `model.adapter: unsloth_local` | [generated-local](../../../configs/eval/generated-local.yaml)、[local-trained-decision](../../../configs/eval/local-trained-decision.yaml) | 保存的 Clef LoRA/full `model/` 或适用的 `merged/`、Linux 单卡 CUDA；不是任意原始 HF 权重路径 |
| 评测 Laya 产物 | `model.adapter: laya_local` | [Laya eval](../../../examples/training/laya/eval.yaml) | 版本化 Laya 产物，官方四位小数读出；保留温度与上游警告 |
| 评测 OpenJev 产物 | `model.adapter: openjev_local` | [OpenJev eval](../../../examples/training/openjev/eval.yaml)、[完整来源 C/N](../../../examples/training/openjev/eval-full/) | 版本化 stage-two 权重；单 rationale + 官方标签 logits；支持 C/N 及完整 77／151 类 Choice；兼容 v1，新训练使用 v2 标签池 |
| **Training-free：AnyJev L0** | `model.adapter: anyjev_local` | [AnyJev](../../../examples/eval/training-free/anyjev/eval.yaml) | 普通固定 HF 底座、content_free、完整 rotations；支持完整 77／151 类 Choice、Noul、Score；互异单 token 标签在运行时验证 |
| **Training-free：Chat baseline** | `model.adapter: chat_api`；`model.provenance.method: chat_baseline` | [Chat Typed Decisions](../../../configs/eval/chat/typed-decisions.yaml) | Chat Completions 服务；固定 zero-shot prompt 要求 JEV JSON，概率由 LLM 自报 |
| **Training-free：LLM2Jev** | `model.adapter: jev`；`model.provenance.method: llm2jev` | [LLM2Jev Typed Decisions](../../../configs/eval/llm2jev/typed-decisions.yaml) | 独立 LLM2Jev 服务，以候选 yes/no logits 构造概率；需要其固定版本及实际底座/服务身份 |
| 只比较或重评分已有预测 | `compare.py` / `eval --score-only`，沿用快照 | 原运行配置与完整快照 | 可离线，不新增模型调用；不是新的推理方法 |

**没有 `adapter: llm2jev`，也没有总开关 `training_free: true`。** LLM2Jev 配置中的 provenance 是方法声明，不会把普通 JEV 服务变成 LLM2Jev。先检查实际服务；training-free 表示不为这条方法训练模型，不表示无 GPU、无服务成本或可以省略数据预检。

AnyJev 固定 `level=L0`、`prior=content_free`、`adaptive_shifts=false`；≤26 类 Choice 保留旧路径，大目录仍使用完整 K 次轮换，保留全部候选和原始概率。[BANKING77](../../../examples/eval/training-free/anyjev/banking77.yaml)／[CLINC150+OOS](../../../examples/eval/training-free/anyjev/clinc150.yaml) 配置传入完整 77／151 类目录。AnyJev 与 OpenJev 分别按自己的编码规则逐个验证互异单 token 标签；A–ZZZ 的 18,278 个候选是理论候选数，实际容量与完整上下文、显存均需运行时核实，不能靠裁剪目录绕过。OpenJev 仍不支持 Score，标签池和 v1／v2 重载约束见[方法说明](../../../examples/training/openjev/README.md)。最大候选概率 confidence 不是正确率校准；tokens/费用未知时保留 null。

用户说“比较 training-free 方法”时，展示 Chat、LLM2Jev、AnyJev 与适用比较选择；若已明确“两者比较”，不再询问方法。使用同一底座与同一组样本时可考虑受控方法比较；不同底座或不同服务条件时只能说明各系统的差异，不能把差值全归因于方法。

## Benchmark 与指标入口

| 选择 | 数据范围与主要指标 | 注意事项 |
| --- | --- | --- |
| `generic` | 用户自己的 eval.v1 或 training.v3；accuracy、概率/校准、MAE 等 | 根据 target 类型选择适用指标，先核对真实分母 |
| `jevbench` | public；accuracy、Brier、ECE、MAE、group agreement | 使用固定数据/benchmark 版本，不伪称独立私有 test |
| `typed_decisions` | 常用 test；gold_accuracy、soft_accuracy、soft_brier、soft_kl、ECE、MAE | gold_accuracy 是与源教师模态标签一致率，不是人工正确率 |
| `banking77` | train/test；77 类 native_accuracy、macro_f1 | 不能仅靠更高概率的 argmax 替换原生 Choice |
| `clinc150_oos` | train/validation/test；151 类 native_accuracy、macro_f1，报告 OOS/域内分项 | OOS 与域内分项一起解释，保留完整标签空间 |

Chat 与 LLM2Jev 两条 HTTP training-free 路线都已有这四个公开 benchmark 的配置目录：[Chat](../../../configs/eval/chat)、[LLM2Jev](../../../configs/eval/llm2jev)。注册项和指标参数以当前 `eval-list` 为准；`demo` 用于通路演示。不因为存在全量示例就默认用户要跑全部 benchmark。

## 重要配置选项

| 配置项 | 可选值／约束 | 需要明确的决定 |
| --- | --- | --- |
| `benchmark` / `benchmark_version` / `metrics` | 已注册名称、版本、参数 | 与问题类型及 target 匹配；对比时保持评分口径一致 |
| `data.path` / `splits` / `subsets` / `limit` | 文件、筛选列表或 null；limit ≥1 或 null | 用户数据或指定 benchmark；全量或 smoke；原始数据版本与实际选中样本 |
| `model.base_url` / `endpoint_path` / `model` / `api_key_env` | HTTP 服务根、支持的 endpoint、服务模型名、环境变量名 | 原生 JEV 支持 `/v1/systemone` 或 `/api/alpha/decisions`；LLM2Jev 要求前者；Chat 使用 `/v1/chat/completions` |
| Chat `model.temperature` / `max_tokens` | temperature ∈[0,2]；max_tokens 为正整数 | 固定生成设置与截断限制；配对时记录实际设置 |
| Chat `model.response_format` | `text` / `json_object` / `json_schema` | 服务的真实支持能力；解析仍严格，不自动补字段或修复概率 |
| Chat `model.serving.enable_thinking` | `true` / `false` / null | thinking 行为需与 provenance 一致；现有 LLM2Jev 配对示例关闭 thinking |
| 本地 `model.path` / `load_in_4bit` / `precision` | 按 adapter 选择产物与支持的精度；AnyJev 使用 `model.model/revision` | adapter 需要基础模型本地可用或已缓存；merged 便于独立复制 |
| 本地 `max_seq_length` / `load_timeout` / `local_files_only` | 长度 32–32768；加载超时 (0,3600] 秒；布尔值 | 与真实 checkpoint、输入长度、模型缓存和资源约束一致 |
| `execution.concurrency` / `timeout` / `max_attempts` | 并发 1–256、超时 >0 秒、尝试 1–20 | HTTP 服务容量和限流；本地 adapter **只允许 concurrency=1、max_attempts=1** |
| `execution.backoff` / `max_retry_delay` | 非负秒数 | 有限重试及服务 Retry-After；不把重放请求算成免费 |
| `model.provenance`（HTTP） | 方法、底座及 revision、上游、backend、依赖、精度、settings | 记录实际环境，不能复制示例里的历史版本后当作本次已验证身份 |
| `output` | 独立空 `evaluation/<run-id>/run/` | 每个方法、模型和样本范围各一个运行；不能让 smoke 与全量共用目录 |

LLM2Jev 的 provenance 要求匹配上游仓库、固定的 40 位上游与底座 revision，以及底座、backend、依赖和精度。不同配置类的字段不能混用：本地模型不用 `base_url`；Chat 的 temperature/response_format 不能塞进 `jev` 配置。评测 YAML 当前没有通用 USD 硬预算字段；用户有限额时先说明可执行的控制方式，不能发明 `budget_usd` 参数。

## Training-free 路线的任务内执行

1. 核对请求是方法咨询、比较已有报告还是新实测。只有已请求的新实测才调用服务；不要额外训练。
2. 确定 benchmark、样本范围、冻结底座、服务与比较目的，展示关键配置。已有 endpoint 直接核查；本机部署服务需已指定资源，按 [示例服务说明](../../../examples/eval/training-free/README.md) 使用独立环境和 `uv sync --extra eval-llm2jev`。
3. 将选中的 Chat/LLM2Jev YAML 复制到任务各自阶段目录，重写数据和输出路径，填真实 model/endpoint/provenance。先各自 `eval --validate-only`，再按请求在线执行。
4. 建议新环境先用小样本 smoke 检查服务和格式；若用户未授权额外试跑，先提出建议。已授权 smoke + full 时使用独立 run，不能 resume 一个改了 limit 的旧运行。
5. 保留完整快照及失败覆盖，用显式运行路径比较。共享示例 `run.sh both` 固定读取示例配置并写示例输出，**任务归档流程不要直接调用它代替这两份任务配置**；也不要用一个 `--run-dir` 覆盖所有 benchmark。

## 比较条件

系统比较默认使用 `compare.py` 的显式运行路径。受控 Chat/LLM2Jev 方法比较再加 `--paired`：当前实现检查相同样本/target 快照、评分身份、声明的冻结底座及服务身份，要求 `serving_backend: sglang` 和明确的 thinking 设置；固定 LLM2Jev 路线关闭 thinking。独立核对服务实际加载的权重，provenance 通过不等于服务权重已获独立验证。

成对校验失败保留原因；不同数据/split/版本不能通过关掉 `--paired` 宣称同口径提升。partial 报告不产生完整 paired delta，费用未知保持未知。比较脚本按第二个输入减第一个输入给差值，报告中应注明方向。
