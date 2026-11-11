# 原生训练：方法、配置与选择依据

用于 `dforge-train` 的路线选择和配置说明。以当前 [训练配置模型](../../../src/decisionforge/training/config.py)、[后端实现](../../../src/decisionforge/training/backends/unsloth.py) 和 [训练指南](../../../docs/zh/training.md) 为准；下面不是新增 CLI 参数或通用训练框架功能。

## 先选训练路线

**Unsloth full、Laya、OpenJev 和免训练 AnyJev 均为实验性方案。** 默认训练仍为 Unsloth LoRA；Laya 默认使用官方 `convaiinnovations/laya`（ModernBERT-large 与已训练头），OpenJev 默认 calibrate → reinforce 两阶段。不能因接入完成就推荐为质量更优路线；先说明[已知问题和实测范围](../../../docs/zh/training.md)。

当前训练后端为 `unsloth`、`laya`、`openjev`。三者都直接读取同一 `decisionforge.training.v3` JSON/JSONL；先按 group 划分角色，再在后端内部映射完整软监督。Unsloth 与 Laya 支持 Choice/Noul/Score 和一个请求多个问题；OpenJev 支持 Noul 与完整多类别 Choice，包括 77／151 类；按全部公开问题的最大类别数分配互异单 token 标签，新训练保存 v2 标签池并在两个阶段及重载时重验。历史 v1 产物仍可读取，但跨标签池的初始化会拒绝。Score 必须在显式、有审计的数据准备步骤中排除，训练不自动过滤。实际支持取决于当前 tokenizer 的标签池、完整 prompt 上下文与显存；A–ZZZ 的 18,278 个候选不代表所有 tokenizer 都能支持该类别数。不能直接把任意 chat-SFT/HF 数据集当作该契约。

| 向用户说明的选择 | 实际配置 | 适用性与边界 |
| --- | --- | --- |
| 4-bit 底座 + LoRA + 决策头（QLoRA 式配置） | `model.load_in_4bit: true` | 示例常用配置，节省底座显存；仍需检查模型和设备兼容性 |
| 非 4-bit 底座 + LoRA + 决策头 | `model.load_in_4bit: false`，`training.precision` 选择计算精度 | 需要更多显存；**不是**全参数微调 |
| Full 全参数 | `backend: unsloth`、`finetuning: full`、`model.load_in_4bit: false`、`lora: null` | 底座与决策头均训练；支持单机 DDP 和完整状态恢复 |
| Laya RLCD | `backend: laya`，默认 full | 官方模型、损失与读出；支持单机 DDP、独立 calibration；不支持完整状态恢复 |
| OpenJev RLCD | `backend: openjev`，默认 full | 官方 calibrate → reinforce 两阶段；单 GPU；不支持完整恢复或独立温度校准 |
| DPO、PPO、GRPO、普通 chat-SFT | 无对应原生配置 | 仍需另一训练栈或单独实现方案 |

没有 `training.method`、`finetuning_type`、`method: qlora` 等切换字段。默认 `finetuning: lora` 保留 `target_modules="all-linear"`；`finetuning: full` 使用官方 `full_finetuning=True`，不启用 LoRA/量化。各后端只接受其支持的类型化字段，不接受任意上游 kwargs。示例名 `qwen35-4b-full.yaml` 指完整数据训练，`output.save_merged` 指合并导出；二者都不表示 Full 微调。

## 选择起始示例

| 目的 | 起始文件 | 使用前必须核对 |
| --- | --- | --- |
| 小规模通路检查 | [20-step smoke](../../../examples/training/unsloth/qwen35-4b-smoke.yaml) | 随附 120 records / 600 questions，仅演示；换成用户数据后重新验证 |
| 已生成 JEV 小数据 | [generated](../../../configs/training/generated.yaml) | 改为真实 accepted.json；示例关闭校准，不能不看数据直接套用 |
| 完整数据训练 | [full](../../../examples/training/unsloth/qwen35-4b-full.yaml) | 完整输入需准备；示例训练量不保证适合用户资源 |
| Full 单机双卡 | [Unsloth DDP](../../../examples/training/unsloth/full-ddp.yaml) | full 需要非量化底座；记录 world_size 与各 rank 恢复状态 |
| Laya 单机双卡 | [Laya DDP](../../../examples/training/laya/ddp.yaml) | 固定模型/依赖；calibration 来自独立角色；使用 epoch 预算 |
| OpenJev 两阶段 | [OpenJev](../../../examples/training/openjev/single.yaml) | 先显式准备 C/N；分别记录 max_steps 与 reinforce_steps、总更新数和 stage lineage |
| 完整来源 Unsloth full | [Typed Decisions full](../../../examples/training/unsloth/typed-decisions-full.yaml) | 先按[完整数据指南](../../../examples/training/data/FULL.md)冻结角色；固定 Qwen3-0.6B、单卡 BF16、1 epoch；命令显式禁用上游编译 |
| 完整来源 Laya | [Typed Decisions Laya](../../../examples/training/laya/typed-decisions-full.yaml) | 使用相同冻结 group 角色、官方固定 Laya；1 epoch、FP16 |
| 完整来源 OpenJev | [Typed Decisions OpenJev](../../../examples/training/openjev/typed-decisions-full.yaml) | 使用同一冻结角色的显式 C/N 投影；400 calibrate + 400 reinforce；不使用独立 calibration |
| 自定义原生训练 | [通用模板](../../../configs/training/unsloth.yaml) | 从支持的字段构造配置并检查实际数据和模型 |

选择 smoke 不等于用户已授权用小数据替代完整训练。若用户授权“先小跑再全量”，使用两个 run 目录，并记录两次各自的配置与终点。

## 重要选项

未明确的关键项按 [交互规则](interaction.md) 提问；普通超参数可按已授权示例默认值提出整套方案，不逐字段盘问。

| 配置项 | 可选值／限制 | 选择时应说明什么 |
| --- | --- | --- |
| `model.name` / `revision` | HF ID 或本地模型路径；revision 可为空 | 基础模型还是已有 Clef；已知版本、缓存、访问方式；本地路径随 YAML 搬移重写 |
| `model.load_in_4bit` | `true` / `false` | 量化加载与显存差异，和 LoRA 方法、计算精度分开说明 |
| `model.max_seq_length` | 整数 ≥32；配置类默认 2048 | 请求实际编码长度；不能遇 OOM 就静默截短 |
| `lora.rank` / `alpha` / `dropout` | rank/alpha ≥1；dropout ∈[0,1) | 适配器容量和正则；配置类默认 16/16/0，smoke 显式为 8/8/0 |
| `training.num_train_epochs` / `max_steps` | **恰好一个非 null**；epochs >0 或 steps ≥1 | 按 epoch 还是固定步数结束；切换时将另一个设 `null` |
| `training.batch_size` / `gradient_accumulation_steps` | 均 ≥1 | 单卡每步样本与累积；二者乘积可描述名义有效 batch，不能保证改变后训练等价 |
| `training.learning_rate` / `head_learning_rate` | 均 >0 | Unsloth/Laya 分别用于底座或 LoRA 与决策头；OpenJev 只有一个优化器学习率，head_learning_rate 必须相同 |
| `training.precision` | 按后端：Unsloth auto/bf16/fp16；Laya auto/fp16；OpenJev bf16 | 设备支持与已有实验要求；不是量化位数 |
| `training.lr_scheduler_type` | `linear` / `cosine` / `constant` / `constant_with_warmup` | 与 `warmup_steps`（≥0）、训练总量一起确认 |
| `training.seed` / `weight_decay` | seed 为非负整数；weight_decay ≥0 | 复现信息与正则；沿用确认配置 |
| `training.gradient_checkpointing` | `true` / `false` | 显存与计算开销折中，不等于磁盘恢复 checkpoint |
| `training.logging_steps` / `save_steps` | 后端相关 | Unsloth 均 ≥1，保存完整 Trainer checkpoint；Laya/OpenJev 的 save_steps 为 null，不支持完整状态恢复 |
| `data.split_map` / `auto_holdout` | 原标准角色不能互换；留出比例非负且和 <1 | 先查已有 train/validation/calibration/test；不要覆盖已有留出角色 |
| `calibration.enabled` | `true` / `false` | 独立校准用途与数据是否足够；拟合至少需要 10 个可用编码问题，按题型报告 fitted/fallback |
| `data.on_skipped` / `on_truncated` | `error` / `allow`，默认 `error` | `allow` 会接受编码丢弃/截断，必须说明影响，不能为跑通静默开启 |
| `output.directory` / `save_merged` | 新运行的空 `run/`；merged 为布尔值 | LoRA 保存 adapter/决策头，可选 merged；full 保存完整 Clef；Laya/OpenJev 为独立版本化产物，不接受 save_merged |

训练需要 NVIDIA CUDA；Windows 原生支持 Unsloth 单卡 LoRA，其余方法使用 Linux。Unsloth full 与 Laya 支持单机 `distributed.strategy: ddp`、`world_size` ≥2；OpenJev 仅单卡，拒绝 DDP。默认 LoRA 使用原有单卡路径。Laya/OpenJev 的 `init_checkpoint` 是带来源记录的新运行权重初始化，不是完整状态 resume。主机、环境和 GPU 由用户指定；训练 YAML 没有 SSH/队列/租卡配置。`CUDA_VISIBLE_DEVICES` 属于启动环境，记录到命令与资源信息中。不要把 `train --validate-only` 当成显存、模型加载或依赖检查。

## 配置展示与证据

执行前展示方法、模型/revision、数据及实际划分、GPU、量化/精度、长度、LoRA 参数、训练量、后端适用的学习率组、batch/累积、校准、产物格式和输出路径。只把偏离选定示例的项单独列为变更理由；保存 `configuration.md` 和实际 YAML。

Unsloth 完整状态恢复比较的是有效配置，当前只允许输出位置与 `training.logging_steps` 覆盖；不要把调整 batch、量化、学习率、总步数后的运行叫作原实验恢复。需要改变实验配置时按授权建立新运行并保留旧记录。
