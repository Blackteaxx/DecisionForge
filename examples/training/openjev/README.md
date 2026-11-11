# OpenJev RLCD（实验性，默认两阶段）

公共训练入口默认 **calibrate → reinforce 两阶段**；直接 REINFORCE 仅为单独对照实验，不替换默认。本轮两阶段在 Bank/CLINC 的 top-1 全落在前 26 个候选位置，原因尚未确认；这不等同于实现只能表示 26 类。Score 排除、完整目录与比较限制见[已知问题](../../../docs/zh/training.md#openjev-保留两阶段默认)和[实测报告](../../../docs/zh/training.md)。

OpenJev 支持 Noul 和完整多类别 Choice，包括 BANKING77 的 77 项与 CLINC150+OOS 的 151 项。先用 [prepare_cn.py](prepare_cn.py) 显式排除 Score 并保存逐题审计；训练后端不会自动过滤。安装仓库根目录的 `uv sync --extra train-openjev`，使用单卡 CUDA 环境。阶段预算、限制和产物契约见[训练指南](../../../docs/zh/training.md#openjev-calibrate--reinforce)。

```bash
python examples/training/openjev/prepare_cn.py
uv run dforge train run --config examples/training/openjev/single.yaml --validate-only
uv run dforge train run --config examples/training/openjev/single.yaml
uv run dforge eval run --config examples/training/openjev/eval.yaml
```

默认准备数据位于 `data/local/openjev/typed-decisions-cn.v3.jsonl`，包含原始软监督中保留的 C/N 问题。[single.yaml](single.yaml) 分别设置 calibrate 的 `training.max_steps` 和 reinforce 的 `openjev.reinforce_steps`。

[init.yaml](init.yaml) 从导出权重开始新运行；完整状态 `--resume` 不受支持。`eval.yaml` 默认读取 single 的模型与留出 test；覆盖训练输出时同步调整评测路径。评测支持 `--resume` 和 `--score-only`。

[acceptance.yaml](acceptance.yaml) 和 [eval-acceptance.yaml](eval-acceptance.yaml) 使用已有审计的[共享 C/N 验收数据](../data/acceptance/README.md)，无需先准备默认数据。小样本验收与真实质量评测分别记录。

## 完整 Typed Decisions 的 C/N 配方

先执行[完整数据指南](../data/FULL.md)中的角色冻结、显式训练 C/N 准备和 benchmark C/N 投影命令。[typed-decisions-full.yaml](typed-decisions-full.yaml) 使用固定 Qwen3-0.6B，训练 400 个 calibrate 更新和 400 个 reinforce 更新；两个阶段各自的预算都在配置中。冻结的 120 个 calibration group 保留来源但不参与训练或温度拟合。

```bash
uv run dforge train run --config examples/training/openjev/typed-decisions-full.yaml --validate-only
uv run dforge train run --config examples/training/openjev/typed-decisions-full.yaml
uv run dforge eval run --config examples/training/openjev/eval-full/typed_decisions.yaml
uv run dforge eval run --config examples/training/openjev/eval-full/jevbench.yaml
```

[eval-full/](eval-full/) 使用完整来源中显式保留的 C/N 子集：Typed Decisions 1,200 题、JevBench 213 题。它们不等于完整 benchmark；每个投影都有逐题排除审计。BANKING77 与 CLINC150+OOS 可直接使用全部 77 / 151 个原始选项；不裁目录、不合并类别，也不把原始 option key 改写成内部标签。完整命令见下面的多类别说明。

## 多类别标签、旧模型与新训练

无需新的类别数配置。训练先扫描全部角色的公开问题，按最大选项数分配至少 10 个内部标签；保留 A–J，然后依次尝试 K–Z、AA–ZZZ。每个候选必须经当前 tokenizer 验证 `encode(" " + label, add_special_tokens=False)` 恰好一个非特殊 token，且 token ID 不重复；多 token 或碰撞候选会跳过。候选池不足会明确报错，因此 151 是目标目录大小示例，不代表所有 tokenizer 均有无限容量。

新导出使用 `decisionforge.openjev.artifact.v2`，保存标签、token ID、实际编码字符串、协议 hash、训练最大选项数和 tokenizer 文件身份。两个训练阶段及重载均复核同一协议。原始完整 state、多题请求、兄弟问题、每题原始 key 顺序与完整软分布 `q` 都保留；Noul 继续使用 false/true，Score 仍须显式排除并审计。完整提示、rationale 与 `Answer:` 标记预算超限会拒绝，不截断目录。

历史 v1 模型可继续加载。旧 A–J 请求的提示、token 读出与请求 seed 保持一致。对 77 / 151 项请求，运行时使用完整词表按请求扩展读出标签，不调整模型结构；输出 `inference_only_label_extension`、`trained_pool_size` 和实际 `label_protocol`。这表示旧权重可在完整目录上推理，不能据此声称旧训练见过新增标签或对应类别。扩展协议进入 seed；训练 pool、tokenizer、协议和读出策略进入评测恢复身份，变更后拒绝混用旧 journal。

从旧权重发起新训练仍要求相同 backbone、tokenizer、生成配置和训练标签 pool；跨 pool 的 `init_checkpoint` 会拒绝。需要大 K 新训练时使用普通 Qwen3 backbone 与原始 v3 数据；完整状态训练 `--resume` 仍不支持。模型、BF16、量化及单卡限制保持原配置契约。
