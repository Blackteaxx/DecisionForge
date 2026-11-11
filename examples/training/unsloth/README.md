# Qwen3.5-4B 决策训练示例

默认训练方式为 **LoRA**。**Unsloth full 全参数训练属于实验性方案**，本轮四个主指标均低于 LoRA；`qwen35-4b-full.yaml` 的 full 表示完整数据，训练方式仍是 LoRA。全参数配置与[实验性限制](../../../docs/zh/training.md)应分别核对，完整对照见[本轮报告](../../../docs/zh/training.md)。

这个目录展示：输入数据 → 训练与校准 → 全部 benchmark 评测 → 汇总报告。完整数据格式与训练参数见 [训练指南](../../../docs/zh/training.md)。

| 文件 | 内容 |
| --- | --- |
| `../data/typed-decisions-sample.jsonl` | 随仓库提供的 120 案例 / 600 问题小样本 |
| `../data/SOURCE.md` / `../data/LICENSE` | 固定数据版本、转换说明和许可证 |
| `qwen35-4b-smoke.yaml` | 20 步短训练配置，使用随附数据 |
| `qwen35-4b-full.yaml` | 完整数据训练 2 epochs 的配置 |
| `train.sh` | 简单训练入口，默认使用 smoke 配置 |
| `../prepare_data.sh` / `../prepare_typed_decisions.py` | 下载固定数据、转换成训练格式 |
| `run_full.sh` | 按顺序展示完整流程的 shell 脚本 |
| `eval/*.yaml` | 四个完整 benchmark 与 demo/generic 的评测配置 |
| `summarize.py` | 读取已有报告，汇总指标；无需模型依赖 |
| `predict.py` | 加载保存的模型，预测公开请求 |
| [reports/](../reports/2026-10-12-typed-decisions-qwen35-4b.md) | 已完成实验的真实报告与指标 JSON |

模型保存与恢复的独立验收见 [checkpoint 记录](../reports/checkpoint-recovery.json) 和 [本地评测记录](../reports/local-evaluation.md)。

## 安装与配置

使用 Python 3.12、Linux 单卡 CUDA 环境，先激活虚拟环境并准备适合驱动的 CUDA PyTorch，再安装：

```bash
uv sync --extra train-unsloth
python -m pip install -e '.[eval-data]'
```

训练配置默认使用模型 ID `Qwen/Qwen3.5-4B`。已有本地模型时，修改 YAML 的 `model.name`，例如 `../../../models/Qwen3.5-4B`。YAML 中的本地路径相对于该配置文件；本示例没有机器专属路径。

下面的命令从仓库根目录执行。脚本使用当前环境的 `python3`，也可以设置 `PYTHON_BIN` 指定解释器。

## 先跑小样本

基础安装即可预检，不需要 GPU 或下载模型：

```bash
python -m pip install -e .
bash examples/training/unsloth/train.sh --validate-only
```

实际训练与恢复：

```bash
bash examples/training/unsloth/train.sh
bash examples/training/unsloth/train.sh --resume
```

默认产物在 `outputs/train/qwen35-4b-smoke/`。需要调整步数、学习率或模型时直接编辑 smoke YAML；也可通过 `--config` 选择自己的配置。

## 完整流程

先查看 [run_full.sh](run_full.sh)，脚本只有四个步骤：

1. 准备完整训练数据及 benchmark 数据。
2. 使用 `qwen35-4b-full.yaml` 训练、校准并保存模型。
3. 使用 `eval/*.yaml` 逐项评测。
4. 用 `summarize.py` 生成汇总报告。

```bash
bash examples/training/unsloth/run_full.sh
```

完整原 train 的 1,200 案例按 group 划分为 train 960 / validation 120 / calibration 120，原 test 的 400 案例只做诊断与评测。默认 LoRA rank 16、有效 batch 8、训练 2 epochs。启动脚本设置 `UNSLOTH_COMPILE_DISABLE=1`，以避免编译 backward 的 custom-op stride 兼容性问题。

正式 benchmark 共 10,811 题：Typed Decisions 2,000、JevBench 231、BANKING77 3,080、CLINC150+OOS 5,500。demo/generic 各 9 条用于检查协议，单独列出。

输出位置与 YAML 对应：

```text
outputs/train/qwen35-4b-full/
  model/                  # adapter、决策头与校准
  merged/                 # 合并后的模型
  checkpoints/            # 可恢复的完整 checkpoint
  report.json             # 训练诊断

outputs/eval/qwen35-4b-full/
  <benchmark>/            # 逐题预测、journal、分项报告
  results/                # 每次 eval 返回的报告位置
  report.md               # 指标汇总
  results.json            # 机器可读汇总
```

merged 约 9.4 GB，adapter 约 205 MiB，完整 checkpoints 约 2.3 GiB；同时为模型缓存和导出临时文件预留空间。需要更换输出目录时，修改训练和评测 YAML 中对应路径及脚本最后的汇总位置。

## 分步运行与恢复

每步都使用项目现有 CLI，便于修改和排错：

```bash
# 准备与预检训练数据
bash examples/training/prepare_data.sh
uv run dforge train run --config examples/training/unsloth/qwen35-4b-full.yaml --validate-only

# 训练；训练中断时添加 --resume
uv run dforge train run --config examples/training/unsloth/qwen35-4b-full.yaml
uv run dforge train run --config examples/training/unsloth/qwen35-4b-full.yaml --resume

# 单独评测；评测中断时添加 --resume
uv run dforge eval run --config examples/training/unsloth/eval/typed_decisions.yaml
uv run dforge eval run --config examples/training/unsloth/eval/typed_decisions.yaml --resume
```

已完成训练无需重新运行训练命令。其他 benchmark 选择对应的 YAML；已保存的预测由 eval 的 journal 管理。

所有已保存结果可离线重评分并重新汇总，无需 GPU 或模型文件：

```bash
bash examples/training/unsloth/run_full.sh --score-only
```

需要保留训练 `report.json`、各项评测的 manifest、samples.jsonl 和 `.eval/events.jsonl`。若只完成部分 benchmark，可用对应的 eval CLI 单独重评分。

重新加载模型预测：

```bash
python examples/training/unsloth/predict.py --model outputs/train/qwen35-4b-smoke/model \
  --request examples/training/data/public-request.json --load-in-4bit
```

## 实测报告

[正式实验报告](../reports/2026-10-12-typed-decisions-qwen35-4b.md) 保留训练监控、校准、题型/workflow/难度/OOS 分项和运行问题。所有指标为真实测量；JSON 中的机器专属路径已转换为示例相对路径。

| Benchmark | 实测指标 |
| --- | ---: |
| Typed Decisions 教师标签一致率 | 78.95% |
| JevBench accuracy | 74.03% |
| BANKING77 macro-F1 | 65.33% |
| CLINC150+OOS macro-F1 | 77.01% |

训练数据参考是教师概率。Typed Decisions 使用逐题评测，与上游五题联合协议不同，详细口径见报告。

## 全参数训练

[full-single.yaml](full-single.yaml) 和 [full-ddp.yaml](full-ddp.yaml) 使用随附完整 C/N/S 样本，显式设置 `finetuning: full`、非量化底座和 `lora: null`。安装仓库根目录的 `uv sync --extra train-unsloth-full`；双卡配置由 coordinator 启动 torchrun。

```bash
uv run dforge train run --config examples/training/unsloth/full-single.yaml --validate-only
uv run dforge train run --config examples/training/unsloth/full-ddp.yaml
```

[full-acceptance.yaml](full-acceptance.yaml) 使用共享 C/N 验收数据，训练四步；[eval-full.yaml](eval-full.yaml) 读取该配置默认产物的留出 test。复制或覆盖训练输出时同步调整评测路径。静态跨方法检查见[共享验收说明](../data/acceptance/README.md)。

## 完整 Typed Decisions 的全参数配方

[typed-decisions-full.yaml](typed-decisions-full.yaml) 使用固定 Qwen3-0.6B、完整数据冻结的 960/120/120/400 group 角色、单卡 BF16 和一个 epoch。先执行[完整数据与 benchmark 准备](../data/FULL.md)，在独立环境安装仓库根目录的 `uv sync --extra train-unsloth-full`。

```bash
bash examples/training/unsloth/train.sh \
  --config examples/training/unsloth/typed-decisions-full.yaml --validate-only
UNSLOTH_COMPILE_DISABLE=1 bash examples/training/unsloth/train.sh \
  --config examples/training/unsloth/typed-decisions-full.yaml
for benchmark in typed_decisions jevbench banking77 clinc150_oos; do
  uv run dforge eval run --config "examples/training/unsloth/eval-full/$benchmark.yaml"
done
```

该训练命令显式设置 `UNSLOTH_COMPILE_DISABLE=1`，与已执行配方一致，避免上游编译 backward 的 stride 兼容性问题；它不改变数据、训练预算或精度。现有 `run_full.sh` 仍是 Qwen3.5-4B LoRA 完整流程，选择此 Qwen3-0.6B full 配方时使用上面的独立命令。`eval-full/` 指向新 full 模型；`eval/` 保留原 LoRA benchmark 配置。
