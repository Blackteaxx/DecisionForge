# Qwen3-4B：四个 benchmark 全量评测

[AnyJev L0](anyjev/README.md) 属于实验性免训练方案；OpenJev、Laya 和 Unsloth full 同为实验性方案。默认与已知问题见[方法状态](../../../docs/zh/training.md)，后续同底座对照见[统一 Qwen3.5-4B 报告](../../../docs/zh/evaluation.md)。

用同一份 **Qwen3-4B-Instruct-2507** 对比 Chat 和 LLM2Jev。运行全部公开 / test 数据，每条路线 **10,811 条**，两条路线共 **21,622 条**。

| Benchmark | Split | 每条路线样本数 | 配置文件名 |
|---|---|---:|---|
| JevBench | public（original + easy + hard） | 231 | `jevbench-public.yaml` |
| Typed Decisions | test | 2,000 | `typed-decisions.yaml` |
| BANKING77 | test，完整 77 类 | 3,080 | `banking77.yaml` |
| CLINC150+OOS | test，完整 151 类 | 5,500 | `clinc150-oos.yaml` |

```text
training-free/
├── README.md
├── configs/
│   ├── chat/              # 四个 benchmark 的 Chat 配置
│   └── llm2jev/           # 四个 benchmark 的 LLM2Jev 配置
└── scripts/
    ├── prepare.sh         # 获取固定版本数据并转换
    ├── serve.sh           # 启动两条路线共用的模型服务
    ├── run.sh             # 运行四个 benchmark，自动汇总
    └── compare.py         # 离线生成报告
```

以下命令均在仓库根目录执行，数据、权重和结果使用仓库相对目录。

## 1. 准备数据

```bash
python -m pip install -e '.[eval-data]'
bash examples/eval/training-free/scripts/prepare.sh
bash examples/eval/training-free/scripts/run.sh both --validate-only
```

数据保存在 `data/benchmarks/<dataset>/raw/` 和 `processed/`。已有 raw 缓存时，可以用 `prepare.sh --convert-only` 离线转换。预检读取全部评测数据，无需 GPU、模型服务或 API key。

## 2. 启动模型服务

在 Linux / NVIDIA GPU 环境、独立的 Python 3.12 环境运行：

将 `Qwen/Qwen3-4B-Instruct-2507` 的权重准备到 `data/local/models/Qwen3-4B-Instruct-2507/`，或在下面的命令中替换为自己的相对目录。模型 revision 见配置文件。

```bash
python3.12 -m venv .venv-llm2jev
source .venv-llm2jev/bin/activate
uv sync --extra eval-llm2jev
bash examples/eval/training-free/scripts/serve.sh data/local/models/Qwen3-4B-Instruct-2507
```

服务监听 `127.0.0.1:30000`，同时提供 Chat 和 LLM2Jev 接口。LLM2Jev 通过本仓库的可选依赖安装，固定上游 Git revision。

## 3. 运行四个 benchmark

在安装了 DecisionForge 的另一个终端运行：

```bash
bash examples/eval/training-free/scripts/run.sh both
```

默认执行四组 Chat / LLM2Jev 全量评测，完成后自动生成报告。单独跑一条路线或恢复运行：

```bash
bash examples/eval/training-free/scripts/run.sh chat
bash examples/eval/training-free/scripts/run.sh llm2jev
bash examples/eval/training-free/scripts/run.sh both --resume
```

八个运行分别保存到 `outputs/eval/training-free/<benchmark>-<route>/`。配置不设 `data.limit`；两条路线均使用 BF16、关闭 thinking、并发 8。修改服务地址或模型时，同步更新 `configs/chat/` 和 `configs/llm2jev/`；`provenance.dependencies` 记录的是历史实验环境，换环境时更新实际版本。相对路径以 YAML 所在目录为准。

## 4. 查看或重新生成报告

```bash
python examples/eval/training-free/scripts/compare.py
# 仅对已保存预测重评分，再生成报告
bash examples/eval/training-free/scripts/run.sh both --score-only
```

报告位于 `outputs/eval/training-free/comparison/`：`report.md` 是简表，`comparison.md` / `comparison.json` 包含完整指标、分母、失败统计和模型身份。比较要求每组的数据、评分规则及声明的底座和服务参数一致。

JSON 中的 `run_directory` 相对 `comparison.json` 所在目录解析，可以将整个 `outputs/eval/training-free/` 目录复制给其他人查看。

## AnyJev L0

[AnyJev CUDA 示例与方法说明](anyjev/README.md) 使用独立本地 worker，支持 Choice／Noul／Score；[BANKING77](anyjev/banking77.yaml) 与 [CLINC150+OOS](anyjev/clinc150.yaml) 配置传入完整 77／151 类目录。≤26 类 Choice 保留旧路径，大目录逐个验证互异单 token 标签，并维持完整 K 次 L0 轮换。A–ZZZ 的 18,278 个候选是理论候选数；当前 tokenizer 的实际容量、完整 prompt 上下文和显存仍需运行时核实，不裁剪候选。配置存在不表示完整实测已完成，结果以对应运行记录为准。
