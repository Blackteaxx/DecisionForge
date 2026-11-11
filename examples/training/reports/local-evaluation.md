# 本地模型评测与恢复验收

2026-10-12，在 Linux / NVIDIA H200 单卡上直接加载短训练的 Qwen3.5-4B 决策模型，评测 Typed Decisions `1-single-question` 原 test 集的 400 案例 / 2,000 问题。使用保存的 decision head 与温度参数，没有重新训练或在 test 上拟合校准。

| 产物 | 题数 | 严格有效概率 | soft Brier ↓ |
| --- | ---: | ---: | ---: |
| LoRA `model/` | 2,000 | 2,000 | 0.24813527753499282 |
| 独立 `merged/`，4-bit 加载 | 2,000 | 2,000 | 0.24859106578809376 |

两种产物覆盖 Choice/Noul/Score，scoring_invalid 均为 0。重新量化后的 merged 与 adapter 数值可能不同。这是短训练的接入验收；完整 2 epochs 训练后的结果见 [正式实验报告](2026-10-12-typed-decisions-qwen35-4b.md)。

验证了以下行为：

- 已完成运行 resume 不新增预测，score-only 不修改原预测。
- 在保存 3 条预测后中断，resume 完成 10 条且前 3 条保持不变。
- 模型和原数据路径不可用时，仍可从保存的快照和 journal 离线评分。
- 每种产物另用 12 题复查，与官方 `FastDecisionModel.predict` 对照三个题型。本地完整概率与官方四位小数展示概率的最大差为 `4.915651304926438e-5`，在显示舍入范围内。

运行时为 Unsloth/Zoo 2026.10.3、Torch 2.11.0+cu129、Transformers 5.17.0、PEFT 0.21.2、bf16。运行 manifest 保存实际依赖、底座、校准、设备和 checkpoint 内容身份；不估算 GPU 成本。

复查使用训练依赖及 CUDA 环境，在仓库根目录指定自己的训练产物目录：

```bash
python tests/acceptance/local_eval_gpu_acceptance.py \
  --trained-run outputs/train/qwen35-4b-full \
  --work-dir outputs/eval/local-acceptance \
  --limit 12 --compare-official
```

脚本要求同时存在 `model/` 和 `merged/`，额外验证 10 题的中断恢复。默认数据为准备后的完整 Typed Decisions test；`--limit 2000` 可执行全量复查。每次使用新的 work-dir，输出保留在该目录。已有预测的重评分使用普通 eval 的 `--score-only`。
