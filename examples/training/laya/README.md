# Laya RLCD（实验性，默认 ModernBERT）

默认使用 `convaiinnovations/laya` 的 **ModernBERT-large 编码器与已训练决策头**，固定 revision 为 `7b928d828b7b0e022f929d9bd2e44165aa270148`。Qwen 适配仅为单独实验，不替换此默认。历史 BERT 继续训练结果与本轮适配退化见[实验性方案及已知问题](../../../docs/zh/training.md#laya-保留-bert-默认)和[完整分数](../../../docs/zh/training.md)。

[single.yaml](single.yaml) 与 [ddp.yaml](ddp.yaml) 使用共享 Typed Decisions C/N/S 样本及按 group 留出的 validation/calibration/test。安装仓库根目录的 `uv sync --extra train-laya`；模型与依赖固定版本见[训练指南](../../../docs/zh/training.md#laya-rlcd)。

```bash
uv run dforge train run --config examples/training/laya/single.yaml --validate-only
uv run dforge train run --config examples/training/laya/single.yaml
uv run dforge eval run --config examples/training/laya/eval.yaml
```

双卡选择 `ddp.yaml`，由 coordinator 启动 torchrun。[init.yaml](init.yaml) 从导出权重开始新运行，不恢复优化器，完整 `--resume` 不受支持。`eval.yaml` 默认读取 single 的模型与留出 test；覆盖输出后同步调整这些路径。

[acceptance.yaml](acceptance.yaml) 和 [eval-acceptance.yaml](eval-acceptance.yaml) 使用[共享验收数据](../data/acceptance/README.md)及独立输出。该小样本用于检查集成，不能用来证明质量提升。

## 完整 Typed Decisions 配方

先按[共享完整数据指南](../data/FULL.md)冻结角色并准备 benchmark。[typed-decisions-full.yaml](typed-decisions-full.yaml) 使用固定官方 Laya、一个 epoch、batch 16、FP16 和 8192 最大长度；独立 calibration 只使用冻结的 120 个 group。

```bash
uv run dforge train run --config examples/training/laya/typed-decisions-full.yaml --validate-only
uv run dforge train run --config examples/training/laya/typed-decisions-full.yaml
for benchmark in typed_decisions jevbench banking77 clinc150_oos; do
  uv run dforge eval run --config "examples/training/laya/eval-full/$benchmark.yaml"
done
```

[eval-full/](eval-full/) 对四个完整 benchmark 评测，默认读取该完整训练配方的产物，不对 test 重新拟合温度。需要新输出时一起改训练和评测 YAML 的路径。
