# 原生决策训练示例

先按训练方法选择目录。所有 YAML 内的相对路径以配置文件所在目录解析；命令从仓库根目录执行。训练参数、依赖和产物契约见[训练指南](../../docs/zh/training.md)。

**OpenJev、Laya、AnyJev 和 Unsloth full 均为实验性方案。** Unsloth LoRA 保留为默认训练入口；Laya 默认基于官方 ModernBERT-large，OpenJev 默认 calibrate → reinforce 两阶段。方法限制、负结果和使用建议见[实验性方法及已知问题](../../docs/zh/training.md)。

| 方法 | 配置与入口 | 本地评测 |
| --- | --- | --- |
| [Unsloth LoRA / full](unsloth/README.md) | LoRA 默认；**full 实验性**；支持单卡/DDP、完整 checkpoint 恢复 | [LoRA benchmark](unsloth/eval/)、[full 留出集](unsloth/eval-full.yaml) |
| [Laya RLCD（实验性）](laya/README.md) | 默认官方 ModernBERT-large；单卡/DDP、导出权重初始化 | [留出集](laya/eval.yaml) |
| [OpenJev RLCD（实验性）](openjev/README.md) | 显式 C/N 数据准备、默认单卡 calibrate → reinforce 两阶段、权重初始化 | [留出集](openjev/eval.yaml) |

[AnyJev L0（实验性）](../eval/training-free/anyjev/README.md) 是无训练的本地评测方法，入口位于评测目录。

完整来源训练和全部适用 benchmark 的可复现步骤见[完整 Typed Decisions 配方](data/FULL.md)，各方法的 `typed-decisions-full.yaml` 使用共同冻结的 group 角色。

## 共享数据与验收

- [data/](data/SOURCE.md) 保留固定版本的 Typed Decisions 120 案例 / 600 问题、来源与许可证。
- [prepare_data.sh](prepare_data.sh) 与 [prepare_typed_decisions.py](prepare_typed_decisions.py) 下载并转换完整 Typed Decisions 数据，供各方法使用。
- [data/acceptance/](data/acceptance/README.md) 保存跨方法 C/N/S 和 C/N 验收 fixture 及逐项审计；各方法自己的 `*acceptance.yaml` 引用同一份数据。
- [tests/acceptance/method_acceptance.py](../../tests/acceptance/method_acceptance.py) 提供 CPU 静态矩阵和单次 CLI 证据记录；静态检查不代表 GPU 训练通过。
- [reports/](reports/2026-10-12-typed-decisions-qwen35-4b.md) 保留历史 LoRA 实测与恢复记录；[多方法运行报告](../../docs/zh/training.md) 记录已有 GPU 验收范围。

```bash
# 基础安装可运行；不加载模型或占用 GPU。
bash examples/training/unsloth/train.sh --validate-only
python tests/acceptance/method_acceptance.py static --work-dir /tmp/method-static-001
```

真实训练前核对模型、设备、预算、数据角色和输出路径。`qwen35-4b-full.yaml` 的 full 表示完整数据，训练方式仍为 LoRA；全参数训练使用 `unsloth/full-*.yaml`。

- [ms-swift OmniJev](swift/README.md)：实验性图文/离线视频继续训练、公开图文 benchmark 和本地评测。
