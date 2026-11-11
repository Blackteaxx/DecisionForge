# 训练后端贡献

**Unsloth full、Laya、OpenJev 及免训练 AnyJev 均为实验性方案。** Laya 默认官方 ModernBERT 模型，OpenJev 默认两阶段；接入与验证要求应连同[已知问题](../../../../docs/zh/training.md)一起阅读。已有 Unsloth LoRA 默认配置保持不变。

`unsloth.py` 是当前 Unsloth 后端实现，公共协议在 `decisionforge.training.backend`，训练编排在 `decisionforge.training.runner`。旧 `unsloth_backend` 导入保留兼容。真正训练时才加载 Unsloth、PyTorch 等可选依赖；基础安装和配置预检不应导入 GPU 栈。

内置训练后端接受 `backend: unsloth`、`backend: laya`、`backend: openjev` 和 `backend: swift`，这里没有自动发现机制。新增后端需要实现协议，并显式修改类型化配置、runner 的后端选择、依赖声明和测试，不能只添加文件或任意 backend 名称。保持监督契约、分组划分、checkpoint/恢复、校准、导出和重载校验语义。后端负责训练机制，方法负责推理语义，eval runner 负责调用、检查点和报告。

参见 [训练指南](../../../../docs/zh/training.md) 和 [扩展入口](../../../../extensions/README.md)。

训练策略、引擎接入及 checkpoint 兼容步骤见 [开发者指南](../../../../docs/zh/extending.md)。

`openjev.py` coordinates two isolated invocations of the pinned original OpenJev `main()`.
`openjev_task.py` owns the reversible A–J v3 mapping and strict probability contract;
`openjev_stage.py` registers the task and observes original AdamW updates. No trainer/loss/reward
implementation is copied. Its distinct `openjev_rlcd` artifact has both weights-only stages,
explicit lineage and config/tokenizer resources. It supports full single-GPU C/N only, no
independent calibration or complete state resume. The local worker calls official sample/forward
with one rationale and preserves raw output separately from the common scoring view.

`swift.py` invokes the pinned ms-swift `decision` trainer in an isolated process.
`swift_task.py` owns the reversible v3 mapping and abstain conditioning;
`swift_plugin.py` registers local data and the sole independent calibration callback.
The explicit `swift_routes` registry enables source-pinned OmniJev, Clef, Kev, JEV
and noncommercial-research StartLux; THX-02 stays blocked until its source/license can
be verified. Each route supplies conversion, limits, native-column readout, loader assets
and lineage. OmniJev v1 artifacts retain their original loader/runtime. Other routes use
`decisionforge.swift.decision.v1`, with full or continued-LoRA modes gated by source structure.
Adding a model_type requires verified licensing, mapping, restriction checks and tests
before changing the typed registry; laya and from-scratch routes are excluded. `uv sync --extra train-swift` installs a
separate, conflicting `train-swift` extra. CPU exact-template preflight loads no weights;
real training and local evaluation require a user-assigned single BF16 CUDA GPU.
