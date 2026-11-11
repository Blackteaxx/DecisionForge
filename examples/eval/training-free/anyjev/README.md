# AnyJev L0 local evaluation（实验性）

**AnyJev 是实验性免训练方案。** 完整大目录评测已有实测结果，但 K 次完整轮换和三个 content-free probes 带来较高推理及日志成本，不保证速度或跨数据集优势。默认算法、完整候选和原始概率保持不变；见[已知问题](../../../../docs/zh/training.md)及[统一 4B 结果](../../../../docs/zh/evaluation.md)。

This authored three-question fixture exercises Choice, Noul and ordered Score on an ordinary Qwen3-1.7B backbone. It is a functional example, not a benchmark result. AnyJev is pinned to `35311c58955716cb8955ce1368a665988bc63735` (0.3.0). The example backbone uses an immutable HF revision. `training_free: true` is a declaration; an arbitrary model identifier cannot prove its training history. Known Tacit and local decision artifacts are rejected.

Run from the repository root. Static validation needs only the base installation and neither imports the HF stack nor downloads weights:

```bash
python -m pip install -e .
uv run dforge eval run --config examples/eval/training-free/anyjev/eval.yaml --validate-only
```

Use a separate Linux/NVIDIA CUDA environment for inference:

```bash
python3.12 -m venv .venv-anyjev
source .venv-anyjev/bin/activate
uv sync --extra eval-anyjev
uv run dforge eval run --config examples/eval/training-free/anyjev/eval.yaml
uv run dforge eval run --config examples/eval/training-free/anyjev/eval.yaml --resume
uv run dforge eval run --config examples/eval/training-free/anyjev/eval.yaml --score-only
```

The dependency pins were installed for CPU tests against the actual upstream decider; these checks do not establish CUDA model compatibility or GPU acceptance. Use the issue acceptance report for real inference evidence. The worker requires the pinned VCS installation recorded in package `direct_url.json` and rejects another commit even if its version remains 0.3.0.

A fully materialized local HF model directory can replace `model`; relative paths starting with `./` or `../` resolve against YAML. Local model/tokenizer content hashes enter resume identity, so hashing large weights adds startup work. The HF backend does not forward offline options to its tokenizer; for fully offline startup use local weights/tokenizer files and set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` before starting the command. Source data and model files may be removed for `--score-only`, which uses the saved sample snapshot and never loads optional packages.

The method is fixed to upstream `Decider(level="L0")`, `prior="content_free"`, `prior_strength=1.0`, `adaptive_shifts=False`, unlimited complete rotations, `combine="logmean"`, probes `("N/A", "", "[MASK]")`, and the pinned upstream system instruction. Shared prefix is `auto`, its minimum prefix length is 256, and canonical option sorting is disabled. There are no task labels, few-shot examples, optimizer steps or Tacit checkpoints in inference.

Choice 保留每个原始 key 和 description，使用可逆 JSON 描述传递完整目录。2–26 类保留上游 A–Z 标签和原有数值路径；更大目录使用当前 tokenizer 逐个验证为单 token 的互异大写字母标签（A–Z 后依次筛选 AA–ZZZ），不会假设 AA 等字符串一定是单 token。BANKING77 的 77 类与 CLINC150+OOS 的 151 类均可完整传入。 Noul uses upstream `p_true` (Yes index 0), with all public false/true criteria included in the question. Score supports 2–10 explicit ordered levels, including duplicate descriptions, and uses the zero-based level expectation. Structured state/instructions/descriptions are deterministic JSON; role/content lists retain extra public fields.

Choice uses K cyclic permutations; Noul uses two Yes/No phrasings. **Score uses exactly one ordered pass.** Three content-free probes per permutation are normally cached after the first occurrence of a question; upstream prompt deduplication may reduce logical work. Raw results retain full indexed probabilities, original mapping, settings, diagnostics and per-request counter deltas. `backend_calls`, `flat_prompts` and shared-prefix counters are upstream counters, not measured GPU forwards. Confidence means maximum candidate probability, not calibrated correctness probability. Token counts, cost and unmeasured timing decompositions remain unknown.

大目录扩展固定基于 AnyJev `35311c58955716cb8955ce1368a665988bc63735`：仅放开官方 Question.choice 的 26 类构造上限（保留数量下限、str 转换和选项唯一校验），通过局部 Decider 子类覆盖官方 `_labels_for` 钩子。大目录的内部 question.text 追加“完整标签可由多个字母组成”的说明；原始公开字段和小目录 prompt 保持不变。官方 prompt 构造、K 次完整 cyclic rotations、content-free prior 和 logmean 数值流程全部复用，不修改安装包或全局对象。标签沿用上游 bare 优先、前置空格次选的编码规则；多 token、重复 token ID 和 special token 均不可入选。A–ZZZ 有限池的理论上限为 18,278，实际支持取决于 tokenizer 和上下文长度；不足会明确失败。每条大目录预测保存 schema、labels、token_ids、实际 encodings 和 SHA-256 协议 hash，可独立重验。目录不裁剪。 Tokenizer single-token label compatibility, complete rendered prompt lengths, actual CUDA placement and requested precision are checked at runtime; static validation reports those checks as deferred. Invalid distributions remain invalid with JSON-safe raw evidence; outputs are never clipped or normalized.

The standard evaluation journal supports interruption/resume without duplicate completed samples. Changes to data, source, method settings, dependencies, backbone revision/content or inference precision invalidate resume. Cancellation and deadlines terminate the owned worker. Reports preserve the AnyJev probability source and unknown local GPU cost.

[acceptance.yaml](acceptance.yaml) uses the shared training.v3 C/N/S fixture and a separate output directory for cross-method integration checks. See the [shared acceptance recorder](../../../training/data/acceptance/README.md) for static checks and invocation evidence.

## Complete canonical benchmarks

After the [shared data preparation](../../../training/data/FULL.md), use [typed_decisions.yaml](typed_decisions.yaml) for all 2,000 test questions and [jevbench.yaml](jevbench.yaml) for all 231 public questions. These recipes use ordinary Qwen3-0.6B at revision `c1899de289a04d12100db370d81485cdf75e47ca`, without decision training or a Tacit checkpoint. The existing three-question demo remains separate.

```bash
uv run dforge eval run --config examples/eval/training-free/anyjev/typed_decisions.yaml --validate-only
uv run dforge eval run --config examples/eval/training-free/anyjev/jevbench.yaml --validate-only
uv run dforge eval run --config examples/eval/training-free/anyjev/typed_decisions.yaml
uv run dforge eval run --config examples/eval/training-free/anyjev/jevbench.yaml
```

上述完整 benchmark 配置不限制样本数。新增 [banking77.yaml](banking77.yaml) 与 [clinc150.yaml](clinc150.yaml) 分别评测全部 3,080／5,500 个测试样本，包含完整 77／151 类目录。首次遇到每份问题时需要 K 个真实轮换与 3K 个 content-free probe prompts，缓存后每个样本仍需要 K 个真实轮换；大目录的计算成本会明显增加。配置本身不代表实际验收通过，运行结果与失败另行记录。

大于 26 类的 Choice 会将全部诊断矩阵无损保存到 `anyjev.diagnostics.compressed`：`schema_version: 1`、`codec: gzip+base64`，并记录未压缩 JSON 的字节数和 SHA-256。`permutations`、prior 等轻量标量仍可直接阅读；所有矩阵、排列和非有限数值的原始 JSON-safe 字符串证据均保留。小目录、Noul 和 Score 的诊断格式保持原样。这一表示减少完整 CLINC150+OOS 在评测 journal 双表和副本中持有大量 Python float/list 对象的内存；仍须为完整预测与 journal 预留磁盘空间。

读取任一版本诊断均可使用基础安装自带的解码函数，无需加载模型或任何 SDK：

```python
from decisionforge.training_free.anyjev.diagnostics import decode_diagnostics

diagnostics = decode_diagnostics(response["anyjev"]["diagnostics"])
```

解码会重验 schema、codec、未压缩长度、SHA-256 与可读标量；损坏的证据会明确报错。完整编码随 journal 保存，`--score-only` 可在数据和模型文件已删除时直接重读；诊断压缩不改变概率、置信度、选项映射、轮换数量或 L0 算法。

完整大目录运行会在每次预测内部复用已验证的完整 prompt token 序列：长度检查、上游长度排序和同批次 padding 使用同一编码结果；模型 forward、输入顺序、K 个轮换及概率读取保持原流程。该优化依据 H200 warm151 性能分析中显著的重复分词开销实现；实际墙钟收益以同设备 A/B 记录为准。

缓存仅用于大于 26 类的 Choice、已知 fast tokenizer 的 `input_ids`／`attention_mask` 和上游固定调用参数。额外字段、未知参数或缓存缺失会走原 tokenizer；上限为 1,024 条、总计 2,000,000 个 token，每次预测结束（包括异常）恢复原 tokenizer 并清空缓存。小目录、Noul 和 Score 保持原路径。缓存模块纳入推理源码身份，旧预测恢复时仍执行兼容性检查。
