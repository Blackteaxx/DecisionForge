# Shared method acceptance

Run from a committed repository checkout with the selected environment's Python.
This finite local driver records public CLI invocations; it does not install
dependencies, allocate compute, or prove GPU acceptance from a static result.
Optional environments and source pins are documented in [training](../../../../docs/zh/training.md).

```bash
python tests/acceptance/method_acceptance.py static --work-dir /tmp/method-static-001
```

The static matrix blocks optional SDK imports in every child CLI. It feeds the
**same C/N file and SHA256** to Unsloth full, Laya, OpenJev and canonical generic
evaluation. 新静态矩阵保持 group 隔离，允许 AnyJev 的完整 27／77／151 类目录，并继续在创建输出前拒绝 OpenJev Score 与 Laya/OpenJev full resume。类别标签的 tokenizer 单 token 验证仍在运行时完成；静态预检不会导入模型 SDK。旧报告里的类别拒绝记录属于旧版本历史事实，已由新的标签协议取代。
`static.json` is CPU contract evidence only.

`source.json` records explicit role assignment for 15 existing example
records (2 train, 1 validation, 10 calibration, 2 test). Public requests and all
soft supervision are unchanged. `common-cns.v3.jsonl` contains 75 questions;
`common-cn.v3.jsonl` contains 45, with only Score questions and their answers
explicitly excluded. `common-cn.v3.source.json` records every exclusion. The Unsloth full, Laya and OpenJev
`*acceptance.yaml` templates use that same C/N file. For C/N/S acceptance of
Unsloth/Laya, copy their YAML and change only the data path to the C/N/S fixture;
record the effective config. OpenJev must reject that input. The small fixtures
check integration and make no quality claim; the external real training jobs
use the original 120-record fixture and its group-disjoint holdouts.

For real work, select your authorized local GPU/environment, copy and adjust the
YAML's model/data/output paths, then record each invocation separately:

```bash
python tests/acceptance/method_acceptance.py record --work-dir /tmp/full-command-001 \
  train --config examples/training/unsloth/full-acceptance.yaml --run-dir /tmp/full-run-001
python tests/acceptance/method_acceptance.py record --work-dir /tmp/full-resume-001 \
  train --config examples/training/unsloth/full-acceptance.yaml --run-dir /tmp/full-run-001 --resume
python tests/acceptance/method_acceptance.py record --work-dir /tmp/anyjev-command-001 \
  eval --config examples/eval/training-free/anyjev/acceptance.yaml --run-dir /tmp/anyjev-run-001
python tests/acceptance/method_acceptance.py record --work-dir /tmp/anyjev-score-001 \
  eval --config examples/eval/training-free/anyjev/acceptance.yaml --run-dir /tmp/anyjev-run-001 --score-only
```

Each recorder directory must be new. Original stdout/stderr, exit status,
command, copied input config, config SHA, Git revision/dirty state, Python,
installed package metadata and visible-device setting are retained. Static and
score-only calls use the optional-import guard. Actual GPUs, tensor deltas,
optimizer/RNG continuity and native workload diagnostics must be checked in the
run's own artifacts. A recorded successful CLI is not a substitute for those
checks. The recorder reads **the report directory returned by that invocation**
and then `metrics.json`; there is no `evaluation/report.json` contract.

The recorder uses the existing Linux subreaper to own each CLI launch, including
detached descendants. It persists command metadata before launch and streams
unbuffered stdout/stderr directly into evidence files while the CLI runs.
SIGINT/SIGTERM forward to the owned session; the supervisor terminates and
reaps descendants before the recorder writes its terminal record and returns
130/143. Repeated cancellation cannot interrupt cleanup. The command record
distinguishes completed, failed and interrupted states, records the requested
signal and scope exit code, and retains partial logs. This recorder requires
Linux; it does not signal unrelated processes or treat interruption as completion.

Full resume requires a genuinely interrupted, compatible complete checkpoint.
Laya/OpenJev reject full resume; weight initialization creates a new run.
OpenJev's `max_steps` and `reinforce_steps` are separate stage budgets. Its
calibrate stage is training, not held-out temperature fitting. Local reload
templates are [Unsloth full](../../unsloth/eval-full.yaml),
[Laya](../../laya/eval-acceptance.yaml), and [OpenJev](../../openjev/eval-acceptance.yaml);
they point to their corresponding default training output and held-out test
data, so rewrite paths when overriding `--run-dir`.

The [historical acceptance report](../../../../docs/zh/training.md) records verified
finite GPU acceptance for all four methods: full and Laya dual-H200 runs, fresh
OpenJev v2 single-H200 training, and separately recovered AnyJev assertions.
Original AnyJev/OpenJev attempt1 Failed records remain Failed. The report gives
actual sources/steps/metrics/retention, warnings and limited measurement scopes;
final controller review is separate from runtime execution and no quality
improvement or cross-model ablation is claimed.
