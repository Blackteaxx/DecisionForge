# Full Typed Decisions preparation and benchmark recipes

The method directories contain separate `typed-decisions-full.yaml` recipes for the full source dataset. The existing sample recipes remain useful for integration checks. Run these commands from the repository root with the appropriate installed environment; full preparation needs the `eval-data` extra, while model execution needs the method's pinned requirements file.

## Prepare and freeze the shared roles

```bash
python -m pip install -e '.[eval-data]'
bash examples/training/prepare_data.sh
python examples/training/prepare_full.py
```

The existing converter writes `data/local/typed-decisions-training.jsonl` from pinned original train/test sources without sample limits. `prepare_full.py` writes `data/local/typed-decisions-frozen-roles.v3.jsonl` and its `.roles.json` audit using the native splitter with seed 3407. It refuses existing output or audit files. The original 1,200 train groups become 960 optimizer-training, 120 validation and 120 independent-calibration groups; the original 400 test groups remain test. All public requests and soft supervision remain unchanged. Frozen roles ensure OpenJev uses the same 960 optimizer-training groups even though it disables independent calibration.

Unsloth and Laya retain all 8,000 source questions: 4,800 optimizer-training, 600 validation, 600 calibration and 2,000 test. These are synthetic teacher distributions, not human correctness labels. The held-out test and benchmark references are used for evaluation rather than optimizer training.

## Prepare the canonical benchmarks

```bash
# prepare_data.sh already prepares Typed Decisions train/test.
python data/benchmarks/jevbench/prepare.py
python data/benchmarks/banking77/prepare.py
python data/benchmarks/clinc150_oos/prepare.py
```

The existing preparation tools pin their sources and retain raw/source manifests; `--convert-only` reuses already downloaded inputs. No recipe sets a sample limit or changes any candidate catalog. Typed Decisions evaluation is the registered one-question protocol, not the upstream five-question leaderboard protocol.

| Benchmark | Canonical split/count | Unsloth / Laya | AnyJev L0 | OpenJev |
| --- | --- | --- | --- | --- |
| Typed Decisions | test / 2,000 | all 2,000 | all 2,000 | explicit 1,200 C/N subset |
| JevBench | public / 231 | all 231 | all 231 | explicit 213 C/N subset |
| BANKING77 | test / 3,080 | all 3,080 | 完整 3,080，保留 77 类 | 完整 3,080，保留 77 类 |
| CLINC150+OOS | test / 5,500 | all 5,500 | 完整 5,500，保留 151 类 | 完整 5,500，保留 151 类 |

CLINC150+OOS 是意图分类 benchmark，不是临床或医疗数据集。上表描述可配置的数据覆盖和当前程序能力，完整实测是否完成以各次运行的实际终态为准。AnyJev 与 OpenJev 均可传入完整 77／151 类目录，并各自验证互异、非特殊的单 token 标签；A–ZZZ 的 18,278 个候选是理论候选数，不保证当前 tokenizer 均能全部使用。完整 prompt 的上下文预算、显存和运行成本仍需核实，不裁剪目录。AnyJev ≤26 类保留旧路径，大目录维持完整 K 次轮换；OpenJev 仍不支持 Score，历史 v1 可读，新训练保存并重验 v2 标签池，详见[方法说明](../openjev/README.md)。

## Prepare OpenJev's explicit C/N projections

```bash
python examples/training/openjev/prepare_cn.py \
  --source data/local/typed-decisions-frozen-roles.v3.jsonl \
  --output data/local/typed-decisions-openjev-cn-frozen-roles.v3.jsonl
python examples/training/openjev/prepare_eval.py \
  --source data/benchmarks/typed_decisions/processed/test.jsonl \
  --output data/local/openjev/typed_decisions-test-cn.jsonl
python examples/training/openjev/prepare_eval.py \
  --source data/benchmarks/jevbench/processed/public.jsonl \
  --output data/local/openjev/jevbench-public-cn.jsonl
```

上述训练准备显式排除 3,200 个 Score 问题，保留 1,600 个 case/group ID 及冻结角色：2,880 个 C/N 训练问题、360 个验证问题、360 个保留但不使用的 calibration 问题，以及 1,200 个测试问题；sidecar 逐题记录排除项。评测准备会为 Typed Decisions 的 800 个 Score 问题及 JevBench 的 18 个 Score 问题分别写入 `.source.json` 排除审计，拒绝覆盖已有输出；完整 Choice 目录保留原样，包括 77／151 类，标签池与完整上下文在模型运行时验证。保留问题的 request、target 和 metadata 不变。Typed Decisions／JevBench 的这两份评测应注明支持的 C/N 子集，其覆盖与完整 benchmark 不同。

## Train and evaluate by method

The portable templates use `Qwen/Qwen3-0.6B` revision `c1899de289a04d12100db370d81485cdf75e47ca`, or official `convaiinnovations/laya` revision `7b928d828b7b0e022f929d9bd2e44165aa270148`. Full recipe budgets reproduce the recorded experiment: Unsloth full one epoch, batch 2 with accumulation 4 and BF16; Laya one epoch, batch 16 and FP16; OpenJev 400 calibrate plus 400 reinforce updates, batch 8. These are separate method runs, not matched-compute experiments.

| Method | Train configuration | Benchmark configurations |
| --- | --- | --- |
| Unsloth full | [typed-decisions-full.yaml](../unsloth/typed-decisions-full.yaml) | [eval-full/](../unsloth/eval-full/) |
| Laya RLCD | [typed-decisions-full.yaml](../laya/typed-decisions-full.yaml) | [eval-full/](../laya/eval-full/) |
| OpenJev RLCD | [typed-decisions-full.yaml](../openjev/typed-decisions-full.yaml) | [eval-full/](../openjev/eval-full/) |
| AnyJev L0, no training | fixed ordinary backbone | [Typed Decisions](../../eval/training-free/anyjev/typed_decisions.yaml)、[JevBench](../../eval/training-free/anyjev/jevbench.yaml)、[BANKING77](../../eval/training-free/anyjev/banking77.yaml)、[CLINC150+OOS](../../eval/training-free/anyjev/clinc150.yaml) |

See each method's README for the commands and requirements. YAML paths resolve relative to their configuration file. Native benchmark templates point to their method's default full-training output; rewrite model and output paths together when copying configs or overriding a training `--run-dir`. New runs need unused output directories. `--validate-only` checks data/contracts without loading a model or using a GPU; `--score-only` reuses completed predictions.
