# 历史 v1 复现

本页只说明历史 v1：专用教师构造完整病例，再调用原生 build 检查。常规新任务使用 [原生流程](usage.md)。迁移只调整模块位置，历史生成、训练和评分语义保持不变。命令从包含本目录的 DecisionForge 仓库根目录执行；不依赖历史任务目录或私人挂载路径。新运行的数据、日志和模型放到自选 `TASK`/`STORE`，不要写进版本化的示例目录。

## 0. 直接复用 v1 冻结数据

[完整数据](../data/README.md)已随案例提供，无需重新付费生成。可先运行 `uv run dforge train run --config examples/domain/medical_triage/configs/training.yaml --validate-only` 离线检查原始6065条与角色划分。

沿用下文的工作目录布局时，在设置 `STORE` 后复制三份原样数据：

```bash
mkdir -p "$STORE/data/prepared/joint-001"
cp examples/domain/medical_triage/data/v1/training.json "$STORE/data/prepared/joint-001/"
cp examples/domain/medical_triage/data/v1/validation.jsonl "$STORE/data/prepared/joint-001/"
cp examples/domain/medical_triage/data/v1/calibration.jsonl "$STORE/data/prepared/joint-001/"
```

训练、选模、校准可以跳过第2、3节，接第4节。完整公开评测仍需第1节的测试原文件和扩展准备。第2、3节仅用于复查历史生成方法；新调用的随机输出不能替代原冻结数据。

## 1. 安装与准备输入

使用 Python 3.12。CPU 准备阶段无需 GPU：

```bash
python -m pip install -e '.[dev,eval-data]'
export TASK=/absolute/path/to/new-medical-task
export STORE=/absolute/path/to/new-medical-storage
export BACKBONE=/absolute/path/to/frozen-Qwen3.5-9B
export PYTHONPATH="$PWD/src:$PWD"
```

按锁定来源下载原始文件并核对哈希，然后物化来源清单。这里只用标准库做获取和转换，不额外维护下载器：

```bash
python - <<'PYINPUT'
import hashlib, json, os, urllib.request
from pathlib import Path
from examples.domain.medical_triage.scripts.prepare_sources import assign_source_roles, load_nlm_topics
lock = json.loads(Path('examples/domain/medical_triage/configs/sources.lock.json').read_text())
task = Path(os.environ['TASK'])
for item in lock['files']:
    path = task / item['task_relative_path']
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(item['url'], timeout=120) as response:
            data = response.read()
        assert hashlib.sha256(data).hexdigest() == item['sha256'], path
        path.write_bytes(data)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], path
xml = task / lock['files'][-1]['task_relative_path']
sources = {row['id']: row for row in assign_source_roles(load_nlm_topics(xml))}
prepared = task / 'data/prepared'
prepared.mkdir(parents=True, exist_ok=True)
for run in lock['generation_rounds']:
    rows = [sources[key] for key in run['source_ids']]
    for row in rows:
        reference = lock['source_summaries'][row['id']]
        assert row['summary_sha256'] == reference['sha256'] and row['split'] == reference['split']
    (prepared / (run['id'] + '-sources.jsonl')).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
PYINPUT
python -m examples.domain.medical_triage.scripts.prepare_benchmarks \
  --raw-root "$TASK/data/inputs/benchmarks" --output "$TASK/evaluation/frozen"
```

独立选模和报告需要任务所属的扩展身份，复制实体包并记录文件哈希：

```bash
mkdir -p "$TASK/benchmark_ext"
cp -R examples/domain/medical_triage/benchmark_ext/. "$TASK/benchmark_ext/"
python - <<'PYEXT'
import hashlib, json, os
from pathlib import Path
root = Path(os.environ['TASK']) / 'benchmark_ext'
files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*')) if p.is_file() and p.name != 'IDENTITY.json' and '__pycache__' not in p.parts and not any(part.endswith('.egg-info') for part in p.parts)}
(root / 'IDENTITY.json').write_text(json.dumps({'distribution': 'dforge-medical-triage-benchmark', 'version': '0.1.0', 'benchmark': 'medical_decisions', 'benchmark_version': '1', 'files': files}, indent=2) + '\n')
PYEXT
python -m pip install --no-deps -e "$TASK/benchmark_ext"
uv run dforge eval list
```

五个原始文件的固定 URL、revision、大小、许可均在锁定清单。CARE 取官方 `public_test_1`，Synth 取 INBOX_A/B，Reddit 取官方 test parquet。NLM 为 2026-10-27 XML，仅提取英文 full-summary；日期 URL 可能失效，需自行取得 SHA256 相同的原始快照，不以最新 XML 冒充原版本。本交付没有重新分发原始文件。

底座固定 `Qwen/Qwen3.5-9B` revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`；自行下载完整 snapshot 到 `BACKBONE`。训练、本地选模与推理使用 Linux、NVIDIA CUDA，实测单 H200。训练环境按 [运行环境](../../../../pyproject.toml) 安装：

```bash
uv sync --extra train-unsloth
python -m pip install --no-deps -e "$TASK/benchmark_ext"
```

原实测 Torch 2.11.0+cu130、Transformers 5.17.0、Unsloth 2026.10.3，Unsloth decision API 源码固定到 requirements 中的提交。CUDA wheel 按机器匹配；CPU 核心测试环境不安装 benchmark 扩展，以免改变内置注册表的测试前提。

## 2. 教师构造与 checker

本地 `.env` 配置 `BOYUE_API_BASE`、`BOYUE_API_KEY`，模型为 `deepseek-flash`；不要提交凭据。下面会产生真实模型请求与费用。

```bash
python -m examples.domain.medical_triage.scripts.v1.generate \
  --source-manifest "$TASK/data/prepared/construct-001-sources.jsonl" \
  --output "$STORE/data/runs/construct-001" --round 1 --cases-per-source 8 --concurrency 8
uv run dforge data inspect --config "$STORE/data/runs/construct-001/build.yaml"
uv run dforge data build --config "$STORE/data/runs/construct-001/build.yaml"
```

第二至四轮分别改成 `construct-002/003/004`、`--round 2/3/4`，并发 4，其他配置沿用。锁定来源数量依次为 496、309、200、80。每轮使用全新目录，保存日志及 `build` 实际返回的 `run_directory`；发生版本后缀时使用实际路径。

教师构造中断用相同参数加 `--resume`；成功响应从 journal 复用，结果未知的请求保留记录。`--finalize-existing` 仅在确认构造已停止后离线物化。checker 中断用原生 `uv run dforge data resume --run-dir ACTUAL_BUILD_RUN`。不要从头重放成功 API 请求。

教师具有随机性，服务别名也可变化；重新调用不保证原始字节、6088 条接受或最终6065条。原成绩对应 [随案例提供的冻结训练数据](../data/README.md)，其 SHA256 在 [模型与数据身份](../results/identities.json)。新生成数据的成绩必须另报，不能冒充原 v1。

## 3. 汇总清洗与审计

把四个实际 `build` 输出目录填入下列命令；默认首次无版本后缀时为各构造目录的 `build/run/`：

```bash
python -m examples.domain.medical_triage.scripts.v1.assemble \
  --runs ACTUAL_BUILD_RUN_1 ACTUAL_BUILD_RUN_2 ACTUAL_BUILD_RUN_3 ACTUAL_BUILD_RUN_4 \
  --output "$STORE/data/prepared/joint-001"
python -m examples.domain.medical_triage.scripts.v1.audit \
  --assembled "$STORE/data/prepared/joint-001" \
  --benchmarks "$TASK/evaluation/frozen/care.jsonl" "$TASK/evaluation/frozen/synth.jsonl" "$TASK/evaluation/frozen/reddit.jsonl" \
  --output "$TASK/data/prepared/overlap-audit.json"
uv run dforge data validate --input "$STORE/data/prepared/joint-001/training.json"
```

`assemble` 保留来源角色、固定候选语义和正确答案内容，隔离冲突与跨角色患者复用，不重标。`audit` 检查来源家族、患者消息/病史与公开测试的精确及近重复；发现重叠时失败。训练只在清洗、覆盖和审计通过后开始。具体算法见 [方法与消耗](methods.md)。

## 4. 训练、选模、校准

将 [training.yaml](../configs/training.yaml) 的数据、底座和输出重写到新工作目录：

```bash
python - <<'PYCONFIG'
import os
from pathlib import Path
import yaml
source = Path('examples/domain/medical_triage/configs/training.yaml')
config = yaml.safe_load(source.read_text())
task, store = Path(os.environ['TASK']), Path(os.environ['STORE'])
config['data']['path'] = str(store / 'data/prepared/joint-001/training.json')
config['model']['name'] = os.environ['BACKBONE']
config['output']['directory'] = str(store / 'training/joint-001/run')
folder = task / 'training/joint-001'
folder.mkdir(parents=True, exist_ok=True)
(folder / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
PYCONFIG
export UNSLOTH_COMPILE_DISABLE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
uv run dforge train run --config "$TASK/training/joint-001/config.yaml" --validate-only
uv run dforge train run --config "$TASK/training/joint-001/config.yaml"
python -m examples.domain.medical_triage.scripts.v1.select_model \
  --training-run "$STORE/training/joint-001/run" \
  --validation "$STORE/data/prepared/joint-001/validation.jsonl" \
  --control-root "$TASK/training/selection-001" --run-root "$STORE/evaluation/validation-selection-001"
python -m examples.domain.medical_triage.scripts.v1.calibrate fit \
  --selection "$TASK/training/selection-001/selection.json" \
  --data "$STORE/data/prepared/joint-001/training.json" \
  --output "$STORE/training/calibration-001/run" --batch-size 2
```

新运行的 `run/` 必须为空，stdout/stderr 写在其外。原 v1 5024条训练、两轮628更新、七个checkpoint，在446条独立validation上选step200；595条calibration只拟合温度，不更新模型权重。新进程重载要求四探针原生Choice一致、概率误差≤0.002。

原模型是 LoRA+决策头，依赖固定底座，不能只取 adapter 或将其当作 merged 模型。已清理的历史 checkpoint 不能直接 resume；重现选模轨迹须重新训练。

## 5. 启动同底座服务并生成15份配置

LLM2Jev/AnyJev 普通 HF 环境安装 `.[eval-anyjev,eval-llm2jev]`；各评测环境安装同一任务扩展。上游 revision 均已固定在仓库依赖中。GPU 方法顺序执行，同一时间仅保留当前方法需要的模型。

LLM2Jev 服务：

```bash
python -m llm2jev.server.server --backend transformers --model-path "$BACKBONE" \
  --served-model-name qwen35-9b --device cuda:0 --dtype bfloat16 --batch-size 4 \
  --host 127.0.0.1 --port 30000
```

Chat 服务（执行 Chat 时启动）：

```bash
python -m examples.domain.medical_triage.scripts.v1.serve_chat \
  --model-path "$BACKBONE" --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --host 127.0.0.1 --port 30001
```

独立终端生成配置；此步骤无需服务正在运行：

```bash
python -m examples.domain.medical_triage.scripts.v1.experiment configs \
  --task-root "$TASK" --store "$STORE" --backbone "$BACKBONE" \
  --calibrated-model "$STORE/training/calibration-001/run/model"
```

得到 trained/jev/llm2jev/anyjev/chat × care/synth/reddit 全15份配置，完整test、无limit。依次对每份执行 `uv run dforge eval run --config ACTUAL_CONFIG --validate-only`，再执行不带 `--validate-only` 的真实推理。trained 使用Unsloth环境，AnyJev使用原版HF环境，HTTP方法在服务可达的环境执行。不得把离线预检当成实测。

AnyJev 固定 L0/content_free/全部K次轮换/logmean，`shared_prefix: false` 使用官方完整prompt路径；不修改上游算法。Chat 为BF16、温度0、关闭thinking、max_tokens2048、text输出，不修补无效JSON。

外部 Jev 使用 [jev-openrouter.yaml](../configs/jev-openrouter.yaml) 所示的 OpenRouter 原生 Decisions API，`.env` 提供 `OPENROUTER_API_KEY`。原响应模型 `typesafe/jev-1.14-20261014`；服务别名未来可能变化，须保存实际版本。没有出口限制时直接执行生成的 Jev 配置；需要代理时可使用 [run_jev.py](../scripts/v1/run_jev.py) 的 `--help`，自行提供私有代理配置。该脚本会检查选模/校准/重载完成，不依赖预置代理节点。

## 6. 报告与离线复算

从本次运行记录中明确选定15个实际目录，不能用宽泛glob混入旧失败或第二轮结果：

```bash
python -m examples.domain.medical_triage.scripts.v1.report \
  --task-root "$TASK" --calibration-manifest "$STORE/training/calibration-001/run/manifest.json" \
  --runs RUN1 RUN2 RUN3 RUN4 RUN5 RUN6 RUN7 RUN8 RUN9 RUN10 RUN11 RUN12 RUN13 RUN14 RUN15 \
  --output "$TASK/evaluation/comparison-001"
```

报告器校验完整矩阵、相同样本/target/扩展、最终校准模型内容身份。已有完整原生运行记录时，`uv run dforge eval run --score-only --config ACTUAL_CONFIG --run-dir ACTUAL_RUN` 仅离线复算，不产生新推理。本交付只有汇总结果，未包含逐题预测，不能凭汇总JSON重建原始预测。

历史API和coding agent汇总统计见 [方法与消耗](methods.md)，一次性会话统计、迁移和交付检查工具不随案例维护。删除的中间checkpoint恢复限制见 [清理收据](../results/storage-cleanup.json)。
