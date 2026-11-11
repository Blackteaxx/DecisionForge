# 使用仓库原生流程

常规任务从原生 CLI 与配置开始。这里保留两个来源转换脚本；生成、检查、修复、缓存和恢复使用仓库已有实现。本页是后续使用方式，未产生替代 v1 的新结果。

若只是查看或复用 v1 数据，直接使用 [已打包的冻结数据](../data/README.md)，无需运行本页的材料出题步骤。

## 准备原始资料

在 DecisionForge 仓库根目录使用 Python3.12，安装 `python -m pip install -e '.[dev,eval-data]'`。设置工作位置：

```bash
export TASK=/absolute/path/to/new-medical-task
export XML=/absolute/path/to/medlineplus-topics-2026-10-27.xml
```

原始 URL、revision、SHA256、文件大小和许可见 [sources.lock.json](../configs/sources.lock.json)。其中 NLM 日期 XML 可能下线，复现历史版本需要相同哈希的快照；不要把最新资料标成历史版本。只读取 NLM 自有英文摘要，不抓取所链接的第三方文章。源文件保留不变，工作副本和转换结果放在 TASK。

## 来源清洗为 TXT

```bash
python -m examples.domain.medical_triage.scripts.prepare_sources \
  --xml "$XML" --output "$TASK/data/source-candidates.jsonl" \
  --text-directory "$TASK/data/materials"
```

脚本做 XML/HTML 清洗、行动相关候选筛选、来源记录和确定性角色划分；按 `train/validation/calibration/<source-id>.txt` 写出标题及原摘要，不生成问题、患者或答案。同一来源只进入一个角色。已有不同内容的 TXT 会拒绝覆盖。默认seed3407。

## 原生出题与检查

`.env` 提供 `BOYUE_API_BASE` 和 `BOYUE_API_KEY`。先选一份已准备的 TXT，设置它的真实角色和新的输出目录：

```bash
export MEDICAL_SOURCE_TXT="$TASK/data/materials/train/SOURCE_ID.txt"
export MEDICAL_SOURCE_SPLIT=train
export MEDICAL_RUN_DIR="$TASK/data/native/SOURCE_ID"
uv run dforge data inspect --config examples/domain/medical_triage/configs/native-generation.yaml
uv run dforge data build --config examples/domain/medical_triage/configs/native-generation.yaml
```

`inspect` 离线，`build` 才调用模型。TXT 无已有问答时，原生 extractor → text_assessor → qa_constructor → Choice 补全 → checker/定向修复 → 导出依次执行；没有足够依据可以不出题。每份最多5问，同一来源保留共同分组，关闭问题级随机留出；validation/calibration 来源分别用相应 `MEDICAL_SOURCE_SPLIT`。

按需对其余来源重复相同命令，保存实际 `run_directory`；不要把所有来源拼在一起再随机拆分。中断和验收直接用现有命令：

```bash
uv run dforge data resume --run-dir ACTUAL_RUN
uv run dforge data validate --input ACTUAL_RUN/accepted.json
```

不要把重跑 `build` 当恢复。只有核对 accepted/fail/meta、来源与角色覆盖后，才将已验收数据交给仓库原生 `train`。训练配置起点是 [training.yaml](../configs/training.yaml)，须替换数据/输出/模型位置，先执行 `train --validate-only`。

该配置构造的是原文支持的医疗决策问答。原生构造器不添加不存在的患者观测，Choice 补全也不保证固定四类临床行动；双患者比较、角色配额等 v1 实验约束并未由这个配置实现。需要这些约束时，应先识别能力缺口，再决定扩展；不能另写生成器却仍称为原生流程。

## 测试转换与接入

把冻结 CARE/Synth/Reddit 原始文件按来源清单放在 `RAW/care`、`RAW/synth`、`RAW/reddit`，运行：

```bash
python -m examples.domain.medical_triage.scripts.prepare_benchmarks \
  --raw-root "$RAW" --output "$TASK/evaluation/frozen"
python -m pip install --no-deps -e examples/domain/medical_triage/benchmark_ext
uv run dforge eval list
```

完整转换为135/869/1502条，不向模型发送标签或解释。配置现有 `eval` adapter 后执行 `eval --validate-only`，再按指定资源运行评测。扩展的指标契约见 [benchmark_ext](../benchmark_ext/README.md)。v1 五方法的准确运行配置和选模流程单独保存在 [历史复现](reproduce-v1.md)。
