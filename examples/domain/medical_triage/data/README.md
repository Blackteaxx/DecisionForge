# v1 冻结训练数据

这里提供产生 v1 正式结果的实际数据，可以直接查看或交给仓库训练入口，无需重新调用教师生成。原始 `training.json` 按字节保留，SHA256 与实验记录一致。

## 文件与划分

| 文件 | 内容 |
| --- | --- |
| [v1/training.json](v1/training.json) | 原始完整训练输入，6065 条，保留各条 `split` |
| [v1/validation.jsonl](v1/validation.jsonl) | 原独立选模使用的446条验证请求及金标 |
| [v1/calibration.jsonl](v1/calibration.jsonl) | 原595条独立校准记录的评测格式视图 |
| [v1/examples.json](v1/examples.json) | 从训练角色抽出的两条原样记录，分诊与消息优先级各一条，方便预览 |
| [v1/provenance.jsonl](v1/provenance.jsonl) | 每条记录的任务、来源ID、原文证据、教师理由与合成标签说明 |
| [v1/sources.jsonl](v1/sources.jsonl) | 309个实际来源的标题、URL、冻结摘要、哈希与原角色，去除重复存储 |
| [v1/manifest.json](v1/manifest.json) | 文件大小、SHA256、数量与准入统计 |

| 角色 | 分诊行动 | 消息优先级 | 合计 | 患者/来源连通组 |
| --- | ---: | ---: | ---: | ---: |
| train | 2650 | 2374 | 5024 | 218 |
| validation | 238 | 208 | 446 | 23 |
| calibration | 329 | 266 | 595 | 28 |

`training.json` 是原实验的完整输入文件名，其中只有 `split=train` 的5024条用于权重优化；validation用于checkpoint选择，calibration用于选模后的温度拟合。没有公开测试记录。不要将6065条全部改成训练角色，或在训练时重新随机划分。

## 字段和标签怎么读

- `record_id`：数据内唯一记录ID，可连接 provenance 中同名字段。
- `group_id`：患者/来源连通组。各角色之间不共享组。
- `request.state`：模型可见的患者事实；保留原格式，部分记录是包含患者对象的 JSON 字符串。
- `request.questions`：问题和候选内容。`supervision.answers` 以相同问题ID关联标签。
- `supervision.answers.<qid>.choice`：当前候选映射下的正确选项。选项已经置换，不能把字母A统一解释为某个临床行动。
- `confidence` 与 one-hot `probabilities`：教师硬标签的监督编码，不代表临床置信度。

provenance 通过 `source_id` 连接 sources，可以查看原摘要、连续证据和构造理由。证据与理由是审计材料，不应拼入模型的患者输入。validation/calibration JSONL 是已有角色的格式转换，不能再次追加到训练JSON造成重复。

## 来源与清洗

资料来自2026-10-27 MedlinePlus/NLM官方XML的英文健康主题摘要，仅使用NLM自有 `full-summary`，未抓取链接的第三方正文。署名：Source: MedlinePlus, National Library of Medicine. 来源身份与原文件许可/使用说明见 [来源锁定清单](../configs/sources.lock.json)。sources 文件保留这些实际使用的摘要，阅读依据无需再访问私人存储。

`deepseek-flash` 构造虚拟患者和监督，再由仓库原生build/checker验收；teacher和checker为同一模型，没有独立临床专家逐条裁定。四轮checker接受6088条、review37条；全局隔离23条后准入6065条，没有重标或改变来源角色。来源清洗脚本是 [prepare_sources.py](../scripts/prepare_sources.py)，历史合并和隔离代码在 [scripts/v1/](../scripts/v1/)。完整过程及审计局限见 [方法文档](../docs/methods.md)。

这份数据对应历史专用病例构造流程，不是后来新增原生TXT配置的产物。人物和病史为合成内容，不等于真实患者记录或临床真值；字面隔离也不证明语义或预训练完全独立。

## 直接验证与使用

从DecisionForge仓库根目录、在已安装项目依赖的Python环境运行，以下两条均不调用模型或GPU：

```bash
uv run dforge data validate --input examples/domain/medical_triage/data/v1/training.json
uv run dforge train run --config examples/domain/medical_triage/configs/training.yaml --validate-only
```

[训练配置](../configs/training.yaml)已指向这份数据，保留原角色、关闭自动留出和自动最后checkpoint校准。实际训练需配置相应GPU、底座与新的输出位置；独立选模、校准和重载见 [历史复现](../docs/reproduce-v1.md)。

`training.json` SHA256：

```text
1c60d6d832ead5b11a78e45cc0a64810747548dcf66248121ed0208b7eb0442d
```
