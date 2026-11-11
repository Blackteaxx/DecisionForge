# 示例

- standard.json：三种题型的标准输入模板，字段齐全，仅调用 checker。
- minimal.json：只有 question/answer，使用 configs/qa_choice.yaml、qa_noul.yaml、qa_score.yaml 补齐缺项。
- custom.json + mapping.json：自定义嵌套 JSON，通过 configs/flexible_qa.yaml 映射。
- source.txt：TXT 原文问答提取，通过 configs/txt.yaml 运行；超长TXT会先进入[自动分批](../docs/zh/data-generation.md)，生成源目录batches及逐批YAML。
- [domain/medical_triage/](domain/medical_triage/README.md)：医疗分诊与患者消息优先级的完整任务，包括数据来源、冻结训练数据、benchmark 扩展、五种方法的结果与复现说明。
- [training/](training/README.md)：按 Unsloth、Laya、OpenJev 方法组织的训练与本地评测示例。
- [eval/unsloth/](eval/unsloth/README.md)：加载训练产物进行本地评测。
- [eval/training-free/](eval/training-free/README.md)：Chat baseline、LLM2Jev 服务与成对报告。
- eval/：原生评测协议、HTTP 服务和 benchmark 插件示例。

参考 [生成流程](../docs/zh/data-generation.md)。所有配置路径均相对于 YAML 文件。

可安装的 benchmark/metric 插件示例位于 [extensions/example_plugin](../extensions/example_plugin/README.md)。
