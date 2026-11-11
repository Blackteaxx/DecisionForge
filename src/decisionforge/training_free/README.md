# 免训练方法贡献

`chat_baseline/prompt.py` 与 `processing.py` 拥有固定 zero-shot prompt/schema 和响应解析语义；`llm2jev` 拥有包装常量、验证及 workload 语义。HTTP/本地传输与 worker 在 `decisionforge.eval.adapters`，eval runner 保持通用调用、日志、恢复与报告职责。

此目录不是任意方法名称的插件注册表。新增方法需要显式接入配置验证、adapter 选择、provenance、推理 source identity 和测试，不能只添加子目录。保持现有请求协议、严格解析和评分语义，按需延迟导入可选依赖；benchmark 继续拥有评分。

推理代码变化会拒绝普通 resume。跨旧布局的运行也保守拒绝恢复：如需继续原推理，使用匹配的旧 checkout；重新推理使用新运行目录；已有预测可使用 `--score-only` 离线重评。详见 [评测参考](../../../docs/zh/evaluation.md) 和 [扩展入口](../../../extensions/README.md)。

新增方法的配置、adapter、来源身份和测试接线见 [开发者指南](../../../docs/zh/extending.md)。
