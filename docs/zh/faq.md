# 常见问题

[English](../en/faq.md) · [README](../../README.md)

## 离线检查为什么不需要 key？

`data inspect`、`data validate`、`data export`、schema、doctor、训练预检、评测预检和重评分不调用模型。使用 `configs/offline.yaml` 可避开未设置的模型环境变量；生成配置若含 `${OPENAI_MODEL}`，仍需能解析该变量。

## 模型接口不支持 strict 输出怎么办？

按服务能力选择 `llm.api: responses/chat_completions`、`structured_output: strict/json/prompt` 和 `chat_token_limit_param`。兼容服务不一定支持所有组合；先验证小输入，再扩大任务。不要用无限重试掩盖参数错误。

## 退出码是什么？

0 为成功，1 为配置、文件或执行错误，2 为 CLI 参数错误或已保存的预算/模型暂停状态，130 为用户中断。检查 JSON 的 status 与实际产物，不只看日志结尾。

## 为什么安装 GPU extra 失败？

后端要求特定 Python、平台和 CUDA。先运行 `dforge doctor`，使用 3.12 独立环境，并按对应 extra 安装。多个训练组互斥；不要用 `--all-extras`。OpenJev Arrow 固定为 21.0.0。macOS 基础环境不验证 GPU 训练。

## 配置路径和运行目录为什么不同？

YAML 内路径相对配置文件；CLI 覆盖路径相对当前目录。输入/配置身份变化会创建版本目录。后续命令使用 JSON 返回的实际 run_directory。运行锁用于保护检查点，不能直接删除活动锁。

## 旧 Web UI 或 deep_search 配置还能用吗？

这两个功能已移除。旧 `input.type: deep_search` 会收到明确错误；资料搜寻由 Agent 工具负责，仓库保留 JSON/TXT 输入及 MinerU PDF 解析。界面路径是 dsh 插件，见 [Agent](agent.md)。

## 数据与模型结果是否可直接对外声称有效？

checker 通过说明符合配置检查，并不证明领域事实或临床有效性。确认标签、来源独立、测试集隔离和指标口径；报告失败率、限制和负结果，保留可追溯原始证据。

## Pages 为什么暂时不能访问？

仓库仍为私有。workflow 已准备；公开和启用 Pages 后验收部署。非默认分支需允许相应 environment 分支。当前可本地打开 `demo/index.html` 或离线 bundle。
