# DecisionForge 插件

为 DeepSeek Harness Web UI 添加 DecisionForge Logo、名称、首页 slogan「让数据成为决策的力量」和中文使用指南。使用指南包含五个项目 skills、阶段流程、可复制示例和任务接续；点击侧栏 **DecisionForge 使用指南** 或首页 Logo 打开。

这是通过 `dsh.bundle` 注册的原生插件包 `@decisionforge/dsh-plugin`。安装后的包在当前 dsh profile 中持久保留；本页的开关控制品牌与指南，移除按钮卸载插件。若界面提示需要重启，请重启同一个 profile。

## 使用

1. 在原生模型设置中配置 Agent 模型。
2. 使用项目的 `npm start --prefix extensions/dsh-decisionforge` 启动，会自动添加并选择当前 checkout；直接运行 `dsh web` 时手动选择完整 DecisionForge 仓库。准备 Python 环境。
3. 返回对话，输入 `/dforge` 加任务描述，或显式调用 `/dforge-data`、`/dforge-train`、`/dforge-eval`、`/dforge-analyze`。

插件提供品牌、指南及项目启动时的默认工作区选择（之后可手动切换）。skills 与业务命令来自你选择的 DecisionForge 仓库；安装本插件不会全局复制 skills、自动执行任务或配置模型密钥。

## 分享与安装

把构建后的 `.tgz` 文件交给使用者，在其 dsh 的 **插件 → 添加插件** 中填入该文件的绝对路径，或运行：

```bash
dsh plugin --profile web add /absolute/path/decisionforge-dsh-plugin-0.3.0.tgz
dsh web
```

要求 Node >=24.9.0、pnpm 可用；兼容版本固定为 dsh 0.2.2-rc.1。当前未发布到 npm，不能直接按包名从公共 registry 安装。安装和启动须使用同一 `DSH_HOME` 与 profile。其他 profile 需使用自己的名称安装，且应具备 Web UI 的原生 slots。

停用恢复官方品牌并移除指南；不删除聊天记录或 DecisionForge 任务。浏览器标题和 favicon 仍由上游提供。

[DecisionForge 项目](https://github.com/Blackteaxx/DecisionForge) · [完整接入教程](https://github.com/Blackteaxx/DecisionForge/blob/main/docs/zh/agent.md)
