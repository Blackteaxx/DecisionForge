# DecisionForge plugin

A native DeepSeek Harness Web UI plugin providing DecisionForge branding, the homepage slogan “让数据成为决策的力量” (Turn data into the power of decisions), and a Chinese guide for five project skills, stage planning, examples and task continuation.

The package registers a `dsh.bundle` and appears in Plugins as **DecisionForge**. Enable, disable or remove it through the native manager. Follow any restart prompt. Disabling restores the official branding and removes the guide while retaining chat history and DecisionForge task files.

The project launcher (`npm start --prefix extensions/dsh-decisionforge`) automatically registers and selects its current checkout. You can switch workspaces afterwards. When running standalone `dsh web`, select a complete DecisionForge repository as your workspace. Then configure an Agent model and prepare Python, then use `/dforge` or the stage skills in chat. This plugin does not install global skills, copy their instructions, configure credentials or start business tasks.

To install a shared build, enter the absolute `.tgz` path in **Plugins → Add plugin**, or run:

```bash
dsh plugin --profile web add /absolute/path/decisionforge-dsh-plugin-0.3.0.tgz
dsh web
```

Requires Node >=24.9.0 and pnpm. Tested against dsh 0.2.2-rc.1 only. Use the same profile and `DSH_HOME` for installation and startup. The package has not been published to npm; install a built tarball rather than its registry name. Browser title and favicon remain upstream-owned.

[DecisionForge repository](https://github.com/Blackteaxx/DecisionForge) · [Full setup guide](https://github.com/Blackteaxx/DecisionForge/blob/main/docs/zh/agent.md)
