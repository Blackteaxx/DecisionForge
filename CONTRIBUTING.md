# 贡献指南

[English](CONTRIBUTING.en.md)

先通过 issue 描述可复现的问题或建议；小修复可以直接提交 PR。提交前阅读对应的 [中文指南](docs/zh/extending.md)。

```bash
uv sync --locked --group dev
uv run ruff check
uv run pytest
uv run python tests/acceptance/check_public.py
```

插件修改还需 Node >=24.9 与 npm：

```bash
npm ci --prefix extensions/dsh-decisionforge
npm run build --prefix extensions/dsh-decisionforge
npm test --prefix extensions/dsh-decisionforge
```

为行为变化添加有意义的回归验证，保留 JEV 数据语义、来源隔离、检查点恢复和离线命令的凭据边界。可选依赖按命令延迟加载。GPU 测试与付费模型调用需在事先确认的资源和预算中运行；离线通过不代表 GPU 路线已经验收。

PR 说明问题、最终行为、测试结果与限制。中英文文档同步更新；品牌资产只从 `docs/assets/brand/` 生成。不要提交凭据、私有地址、模型权重、完整个人会话或本机路径。使用合成测试数据或明确许可的最小夹具，并保留第三方来源和许可证。

贡献代码按 [Apache-2.0](LICENSE) 提供；第三方代码、模型和数据保留各自许可证。发现凭据泄漏时，先撤销/轮换凭据，再私下联系维护者；不要把值贴到公开 issue。
