# DecisionForge Demo

自包含静态页面，展示真实的医疗分诊与材料性能预测任务，支持中英切换、浅色/深色主题和会话回放。回放消息保留原语言，已按当前命令名替换，并隐藏内部路径、凭据和人员信息。

```bash
uv run python -m http.server 8000 --directory demo
# http://localhost:8000/
uv run python demo/scripts/bundle.py
# demo/dist/decisionforge-demo.html：离线单文件，不提交
```

页面加载与交互均不请求外部资源。GitHub 和快速开始是用户点击后打开的文档链接。品牌资产由 `docs/assets/brand/` 复制到 `demo/assets/brand/`，更新品牌时同时复制这些文件。

重新生成真实回放需要原始会话导出目录（不随公开仓库分发）：

```bash
uv run python demo/scripts/build_trajectories.py /path/to/trajectory-export
```

`curation.py` 记录编辑规则和明确的旧→新命令替换表；生成器对全部字符串进行脱敏和迁移。性能数字来自原始任务结果，本次品牌迁移不改动这些数字。

`.github/workflows/pages.yml` 支持 main 推送和手动发布。预期地址为 <https://blackteaxx.github.io/DecisionForge/>。首次发布前需在仓库 Settings → Pages 中将来源设为 GitHub Actions；非默认分支部署还需允许对应的 Pages environment 分支。
