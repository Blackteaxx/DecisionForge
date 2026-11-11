# 验证方式

在仓库根目录运行 `uv run python .agents/skills/validation/check_package.py --offline-cli` 检查 Skills、共享模板、相对软链接与离线命令。

这不替代实际模型调用、GPU 训练或跨会话接续验收。任务执行仍按各阶段 Skills 保存配置、授权、进程和产物事实。
