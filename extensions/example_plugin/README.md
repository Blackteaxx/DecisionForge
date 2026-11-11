# 可安装的评测扩展

从仓库根目录执行：

```bash
python -m pip install -e extensions/example_plugin
dforge eval-list
dforge eval --config extensions/example_plugin/config.yaml --validate-only
```

distribution 名称为 `dforge-eval-example`。`jev_eval_example/benchmark.py` 定义 benchmark，`metrics.py` 定义 group metric；`__init__.py` 保持 entry-point exports，pyproject.toml 注册 `custom_groups` 和 `unanimous_groups`。accuracy 复用已有指标，group 评分由本扩展拥有。

配置的数据路径 `../../examples/eval/samples.jsonl` 和输出路径 `../../outputs/eval/custom-groups` 均相对 config.yaml。安装后可以从任意工作目录使用配置的绝对路径验证。validate-only 不需要服务或密钥，不调用模型。

真实推理沿用标准参数：

```bash
dforge eval --config extensions/example_plugin/config.yaml \
  --env-file .env.eval --run-dir outputs/eval/custom-groups/example-run
```

复制该包并修改 distribution、模块、benchmark/metric 名称和评分函数即可开发自己的插件。保留规范 Sample、原始预测和失败分母语义。详见 [扩展开发指南](../../docs/zh/extending.md)。
