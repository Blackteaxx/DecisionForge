# 内置 benchmark 贡献

每个公开子包拥有 benchmark、评分规则和相关 metric；`_classification.py`、`_intent.py`、`_registration.py` 是私有复用实现。`generic`、`jevbench`、`typed_decisions`、`banking77`、`clinc150_oos`、`demo` 是当前实现。

新增 `my_benchmark/benchmark.py`，继承 `decisionforge.eval.BaseBenchmark`，并添加 `my_benchmark/__init__.py`：

```python
def register(registry):
    from .benchmark import MyBenchmark
    from .metrics import my_metric

    registry.register_metric(my_metric())
    registry.register_benchmark(MyBenchmark())
```

`load_registry()` 按稳定名称顺序自动发现已安装的公开子包并调用 hook，不需要修改核心类列表。不要依赖另一个子包的注册顺序；可以显式导入复用的实现。Metric 源码归 benchmark 拥有，注册名称仍全局唯一，现有别名继续兼容；新名称应避免碰撞。

新增实现须维持题型、参考来源、失败分母、原始响应保留和评分版本语义，并覆盖离线评分测试。数据准备放在 `data/benchmarks/<name>/`，评分代码不能依赖仓库数据脚本。外部 distribution 用 entry points，见 [extensions](../../../extensions/README.md) 和 [API 指南](../../../docs/zh/extending.md)。

逐步实现和验证流程见 [开发者指南](../../../docs/zh/extending.md)。
