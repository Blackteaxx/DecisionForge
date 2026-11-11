# Typed Decisions 小样本

来源：[LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/tree/e039ebffcc280174dd354227424fb2b249f191de)，`all` 配置，固定 revision `e039ebffcc280174dd354227424fb2b249f191de`。源数据卡声明 Apache-2.0；随附 [许可证](LICENSE)。

`typed-decisions-sample.jsonl` 是派生的 DecisionForge v3 训练文件。按 workflow 名称排序轮流选择，保留各 workflow 的源行顺序，取 100 个原 train 案例与 20 个原 test 案例，共 120 records / 600 questions。四个 workflow 为 agent_trace_observability、customer_service、invoice_processing、security_incidents。原始五问题请求和结构化 state 保留，原 train/test 归属保留。

修改：记录/group ID 映射为唯一正整数；Choice/Score 完整软分布在六位小数误差不超过 1e-5 时显式归一化；Choice 的派生选择、Score 的概率期望及两者的 JEV confidence 重算为 v3 合法值；Noul 保留原 p_true。参考是单教师 temperature=0.7 下三次采样的平均概率，不是人工标签。

逐条原 case ID、workflow 和 split 对应关系见 `typed-decisions-sample.source.json`。`public-request.json` 是第一条样本的公开 request，不含监督。

重新准备相同选择：

```bash
bash examples/training/prepare_data.sh --train-limit 100 --test-limit 20
# 输出位于 data/local/，保留仓库随附样本不动。
```
