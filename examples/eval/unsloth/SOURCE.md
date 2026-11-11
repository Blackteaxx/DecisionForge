# Local evaluation sample

100 questions from the original 20 test cases included in the training example (5 per case). Source: [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/tree/e039ebffcc280174dd354227424fb2b249f191de), revision `e039ebffcc280174dd354227424fb2b249f191de`, Apache-2.0. License: [LICENSE](../../training/data/LICENSE). Conversion uses the pinned repository benchmark converter, retaining original teacher gold labels, soft distributions and question names in metadata.

The request contains exactly one decision question, matching benchmark version `1-single-question`. Targets and metadata are retained solely for scoring and never sent into the local model worker.
