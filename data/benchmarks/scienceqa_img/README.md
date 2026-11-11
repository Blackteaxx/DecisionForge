# ScienceQA-IMG

ScienceQA's image subset is the public图文 choice benchmark used by the experimental
ms-swift route. The Hugging Face dataset revision is pinned to
`f18b0a70359ebfb41f658fd564208d0355b013f4`; the upstream project revision is pinned to
`2cbf8318e07b9ece895bb2ae605e71e38d623264`. The dataset declares CC-BY-NC-SA-4.0;
the upstream code is MIT. Only rows with one image are prepared, and every prepared
image is copied to the cache and hashed. Train/validation/test are kept separate.

```bash
python data/benchmarks/scienceqa_img/prepare.py
python data/benchmarks/scienceqa_img/prepare.py --fetch-only
python data/benchmarks/scienceqa_img/prepare.py --convert-only --data-dir /path/to/cache
```

The output is `processed/<split>.jsonl` with `decisionforge.eval.v1` requests. The image is
referenced by a relocatable local path and SHA-256, never silently dropped. The standard
metrics are accuracy, ECE, Brier and NLL; the choice probability contract conditions the
OmniJev K+1 output on the K ordinary options and reports `abstain_probability` separately.
