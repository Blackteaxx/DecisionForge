---
license: cc-by-nc-4.0
language:
- en
tags:
- decision-model
- typed-decisions
- classification
- multimodal
- reasoning
---

# StartLux-Decision-v1.1-4B-AutoThink

StartLux-Decision-v1.1-4B-AutoThink is a 4B decision model that reasons only where it is unsure. You give it a state (text or JSON, with up to 8 images) and typed questions about it: pick one of several options (`choice`), yes or no (`noul`), or a level on a scale of 2 to 10 (`score`). It reads a probability for every option in one pass, and when the top probability is below the gate, 0.6 by default, it writes a short thought, reads the answer again and mixes the two read-outs. The request and response format is compatible with the System One `/v1/systemone` API.

Formats: **BF16** · [GGUF](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-4B-AutoThink-GGUF) · [MLX 8-bit](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-4B-AutoThink-MLX-8bit) · [MLX 4-bit](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-4B-AutoThink-MLX-4bit) · Code: [StartLuxLabs/StartLux-Decision](https://github.com/StartLuxLabs/StartLux-Decision) · Paper: [arXiv 2610.17309](https://arxiv.org/abs/2610.17309)

Family: [0.8B Instant](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-0.8B-Instant) · [2B Instant](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-2B-Instant) · [4B Instant](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-4B-Instant) · **4B AutoThink** · [9B Instant](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-9B-Instant) · [9B AutoThink](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-9B-AutoThink) · [27B Instant](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-27B-Instant) · [27B AutoThink](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-27B-AutoThink)

<p align="center"><img src="https://huggingface.co/startlux-models/StartLux-Decision-v1.1-4B-AutoThink/resolve/main/figures/score_vs_size.png" width="860" alt="Decision Index 0.3 public score against model size"></p>

<p align="center"><em>Decision Index 0.3, public subset, as of 11 October 2026: the StartLux-Decision Instant models against other decision models of each size. Figure 1 of the <a href="https://arxiv.org/abs/2610.17309">technical report</a>.</em></p>

## How it answers

Each question is read as its own prompt that holds the whole state. The model first reads the logits of the option letters at the last prompt position, divided by a temperature for the question type and normalised over the listed options. If the top probability reaches the gate (0.6 by default, 0.9 in the results below), that read-out is the answer. Otherwise the same weights write a thought of at most 4,096 tokens (temperature 0.6, top-p 0.95, top-k 20), read the letters again after it, and answer 0.25 × the first read-out + 0.75 × the second. The gate applies to each question on its own, and the server closes a thought that reaches the limit or starts to loop.

Stages 1 and 2 used the data of the Instant models but kept causal attention. Stage 3 trained the thought with supervised reasoning examples and then with RLCD², reinforcement learning for calibrated deliberative decisions.

## Results

| Model | Decision Index 0.3, public | DecisionPivot, pair accuracy |
|---|---|---|
| 27B AutoThink | 73.37 | 86.9 |
| 27B Instant | 72.83 | 75.4 |
| 9B AutoThink | 70.45 | 59.9 |
| 9B Instant | 69.96 | 47.5 |
| **4B AutoThink (this model)** | **68.21** | **59.9** |
| 4B Instant | 67.88 | 41.5 |
| 2B Instant | 59.14 | 21.1 |
| 0.8B Instant | 53.75 | 8.1 |

Decision Index 0.3 scores are our own runs of the public kit on its public subset, not leaderboard submissions. We compared training recipes and seed averages by these scores, so they reflect model selection and are not independent test results. At every size the Instant model leads the strongest evaluated decision model of comparable scale by 7.4 to 16.9 points. [DecisionPivot](https://huggingface.co/datasets/startlux-models/DecisionPivot) was not used for any data, checkpoint or configuration choice. Its core track has 7,800 program-generated examples in 13 families, mostly pairs that differ in one answer-changing detail, and a pair counts only when both examples are right. AutoThink rows use a gate of 0.9.

## Speed

Median latency of one read-out in milliseconds, one request in flight, by prompt length. vLLM 0.31 on one NVIDIA H200, MLX on an Apple M5 Pro with 64 GB of memory (a dash: does not fit in memory).

| Engine and precision | 256 tokens | 1,024 | 4,096 | 8,192 |
|---|---|---|---|---|
| vLLM BF16 | 29 | 39 | 70 | 131 |
| vLLM FP8 | 30 | 38 | 56 | 105 |
| MLX BF16 | 86 | 587 | 2,257 | 4,800 |
| MLX 8-bit | 94 | 466 | 2,339 | 4,875 |
| MLX 4-bit | 88 | 527 | 2,275 | 4,730 |

A question that thinks adds the time of its thought and of the second read-out. Questions that do not think cost one read-out.

## Quick start

```bash
git clone https://github.com/StartLuxLabs/StartLux-Decision.git
cd StartLux-Decision
pip install -e '.[vllm]'      # vLLM 0.19.1, the version of the published scores; vLLM 0.31 is also supported
hf download startlux-models/StartLux-Decision-v1.1-4B-AutoThink --local-dir models/StartLux-Decision-v1.1-4B-AutoThink
startlux-decide serve --model models/StartLux-Decision-v1.1-4B-AutoThink --launch-vllm --gpus 0 --port 8090
```

Images and prompts over 65,536 tokens need vLLM 0.31. With several GPUs, `--gpus 0,1,2,3` starts one vLLM server per GPU behind the same decision server.

```bash
curl -s localhost:8090/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": {"ticket": "I was charged twice for order #4411 and the app still shows it as unpaid."},
  "questions": {
    "team":     {"type": "choice", "instructions": "Which team should handle this ticket?",
                 "criteria": {"billing": "Payments and refunds", "shipping": "Delivery", "technical": "App and login"}},
    "urgent":   {"type": "noul", "instructions": "Should this ticket be answered today?"},
    "severity": {"type": "score", "instructions": "How severe is the impact?",
                 "criteria": ["cosmetic", "annoying", "blocks the customer"]}
  }
}'
```

The response has this shape:

```text
{"answers": {
   "team":     {"type": "choice", "choice": <id>, "confidence": <c>,
                "probabilities": {"billing": <p>, "shipping": <p>, "technical": <p>},
                "none_probability": <p>},
   "urgent":   {"type": "noul", "noul": <P(yes)>},
   "severity": {"type": "score", "score": <expected level>, "confidence": <c>,
                "legend": {"0": "cosmetic", "1": "annoying", "2": "blocks the customer"},
                "probabilities": {"0": <p>, "1": <p>, "2": <p>}}},
 "usage": {"input_tokens": <n>, "output_tokens": <thought tokens>},
 "model": "StartLux-Decision-v1.1-4B-AutoThink"}
```

A request can set `think_mode` (`auto`, `never` or `always`), `gate`, `think_budget` and `mix`, the weight of the read-out after the thought. `return_details` adds both read-outs, the thought's length and how it ended, and `return_thought` the thought text.

Every choice answer carries `none_probability`, the share of the read-out that goes to an extra "None of the above" option the server adds internally. `choice` stays one of your options, and `probabilities` covers only your options. With `"allow_none": true`, `choice` is null when "None of the above" is the most likely option. [docs/none_probability.md](https://github.com/StartLuxLabs/StartLux-Decision/blob/main/docs/none_probability.md) explains how to read it and why we suggest acting on it from 0.2. [docs/api.md](https://github.com/StartLuxLabs/StartLux-Decision/blob/main/docs/api.md) has the full format, including images (base64 strings or data URIs, at most 8 per request).

## Engines and formats

The model reads its prompt causally, so it runs on vLLM, transformers, llama.cpp and MLX through the package. The option-letter read-out, the gate and the thought loop are in the package, and plain chat with the weights does not give these decisions.

| Build | Decision Index 0.3 | DecisionPivot, chance-corrected |
|---|---|---|
| BF16, vLLM | 68.11 | 65.3 |
| FP8 at load (`--quantization fp8`), vLLM | 68.09 | 65.6 |
| GGUF Q8_0, llama.cpp | | 65.6 |
| GGUF Q4_K_M, llama.cpp | | 63.5 |

These are runs of the served engines, so the BF16 row differs slightly from Results. The DecisionPivot column here is the mean chance-corrected accuracy over the benchmark's seven scoring layers, not pair accuracy. Q8_0 is the recommended GGUF file. MLX 8-bit builds agree with BF16 on the top option for 99.5% to 99.6% of questions, 4-bit builds for 96.3% to 97.5%.

## Limitations

One read covers at most 26 options. Longer choice lists go through group rounds (groups of at most 25, the three best of each to a final read, the rest sharing 0.001 of the probability). `none_probability` is an alarm score, not a calibrated probability, and it is weak where no single answer is right (preferences, moves in a game) and on questions the model cannot answer itself. A confidence used to select or route answers overstates their accuracy, so set thresholds on the answers your own policy returns. Thoughts are sampled, so a question that thinks can get a different answer on another run. Callers that never need a thought are better served by the Instant model of the same size.

## License

The weights are released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/): free for research and other non-commercial use, with attribution. Commercial use requires a separate license from StartLux Labs (contact@startlux.com). The inference package is Apache-2.0. See [LICENSE](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-4B-AutoThink/blob/main/LICENSE) and [NOTICE](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-4B-AutoThink/blob/main/NOTICE).

## Citation

```bibtex
@misc{startlux2026decision,
  title         = {StartLux-Decision: AutoThink Decision Models with Reinforcement Learning for Calibrated Deliberative Decisions},
  author        = {Zhuang, Xialie and Su, Zunhai and Pan, Jia and Wong, Ngai and Chen, Guanhua and Kuo, Chuan-Wei},
  year          = {2026},
  eprint        = {2610.17309},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2610.17309}
}
```
