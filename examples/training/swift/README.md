# ms-swift decision routes (experimental)

## Registered decision routes (issue #29)

`backend: swift` now selects an explicit route registry. The fixed SDK remains
`680df5c670d7e8add906ac49381d603869410612`; install it in the separate environment
from `uv sync --extra train-swift`. The installed API was checked with ms-swift
4.7.0.dev0 from that commit, Transformers 5.17.0 and CPU Torch 2.11.0.

| `swift.model_type` | Fixed starting checkpoint | Public input / limits | Training mode / status |
| --- | --- | --- | --- |
| `omnijev` | Qwen3.5-4B + published OmniJev adapter | C≤255, N, S2–10; verified images or offline video mosaic | Continue published LoRA; **real training accepted in #28** |
| `clef` | Cloudflare/clef, merged 27B | C≤255, N, S2–10; ≤32 images; all questions in one forward | New LoRA on published model or full; **code + CPU only** |
| `kev` | Qwen3.5-0.8B-Base + jaredpalmer/kev-0.8b | C≤255, N, S2–10; **text only** | Continue published LoRA; **code + CPU only** |
| `jev` | autotrust/JEV-27B-VL + `adapter_vllm` | C≤16, N, S**exactly6**; ≤32 images | Continue published LoRA; **code + CPU only** |
| `startlux` | StartLux-Decision-v1.1-4B-AutoThink, merged | C≤26 total columns (≤25 ordinary options with `allow_none`), N, S2–10; ≤8 images | New LoRA on published model or full; **noncommercial research, code + CPU only** |
| `thx02` | Unverified | Not enabled | Official HF source401 / ModelScope404; no verified license/revision |

C/N/S denote choice/noul/score. All enabled routes reject unsupported input before
model allocation, including options/images/pixels beyond configured caps, videos
outside OmniJev's contract, reserved template markers and excessive sequence length.
New image routes cap each raw **and processor-expanded** image at 602,112 pixels by
default (the configured cap may be lower). Examples set 4096 tokens; complete records
must fit that limit and the pinned model context. No option subsampling, packing,
padding-free execution, silent state truncation or question deletion is enabled.
`laya` remains its existing separate backend; no from-scratch route is exposed.

### Sources and permitted use

[model-sources.json](model-sources.json) fixes every enabled checkpoint/revision,
head file, environment variable and license. `sources/` retains the pinned cards,
licenses and configuration evidence; OmniJev's existing `model-source.json` remains.
Clef, Kev and JEV sources declare Apache-2.0. StartLux weights are **CC-BY-NC-4.0**;
this integration is explicitly restricted to **noncommercial research** and both
training/eval require `usage: noncommercial_research`. Commercial use requires
separate authorization from the rightsholder. Export retains its license and notice.
THX-02 is rejected by typed configuration until its source and applicable license
can be verified; there is deliberately no runnable example or fabricated proof.

The actual pinned Clef/JEV cards identify Qwen3.8-27B with
`Qwen3_5ForConditionalGeneration` / `model_type=qwen3_5` configs. Older Qwen3-VL
wording in the issue background does not override these source snapshots.

### Conversion, probabilities and calibration

Clef preserves joint questions and uses the native lexical choice-column order;
the reversible mapping restores original v3 option keys. Its fixed native template
rejects images and clips long states, so `dforge_clef` retains the native joint
text rendering and ports Qwen image expansion before all question/option spans,
shifting those spans by the exact inserted token count. CPU tests decode the spans
and exercise real pixel tensors and native RoPE/collation. `dforge_kev` preserves
the native pointer layout but rejects excessive state length instead of clipping.
JEV and StartLux retain their native prompts with an additional encoded-pixel guard.

Kev/JEV use native false,true noul columns; StartLux uses **true,false**. Score
columns map to original ordinal indices. JEV's published head uses six fixed score
verbalizers, so other rubric sizes fail preflight; descriptions are retained in
instructions. StartLux `allow_none: true` appends exactly one zero target column;
readout retains raw none mass, then conditions onto the original K options.
All-none, invalid widths, nonfinite/out-of-range values or unnormalized native
vectors are errors. Every local result retains raw native records and the route
probability contract; `swift_local` independently rechecks its common answer.
This uses StartLux's native decision readout; the published AutoThink service's
conditional thought generation/mixing policy is not this training/inference API.

Only OmniJev, JEV and StartLux enable independent calibration with the native Swift
callback. Clef has no native post-hoc buffer and Kev's scalar temperature is not
supported by that callback's per-kind contract: explicit calibration is rejected
for these routes. Post-hoc buffers are reset for fitting; learned head parameters
remain trainable. Test is never fitted. No extra DecisionForge temperature is applied.

### CPU verification and local evaluation

The four new YAMLs use committed synthetic C/N/S fixtures (7 records, 21 questions)
and a generated 256×256 red square. These are encoding examples, **not benchmark
results or evidence of training quality**. All fields, questions and options are
retained. Run each route without weights:

```bash
export CUDA_VISIBLE_DEVICES=''
export DFORGE_SWIFT_METADATA_CACHE=/path/to/metadata-only-cache
uv run dforge train run --config examples/training/swift/clef.yaml --validate-only
uv run dforge train run --config examples/training/swift/kev.yaml --validate-only
uv run dforge train run --config examples/training/swift/jev.yaml --validate-only
uv run dforge train run --config examples/training/swift/startlux.yaml --validate-only
python examples/training/swift/validate_routes.py --output /path/to/new-evidence-directory
python -m pytest -q
```

First use downloads only an exact allowlist of tokenizer/processor/config/license
files at the pinned revision; no tensor/checkpoint files are requested. A `.json`
suffix alone is insufficient: JEV `decision_head.json` contains learned bias and is
**excluded** from CPU metadata downloads. After populating the cache, setting
`HF_HUB_OFFLINE=1` enables offline repetition. With the explicit cache environment,
`tests/test_swift_routes_cpu.py` additionally runs native encode/collator/span,
no-truncation and pixel-limit checks with model-weight loading functions forbidden.
Base installations without the optional SDK/cache keep pure contracts testable.

`eval-{clef,kev,jev,startlux}.yaml` demonstrate the matching `swift_local` route and
all three question types. They refer to future exported model directories; changing
`model.path` to that route's fixed source ID evaluates the original published
model. These GPU commands were **not run** for #29. Training/eval check for an
assigned single BF16 CUDA device before any model-weight download. Quantization,
external DDP and optimizer-state resume are rejected.

New artifacts use `decisionforge.swift.decision.v1` with explicit model_type/finetuning,
source/adapter/subdirectory/Swift revision, parent identity and usage declaration.
They retain trained `scoring_head.*`, original loader assets and processor/notices;
LoRA requires adapter weights, full requires backbone weights and complete shard
indices. SHA-256 validation is streamed to avoid loading multi-GB files into memory.
`swift.init_checkpoint` initializes another run in the same mode and route. Every
future real run must observe backbone-or-adapter **and** head gradients/updates,
complete native calibration when enabled, and compare exported/reloaded outputs.
OmniJev keeps `decisionforge.swift.omnijev.v1` and its unchanged runtime paths.

### Resource planning and evidence boundary

These are **theoretical estimates**, not measured requirements: a 0.8B BF16 backbone
is about 1.6GB of weights; 4B about 8GB; 27B about 54GB before activations, optimizer,
head and temporary storage. Kev LoRA is a few-GB-weight route; 4B LoRA generally
needs tens of GB total; 27B LoRA warrants an 80GB-class device or careful measured
planning. Full 27B optimizer/gradient state can require several hundred GB. The
single-GPU integration does not guarantee those full-mode runs fit a given GPU.
OmniJev's actual H200 measurements remain in its separate #28 report. No real
training, GPU evaluation, GPU allocation, new accuracy or throughput measurement
was performed for the four new routes.

[CPU evidence and compatibility report](../../../docs/zh/training.md)
archives genuine CLI results, safe file inventories and replay of all 2,017 saved
OmniJev predictions. It distinguishes metadata/artifact validation and readout replay
from a new GPU forward pass.

## OmniJev: existing real-training example

This example continues the published `tinnel123/OmniJev` adapter on the pinned
`Qwen/Qwen3.5-4B` base. It does not create a new LoRA and it rejects truncation,
option subsampling, raw videos, and unverified media files. Videos must first be
converted with `swift.model.omnijev_media.video_state` into a 4x4 mosaic plus its
16 timestamps. `backend: swift` is validate-only safe on CPU; training/evaluation
requires one BF16 CUDA GPU.

Use a separate Python 3.12 environment, then prepare the fixed benchmark:

```bash
uv sync --extra train-swift
python data/benchmarks/scienceqa_img/prepare.py
uv run dforge train run --config examples/training/swift/omnijev.yaml --validate-only
uv run dforge train run --config examples/training/swift/omnijev.yaml
```

The model artifact includes adapter weights, `scoring_head.*`, original OmniJev head
files, SHA-256 lineage, upstream license notices, and the single ms-swift calibration record. Optimizer-state
resume is unsupported; start a new run from a verified exported artifact with
`swift.init_checkpoint`.

The backend keeps layer gradient checkpointing off and sets the native checkpoint kwargs
to `use_reentrant=false`. The pinned Swift trainer otherwise overrides OmniJev's branch
checkpoint and disconnects the LoRA gradient. First-update gradient checks and final
adapter/head weight-delta checks reject that failure before accepting any model.

The split recipe keeps 5,577 training, 2,097 validation, 641 calibration and all 2,017
ScienceQA-IMG test questions. Image preparation converts RGB PNG and explicitly rescales
oversize images with preserved aspect ratio; original/prepared dimensions and hashes are
recorded in the manifest. The upstream dataset license is CC-BY-NC-SA-4.0; the mirror's
more permissive tag is not used to relax the upstream terms. See `model-source.json` for
OmniJev source/license evidence and original-training-manifest limitations.

`eval-original.yaml`, `eval-continued.yaml` and `eval-chat.yaml` cover the same full test
split. Set the original adapter path to the downloaded pinned snapshot. The new
`swift_chat_local` baseline uses the pinned untrained Qwen base, thinking disabled,
temperature zero and at most 4,096 generated tokens, with strict JSON/probability parsing.
Existing `chat_api` remains text-only. Failed/illegal results are retained as failures;
no generated JSON or probability vector is repaired.

The pinned Swift decoder prepends the assistant template prefix to its returned text,
including an empty non-thinking block. Chat requests return the actual generated token
IDs and use Swift's native decoder with `first_token=false` for protocol content. Raw
results retain both SDK text and generated token IDs. Text actually generated by the
model, including thinking tags or malformed JSON, is preserved and strictly validated.

The shared probability contract appends zero abstain target mass for choice training,
then conditions OmniJev's K+1 predictions on non-abstain mass for scoring. The raw abstain
mass remains available per question. Chat emits K normalized probabilities directly
(equivalent abstain mass zero). All-abstain output is illegal. Noul native yes/no columns
map to true/false; score levels map back to their original ordinal indices.

DecisionForge uses the upstream temperature fitting code only on the independent calibration
role. The callback collects logits with shipped post-hoc temperatures reset to one,
then stores the single replacement global × relative temperature in `scoring_head.json`.
The learned `log_tau` and the original noul bias remain part of the model. DecisionForge does
not fit or apply an additional temperature. Before/after ECE, Brier and NLL are recorded;
test is never used for fitting. NLL uses a 1e-12 logarithm floor.

A new run can set `swift.init_checkpoint` to a verified exported artifact. This is
weights initialization with recorded parent lineage, not optimizer/scheduler/RNG resume;
`train --resume` deliberately rejects this backend. Merged export and distributed
FSDP/ZeRO-3 are unsupported. The adapter and native head are saved together, and export
acceptance actually reloads the model and compares probabilities and calibration.

After all real runs complete, `report.py TASK_ROOT --output NEW_REPORT_DIRECTORY` archives
the three full per-question outputs, metrics, calibration/reload evidence, configurations,
and measured resources. It rejects incomplete coverage; it does not create a placeholder
result report. The common evaluator retains raw failures in its durable journal and can
rescore without model weights or image files.

Media references are objects with `path` and lowercase hexadecimal `sha256` fields.
Relative paths resolve against the source JSON/JSONL directory; moving that directory
together with its image subdirectory preserves the references. Training/eval snapshots
record verified absolute locations plus the same byte hashes. `state` contains context,
not image placeholders. The OmniJev template inserts the image or numbered panel before
the complete state/instruction text; offline video descriptors insert timestamped frames.
Literal `<image>`, `<video>`, `<|opt|>` and `<|/opt|>` in public text are rejected so they
cannot alter the model's span or media layout. Multiple still images are composed by the
pinned upstream panel implementation; none is discarded.

For video, first call the installed `swift.model.omnijev_media.video_state(clip, mosaic)`
offline. Place the returned descriptor (`n_frames`, `cols`, `tile`, `timestamps`,
`duration`) under a single `request.videos` entry, add `mosaic: {path, sha256}` and
`source_sha256` for the original clip. Do not also provide `request.images`, a raw video
path, URL, data URI or audio. The converter verifies the 1,536×1,536 mosaic geometry,
frame metadata, local bytes and hashes before native preprocessing.

The callback's own fitting diagnostics use the native K+1 column distribution. DecisionForge's
role diagnostics and three-arm benchmark metrics use the documented conditional K-option
view. Both before/after views are kept so improved native-head calibration is not silently
presented as improved conditional calibration. Peak inference memory is PyTorch's measured
`max_memory_allocated`; native/generation QPS excludes model loading and IPC, while the
common evaluator also records complete attempt latency and session throughput.

Native intermediate checkpoints also receive `swift_lineage.json` through the save callback. Export validation requires the pinned base, adapter and ms-swift lineage; a directory with missing source fields is rejected.

The [real ScienceQA-IMG acceptance report](../../../docs/zh/training.md)
records 1,395 native AdamW updates on one H200, independent calibration and zero
export/reload probability difference. All three models cover the full 2,017-question test:
95.24% original OmniJev, 98.07% continued OmniJev and 82.60% Qwen Chat. The report retains
all raw records, including 56 strict Chat protocol failures, plus effective configs and
resource/validation evidence.
