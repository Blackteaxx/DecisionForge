"""SDK calls shared by the training worker and the persistent local eval worker.

Importing this module does not import torch/ms-swift. CPU encoding is available
without CUDA; model allocation is confined to Runtime.
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path

from decisionforge.training.backends.swift_task import (
    PROBABILITY_CONTRACT,
    SWIFT_REVISION,
    map_request,
    readout,
)


def installation():
    dist = importlib.metadata.distribution("ms-swift")
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    revision = direct.get("vcs_info", {}).get("commit_id")
    if revision != SWIFT_REVISION and SWIFT_REVISION not in direct.get("url", ""):
        raise ValueError(f"ms-swift must be installed from pinned revision {SWIFT_REVISION}")
    return {"ms_swift": dist.version, "revision": SWIFT_REVISION, "direct_url": direct}


def adapter_dir(settings):
    if settings.init_checkpoint:
        from decisionforge.training.backends.swift import validate_artifact

        return validate_artifact(settings.init_checkpoint)
    from huggingface_hub import snapshot_download

    from decisionforge.training.backends.swift import validate_artifact

    return validate_artifact(
        snapshot_download(settings.adapter, revision=settings.adapter_revision)
    )


def head_environment(adapter):
    adapter = Path(adapter)
    # Trained artifacts retain these original loader files, then overlay scoring_head.*.
    for env, file in {
        "OMNIJEV_HEAD": "head.pt",
        "OMNIJEV_ORD": "ord.pt",
        "OMNIJEV_HEAD_META": "head_meta.json",
        "OMNIJEV_NEW_TOK_EMB": "new_tok_emb.pt",
    }.items():
        path = adapter / file
        if not path.is_file():
            raise ValueError(f"Missing OmniJev loader artifact: {path}")
        os.environ[env] = str(path)
    os.environ["MSO_BRANCH_CKPT"] = "1"
    os.environ["USE_HF"] = "1"
    if os.environ.get("MSO_NO_PANELS", "").lower() in {"1", "true", "yes", "on"}:
        raise ValueError("MSO_NO_PANELS would discard public images")


def cpu_template(config):
    installation()
    from swift.model import get_model_processor
    from swift.template import get_template

    from decisionforge.training.backends.swift_routes import get_route
    from decisionforge.training.backends.swift_routes.sdk import (
        metadata_snapshot,
        register_templates,
    )

    route = get_route(config.swift.model_type)
    register_templates()
    snapshot, _ = metadata_snapshot(route)

    _, processor = get_model_processor(
        str(snapshot),
        model_type=route.name,
        task_type="decision",
        load_model=False,
        download_model=False,
        use_hf=True,
        device_map="cpu",
        model_kwargs={"local_files_only": True},
    )
    limit = processor.model_info.max_model_len
    if limit is not None and config.model.max_seq_length > limit:
        raise ValueError(f"max_seq_length exceeds the pinned backbone context limit {limit}")
    if route.name == "omnijev":
        processor.tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<|opt|>", "<|/opt|>"]}
        )
    template = get_template(
        processor,
        template_type=route.template,
        max_length=config.model.max_seq_length,
        truncation_strategy="raise",
        remove_unused_columns=False,
        # Preserve OmniJev's native composed-panel/video resolution. The new
        # image routes enforce their own raw and encoded pixel limits.
        max_pixels=None if route.name == "omnijev" else config.swift.max_pixels,
    )
    template.set_mode("train")
    return template


def encode_check(config, rows):
    from concurrent.futures import ThreadPoolExecutor

    from decisionforge.training.backends.swift_routes.sdk import preprocess

    template = cpu_template(config)

    def check(item):
        i, row = item
        normalized = preprocess(config.swift.model_type, row)
        try:
            enc = template.encode(normalized)
        except Exception as exc:
            raise ValueError(
                f"{config.swift.model_type} encoding failed at row {i}; no truncation or deletion allowed: {exc}"
            ) from exc
        length = len(enc["input_ids"])
        if length > config.model.max_seq_length:
            raise ValueError(f"Row {i} exceeds max_seq_length; refusing truncation")
        return length, len(enc["decision_meta"]["kinds"])

    # ms-swift's own inference engine uses concurrent template encoding as well.
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as executor:
        results = list(executor.map(check, enumerate(rows)))
    lengths = [length for length, _ in results]
    return {
        "questions": sum(count for _, count in results),
        "rows": len(lengths),
        "max_tokens": max(lengths, default=0),
        "max_seq_length": config.model.max_seq_length,
        "mode": "exact_cpu_swift_template_no_model_weights",
    }


class Runtime:
    def __new__(cls, config=None, adapter=None, *, device="cuda:0", chat=False):
        if (
            cls is Runtime
            and config is not None
            and not chat
            and config.swift.model_type != "omnijev"
        ):
            from decisionforge.training.backends.swift_routes.runtime import RouteRuntime

            return object.__new__(RouteRuntime)
        return object.__new__(cls)

    def __init__(self, config, adapter, *, device="cuda:0", chat=False):
        self.config, self.chat = config, chat
        source = installation()
        import torch
        from swift.infer_engine import TransformersEngine

        if not torch.cuda.is_available():
            raise ValueError("swift requires an explicitly assigned BF16 CUDA device")
        self.torch = torch
        torch.cuda.set_device(device)
        if not torch.cuda.is_bf16_supported() or int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError("swift requires one BF16-capable CUDA device without external DDP")
        if not chat:
            head_environment(adapter)
        start = time.monotonic()
        self.engine = TransformersEngine(
            config.model.name,
            revision=config.model.revision,
            model_type="qwen3_5" if chat else "omnijev",
            task_type="causal_lm" if chat else "decision",
            adapters=[] if chat else [str(adapter)],
            torch_dtype=torch.bfloat16,
            attn_impl="eager",
            device_map=device,
            use_hf=True,
        )
        self.engine.template.max_length = config.model.max_seq_length
        self.engine.template.truncation_strategy = "raise"
        self.engine.template.remove_unused_columns = False
        if chat:
            self.engine.template.enable_thinking = False
        self.metadata = {
            "model": config.model.name if chat else str(adapter),
            "model_identity_source": "loaded_swift_checkpoint",
            "backbone": {"name": config.model.name, "revision": config.model.revision},
            "checkpoint": None
            if chat
            else {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in Path(adapter).iterdir()
                if p.name
                in {
                    "adapter_model.safetensors",
                    "scoring_head.safetensors",
                    "scoring_head.json",
                    "head_meta.json",
                    "new_tok_emb.pt",
                }
            },
            "calibration": None
            if chat
            else json.loads(
                (
                    Path(adapter)
                    / (
                        "scoring_head.json"
                        if (Path(adapter) / "scoring_head.json").is_file()
                        else "head_meta.json"
                    )
                ).read_text()
            ),
            "generation": {
                "enable_thinking": False,
                "max_tokens": 4096,
                "temperature": 0,
                "response_decoding": "native_generated_token_ids_without_template_prefix",
            }
            if chat
            else None,
            "model_revision": config.model.revision,
            "adapter_revision": None if chat else config.swift.adapter_revision,
            "upstream": source,
            "probability_contract": PROBABILITY_CONTRACT,
            "runtime": {
                "device": device,
                "gpu": torch.cuda.get_device_name(device),
                "cuda": torch.version.cuda,
                "precision": "bfloat16" if chat else "bfloat16_backbone_float32_decision_head",
                "load_seconds": time.monotonic() - start,
            },
            "dependencies": {
                k: importlib.metadata.version(k)
                for k in ("torch", "transformers", "peft", "datasets", "ms-swift")
            },
        }

    @contextlib.contextmanager
    def temperature(self, calibrated=True):
        if self.chat or calibrated:
            yield
            return
        from swift.model.decision_head import get_scoring_head

        head = get_scoring_head(self.engine.model)
        ht, ts = head.head_temperature.clone(), head.temps.clone()
        try:
            head.head_temperature.fill_(1)
            head.temps.fill_(1)
            yield
        finally:
            head.head_temperature.copy_(ht)
            head.temps.copy_(ts)

    def predict(self, public, *, calibrated=True):
        from swift.dataset.preprocessor.decision import OmniJevPreprocessor
        from swift.infer_engine import RequestConfig

        rows, mapping = map_request(public, self.config.swift)
        if self.chat:
            start = time.monotonic()
            self.torch.cuda.reset_peak_memory_stats()
            raw = self.predict_chat(public, rows)
            self.torch.cuda.synchronize()
            raw["swift_chat"].update(
                seconds=time.monotonic() - start,
                peak_memory_bytes=self.torch.cuda.max_memory_allocated(),
            )
            return raw
        answers, raw = {}, []
        start = time.monotonic()
        self.torch.cuda.reset_peak_memory_stats()
        with self.temperature(calibrated):
            for row, item in zip(rows, mapping):
                normalized = OmniJevPreprocessor().preprocess(row)[0]
                response = self.engine.infer([normalized], RequestConfig(), use_tqdm=False)[0]
                value = response.choices[0].message.content
                # Decision response content is the native structured prediction list.
                value = json.loads(value) if isinstance(value, str) else value
                if not isinstance(value, list) or len(value) != 1:
                    raise ValueError("Unexpected ms-swift decision response shape")
                probs = value[0]["probs"]
                entry = {**item, "native": value, "abstain_probability": None}
                raw.append(entry)
                try:
                    answer, abstain = readout(public["questions"][item["question_id"]], probs)
                    answers[item["question_id"]] = answer
                    entry["abstain_probability"] = abstain
                except (ValueError, TypeError) as exc:
                    entry["validation_error"] = str(exc)
        self.torch.cuda.synchronize()
        return {
            "answers": answers,
            "swift": {
                "columns": raw,
                "probability_contract": PROBABILITY_CONTRACT,
                "calibrated": calibrated,
                "seconds": time.monotonic() - start,
                "peak_memory_bytes": self.torch.cuda.max_memory_allocated(),
            },
        }

    def predict_chat(self, public, rows):
        from swift.infer_engine import RequestConfig

        from decisionforge.eval.models import Question
        from decisionforge.training_free.chat_baseline.prompt import messages

        question = Question.model_validate(public["questions"]["decision"])
        prompt = messages(public, question)
        # Reuse the same rendered context and all images as decision inference.
        row = rows[0]
        if "video" in row:
            from swift.model.omnijev_media import video_content

            images, content = video_content(row["images"][0], row["video"], prompt[-1]["content"])
            prompt[-1]["content"] = content
        else:
            images = row["images"]
            prompt[-1]["content"] = "<image>" * len(images) + prompt[-1]["content"]
        response = self.engine.infer(
            [{"messages": prompt, "images": images}],
            RequestConfig(max_tokens=4096, temperature=0, return_details=True),
            use_tqdm=False,
        )[0]
        choice = response.choices[0]
        if choice.token_ids is None:
            raise ValueError("ms-swift did not return requested generated token IDs")
        # Swift's default decoder prepends the assistant prefix from the prompt
        # (including the empty non-thinking block). Decode the actual generated
        # IDs with its native codec; preserve generated text without JSON repair.
        content = self.engine.template.decode_generate_ids(choice.token_ids, first_token=False)
        # Retain even malformed/truncated generations. The parent parser validates
        # them after journaling the complete raw output, just like chat_api.
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": choice.finish_reason,
                }
            ],
            "swift_chat": {
                "sdk_response_content": choice.message.content,
                "generated_token_ids": choice.token_ids,
                "response_decoding": "native_generated_token_ids_without_template_prefix",
                "abstain_probability": 0.0,
                "probability_contract": PROBABILITY_CONTRACT,
            },
        }

    def close(self):
        self.engine = None
        gc.collect()
        self.torch.cuda.empty_cache()
