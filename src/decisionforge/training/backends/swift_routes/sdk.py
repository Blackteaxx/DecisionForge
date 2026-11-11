"""Pinned SDK adapters; heavy imports occur only in explicit encoding/runtime calls."""

from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path

from . import get_route

# A .json suffix alone is not sufficient: JEV's decision_head.json contains
# learned bias parameters. It and every tensor/checkpoint file are excluded.
PROCESSOR_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "tokenizer.model",
    "sentencepiece.bpe.model",
    "processor_config.json",
    "preprocessor_config.json",
    "image_processor_config.json",
    "video_preprocessor_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "joint_head_config.json",
    "decision_config.json",
    "LICENSE",
    "LICENSE.txt",
    "LICENSE.md",
    "NOTICE",
    "README.md",
)


def metadata_snapshot(route):
    from huggingface_hub import snapshot_download

    cache = Path(
        os.environ.get(
            "DFORGE_SWIFT_METADATA_CACHE",
            str(Path(os.environ.get("HF_HOME", "~/.cache/huggingface")) / "dforge-metadata"),
        )
    ).expanduser()
    root = cache / route.name / route.revision
    snapshot_download(
        route.model,
        revision=route.revision,
        allow_patterns=list(PROCESSOR_FILES),
        local_dir=root,
        token=False,
    )
    files = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.iterdir()
        if p.is_file()
    }
    if not set(files) <= set(PROCESSOR_FILES):
        raise ValueError("Non-processor file in strict metadata snapshot")
    return root, files


def register_templates():
    from swift.template import register_template
    from swift.template.base import MaxLengthError
    from swift.template.register import TEMPLATE_MAPPING
    from swift.template.templates.decision import (
        ClefTemplate,
        JevTemplate,
        KevTemplate,
        StartLuxTemplate,
    )
    from swift.template.templates.qwen import QwenTemplateMeta

    if all(f"dforge_{name}" in TEMPLATE_MAPPING for name in ("clef", "kev", "jev", "startlux")):
        return

    class PixelGuard:
        def _encode(self, inputs):
            result = super()._encode(inputs)
            grids = result.get("image_grid_thw")
            if grids is not None and self.max_pixels is not None:
                patch = self.processor.image_processor.patch_size
                if any(int(grid[-2:].prod().item()) * patch**2 > self.max_pixels for grid in grids):
                    raise ValueError(
                        "Encoded image grid exceeds max_pixels after processor resizing"
                    )
            return result

    class StrictJevTemplate(PixelGuard, JevTemplate):
        pass

    class StrictStartLuxTemplate(PixelGuard, StartLuxTemplate):
        pass

    class StrictKevTemplate(KevTemplate):
        def _encode(self, inputs):
            if inputs.images or inputs.videos:
                raise ValueError("Kev supports text-only inputs")
            encoder = copy.copy(self)
            encoder.max_length = None
            result = KevTemplate._encode(encoder, inputs)
            if self.max_length is not None and len(result["input_ids"]) > self.max_length:
                raise MaxLengthError(
                    "Kev record exceeds max_length; refusing native state truncation"
                )
            return result

    class StrictClefTemplate(PixelGuard, ClefTemplate):
        """Native joint schema with a Qwen image-token expansion before all spans."""

        def _encode(self, inputs):
            if inputs.videos:
                raise ValueError("Clef supports still images, not the OmniJev video contract")
            local = copy.copy(inputs)
            local.extra_kwargs = dict(inputs.extra_kwargs)
            images = list(inputs.images or [])
            if images:
                state, _ = self._clef_fields(inputs)
                prefix = "\n".join(["<|vision_start|><|image_pad|><|vision_end|>"] * len(images))
                local.extra_kwargs["state"] = prefix + "\n" + self._clef_render(state)
            local.images = []
            local.videos = []
            encoder = copy.copy(self)
            encoder.max_length = None
            encoded = ClefTemplate._encode(encoder, local)
            if images:
                media = self.processor.image_processor(images=images, return_tensors="pt")
                grids = media["image_grid_thw"]
                token = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
                ids = encoded["input_ids"]
                positions = [i for i, value in enumerate(ids) if value == token]
                if len(positions) != len(images) or len(grids) != len(images):
                    raise ValueError("Clef image placeholders/grid count mismatch")
                merge = self.processor.image_processor.merge_size**2
                old_length = len(ids)
                ids, _, _, mm_mask = self._extend_tokens(
                    ids,
                    None,
                    None,
                    positions,
                    lambda i: [token] * int(grids[i].prod().item() // merge),
                    mm_mask=[False] * len(ids),
                )
                delta = len(ids) - old_length
                meta = encoded["decision_meta"]
                meta["question_spans"] = [(a + delta, b + delta) for a, b in meta["question_spans"]]
                meta["option_spans"] = [
                    [(a + delta, b + delta) for a, b in spans] for spans in meta["option_spans"]
                ]
                patch = self.processor.image_processor.patch_size
                if self.max_pixels is not None and any(
                    int(grid[-2:].prod().item()) * patch**2 > self.max_pixels for grid in grids
                ):
                    raise ValueError(
                        "Encoded image grid exceeds max_pixels after processor resizing"
                    )
                encoded.update(media)
                encoded["input_ids"] = ids
                if self.requires_mm_token_type_ids:
                    encoded["mm_token_type_ids"] = self.create_mm_token_type_ids(ids, mm_mask)
            if self.max_length is not None and len(encoded["input_ids"]) > self.max_length:
                raise MaxLengthError(
                    "Clef record exceeds max_length; refusing native state truncation"
                )
            return encoded

    register_template(
        QwenTemplateMeta("dforge_kev", template_cls=StrictKevTemplate, default_system=None),
        exist_ok=True,
    )
    register_template(
        QwenTemplateMeta("dforge_clef", template_cls=StrictClefTemplate, default_system=None),
        exist_ok=True,
    )

    for name, cls in (("jev", StrictJevTemplate), ("startlux", StrictStartLuxTemplate)):
        register_template(
            QwenTemplateMeta(f"dforge_{name}", template_cls=cls, default_system=None),
            exist_ok=True,
        )


def preprocessor_for(model_type):
    from swift.dataset.preprocessor.decision import (
        ClefPreprocessor,
        JevPreprocessor,
        KevPreprocessor,
        OmniJevPreprocessor,
        StartLuxPreprocessor,
    )

    classes = {
        "omnijev": OmniJevPreprocessor,
        "clef": ClefPreprocessor,
        "kev": KevPreprocessor,
        "jev": JevPreprocessor,
        "startlux": StartLuxPreprocessor,
    }
    return classes[get_route(model_type).name]()


def preprocess(model_type, row):
    value = preprocessor_for(model_type).preprocess(row)
    rows = value if isinstance(value, list) else [value]
    if len(rows) != 1 or rows[0] is None:
        raise ValueError(f"{model_type} preprocessor changed row cardinality")
    return rows[0]


def head_environment(route, model_root, adapter_root=None, *, artifact=None):
    for owner, relative, exported, env in route.assets:
        if artifact and (Path(artifact) / exported).is_file():
            path = Path(artifact) / exported
        else:
            root = adapter_root if owner == "adapter" else model_root
            path = Path(root) / relative
        if not path.is_file():
            raise ValueError(f"{route.name} missing published loader asset: {path}")
        os.environ[env] = str(path)
    os.environ["USE_HF"] = "1"


def temperature_buffers(head):
    """Post-hoc buffers only; learned scales/biases remain model parameters."""
    return {
        key: getattr(head, key)
        for key in ("head_temperature", "temps", "temperatures", "temperature")
        if getattr(head, key, None) is not None
    }
