"""CPU-safe, relocatable multimodal contract shared by training and evaluation."""

from __future__ import annotations

import hashlib
import math
import os
import stat as stat_module
from functools import lru_cache
from pathlib import Path
from typing import Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator


class MediaAsset(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def local_only(self):
        if "://" in self.path or self.path.startswith(("data:", "file:")):
            raise ValueError("Media requires a local path; materialize URLs/data URIs first")
        return self


class VideoMosaic(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    mosaic: MediaAsset
    n_frames: Literal[16] = 16
    cols: Literal[4] = 4
    tile: Literal[384] = 384
    timestamps: list[float] = Field(min_length=16, max_length=16)
    duration: float = Field(ge=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def chronological(self):
        if any(not math.isfinite(t) or not 0 <= t <= self.duration for t in self.timestamps):
            raise ValueError("Video timestamps must lie within its duration")
        if self.timestamps != sorted(self.timestamps):
            raise ValueError("Video timestamps must be chronological")
        return self


class MediaRequest(BaseModel):
    images: list[MediaAsset] | None = Field(default=None, min_length=1)
    videos: list[VideoMosaic] | None = Field(default=None, min_length=1, max_length=1)

    @model_validator(mode="after")
    def exclusive_media(self):
        if self.images and self.videos:
            raise ValueError("A request contains still images OR one offline video mosaic")
        return self

    @model_serializer(mode="wrap")
    def omit_absent_media(self, handler):
        data = handler(self)
        for key in ("images", "videos"):
            if data.get(key) is None:
                data.pop(key, None)
        return data


def has_media(request):
    if isinstance(request, dict):
        return bool(request.get("images") or request.get("videos"))
    return bool(request.images or request.videos)


def reject_media(request, adapter):
    if has_media(request):
        raise ValueError(f"{adapter} does not support multimodal images/videos")


def resolve_asset(asset: MediaAsset, root: Path) -> Path:
    # Local assets may legitimately live outside the dataset root. Lexical
    # normalization avoids walking every GPFS parent for every already-absolute
    # reference; the opened file's bytes and stat still establish its identity.
    path = Path(os.path.abspath(os.path.expanduser(root / asset.path)))
    try:
        stat = path.stat()
    except OSError as exc:
        raise ValueError(f"Media file does not exist: {path}") from exc
    if not stat_module.S_ISREG(stat.st_mode):
        raise ValueError(f"Media file is not a regular file: {path}")
    if _asset_sha256(str(path), stat.st_size, stat.st_mtime_ns, stat.st_ino) != asset.sha256:
        raise ValueError(f"Media SHA-256 mismatch: {path}")
    return path


@lru_cache(maxsize=32768)
def _asset_sha256(path, size, mtime_ns, inode):
    # Reuse verified bytes during repeated preflight phases; a changed file stat
    # invalidates the entry. Different CLI processes always hash independently.
    value = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    after = Path(path).stat()
    if (after.st_size, after.st_mtime_ns, after.st_ino) != (size, mtime_ns, inode):
        raise ValueError(f"Media changed during hashing: {path}")
    return value


def resolve_request(request, root: Path):
    """Resolve relative to the JSON(L) dataset, retaining the verified content digest."""
    for asset in [*(request.images or []), *[v.mosaic for v in request.videos or []]]:
        asset.path = str(resolve_asset(asset, root))
    return request


def dimensions(path):
    with Image.open(path) as im:
        size = im.size
        im.verify()
    return size
