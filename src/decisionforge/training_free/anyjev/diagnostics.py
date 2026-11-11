"""Lossless, base-only diagnostic encoding for large Choice journals."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import zlib

SUMMARY_KEYS = (
    "permutations",
    "prior_method",
    "prior_strength",
    "answer_mass",
    "adaptive",
    "shifts_used",
    "stop_threshold",
    "stop_stat",
    "stop_calibrated",
    "order_flip_raw",
    "order_flip_l0",
)


def encode_diagnostics(diagnostics):
    """Pack every JSON-safe value; keep cheap scalar diagnostics readable.

    Large immutable strings are shared by deepcopy, unlike K-by-K float/list
    trees retained in the evaluator's attempts and predictions tables.
    """
    payload = json.dumps(
        diagnostics, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")
    summary = (
        {
            key: diagnostics[key]
            for key in SUMMARY_KEYS
            if key in diagnostics and not isinstance(diagnostics[key], (dict, list, tuple))
        }
        if isinstance(diagnostics, dict)
        else {}
    )
    summary["compressed"] = {
        "schema_version": 1,
        "codec": "gzip+base64",
        "uncompressed_sha256": hashlib.sha256(payload).hexdigest(),
        "uncompressed_bytes": len(payload),
        "payload": base64.b64encode(gzip.compress(payload, compresslevel=1, mtime=0)).decode(
            "ascii"
        ),
    }
    return summary


def decode_diagnostics(diagnostics):
    """Restore exact JSON values and verify integrity; legacy diagnostics pass through."""
    if not isinstance(diagnostics, dict) or "compressed" not in diagnostics:
        return diagnostics
    try:
        encoded = diagnostics["compressed"]
        if encoded["schema_version"] != 1 or encoded["codec"] != "gzip+base64":
            raise ValueError("Unsupported AnyJev diagnostic codec")
        payload = gzip.decompress(base64.b64decode(encoded["payload"], validate=True))
        if (
            len(payload) != encoded["uncompressed_bytes"]
            or hashlib.sha256(payload).hexdigest() != encoded["uncompressed_sha256"]
        ):
            raise ValueError("AnyJev diagnostic integrity mismatch")
        original = json.loads(payload)
        if any(
            key != "compressed" and (not isinstance(original, dict) or original.get(key) != value)
            for key, value in diagnostics.items()
        ):
            raise ValueError("AnyJev diagnostic summary mismatch")
        return original
    except (KeyError, TypeError, ValueError, OSError, EOFError, zlib.error) as exc:
        raise ValueError(f"Invalid compressed AnyJev diagnostics: {exc}") from exc
