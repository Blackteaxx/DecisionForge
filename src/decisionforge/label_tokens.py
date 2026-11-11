"""Deterministic, tokenizer-verified positional labels shared by decision methods."""

from __future__ import annotations

import hashlib
import itertools
import json
import string

SCHEMA_VERSION = 1
ENCODING_STYLES = ("space_prefixed", "bare_then_space")


def label_protocol_hash(protocol):
    payload = {key: value for key, value in protocol.items() if key != "protocol_hash"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def allocate_labels(tokenizer, count, encoding_style="space_prefixed", legacy_count=26):
    """Keep legacy A–Z prefixes, then select single-token labels from AA–ZZZ.

    ``bare_then_space`` matches upstream AnyJev's exact preference. OpenJev uses
    ``space_prefixed``. Every selected token is non-special and distinct. The
    finite candidate pool makes an inadequate tokenizer fail deterministically.
    """
    if type(count) is not int or not 1 <= count <= 18278:
        raise ValueError("Label count must be an integer in 1..18278 (A–ZZZ)")
    if encoding_style not in ENCODING_STYLES:
        raise ValueError(f"Unsupported label encoding style: {encoding_style}")
    if type(legacy_count) is not int or not 0 <= legacy_count <= 26:
        raise ValueError("legacy_count must be an integer in 0..26")
    special = set(getattr(tokenizer, "all_special_ids", ()) or ())
    labels, token_ids, encodings = [], [], []
    used = set()
    for width in (1, 2, 3):
        for chars in itertools.product(string.ascii_uppercase, repeat=width):
            label = "".join(chars)
            candidates = (
                (" " + label,) if encoding_style == "space_prefixed" else (label, " " + label)
            )
            found = None
            for text in candidates:
                ids = tokenizer.encode(text, add_special_tokens=False)
                if len(ids) == 1:
                    # Match upstream's bare-first choice: a colliding bare token
                    # must not silently switch to a different spaced token.
                    found = (ids[0], text)
                    break
            valid = (
                found is not None
                and type(found[0]) is int
                and found[0] >= 0
                and found[0] not in special
                and found[0] not in used
            )
            if not valid:
                if width == 1 and ord(label) - ord("A") < min(count, legacy_count):
                    raise ValueError(
                        f"Cannot preserve legacy label {label!r}: invalid single token or collision"
                    )
                continue
            token_id, text = found
            labels.append(label)
            token_ids.append(token_id)
            encodings.append(text)
            used.add(token_id)
            if len(labels) == count:
                protocol = {
                    "schema_version": SCHEMA_VERSION,
                    "encoding_style": encoding_style,
                    "legacy_count": legacy_count,
                    "labels": labels,
                    "token_ids": token_ids,
                    "encodings": encodings,
                }
                protocol["protocol_hash"] = label_protocol_hash(protocol)
                return protocol
    raise ValueError(
        f"Tokenizer cannot supply {count} distinct single-token labels from A–ZZZ; found {len(labels)}"
    )


def validate_label_protocol(tokenizer, protocol):
    """Verify integrity and reproduce the exact protocol with the current tokenizer."""
    if not isinstance(protocol, dict) or protocol.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported label protocol schema")
    if protocol.get("protocol_hash") != label_protocol_hash(protocol):
        raise ValueError("Label protocol hash mismatch")
    try:
        expected = allocate_labels(
            tokenizer,
            len(protocol["labels"]),
            encoding_style=protocol["encoding_style"],
            legacy_count=protocol["legacy_count"],
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("Malformed label protocol") from exc
    if expected != protocol:
        raise ValueError("Label protocol differs from current tokenizer allocation")
    return expected
