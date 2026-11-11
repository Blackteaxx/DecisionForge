"""Reversible OmniJev mapping and probability contract; no model SDK imports."""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path

from decisionforge.media import dimensions, resolve_request
from decisionforge.util import json_text

SWIFT_REVISION = "680df5c670d7e8add906ac49381d603869410612"
BASE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
ADAPTER_REVISION = "91248107b7f5230bd5715db2e62e83c9a72fcd81"
PROBABILITY_CONTRACT = {
    "version": "decisionforge.omnijev.probabilities.v1",
    "choice_supervision": "original_K_distribution_plus_zero_abstain",
    "choice_readout": "condition_on_non_abstain",
    "abstain": "retained_separately_in_raw_response",
    "noul_columns": ["true", "false"],
    "score_columns": "ordered_original_level_indices",
}
MARKERS = ("<|opt|>", "<|/opt|>", "<image>", "<video>")


def text(value):
    return value if isinstance(value, str) else json_text(value)


def keys_for(question):
    if question["type"] == "noul":
        return ["true", "false"]
    if question["type"] == "score":
        return [str(i) for i in range(len(question["criteria"]))]
    return list(question["criteria"])


def map_request(request, settings, *, root=Path("."), answers=None, record_id=None):
    """Preserve every public field and option; map one original question to one row."""
    if settings.model_type != "omnijev":
        from decisionforge.training.backends.swift_routes import get_route

        return get_route(settings.model_type).convert(
            request, settings, root=root, answers=answers, record_id=record_id
        )
    from decisionforge.models import JEVInput

    public = request.model_dump(mode="json") if hasattr(request, "model_dump") else request
    public = deepcopy(public)
    # Eval accepts absent instructions; preserve the full question as JSON in that case.
    for q in public["questions"].values():
        if q.get("instructions") is None:
            q["instructions"] = json_text(q)
    req = resolve_request(JEVInput.model_validate(public), Path(root))
    images = [a.path for a in req.images or []]
    if len(images) > settings.max_images:
        raise ValueError(f"record_id={record_id}: image count exceeds max_images")
    for path in images:
        w, h = dimensions(path)
        if w * h > settings.max_pixels:
            raise ValueError(f"record_id={record_id}: image pixels exceed max_pixels: {path}")
    video = None
    if req.videos:
        v = req.videos[0]
        images = [v.mosaic.path]
        if dimensions(images[0]) != (v.cols * v.tile, v.n_frames // v.cols * v.tile):
            raise ValueError("Video mosaic dimensions do not match its 4x4 descriptor")
        if v.tile * v.tile > settings.max_pixels:
            raise ValueError("Video frame pixels exceed max_pixels")
        video = v.model_dump(exclude={"mosaic", "source_sha256"})
    items, mapping = [], []
    for qid, q in req.questions.items():
        question = q.model_dump(mode="json")
        keys = keys_for(question)
        where = f"record_id={record_id}, question={qid}"
        if q.type == "choice" and len(keys) > settings.max_options:
            raise ValueError(f"{where}: option count exceeds max_options; no subsampling")
        if q.type == "score" and len(keys) > settings.max_score_levels:
            raise ValueError(f"{where}: score levels exceed max_score_levels")
        instruction = text(req.state) + "\n" + text(q.instructions)
        native = {"type": q.type, "instructions": instruction}
        if q.type == "choice":
            # Generated native keys prevent collisions with the upstream reserved abstain key.
            native["options"] = [
                {"key": f"k{i}", "text": text(v if v is not None else k)}
                for i, (k, v) in enumerate(q.criteria.items())
            ]
            rendered = [o["text"] for o in native["options"]]
        elif q.type == "score":
            native["levels"] = [text(v) for v in q.criteria]
            rendered = native["levels"]
        else:
            # Noul's described true/false criteria are public context too.
            if q.criteria:
                native["instructions"] += "\nCriteria: " + text(q.criteria)
            rendered = []
        if any(len(t) > 200 for t in rendered):
            raise ValueError(
                f"{where}: OmniJev truncates option/level text at 200 characters; refusing"
            )
        if any(marker in t for t in [native["instructions"], *rendered] for marker in MARKERS):
            raise ValueError(f"{where}: reserved media/option marker in public text")
        if q.type == "score" and len(set(rendered)) != len(rendered):
            raise ValueError(f"{where}: rendered score levels are not unique")
        target = None
        if answers is not None:
            a = answers[qid]
            target = (
                [a.noul, 1 - a.noul] if q.type == "noul" else [a.probabilities[k] for k in keys]
            )
            native["gold"] = target + ([0.0] if q.type == "choice" else [])
        row = {"images": images, "questions": [native]}
        if video:
            row["video"] = video
        items.append(row)
        mapping.append(
            {
                "record_id": record_id,
                "question_id": qid,
                "option_keys": keys,
                "type": q.type,
                "target": target,
                "native_keys": [o["key"] for o in native.get("options", [])],
            }
        )
    return items, mapping


def readout(question, probabilities):
    """Reject illegal output, then perform the documented K+1 -> K conditioning."""
    from decisionforge.jevmath import choice_confidence, score_confidence

    question = question.model_dump() if hasattr(question, "model_dump") else question
    keys = keys_for(question)
    width = len(keys) + (question["type"] == "choice")
    if (
        not isinstance(probabilities, list)
        or len(probabilities) != width
        or any(
            type(p) not in (float, int) or not math.isfinite(p) or not 0 <= p <= 1
            for p in probabilities
        )
        or abs(sum(probabilities) - 1) > 1e-5
    ):
        raise ValueError("Invalid ms-swift full column probability distribution")
    abstain = None
    p = list(probabilities)
    if question["type"] == "choice":
        abstain = p.pop()
        mass = sum(p)
        if mass <= 0:
            raise ValueError("All probability mass is abstain; cannot condition on options")
        p = [v / mass for v in p]
    dist = dict(zip(keys, p))
    if question["type"] == "noul":
        answer = {"type": "noul", "noul": dist["true"]}
    elif question["type"] == "score":
        answer = {
            "type": "score",
            "score": sum(i * v for i, v in enumerate(p)),
            "confidence": score_confidence(p),
            "legend": dict(zip(keys, question["criteria"])),
            "probabilities": dist,
        }
    else:
        answer = {
            "type": "choice",
            "choice": max(dist, key=dist.get),
            "confidence": choice_confidence(p),
            "probabilities": dist,
        }
    return answer, abstain


def diagnostics(predictions):
    """Shared scoring view after conditioning, including soft-supervision proper scores."""
    if not predictions:
        return {"questions": 0}
    brier, nll, bins, correct = 0.0, 0.0, [[] for _ in range(15)], 0
    for p, y in predictions:
        idx = max(range(len(p)), key=p.__getitem__)
        hit = float(idx == max(range(len(y)), key=y.__getitem__))
        correct += hit
        bins[min(14, int(p[idx] * 15))].append((p[idx], hit))
        brier += sum((a - b) ** 2 for a, b in zip(p, y))
        nll -= sum(b * math.log(max(a, 1e-12)) for a, b in zip(p, y))
    n = len(predictions)
    return {
        "questions": n,
        "accuracy": correct / n,
        "brier": brier / n,
        "nll": nll / n,
        "ece": sum(abs(sum(a - b for a, b in group)) for group in bins) / n,
        "ece_bins": 15,
    }
