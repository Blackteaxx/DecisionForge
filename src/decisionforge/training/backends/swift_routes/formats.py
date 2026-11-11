"""Route-specific, reversible v3 converters and native-column readout."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path

from decisionforge.media import dimensions, resolve_request
from decisionforge.models import JEVInput
from decisionforge.training.backends.swift_task import text


def public_request(request, settings, root, record_id):
    from . import get_route

    route = get_route(settings.model_type)
    public = (
        request.model_dump(mode="json") if hasattr(request, "model_dump") else deepcopy(request)
    )
    rendered = json.dumps({k: public[k] for k in ("state", "questions")}, ensure_ascii=False)
    if any(
        marker in rendered
        for marker in (
            "<image>",
            "<video>",
            "<|vision_start|>",
            "<|vision_end|>",
            "<|image_pad|>",
            "<|video_pad|>",
            "<|im_start|>",
            "<|im_end|>",
        )
    ):
        raise ValueError(f"{route.name}: public text contains reserved media/template markers")
    for q in public["questions"].values():
        if q.get("instructions") is None:
            q["instructions"] = text(q)
    req = resolve_request(JEVInput.model_validate(public), Path(root))
    if req.videos:
        raise ValueError(f"{route.name} does not support the OmniJev offline video-mosaic contract")
    images = [asset.path for asset in req.images or []]
    if len(images) > min(settings.max_images, route.max_images):
        raise ValueError(f"{route.name}: record_id={record_id}: image count exceeds route limit")
    for path in images:
        w, h = dimensions(path)
        if w * h > min(settings.max_pixels, route.max_pixels):
            raise ValueError(f"{route.name}: image pixels exceed route limit: {path}")
    for qid, question in req.questions.items():
        if question.type == "choice" and len(question.criteria) > settings.max_options:
            raise ValueError(f"{route.name}: question={qid}: option count exceeds max_options")
        if question.type == "score":
            count = len(question.criteria)
            if count > settings.max_score_levels or (
                route.score_levels and count != route.score_levels
            ):
                raise ValueError(
                    f"{route.name}: question={qid}: unsupported score levels (native requires {route.score_levels or '2–10'})"
                )
    return req, images


def original_keys(q):
    if q.type == "noul":
        return ["false", "true"]
    if q.type == "score":
        return [str(i) for i in range(len(q.criteria))]
    return list(q.criteria)


def target_for(answer, keys, kind):
    return (
        [answer.noul if k == "true" else 1 - answer.noul for k in keys]
        if kind == "noul"
        else [answer.probabilities[k] for k in keys]
    )


def mapping_for(qid, q, record_id, row_index, question_index, native_keys, answers):
    keys = original_keys(q)
    return {
        "record_id": record_id,
        "question_id": qid,
        "type": q.type,
        "option_keys": keys,
        "native_keys": list(native_keys),
        "row_index": row_index,
        "question_index": question_index,
        "target": target_for(answers[qid], keys, q.type) if answers is not None else None,
    }


def clef(request, settings, *, root=Path("."), answers=None, record_id=None):
    req, images = public_request(request, settings, root, record_id)
    questions, gold, mappings = {}, {}, []
    for i, (qid, q) in enumerate(req.questions.items()):
        keys = sorted(q.criteria) if q.type == "choice" else original_keys(q)
        questions[qid] = q.model_dump(mode="json", exclude_none=True)
        if answers is not None:
            gold[qid] = target_for(answers[qid], keys, q.type)
        mappings.append(mapping_for(qid, q, record_id, 0, i, keys, answers))
    row = {"state": req.state, "images": images, "questions": questions}
    if answers is not None:
        row["answers"] = gold
    return [row], mappings


def kev(request, settings, *, root=Path("."), answers=None, record_id=None):
    req, images = public_request(request, settings, root, record_id)
    rows, mappings = [], []
    for i, (qid, q) in enumerate(req.questions.items()):
        keys = original_keys(q)
        if q.type == "choice":
            options = [text(q.criteria[k] if q.criteria[k] is not None else k) for k in keys]
        elif q.type == "score":
            options = [text(level) for level in q.criteria]
        else:
            options = [text((q.criteria or {}).get(k) or k) for k in keys]
        question = {"kind": q.type, "question": text(q.instructions), "options": options}
        if answers is not None:
            question["gold"] = target_for(answers[qid], keys, q.type)
        rows.append({"state": text(req.state), "images": images, "questions": [question]})
        mappings.append(mapping_for(qid, q, record_id, i, 0, keys, answers))
    return rows, mappings


def jev(request, settings, *, root=Path("."), answers=None, record_id=None):
    req, images = public_request(request, settings, root, record_id)
    rows, mappings = [], []
    for i, (qid, q) in enumerate(req.questions.items()):
        keys = original_keys(q)
        instructions = text(q.instructions)
        if q.type == "choice":
            options = [text(v if v is not None else k) for k, v in q.criteria.items()]
        else:
            # The published verbalizers are fixed false/true and digits0..5.
            # Preserve descriptions as context instead of silently discarding them.
            options = keys
            if q.criteria:
                instructions += "\nCriteria (in original level/key order): " + text(q.criteria)
        question = {"kind": q.type, "question": instructions, "options": options}
        if answers is not None:
            question["gold"] = target_for(answers[qid], keys, q.type)
        rows.append({"state": text(req.state), "images": images, "questions": [question]})
        mappings.append(mapping_for(qid, q, record_id, i, 0, keys, answers))
    return rows, mappings


def startlux(request, settings, *, root=Path("."), answers=None, record_id=None):
    req, images = public_request(request, settings, root, record_id)
    rows, mappings = [], []
    for i, (qid, q) in enumerate(req.questions.items()):
        keys = ["true", "false"] if q.type == "noul" else original_keys(q)
        none = settings.allow_none and q.type == "choice"
        if q.type == "choice":
            if len(keys) + none > 26:
                raise ValueError("StartLux allow_none consumes one of its 26 letter columns")
            if any(len(k) > 200 or "\n" in k or "\r" in k for k in keys):
                raise ValueError("StartLux option IDs must be single-line and at most200characters")
            if none and any(" ".join(k.lower().split()) == "none of the above" for k in keys):
                raise ValueError(
                    "StartLux allow_none conflicts with an original None of the above option"
                )
        native = q.model_dump(mode="json", exclude_none=True)
        if none:
            native["allow_none"] = True
        if answers is not None:
            native["gold"] = target_for(answers[qid], keys, q.type) + ([0.0] if none else [])
        rows.append({"state": req.state, "images": images, "questions": {qid: native}})
        item = mapping_for(qid, q, record_id, i, 0, keys, answers)
        item["extra_choice_column"] = "none_of_the_above" if none else None
        mappings.append(item)
    return rows, mappings


CONVERTERS = {"clef": clef, "kev": kev, "jev": jev, "startlux": startlux}


def readout(question, probabilities, mapping=None, *, model_type, allow_none=False):
    from decisionforge.jevmath import choice_confidence, score_confidence

    q = question.model_dump() if hasattr(question, "model_dump") else question
    keys = (
        ["false", "true"]
        if q["type"] == "noul"
        else [str(i) for i in range(len(q["criteria"]))]
        if q["type"] == "score"
        else list(q["criteria"])
    )
    native = (
        mapping["native_keys"]
        if mapping
        else (
            sorted(keys)
            if model_type == "clef" and q["type"] == "choice"
            else ["true", "false"]
            if model_type == "startlux" and q["type"] == "noul"
            else keys
        )
    )
    extra = (
        model_type == "startlux"
        and q["type"] == "choice"
        and (bool(mapping.get("extra_choice_column")) if mapping is not None else allow_none)
    )
    if (
        not isinstance(probabilities, list)
        or len(probabilities) != len(native) + extra
        or set(native) != set(keys)
        or len(native) != len(set(native))
        or any(
            type(p) not in (float, int) or not math.isfinite(p) or not 0 <= p <= 1
            for p in probabilities
        )
        or abs(sum(probabilities) - 1) > 1e-5
    ):
        raise ValueError(f"Invalid {model_type} native probability columns")
    extra_probability = probabilities[-1] if extra else None
    values = probabilities[:-1] if extra else probabilities
    if extra:
        mass = sum(values)
        if mass <= 0:
            raise ValueError(
                "All probability mass is none-of-the-above; cannot condition on original options"
            )
        values = [p / mass for p in values]
    native_dist = dict(zip(native, values))
    dist = {k: native_dist[k] for k in keys}
    p = list(dist.values())
    if q["type"] == "noul":
        return {"type": "noul", "noul": dist["true"]}, None
    if q["type"] == "score":
        return {
            "type": "score",
            "score": sum(i * v for i, v in enumerate(p)),
            "confidence": score_confidence(p),
            "legend": dict(zip(keys, q["criteria"])),
            "probabilities": dist,
        }, None
    return {
        "type": "choice",
        "choice": max(dist, key=dist.get),
        "confidence": choice_confidence(p),
        "probabilities": dist,
    }, extra_probability
