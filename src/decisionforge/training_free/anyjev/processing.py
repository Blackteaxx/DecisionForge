"""Lossless public request mapping and strict, unrepaired upstream result readout."""

from __future__ import annotations

import json
import math

from decisionforge.eval.models import Request


def canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def validate_request(request):
    request = request if isinstance(request, Request) else Request.model_validate(request)
    question = request.questions["decision"]
    if question.type == "choice" and len(question.criteria) > 18278:
        raise ValueError("AnyJev Choice label pool supports at most 18278 options; full catalogs cannot be truncated")
    return request


def prepare_request(request):
    request = validate_request(request)
    q = request.questions["decision"]
    options = (
        [canonical({"key": key, "description": value}) for key, value in q.criteria.items()]
        if q.type == "choice"
        else [canonical(value) for value in q.criteria]
        if q.type == "score"
        else ["Yes", "No"]
    )
    return {
        "state": request.state if isinstance(request.state, str) else canonical(request.state),
        "kind": q.type,
        "text": canonical(
            {"instructions": q.instructions, "criteria": q.criteria if q.type == "noul" else None}
        ),
        "options": options,
        "labels": q.labels,
    }


def json_safe(value):
    """Keep invalid numerical evidence JSON serializable, without altering scoring data."""
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
    return value


def response_from_decision(question, decision):
    raw = {"confidence_basis": "maximum_candidate_probability"}
    # Read each field independently: one malformed property must not erase the
    # remaining upstream evidence. Validation below still reads original values.
    for field in ("probs", "level", "diagnostics", "confidence", "argmax", "p_true", "value"):
        try:
            raw[field] = json_safe(getattr(decision, field))
        except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
            pass
    try:
        raw["result"] = json_safe(decision.to_dict())
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
        pass
    response = {"anyjev": raw}
    try:
        if any(not hasattr(decision, field) for field in ("probs", "level", "diagnostics")):
            raise ValueError("Missing required upstream result field")
        p = decision.probs.tolist() if hasattr(decision.probs, "tolist") else decision.probs
        expected = 2 if question.type == "noul" else len(question.criteria)
        if (
            not isinstance(p, (list, tuple))
            or len(p) != expected
            or any(
                type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in p
            )
            or abs(sum(p) - 1) > 1e-6
        ):
            raise ValueError(
                "Invalid upstream probability vector: expected complete finite unit distribution"
            )
        if decision.level != "L0":
            raise ValueError("Upstream result must require L0")
        confidence = decision.confidence
        if not math.isfinite(confidence) or abs(confidence - max(p)) > 1e-12:
            raise ValueError("Invalid upstream maximum-candidate confidence")
        raw["confidence"] = confidence
        answer = {"type": question.type, "confidence": confidence}
        if question.type == "noul":
            value = decision.p_true
            if value != p[0]:
                raise ValueError("Noul p_true differs from Yes probability")
            answer.update(noul=value, probabilities={"false": p[1], "true": p[0]})
            raw["p_true"] = value
        elif question.type == "choice":
            options = [
                canonical({"key": k, "description": v}) for k, v in question.criteria.items()
            ]
            selected = decision.argmax
            if selected not in options or p[options.index(selected)] != max(p):
                raise ValueError("Invalid upstream selected option")
            answer.update(
                choice=question.labels[options.index(selected)],
                probabilities=dict(zip(question.labels, p)),
            )
            raw["argmax"] = selected
            raw["mapping"] = dict(zip(options, question.labels))
        else:
            value = decision.value
            if not math.isfinite(value) or abs(value - sum(i * v for i, v in enumerate(p))) > 1e-10:
                raise ValueError("Invalid upstream zero-based Score expectation")
            answer.update(
                score=value,
                probabilities=dict(zip(question.labels, p)),
                legend=dict(zip(question.labels, question.criteria)),
            )
            raw["value"] = value
        response["answers"] = {"decision": answer}
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError) as exc:
        raw["invalid_output"] = str(exc)
    if question.type == "choice" and len(question.criteria) > 26 and "diagnostics" in raw:
        from decisionforge.training_free.anyjev.diagnostics import encode_diagnostics

        raw["diagnostics"] = encode_diagnostics(raw["diagnostics"])
    return response
