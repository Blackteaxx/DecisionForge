"""CPU-safe reversible task mapping for the pinned OpenJev task registry."""

from __future__ import annotations

import copy
import json
import math

LABELS = list("ABCDEFGHIJ")
TASK_NAME = "dforge_v3"
MAPPING_VERSION = "decisionforge.openjev.mapping.v1"
MAPPING_VERSION_V2 = "decisionforge.openjev.mapping.v2"
WEIGHT_SOURCE = "sampling_weights_derived_from_dforge_supervision"
SAMPLING = {
    "m": 1,
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "stop_strings": ["Answer:"],
    "enable_thinking": False,
    "generation_micro_batch_size": 64,
    "probability_source": "openjev_official_post_rationale_label_logits_single",
}
PUBLIC_PROTOCOL = {
    "training_and_reload_probe": "complete_original_v3_public_request_plus_focus_key",
    "eval_cli": "complete_provided_canonical_eval_request_single_decision",
}


def option_keys(question):
    kind = question["type"]
    if kind == "noul":
        return ["false", "true"]
    if kind != "choice":
        raise ValueError(f"OpenJev does not support question type {kind}")
    criteria = question.get("criteria")
    if not isinstance(criteria, dict) or len(criteria) < 2:
        raise ValueError("OpenJev Choice needs at least 2 options")
    return list(criteria)


def public_item(public, focus, uid="prediction", *, labels=None):
    if set(public) != {"state", "questions"} or focus not in public["questions"]:
        raise ValueError("OpenJev requires complete public state/questions and a valid focus key")
    keys = option_keys(public["questions"][focus])
    # Validate every question, including public siblings; no unsupported data disappears.
    for question in public["questions"].values():
        option_keys(question)
    if labels is None:
        labels = LABELS if len(keys) <= len(LABELS) else None
    if labels is not None and (
        len(labels) < len(keys)
        or len(set(labels)) != len(labels)
        or any(not isinstance(label, str) or not label for label in labels)
    ):
        raise ValueError("OpenJev label pool must be unique and cover all original options")
    item = {
        "uid": uid,
        "public_request": copy.deepcopy(public),
        "focus": focus,
        "option_keys": keys,
        "label_mapping": dict(zip(labels, keys)) if labels is not None else None,
        "mapping_version": MAPPING_VERSION if labels == LABELS else MAPPING_VERSION_V2,
    }
    if labels is None:
        item["label_validation"] = "deferred_to_tokenizer"
    return item


def message(item):
    if item.get("label_mapping") is None:
        raise ValueError("OpenJev message requires a tokenizer-validated label pool")
    # Structured content/absent instructions are serialized exactly, not paraphrased.
    public = json.dumps(item["public_request"], ensure_ascii=False, allow_nan=False)
    mapping = json.dumps(item["label_mapping"], ensure_ascii=False, allow_nan=False)
    focus = json.dumps(item["focus"], ensure_ascii=False)
    multiletter = any(len(label) > 1 for label in item["label_mapping"])
    heading = (
        "Answer label to original option key"
        if multiletter
        else "Answer letter to original option key"
    )
    tail = (
        "where X is the complete displayed label of the correct option."
        if multiletter
        else "where X is the letter of the correct option."
    )
    return (
        f"Public request: {public}\n\nFocus question key: {focus}\n"
        f"{heading}: {mapping}\n\n"
        "Think step by step, briefly (a few sentences), then finish with exactly one line of the form "
        f"'Answer: X', {tail}"
    )


def map_roles(data, *, labels=None):
    splits, report = {}, {}
    for role, records in data.roles.items():
        items, mapping = [], []
        for record in records:
            for focus, question in record.request.questions.items():
                item = public_item(
                    record.request.model_dump(mode="json", exclude_none=True),
                    focus,
                    json.dumps([record.record_id, focus], ensure_ascii=False),
                    labels=labels,
                )
                if item["label_mapping"] is None:
                    raise ValueError("OpenJev training requires a tokenizer-validated label pool")
                answer = record.supervision.answers[focus]
                q = (
                    [1.0 - answer.noul, answer.noul]
                    if question.type == "noul"
                    else [answer.probabilities[key] for key in item["option_keys"]]
                )
                validate_distribution(q, len(item["option_keys"]))
                item.update(q=q, counts=list(q))
                mapping.append(
                    {
                        "record_id": record.record_id,
                        "question_id": focus,
                        "group_id": record.group_id,
                        "role": role,
                        "item_index": len(items),
                        "option_keys": item["option_keys"],
                        "label_mapping": item["label_mapping"],
                        "source_target": dict(zip(item["option_keys"], q)),
                        "weight_source": WEIGHT_SOURCE,
                    }
                )
                items.append(item)
        if role != "calibration":
            splits["dev" if role == "validation" else role] = items
        report[role] = {
            "mapping": mapping,
            "questions": len(items),
            "used_by_upstream": role != "calibration",
        }
    for role in ("train", "dev", "test"):
        if not splits[role]:
            raise ValueError(f"OpenJev requires a nonempty {role} role")
    return splits, report


def register_task(tasks, splits, *, labels=None):
    labels = LABELS if labels is None else labels
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("OpenJev task labels must be unique")
    tasks[TASK_NAME] = dict(
        labels=list(labels),
        message=message,
        load=lambda: splits,
        k=lambda item: len(item["option_keys"]),
    )


def validate_labels(tok, labels=None):
    labels = LABELS if labels is None else labels
    encodings = [tok.encode(" " + label, add_special_tokens=False) for label in labels]
    if any(len(ids) != 1 for ids in encodings) or len({ids[0] for ids in encodings}) != len(labels):
        raise ValueError("OpenJev labels must encode as unique single tokens")
    return [ids[0] for ids in encodings]


def validate_distribution(values, k):
    if (
        not isinstance(values, list)
        or len(values) != k
        or any(
            type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
            for value in values
        )
        or abs(sum(values) - 1) > 1e-6
    ):
        raise ValueError(
            "Invalid OpenJev probability distribution; no normalization or repair is allowed"
        )
    return values


def answer_from_distribution(question, values):
    keys = option_keys(question)
    validate_distribution(values, len(keys))
    probabilities = dict(zip(keys, values))
    if question["type"] == "noul":
        return {"type": "noul", "noul": probabilities["true"], "probabilities": probabilities}
    return {
        "type": "choice",
        "choice": max(keys, key=probabilities.__getitem__),
        "confidence": max(values),
        "probabilities": probabilities,
    }


def check_prompt_budget(tok, prompt, max_seq_length, max_new_tokens, suffix):
    ids = tok.encode(prompt)
    if getattr(tok, "pad_token_id", None) in ids:
        raise ValueError(
            "OpenJev public prompt contains a padding token removed by the official sampler"
        )
    length = len(ids)
    if length + max_new_tokens + len(suffix) > max_seq_length:
        raise ValueError(
            "OpenJev complete public prompt plus rationale/marker budget exceeds max_seq_length; refusing truncation"
        )
    return length


STATIC_LABEL_CHECKS = {
    "choice_max_options": None,
    "label_candidate_limit": 18278,
    "choice_catalog_capacity": "tokenizer_dependent_runtime_check",
    "runtime_checks": "deferred: CUDA, BF16, tokenizer labels, complete prompt budget",
}


READOUT_POLICY = {
    "version": "decisionforge.openjev.readout.v2",
    "allocation": "tokenizer_verified_space_prefixed_excel_pool_legacy_A_J",
    "extension": "deterministic_request_K_preserve_trained_prefix",
    "training_claim": "extension_does_not_imply_training_on_new_labels_or_classes",
    "seed": "legacy_for_unextended_pool_protocol_hash_for_extended_pool",
}


def inference_label_protocol(tok, manifest, k):
    """Validate trained tokens and derive a request-local readout without resizing weights."""
    from decisionforge.label_tokens import allocate_labels, validate_label_protocol
    from decisionforge.training.backends.openjev import ARTIFACT_VERSION, ARTIFACT_VERSION_V2

    schema = manifest.get("schema_version")
    if schema == ARTIFACT_VERSION:
        if "label_protocol" in manifest:
            raise ValueError("OpenJev v1 artifact cannot declare a v2 label protocol")
        ids = validate_labels(tok)
        if manifest.get("labels") != LABELS or manifest.get("label_token_ids") != ids:
            raise ValueError("OpenJev tokenizer label identity changed")
        trained = allocate_labels(tok, 10, encoding_style="space_prefixed", legacy_count=10)
    elif schema == ARTIFACT_VERSION_V2:
        trained = validate_label_protocol(tok, manifest.get("label_protocol"))
        if (
            trained["encoding_style"] != "space_prefixed"
            or trained["legacy_count"] != 10
            or trained["labels"][:10] != LABELS
            or manifest.get("labels") != trained["labels"]
            or manifest.get("label_token_ids") != trained["token_ids"]
        ):
            raise ValueError("OpenJev manifest label identity differs from its trained protocol")
    else:
        raise ValueError("Unsupported OpenJev artifact schema for inference labels")
    count = max(k, len(trained["labels"]))
    effective = allocate_labels(tok, count, encoding_style="space_prefixed", legacy_count=10)
    size = len(trained["labels"])
    if any(effective[key][:size] != trained[key] for key in ("labels", "token_ids", "encodings")):
        raise ValueError("OpenJev inference label extension changed the trained token prefix")
    return effective
