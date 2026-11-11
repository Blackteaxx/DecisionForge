"""Audit source roles and exact/near public patient text before training admission."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


def public_documents(state):
    """Read patient observations, excluding system instructions and private metadata."""
    if isinstance(state, str):
        try:
            state = json.loads(state)
        except ValueError:
            return [("message", state)]
    if isinstance(state, dict) and isinstance(state.get("message"), str):
        state = {"patients": [state]}
    if isinstance(state, dict) and isinstance(state.get("patients"), list):
        return [
            (kind, patient[kind])
            for patient in state["patients"]
            for kind in ("message", "ehr")
            if isinstance(patient.get(kind), str) and patient[kind].strip()
        ]
    if isinstance(state, list):
        messages = [m["content"] for m in state if m.get("role") == "user" and m.get("content")]
        # A follow-up reply depends on the presenting complaint and prior question.
        # Audit the initial patient message and full patient history, not isolated
        # generic replies such as "No, none of that" as separate patient identities.
        result = [("message", messages[0])] if messages else []
        if len(messages) > 1:
            result.append(("patient_history", "\n".join(messages)))
        return result
    raise ValueError("Unknown public patient-state representation; audit cannot silently skip it")


def normalized_patient_parents(state):
    parents = set()
    for kind, text in public_documents(state):
        tokens = re.findall(r"[a-z0-9]+", text.casefold())
        if len(tokens) >= 4:
            digest = hashlib.sha256(" ".join(tokens).encode()).hexdigest()
            parents.add("normalized-" + kind + ":" + digest)
    return sorted(parents)


def find_overlaps(rows, threshold=0.8, minimum_near_tokens=12):
    """Report cross-role families, normalized exact text, and five-word Jaccard >=0.8.

    This lexical audit detects reuse/paraphrases with large verbatim overlap. It
    cannot establish clinical or semantic independence by itself. Original roles,
    text and labels remain unchanged; findings require admission review.
    """
    if not 0 < threshold <= 1 or minimum_near_tokens < 5:
        raise ValueError("Invalid overlap audit thresholds")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Overlap audit requires globally unique row IDs")
    found, families, exact, inverted, documents = (
        {},
        defaultdict(dict),
        defaultdict(list),
        defaultdict(set),
        [],
    )
    for index, row in enumerate(rows):
        family = row.get("family_id")
        if family:
            for role, previous in families[family].items():
                if role != row["split"]:
                    key = (previous, index, "source_family")
                    found[key] = {
                        "left": rows[previous]["id"],
                        "right": row["id"],
                        "kind": "source_family",
                        "family_id": family,
                    }
            families[family].setdefault(row["split"], index)
        for kind, text in public_documents(row["state"]):
            tokens = tuple(re.findall(r"[a-z0-9]+", text.casefold()))
            # Very short replies such as 'yes' are not patient identities.
            if len(tokens) < 4:
                continue
            fingerprint = (kind, tokens)
            for previous in exact[fingerprint]:
                if rows[previous]["split"] != row["split"]:
                    found[(previous, index, kind)] = {
                        "left": rows[previous]["id"],
                        "right": row["id"],
                        "kind": "exact",
                        "document_kind": kind,
                        "normalized_sha256": hashlib.sha256(" ".join(tokens).encode()).hexdigest(),
                    }
            exact[fingerprint].append(index)
            if len(tokens) < minimum_near_tokens:
                continue
            grams = {tokens[i : i + 5] for i in range(len(tokens) - 4)}
            candidates = set().union(*(inverted[(kind, gram)] for gram in grams))
            for prior in sorted(candidates):
                previous, previous_grams = documents[prior]
                if previous == index or rows[previous]["split"] == row["split"]:
                    continue
                if (
                    min(len(grams), len(previous_grams)) / max(len(grams), len(previous_grams))
                    < threshold
                ):
                    continue
                score = len(grams & previous_grams) / len(grams | previous_grams)
                key = (previous, index, kind)
                if score >= threshold and (
                    key not in found or found[key].get("jaccard", 1) < score
                ):
                    found[key] = {
                        "left": rows[previous]["id"],
                        "right": row["id"],
                        "kind": "near",
                        "document_kind": kind,
                        "jaccard": score,
                    }
            slot = len(documents)
            documents.append((index, grams))
            for gram in grams:
                inverted[(kind, gram)].add(slot)
    return list(found.values())


def audit_dataset(assembled: Path, benchmarks: list[Path], output: Path):
    records = json.loads((assembled / "training.json").read_text())
    lineage = json.loads((assembled / "lineage.json").read_text())
    rows = [
        {
            "id": "generated:" + str(r["record_id"]),
            "split": r["split"],
            "family_id": lineage[str(r["record_id"])]["family_id"],
            "state": r["request"]["state"],
        }
        for r in records
    ]
    inputs = [assembled / "training.json", assembled / "lineage.json", *benchmarks]
    for path in benchmarks:
        for raw in path.read_text().splitlines():
            if raw.strip():
                s = json.loads(raw)
                rows.append(
                    {"id": "benchmark:" + s["id"], "split": "test", "state": s["request"]["state"]}
                )
    overlaps = find_overlaps(rows)
    result = {
        "schema_version": "medical.patient-overlap-audit.v2",
        "status": "needs_review" if overlaps else "passed",
        "generated_records": len(records),
        "all_rows": len(rows),
        "inputs": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        "protocol": {
            "exact": "casefolded alphanumeric patient text with >=4 tokens",
            "near": "five-token shingle Jaccard >=0.8 with >=12 tokens",
            "source_families": "no cross-role reuse",
            "system_prompts": "excluded",
            "dialogue_unit": "initial patient message plus complete patient history; context-dependent isolated follow-up replies are not identities",
            "uses_targets": False,
        },
        "limitations": "Lexical audit does not prove semantic independence or absence from base-model pretraining.",
        "overlaps": overlaps,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembled", type=Path, required=True)
    parser.add_argument("--benchmarks", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_dataset(args.assembled, args.benchmarks, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "overlaps"}))
    if result["status"] != "passed":
        raise SystemExit("Patient overlap findings require review before training")


if __name__ == "__main__":
    main()
