"""Explicitly project canonical evaluation samples to OpenJev C/N and audit Score exclusions.

Choice catalogs retain every original option. Token capacity is checked with the model tokenizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from decisionforge.eval.data import load_samples
from decisionforge.util import json_text


def prepare(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    audit_path = output.with_suffix(".source.json")
    if source == output:
        raise ValueError("C/N evaluation output must differ from the original source")
    for path in (output, audit_path):
        if path.exists():
            raise FileExistsError(path)
    selected, excluded = [], []
    for sample in load_samples(source):
        if sample.question.type == "score":
            excluded.append({"id": sample.id, "group_id": sample.group_id, "type": "score"})
        elif sample.question.type == "choice" and len(sample.question.criteria) < 2:
            raise ValueError(f"sample={sample.id}: OpenJev Choice needs at least 2 options")
        else:
            selected.append(sample)
    if not selected:
        raise ValueError("No C/N evaluation samples remain")
    text = "".join(json_text(sample.model_dump(mode="json")) + "\n" for sample in selected)
    audit = {
        "schema_version": "decisionforge.openjev.eval_cn.v1",
        "source": os.path.relpath(source, output.parent),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "selected": len(selected),
        "excluded": excluded,
        "operation": "explicit_Score_sample_exclusion_only_no_catalog_or_target_rewrite",
        "scope": "supported_C/N_subset_not_full_benchmark",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(text)
    with audit_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = prepare(args.source, args.output)
    print(json.dumps({"selected": audit["selected"], "excluded": len(audit["excluded"])}))


if __name__ == "__main__":
    main()
