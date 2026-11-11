"""Explicitly create/audit the C/N fixture; training itself never filters question types."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from decisionforge.exporter import read_training_records
from decisionforge.util import atomic_write, json_text, write_json


def prepare(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output:
        raise ValueError("OpenJev subset output must differ from the original source")
    rows, excluded = [], []
    for record in read_training_records(source):
        row = record.model_dump(mode="json", exclude_none=True)
        # This explicit dataset-preparation command, never the backend, excludes Score.
        for key, question in copy.deepcopy(row["request"]["questions"]).items():
            if question["type"] == "score":
                excluded.append(
                    {"record_id": record.record_id, "question_id": key, "type": "score"}
                )
                del row["request"]["questions"][key]
                del row["supervision"]["answers"][key]
            elif question["type"] == "choice" and len(question["criteria"]) < 2:
                raise ValueError(
                    f"record_id={record.record_id} question={key}: Choice needs at least 2 options"
                )
        if not row["request"]["questions"]:
            raise ValueError(
                f"record_id={record.record_id}: no C/N questions remain; choose an explicit source subset"
            )
        rows.append(row)
    if not rows:
        raise ValueError("Empty source fixture")
    text = "".join(json_text(row) + "\n" for row in rows)
    atomic_write(output, text)
    audit = {
        "schema_version": "decisionforge.openjev.cn_fixture.v1",
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "records": len(rows),
        "retained_questions": sum(len(row["request"]["questions"]) for row in rows),
        "excluded_questions": excluded,
        "operation": "explicit_Score_question_exclusion_only_no_probability_or_public_field_rewrite",
        "weight_source": "original_soft_supervision_not_human_annotation_counts",
    }
    write_json(output.with_suffix(".source.json"), audit)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=str(Path(__file__).resolve().parents[1] / "data/typed-decisions-sample.jsonl"),
    )
    parser.add_argument(
        "--output",
        default=str(
            Path(__file__).resolve().parents[3] / "data/local/openjev/typed-decisions-cn.v3.jsonl"
        ),
    )
    args = parser.parse_args()
    audit = prepare(args.source, args.output)
    print(
        json.dumps(
            {
                "records": audit["records"],
                "retained_questions": audit["retained_questions"],
                "excluded_score_questions": len(audit["excluded_questions"]),
            }
        )
    )
