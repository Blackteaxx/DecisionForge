"""Explicit, offline conversion of prepared Typed Decisions train/test to local training v3.

Run the existing benchmark prepare.py separately to fetch pinned train/test sources.
This example does not download data and training itself never reads HF datasets.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from decisionforge.jevmath import choice_confidence, score_confidence
from decisionforge.models import TrainingExample
from decisionforge.util import atomic_write, json_text, write_json


def convert(source, output, train_limit=None, test_limit=None):
    import pyarrow.parquet as pq

    records, provenance = [], []
    for split, limit in (("train", train_limit), ("test", test_limit)):
        rows = pq.read_table(Path(source) / "raw" / split / f"{split}.parquet").to_pylist()
        if limit is not None:
            # Round-robin workflows, deterministic within fixed source order.
            from collections import defaultdict

            workflows = defaultdict(list)
            for row in rows:
                workflows[row["workflow"]].append(row)
            selected = []
            while len(selected) < limit and any(workflows.values()):
                for workflow in sorted(workflows):
                    if workflows[workflow] and len(selected) < limit:
                        selected.append(workflows[workflow].pop(0))
            rows = selected
        for row in rows:
            state, questions, gold = (json.loads(row[k]) for k in ("state", "questions", "gold"))
            answers = {}
            for key, q in questions.items():
                g, kind = gold[key], q["type"]
                if kind == "noul":
                    answers[key] = {"type": kind, "noul": g["noul"]}
                    continue
                p = g["probabilities"]
                total = sum(p.values())
                if abs(total - 1) > 1e-5 or any(not 0 <= v <= 1 for v in p.values()):
                    raise ValueError(f"{split}: {row['id']}: {key}: invalid source probabilities")
                p = {k: v / total for k, v in p.items()}
                if kind == "choice":
                    answers[key] = {
                        "type": kind,
                        "choice": max(p, key=p.get),
                        "confidence": choice_confidence(list(p.values())),
                        "probabilities": p,
                    }
                else:
                    answers[key] = {
                        "type": kind,
                        "score": sum(int(k) * v for k, v in p.items()),
                        "confidence": score_confidence(
                            [p[str(i)] for i in range(len(q["criteria"]))]
                        ),
                        "probabilities": p,
                        "legend": {str(i): v for i, v in enumerate(q["criteria"])},
                    }
            identity = len(records) + 1
            record = TrainingExample(
                record_id=identity,
                group_id=identity,
                split=split,
                request={"state": state, "questions": questions},
                supervision={"answers": answers},
            )
            records.append(record)
            provenance.append(
                {
                    "record_id": identity,
                    "source_id": row["id"],
                    "workflow": row["workflow"],
                    "source_split": split,
                    "note": "soft distributions retained; rounding normalized within 1e-5; "
                    "Choice/Score derived summary fields recomputed for v3 contract",
                }
            )
    output = Path(output)
    atomic_write(output, "".join(json_text(r.model_dump()) + "\n" for r in records))
    write_json(
        output.with_suffix(".source.json"),
        {
            "source_path": os.path.relpath(source, output.parent),
            "target_source": "single teacher temperature=0.7, average of three samples; not human labels",
            "records": provenance,
        },
    )
    return len(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/benchmarks/typed_decisions"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/local/typed-decisions-training.jsonl")
    )
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    args = parser.parse_args()
    print(
        {
            "records": convert(args.source, args.output, args.train_limit, args.test_limit),
            "output": str(args.output),
        }
    )


if __name__ == "__main__":
    main()
