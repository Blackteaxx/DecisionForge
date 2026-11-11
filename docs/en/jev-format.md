# JEV data format

[中文](../zh/jev-format.md) · [README](../../README.en.md)

This branding migration changes schema_version namespaces to `decisionforge.*`. Migrate the identifier in older records before use. JEV field structure, decision types and scoring semantics are unchanged.

Accepted and failed training files are arrays of `decisionforge.training.v3` records. Each preserves schema_version, record_id, group_id, split, request and supervision. Generation metadata lives separately.

```json
{
  "schema_version": "decisionforge.training.v3",
  "record_id": 1,
  "group_id": 1,
  "split": "train",
  "request": {
    "state": "France is a European country.",
    "questions": {
      "q1": {"type": "choice", "instructions": "What is the capital of France?",
             "criteria": {"A": "Rome", "B": "Paris"}}
    }
  },
  "supervision": {"answers": {
    "q1": {"type": "choice", "choice": "B", "confidence": 1.0,
           "probabilities": {"A": 0.0, "B": 1.0}}
  }}
}
```

Choice criteria contain 2–255 unique options. Noul may omit criteria; supervision contains type and noul (0 or 1). Score criteria contain 2–10 ordered levels; supervision includes score, legend, probabilities and confidence.

Generated Choice keys use A, B, C, …, AA, AB, …. Options are shuffled deterministically using `execution.seed` and sample identity before assigning keys. The answer choice and probabilities are remapped together. Correct answers are not fixed to one letter, and resuming does not reshuffle an existing sample.

Hard supervision uses one-hot probabilities. `confidence=1` represents certainty of that training target, not measured predictive confidence. Labels must pass the checker; labels in failed records are not accepted supervision.

Empty context is an empty string. Questions belong in instructions; Noul places the original question and proposition in an instructions object. Do not substitute question text for background state.

`meta.json.records` keys are public record IDs and identify output file and source. Failed records without valid JEV have output=null and therefore do not appear in accepted.json or fail.json.

Evaluation reads these arrays and adapts them to decision inputs. Export `requests`, `jev-api`, `jsonl` or `chat-sft` using `dforge data export --format ...`.

```bash
uv run dforge schema export --directory schemas
uv run dforge data validate --input outputs/run_001/accepted.json
```

JEV decision semantics and field structure remain unchanged by project branding.
