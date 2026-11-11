from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from uuid import uuid4

from decisionforge.eval.jevbench import latency_summary
from decisionforge.eval.provenance import method_metadata
from decisionforge.eval.source_identity import scoring_sources
from decisionforge.util import atomic_write, now, write_json


def latency_ms(values):
    result = latency_summary(values)
    return {"n": result["n"], "p50_ms": result["p50_s"], "p95_ms": result["p95_s"]}


def quality_view(samples, predictions, attempts):
    """Include attempted-but-incomplete eligibility without fabricating saved predictions."""
    completed = {p["id"] for p in predictions}
    last = {a["sample_id"]: a for a in attempts}
    result = list(predictions)
    for sample in samples:
        if sample.id in completed or sample.id not in last:
            continue
        attempt = last[sample.id]
        unknown = attempt["status"] == "outcome_unknown"
        result.append(
            {
                **attempt,
                "id": sample.id,
                "terminal": False,
                "status": "outcome_unknown" if unknown else "awaiting_retry",
                "outcome_unknown": unknown,
                "transport_ok": attempt.get("transport_ok", False),
                "protocol_ok": attempt.get("protocol_ok", False),
                "answer": attempt.get("answer"),
                "error": {"kind": "outcome_unknown"} if unknown else attempt.get("error"),
                "model": attempt.get("model", attempt["request"]["model"]),
                "probability_source": attempt.get("probability_source", "unavailable"),
            }
        )
    return result


def performance(store, benchmark, samples, *, local=False):
    attempts, predictions = store.report_attempts(), store.records("predictions")
    by_sample = {s.id: s for s in samples}
    views = {p["id"]: benchmark.prepare(by_sample[p["id"]], p) for p in predictions}
    attempts_by_session, predictions_by_session = defaultdict(list), defaultdict(list)
    for attempt in attempts:
        attempts_by_session[attempt["session_id"]].append(attempt)
    for prediction in predictions:
        predictions_by_session[prediction["session_id"]].append(prediction)
    sessions = []
    for session in store.records("sessions"):
        actual = attempts_by_session[session["id"]]
        terminal = predictions_by_session[session["id"]]
        valid = sum(views[p["id"]]["scoring_valid"] for p in terminal)
        window = session.get("window_seconds")
        sessions.append(
            {
                **session,
                "attempts": len(actual),
                "completed_samples": len(terminal),
                "scoring_valid_samples": valid,
                "success_qps": valid / window if window else None,
                "attempts_per_second": len(actual) / window if window else None,
            }
        )
    known = [a for a in attempts if a.get("attempt_latency_ms") is not None]
    successful = [a for a in known if a.get("transport_ok")]
    usage = [a["usage"] for a in attempts if a.get("usage") is not None]
    costs = [a["cost_usd"] for a in attempts if a.get("cost_usd") is not None]
    local = local or any(a.get("cost_basis") == "local_compute_unpriced" for a in attempts)
    return {
        "sessions": sessions,
        "attempt_latency_ms": latency_ms([a["attempt_latency_ms"] for a in known]),
        "successful_response_latency_ms": latency_ms([a["attempt_latency_ms"] for a in successful]),
        "sample_elapsed_ms": latency_ms([p["sample_elapsed_ms"] for p in predictions]),
        "failed_attempts": sum(not a.get("transport_ok") for a in known),
        "timeout_attempts": sum(
            a.get("error", {}).get("kind", "").endswith("Timeout") for a in known if a.get("error")
        ),
        "unknown_attempts": sum(a["status"] == "outcome_unknown" for a in attempts),
        "usage": {
            "input_tokens": sum(u["input_tokens"] for u in usage) if usage else None,
            "output_tokens": sum(u["output_tokens"] for u in usage) if usage else None,
            "known_attempts": len(usage),
            "missing_attempts": len(attempts) - len(usage),
            "basis": "sum_of_reported_usage_only",
        },
        "reported_cost_usd": sum(costs) if costs else None,
        "missing_cost_attempts": len(attempts) - len(costs),
        "price_per_1000_decisions_usd": sum(costs) * 1000 / len(predictions)
        if predictions and len(costs) == len(attempts)
        else None,
        "cost_basis": "local_compute_unpriced"
        if local
        else "reported_usage_cost"
        if costs
        else "unknown",
        "candidate_tasks": {
            "known_total": sum(a["candidate_task_count"] for a in attempts
                               if a.get("candidate_task_count") is not None),
            "known_attempts": sum(a.get("candidate_task_count") is not None for a in attempts),
            "basis": "logical_binary_tasks_per_http_attempt_not_measured_kernels",
        },
    }


def create_report(
    directory, store, samples, registry, benchmark, metrics, manifest, code, score_only=False
):
    predictions = store.records("predictions")
    view = quality_view(samples, predictions, store.report_attempts())
    result = benchmark.evaluate(samples, view, registry, metrics)
    result["quality_view"] = {
        "attempt_source": "durable_attempt_ledger",
        "incomplete_records": "report_only",
        "accuracy_missing_outcome_policy": "zero_when_attempted",
    }
    result.update(
        created_at=now(),
        inference_manifest=manifest,
        scoring_code=scoring_sources(code),
        performance=performance(
            store,
            benchmark,
            samples,
            local=manifest["config"]["model"].get("adapter") in {"unsloth_local", "laya_local", "openjev_local", "anyjev_local", "swift_local", "swift_chat_local"},
        ),
        performance_source="existing_attempts" if score_only else "recorded_sessions",
    )
    resolved_model_sources = {
        "response", "loaded_checkpoint", "loaded_backbone", "loaded_openjev_stage2_weights", "loaded_swift_checkpoint"
    }
    result["prediction_provenance"] = {
        **method_metadata(manifest["config"]["model"]),
        "actual_models": sorted(
            {
                p["model"]
                for p in predictions
                if p["model_identity_source"] in resolved_model_sources
            }
        ),
        "unresolved_model_predictions": sum(
            p["model_identity_source"] not in resolved_model_sources for p in predictions
        ),
        "probability_sources": sorted({p["probability_source"] for p in predictions}),
    }
    relative = Path("reports") / str(uuid4())
    destination = Path(directory) / relative
    write_json(destination / "metrics.json", result)
    lines = [
        f"# {benchmark.name} evaluation",
        "",
        f"Partial: {result['partial']}",
        f"Method: {result['prediction_provenance']['method']}",
        f"Probability sources: {', '.join(result['prediction_provenance']['probability_sources']) or 'unavailable'}",
        "",
        "| Metric | Value | Used / eligible |",
        "|---|---:|---:|",
    ]
    for name, metric in result["metrics"].items():
        lines.append(
            f"| {name} | {metric['value']} | {metric.get('used')} / {metric.get('eligible')} |"
        )
    lines += ["", "See metrics.json for denominators, versions, failures and session performance."]
    if result.get("request_protocol") == "single_question":
        lines += [
            "",
            "Request protocol: one independent question per request. The upstream Typed Decisions protocol sends five questions per case; these results use a different request protocol.",
            "References: mean of three samples from one teacher at temperature 0.7. Gold accuracy measures agreement with teacher modal labels. Soft Brier is the sum of squared class errors, not a class average.",
        ]
    if "by_tier" in result:
        lines += [
            "",
            "## JevBench public tiers",
            "",
            "| Tier | Samples | Accuracy |",
            "|---|---:|---:|",
        ]
        for tier, summary in result["by_tier"].items():
            lines.append(f"| {tier} | {summary['n_planned']} | {summary['accuracy']} |")
    if "oos" in result:
        oos = result["oos"]
        lines += [
            "",
            "Classification: native Choice, zero-shot over all 151 labels. Headline macro-F1 averages the complete fixed label catalog.",
            f"OOS precision/recall/F1: {oos['precision']} / {oos['recall']} / {oos['f1']}; support: {oos['support']}",
            f"In-scope macro-F1 (150 labels): {result['in_scope_macro_f1']['value']}",
        ]
    if manifest["config"]["model"].get("adapter") == "unsloth_local":
        lines += [
            "",
            "Inference: local saved Unsloth decision model, calibrated full-precision softmax. "
            "No HTTP service or API price is used; local GPU compute cost is not estimated.",
        ]
    if manifest["config"]["model"].get("adapter") == "laya_local":
        lines += ["", "Inference: official Laya saved model and calibrated native readout with "
                  "four-decimal probabilities. Native entropy confidence is preserved; Noul "
                  "scoring derives false=1-P(true). Local GPU compute cost is not estimated."]
    if manifest["config"]["model"].get("adapter") == "openjev_local":
        lines += ["", "Inference: OpenJev two-stage saved weights, one sampled rationale and the official "
                  "post-rationale label-logit softmax. Training phase calibrate is not independent "
                  "holdout calibration. Raw rationale/marker/sampling settings are retained; local GPU "
                  "compute cost is not estimated."]
    if manifest["config"]["model"].get("adapter") == "anyjev_local":
        lines += ["", "Inference: AnyJev L0 with fixed content-free prior, full Choice/Noul rotations "
                  "and one ordered Score pass. Confidence is maximum candidate probability, not "
                  "a calibrated probability of correctness. Tokens and GPU compute cost are unknown; "
                  "upstream workload counters are not GPU forward counts."]
    perf = result["performance"]
    lines += [
        "",
        "## Performance",
        "",
        f"Attempt latency p50/p95 (ms): {perf['attempt_latency_ms']['p50_ms']} / {perf['attempt_latency_ms']['p95_ms']}",
        f"Reported cost (USD): {perf['reported_cost_usd']}; unknown cost attempts: {perf['missing_cost_attempts']}",
        "",
        "| Session | Completed | Valid | Attempts | Success QPS | Attempts/s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for session in perf["sessions"]:
        lines.append(
            f"| {session['id']} | {session['completed_samples']} | {session['scoring_valid_samples']} | {session['attempts']} | {session['success_qps']} | {session['attempts_per_second']} |"
        )
    atomic_write(destination / "summary.md", "\n".join(lines) + "\n")
    return str(relative)
