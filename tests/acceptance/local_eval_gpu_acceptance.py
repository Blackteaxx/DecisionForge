"""Real CUDA acceptance for the local eval adapter using retained training artifacts."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from decisionforge.eval.local_adapter import LocalAdapter
from decisionforge.eval.models import EvalConfig
from decisionforge.eval.runner import run_evaluation
from decisionforge.util import write_json

TRAINED = Path("outputs/train/qwen35-4b-full")


def official_check_worker(connection, options):
    """Acceptance-only cross-check against the SDK's independently formatted API."""
    from decisionforge.eval import local_worker

    original = local_worker.LocalRuntime.predict
    checked = set()

    def predict(runtime, public):
        response = original(runtime, public)
        kind = public["questions"]["decision"]["type"]
        if kind not in checked:
            with runtime.torch.inference_mode():
                official = runtime.decision.FastDecisionModel.predict(
                    runtime.model, runtime.tokenizer, public["state"], public["questions"]
                )["decision"]["probabilities"]
            full = response["answers"]["decision"]["probabilities"]
            assert set(official) == set(full), (official, full)
            delta = max(abs(official[label] - full[label]) for label in full)
            # Official API rounds to four decimal places; allow its display rounding.
            assert delta <= 0.000051, (kind, delta)
            response["official_comparison"] = {"type": kind, "max_probability_delta": delta}
            checked.add(kind)
        return response

    local_worker.LocalRuntime.predict = predict
    local_worker.worker_main(connection, options)


async def main_async(args):
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    evidence = {"checks": {}, "trained_run": str(args.trained_run)}
    for kind in ("model", "merged"):
        config = EvalConfig(
            benchmark="typed_decisions",
            benchmark_version="1-single-question",
            data={"path": str(args.data.resolve()), "splits": ["test"], "limit": args.limit},
            model={
                "adapter": "unsloth_local",
                "path": str(args.trained_run / kind),
                "load_in_4bit": True,
                "precision": "auto",
            },
            execution={"timeout": 300.0},
            output=str(work / kind),
        )

        async def factory(model, execution, transport=None):
            return await LocalAdapter(model, execution, worker_target=official_check_worker).start()

        result = await run_evaluation(
            config, adapter_factory=factory if args.compare_official else None
        )
        report = json.loads((Path(config.output) / result["report"] / "metrics.json").read_text())
        assert result["status"] == "completed", result
        assert report["counts"]["strict_valid"] == args.limit
        assert report["counts"]["scoring_invalid"] == 0
        assert report["metrics"]["soft_brier"]["used"] == args.limit
        assert set(report["by_type"]) == {"choice", "noul", "score"}
        assert report["prediction_provenance"]["unresolved_model_predictions"] == 0
        assert report["performance"]["cost_basis"] == "local_compute_unpriced"
        before = (Path(config.output) / "predictions.jsonl").read_bytes()
        resumed = await run_evaluation(config, resume=True)
        assert resumed["status"] == "completed"
        assert before == (Path(config.output) / "predictions.jsonl").read_bytes()
        scored = await run_evaluation(config, score_only=True)
        assert before == (Path(config.output) / "predictions.jsonl").read_bytes()
        evidence["checks"][kind] = {
            "status": "passed",
            "samples": args.limit,
            "strict_valid": report["counts"]["strict_valid"],
            "soft_brier": report["metrics"]["soft_brier"]["value"],
            "full_precision": True,
            "resume_no_new_predictions": True,
            "score_only": scored["status"],
            "local_model": report["inference_manifest"]["local_model"],
            "report": str(Path(config.output) / result["report"]),
        }
        if args.compare_official:
            comparisons = [
                row["raw_response"]["official_comparison"]
                for row in map(json.loads, before.splitlines())
                if "official_comparison" in (row.get("raw_response") or {})
            ]
            assert {item["type"] for item in comparisons} == {"choice", "noul", "score"}
            evidence["checks"][kind]["official_api_probability_comparison"] = comparisons
        write_json(work / "acceptance.json", evidence)
        print(json.dumps({"artifact": kind, "status": "passed", "samples": args.limit}), flush=True)

    config = EvalConfig(
        benchmark="typed_decisions",
        benchmark_version="1-single-question",
        data={"path": str(args.data.resolve()), "splits": ["test"], "limit": 10},
        model={"adapter": "unsloth_local", "path": str(args.trained_run / "model")},
        execution={"timeout": 300.0},
        output=str(work / "interrupted"),
    )

    class Interrupted(LocalAdapter):
        calls = 0

        async def send(self, payload, question):
            self.calls += 1
            if self.calls == 4:
                raise asyncio.CancelledError()
            return await super().send(payload, question)

    async def factory(model, execution, transport=None):
        return await Interrupted(model, execution).start()

    try:
        await run_evaluation(config, adapter_factory=factory)
        raise AssertionError("Expected intentional interruption")
    except asyncio.CancelledError:
        pass
    snapshot = Path(config.output) / "predictions.jsonl"
    before = [json.loads(line) for line in snapshot.read_text().splitlines()]
    assert len(before) == 3
    resumed = await run_evaluation(config, resume=True)
    after = [json.loads(line) for line in snapshot.read_text().splitlines()]
    assert len(after) == 10 and after[:3] == before
    config.model.path = "/deliberately/unavailable/model"
    config.data.path = "/deliberately/unavailable/data.jsonl"

    class NoSDK:
        def find_spec(self, name, *args):
            if name.split(".")[0] in {"torch", "unsloth", "transformers"}:
                raise AssertionError(f"Offline score imported {name}")

    import sys

    guard = NoSDK()
    sys.meta_path.insert(0, guard)
    try:
        scored = await run_evaluation(config, score_only=True)
    finally:
        sys.meta_path.remove(guard)
    assert scored["status"] == "scored"
    evidence["checks"]["interrupted_resume"] = {
        "completed_before_interrupt": 3,
        "completed_after_resume": 10,
        "previous_predictions_unchanged": True,
        "offline_score_without_model_data_sdk": True,
        "report": str(Path(config.output) / resumed["report"]),
    }
    write_json(work / "acceptance.json", evidence)
    print(json.dumps({"status": "passed", "evidence": str(work / "acceptance.json")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--trained-run", type=Path, default=TRAINED)
    parser.add_argument(
        "--data", type=Path, default=Path("data/benchmarks/typed_decisions/processed/test.jsonl")
    )
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--compare-official", action="store_true")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
