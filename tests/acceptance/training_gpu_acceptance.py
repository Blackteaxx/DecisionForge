"""Real single-GPU acceptance: Typed Decisions, Qwen3.5-4B, interruption and reload.

Run explicitly on a prepared training GPU environment; not part of the CPU test suite.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from decisionforge.training.config import TrainConfig
from decisionforge.training.runner import run_training
from decisionforge.training.unsloth_backend import UnslothBackend
from decisionforge.util import write_json

MODEL = "Qwen/Qwen3.5-4B"


def calibration_fallback_check(work):
    from decisionforge.exporter import read_training_records
    from decisionforge.training.data import to_unsloth_row

    directory = work / "resume"
    records = list(read_training_records(directory / "samples.jsonl"))
    splits = json.loads((directory / "splits.json").read_text())["assignments"]
    calibration_ids = {a["record_id"] for a in splits if a["role"] == "calibration"}
    rows = []
    for record in records:
        if record.record_id not in calibration_ids:
            continue
        row = to_unsloth_row(record)
        keys = [key for key, q in row["questions"].items() if q["type"] == "choice"]
        if keys:
            row["questions"] = {k: row["questions"][k] for k in keys}
            row["gold"] = {k: row["gold"][k] for k in keys}
            rows.append(row)
    config = TrainConfig(
        data={"path": str(directory / "samples.jsonl")}, model={"name": str(directory / "model")}
    )
    backend = UnslothBackend(config, work)
    backend.environment()
    backend.model, backend.tokenizer = backend.api.from_pretrained(
        str(directory / "model"),
        dtype=backend.dtype,
        load_in_4bit=True,
        local_files_only=True,
        use_gradient_checkpointing=False,
    )
    backend.model.decision_config.update(
        temperature=[0.5, 2.0, 3.0], head_temperature=0.5, temperature_by_options={"score:3-5": 4.0}
    )
    public = records[0].request.model_dump(exclude_none=True)
    before = backend.api.predict(backend.model, backend.tokenizer, **public)
    items, report = backend.api.build_dataset(rows, backend.tokenizer, backend.model)
    assert report["skipped"] == 0 and report["truncated"] == 0
    questions = sum(len(i["labels"]) for i in items)
    assert questions >= 10
    backend.items["calibration"] = items
    backend.encoding["calibration"] = {"questions": questions}
    calibration = backend.calibrate()
    after = backend.api.predict(backend.model, backend.tokenizer, **public)
    for key, q in public["questions"].items():
        if q["type"] in {"score", "noul"}:
            assert before[key] == after[key], (key, before[key], after[key])
    assert calibration["relative_temperature_fitted_types"] == ["choice"]
    assert calibration["head_temperature_fitted"] is False
    assert backend.model.decision_config["temperature"][1:] == [2.0, 3.0]
    result = {
        "questions": questions,
        "calibration": calibration,
        "unfitted_predictions_unchanged": True,
        "decision_config": backend.model.decision_config,
    }
    backend.close()
    return result


def terminal_checkpoint_check(work, model_name, data_path):
    class TerminalInterrupt(UnslothBackend):
        once = True

        def prepare(self, data):
            result = super().prepare(data)
            owner = self

            class Interrupt(self.Callback):
                def on_save(self, args, state, control, **kwargs):
                    if owner.once and state.global_step == state.max_steps:
                        TerminalInterrupt.once = False
                        raise KeyboardInterrupt(
                            "Interrupted at terminal on_save before phase commit"
                        )

            self.trainer.add_callback(Interrupt())
            return result

    source = work / "input-terminal.jsonl"
    shutil.copyfile(data_path, source)
    config = TrainConfig(
        data={"path": str(source)},
        model={"name": model_name, "max_seq_length": 2048},
        lora={"rank": 8, "alpha": 8},
        training={
            "num_train_epochs": None,
            "max_steps": 2,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "logging_steps": 100,
            "save_steps": 2,
            "warmup_steps": 0,
        },
        calibration={"enabled": False},
        output={"directory": str(work / "terminal-stage")},
    )
    try:
        run_training(config, backend_factory=TerminalInterrupt)
        raise AssertionError("Expected terminal-step interruption")
    except KeyboardInterrupt:
        pass
    directory = Path(config.output.directory)
    assert json.loads((directory / "manifest.json").read_text())["stage"] == "training"
    checkpoint = directory / "checkpoints/checkpoint-2"
    names = (
        "adapter_model.safetensors",
        "joint_head.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    )
    before = {n: (checkpoint / n).read_bytes() for n in names}
    changed = config.model_copy(deep=True)
    changed.training.learning_rate *= 2
    try:
        run_training(changed, resume=True)
        raise AssertionError("Semantic config change was not rejected")
    except ValueError as exc:
        assert "incompatible" in str(exc)
    header = checkpoint / "adapter_config.json"
    original = header.read_text()
    header.write_text("X" + original[1:])
    try:
        run_training(config, resume=True)
        raise AssertionError("Same-size corrupt checkpoint JSON was not rejected")
    except ValueError as exc:
        assert "Corrupt" in str(exc)
    finally:
        header.write_text(original)
    source.unlink()
    run_training(config, resume=True)
    assert all(before[n] == (checkpoint / n).read_bytes() for n in names)
    report = json.loads((directory / "report.json").read_text())
    assert report["training"]["global_step"] == 2
    assert report["training"]["resume_action"] == "terminal checkpoint; no optimizer updates"
    assert report["training"]["last_logged_loss"] is not None
    assert "error" not in report
    return {
        "global_step": 2,
        "all_checkpoint_states_unchanged": True,
        "semantic_config_rejected": True,
        "same_size_json_corruption_rejected": True,
        "calibration_disabled": True,
        "report": str(directory / "report.json"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument(
        "--calibration-only",
        action="store_true",
        help="Check missing-type calibration on saved resume/model",
    )
    parser.add_argument(
        "--terminal-only",
        action="store_true",
        help="Run terminal-checkpoint boundary on existing evidence directory",
    )
    parser.add_argument(
        "--data", type=Path, default=Path("data/local/typed-decisions-training.jsonl")
    )
    args = parser.parse_args()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    evidence = {"model": args.model, "data": str(args.data.resolve()), "checks": {}}
    if args.calibration_only:
        path = work / "acceptance.json"
        if path.exists():
            evidence = json.loads(path.read_text())
        evidence["checks"]["unfitted_calibration_preserved"] = calibration_fallback_check(work)
        write_json(path, evidence)
        print(json.dumps({"status": "passed", "evidence": str(path)}))
        return
    if args.terminal_only:
        path = work / "acceptance.json"
        if path.exists():
            evidence = json.loads(path.read_text())
        evidence["checks"]["terminal_checkpoint_resume"] = terminal_checkpoint_check(
            work, args.model, args.data
        )
        write_json(path, evidence)
        print(json.dumps({"status": "passed", "evidence": str(path)}))
        return

    class InterruptBackend(UnslothBackend):
        interrupt_once = True

        def prepare(self, data):
            result = super().prepare(data)
            owner = self

            class Interrupt(self.Callback):
                def on_save(self, args, state, control, **kwargs):
                    if owner.interrupt_once and state.global_step == 1:
                        InterruptBackend.interrupt_once = False
                        raise KeyboardInterrupt(
                            "GPU acceptance interruption after complete checkpoint 1"
                        )

            self.trainer.add_callback(Interrupt())
            return result

    def config(name, source, steps, save_steps, merged):
        return TrainConfig(
            data={"path": str(source)},
            model={"name": args.model, "max_seq_length": 2048, "load_in_4bit": True},
            lora={"rank": 8, "alpha": 8},
            training={
                "num_train_epochs": None,
                "max_steps": steps,
                "batch_size": 1,
                "gradient_accumulation_steps": 1,
                "logging_steps": 1,
                "save_steps": save_steps,
                "warmup_steps": 0,
            },
            output={"directory": str(work / name), "save_merged": merged},
        )

    source = work / "input.jsonl"
    shutil.copyfile(args.data, source)
    c = config("resume", source, 3, 1, True)
    try:
        run_training(c, backend_factory=InterruptBackend)
        raise AssertionError("Expected real interruption")
    except KeyboardInterrupt:
        pass
    directory = Path(c.output.directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "interrupted" and manifest["stage"] == "training"
    import torch  # Unsloth has been initialized by the backend before this import.

    cp1 = directory / "checkpoints/checkpoint-1"
    first_optimizer = torch.load(cp1 / "optimizer.pt", weights_only=True, map_location="cpu")
    first_scheduler = torch.load(cp1 / "scheduler.pt", weights_only=True, map_location="cpu")
    assert first_scheduler["last_epoch"] == 1
    assert all(int(v["step"]) == 1 for v in first_optimizer["state"].values())
    frozen_splits = (directory / "splits.json").read_bytes()
    frozen_samples = (directory / "samples.jsonl").read_bytes()
    source.write_text("Original input deliberately modified after checkpoint.\n")
    run_training(c, resume=True)
    assert frozen_splits == (directory / "splits.json").read_bytes()
    assert frozen_samples == (directory / "samples.jsonl").read_bytes()
    final_optimizer = torch.load(
        directory / "checkpoints/checkpoint-3/optimizer.pt", weights_only=True, map_location="cpu"
    )
    final_scheduler = torch.load(
        directory / "checkpoints/checkpoint-3/scheduler.pt", weights_only=True, map_location="cpu"
    )
    assert final_scheduler["last_epoch"] == 3
    assert all(int(v["step"]) == 3 for v in final_optimizer["state"].values())
    report = json.loads((directory / "report.json").read_text())
    assert report["training"]["parameter_update_probe"] == {"lora": True, "head": True}
    for folder in ("model", "merged"):
        assert report["reload"][folder]["protocol_valid"]
        assert set(report["reload"][folder]["types"]) == {"choice", "noul", "score"}
    evidence["checks"]["real_checkpoint_resume"] = {
        "start_step": 1,
        "end_step": 3,
        "optimizer_steps": [1, 3],
        "scheduler_steps": [1, 3],
        "snapshot_preserved": True,
        "training": report["training"],
        "reload": report["reload"],
        "report": str(directory / "report.json"),
    }
    write_json(work / "acceptance.json", evidence)

    class ExportInterrupt(UnslothBackend):
        once = True

        def save(self):
            if self.once:
                ExportInterrupt.once = False
                raise KeyboardInterrupt("GPU acceptance interruption before export")
            return super().save()

    source2 = work / "input-final.jsonl"
    shutil.copyfile(args.data, source2)
    c2 = config("final-stage", source2, 2, 100, False)
    try:
        run_training(c2, backend_factory=ExportInterrupt)
        raise AssertionError("Expected export interruption")
    except KeyboardInterrupt:
        pass
    directory2 = Path(c2.output.directory)
    final2 = directory2 / "checkpoints/checkpoint-2"
    assert (final2 / "dforge_checkpoint.json").exists()
    before_optimizer = torch.load(final2 / "optimizer.pt", weights_only=True, map_location="cpu")
    before_head = (final2 / "joint_head.safetensors").read_bytes()
    before_lora = (final2 / "adapter_model.safetensors").read_bytes()
    source2.unlink()
    run_training(c2, resume=True)
    assert before_head == (final2 / "joint_head.safetensors").read_bytes()
    assert before_lora == (final2 / "adapter_model.safetensors").read_bytes()
    after_optimizer = torch.load(final2 / "optimizer.pt", weights_only=True, map_location="cpu")
    assert [int(v["step"]) for v in before_optimizer["state"].values()] == [
        int(v["step"]) for v in after_optimizer["state"].values()
    ]
    evidence["checks"]["short_training_final_stage_resume"] = {
        "steps": 2,
        "save_steps": 100,
        "full_final_checkpoint": True,
        "head_and_lora_unchanged": True,
        "optimizer_steps_unchanged": True,
        "report": str(directory2 / "report.json"),
    }
    write_json(work / "acceptance.json", evidence)
    evidence["checks"]["terminal_checkpoint_resume"] = terminal_checkpoint_check(
        work, args.model, args.data
    )
    evidence["checks"]["unfitted_calibration_preserved"] = calibration_fallback_check(work)
    write_json(work / "acceptance.json", evidence)
    print(json.dumps({"status": "passed", "evidence": str(work / "acceptance.json")}, indent=2))


if __name__ == "__main__":
    main()
