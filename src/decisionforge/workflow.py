"""Connect the current generation, training and held-out local evaluation contracts."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from decisionforge.cli import execute as generate
from decisionforge.eval.models import EvalConfig, LocalModelConfig
from decisionforge.eval.runner import run_evaluation
from decisionforge.exporter import read_training_records
from decisionforge.training.config import TrainConfig
from decisionforge.training.runner import run_training, semantic_config
from decisionforge.util import digest


async def run_workflow(generation, training, evaluation, *, resume=False):
    """Use the actual outputs of each stage; never train on a paused generation."""
    if not isinstance(evaluation.model, LocalModelConfig):
        raise ValueError("The training workflow requires model.adapter: unsloth_local")
    splits = evaluation.data.splits
    if splits is None or len(splits) != 1 or splits[0] not in {"validation", "test"}:
        raise ValueError("The training workflow requires one held-out split: validation or test")
    generation = generation.model_copy(deep=True)
    training = training.model_copy(deep=True)
    evaluation = evaluation.model_copy(deep=True)
    requested_directory = Path(generation.output.directory).resolve()
    generated = await generate(generation)
    if generated["status"] != "complete":
        raise ValueError(
            f"Generation is {generated['status']}; resume it before training. "
            f"Checkpoint: {generated['run_directory']}"
        )
    directory = Path(generated["run_directory"]).resolve()
    training.data.path = str(directory / "accepted.json")
    # A changed generation must not collide with the previous model or evaluation.
    if directory != requested_directory:
        suffix = "_g_" + digest(str(directory))[:10]
        for owner, field in ((training.output, "directory"), (evaluation, "output")):
            path = Path(getattr(owner, field))
            setattr(owner, field, str(path.with_name(path.name + suffix)))
    training = TrainConfig.model_validate(training.model_dump())
    train_dir = Path(training.output.directory).resolve()
    manifest_path = train_dir / "manifest.json"
    trained = None
    if resume and manifest_path.is_file():
        saved = TrainConfig.model_validate(
            yaml.safe_load((train_dir / "config.yaml").read_text(encoding="utf-8"))
        )
        if semantic_config(training) != semantic_config(saved):
            raise ValueError("Workflow training configuration changed; use a new output directory")
        if list(read_training_records(Path(training.data.path))) != list(
            read_training_records(train_dir / "samples.jsonl")
        ):
            raise ValueError("Generated data changed since training; use a new output directory")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["status"] == "completed":
            trained = {
                "status": "completed",
                "run_directory": str(train_dir),
                "report": str(train_dir / "report.json"),
                "reused": True,
            }
    if trained is None:
        trained = run_training(training, resume=resume and manifest_path.is_file())
    train_dir = Path(trained["run_directory"]).resolve()
    evaluation.data.path = str(train_dir / "data" / f"{splits[0]}.jsonl")
    evaluation.model.path = str(train_dir / "model")
    evaluation = EvalConfig.model_validate(evaluation.model_dump())
    evaluated = await run_evaluation(
        evaluation,
        resume=resume and (Path(evaluation.output) / "manifest.json").is_file(),
    )
    return {"generation": generated, "training": trained, "evaluation": evaluated}
