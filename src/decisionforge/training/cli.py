from pathlib import Path

from decisionforge.training.config import load_config


def execute(args):
    config = load_config(args.config)
    if args.run_dir:
        config.output.directory = str(Path(args.run_dir).resolve())
    if args.validate_only:
        from decisionforge.training.runner import run_training

        return run_training(config, validate_only=True)
    from decisionforge.training.runner import run_training

    return run_training(config, resume=args.resume)
