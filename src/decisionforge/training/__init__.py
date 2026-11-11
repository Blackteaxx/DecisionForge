"""Decision training; importing this package never initializes training dependencies."""

from decisionforge.training.config import TrainConfig, load_config

__all__ = ["TrainConfig", "load_config"]
