"""Standard logging setup, driven by configs/logging.yaml. Call setup_logging()
once at every entry point (scripts, ml/training/train.py, api-services/main.py).
"""
from __future__ import annotations

import logging.config
from pathlib import Path

import yaml

CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs"


def setup_logging(config_name: str = "logging") -> None:
    path = CONFIGS_DIR / f"{config_name}.yaml"
    with path.open() as f:
        config = yaml.safe_load(f)

    log_file = config.get("handlers", {}).get("file", {}).get("filename")
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logging.config.dictConfig(config)
