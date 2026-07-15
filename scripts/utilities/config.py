"""Shared YAML config loader for scripts/ and ml/.

Usage:
    from scripts.utilities.config import load_config
    data_cfg = load_config("data")          # loads configs/data.yaml
    infer_cfg = load_config("inference")    # loads configs/inference.yaml
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs"


@lru_cache(maxsize=None)
def load_config(name: str) -> dict:
    path = CONFIGS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No such config: {path}")
    with path.open() as f:
        return yaml.safe_load(f) or {}
