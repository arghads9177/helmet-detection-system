"""Leakage guard (R4): no capture session may span two splits.

Cheap to run, catches the failure that would silently invalidate every downstream metric --
see docs/architecture/TDD.md SS3.
"""
from pathlib import Path

import pytest

from scripts.utilities.sessions import session_id_for

METADATA_DIR = Path("data/metadata")
CLEANED_DIR = Path("data/interim/cleaned")


def _sessions_in_split(split: str) -> set[str]:
    list_path = METADATA_DIR / f"{split}.txt"
    if not list_path.exists():
        pytest.skip(f"{list_path} not generated yet -- run scripts.dataset_split.split_by_session")
    sessions = set()
    for line in list_path.read_text().splitlines():
        if line.strip():
            sessions.add(session_id_for(Path(line.strip()), CLEANED_DIR))
    return sessions


def test_no_session_spans_two_splits():
    train_sessions = _sessions_in_split("train")
    val_sessions = _sessions_in_split("val")
    test_sessions = _sessions_in_split("test")

    assert not (train_sessions & val_sessions), f"Sessions leak train<->val: {train_sessions & val_sessions}"
    assert not (train_sessions & test_sessions), f"Sessions leak train<->test: {train_sessions & test_sessions}"
    assert not (val_sessions & test_sessions), f"Sessions leak val<->test: {val_sessions & test_sessions}"


def test_every_split_nonempty():
    for split in ("train", "val", "test"):
        list_path = METADATA_DIR / f"{split}.txt"
        if not list_path.exists():
            pytest.skip(f"{list_path} not generated yet -- run scripts.dataset_split.split_by_session")
        lines = [line for line in list_path.read_text().splitlines() if line.strip()]
        assert lines, f"{split}.txt is empty"
