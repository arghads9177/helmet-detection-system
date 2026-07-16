"""Split the annotated dataset into train/val/test by capture session -- never by random
frame (R4): whole sessions go to exactly one split, so near-duplicate frames from the same
video never leak across splits.

Additional constraint (TDD SS3): the test split must contain no_helmet instances from a
session that contributes zero frames to train, so test measures generalization to a new
scene rather than memorization. If only one session has no_helmet, that session stays in
train instead (so the model can learn the class at all) and this constraint is logged as
unmet rather than silently forced.

Images with no matching label file are excluded from the split (ambiguous: could be an
intentional negative sample or a missed annotation -- see scripts/validation/check_dataset.py).

Usage:
    uv run python -m scripts.dataset_split.split_by_session
"""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from scripts.utilities.config import load_config
from scripts.utilities.logging_setup import setup_logging
from scripts.utilities.sessions import find_images, session_id_for

logger = logging.getLogger(__name__)


def load_labels(annotations_dir: Path, stem: str) -> list[tuple[int, float, float, float, float]] | None:
    label_path = annotations_dir / f"{stem}.txt"
    if not label_path.exists():
        return None
    boxes = []
    for line in label_path.read_text().splitlines():
        tokens = line.split()
        if len(tokens) != 5:
            continue
        try:
            class_id = int(tokens[0])
            cx, cy, w, h = (float(t) for t in tokens[1:])
        except ValueError:
            continue
        boxes.append((class_id, cx, cy, w, h))
    return boxes


def group_by_session(
    images: list[Path], annotations_dir: Path, cleaned_dir: Path, no_helmet_class_id: int,
) -> tuple[dict[str, list[Path]], dict[str, int], list[Path]]:
    """Returns (session -> included images, session -> no_helmet instance count, excluded images)."""
    session_images: dict[str, list[Path]] = {}
    session_no_helmet: dict[str, int] = {}
    excluded: list[Path] = []

    for path in images:
        boxes = load_labels(annotations_dir, path.stem)
        if boxes is None:
            excluded.append(path)
            continue
        session = session_id_for(path, cleaned_dir)
        session_images.setdefault(session, []).append(path)
        no_helmet_count = sum(1 for class_id, *_ in boxes if class_id == no_helmet_class_id)
        session_no_helmet[session] = session_no_helmet.get(session, 0) + no_helmet_count

    return session_images, session_no_helmet, excluded


def assign_sessions(
    session_sizes: dict[str, int], no_helmet_sessions: set[str], ratios: dict[str, float],
) -> dict[str, str]:
    total = sum(session_sizes.values())
    targets = {split: total * ratio for split, ratio in ratios.items()}
    assigned: dict[str, str] = {}
    current_counts = {split: 0 for split in ratios}
    remaining = dict(session_sizes)

    if len(no_helmet_sessions) >= 2:
        test_target = targets["test"]
        forced_test = min(no_helmet_sessions, key=lambda s: abs(session_sizes[s] - test_target))
        assigned[forced_test] = "test"
        current_counts["test"] += remaining.pop(forced_test)
    elif len(no_helmet_sessions) == 1:
        logger.warning(
            "Only one no_helmet-bearing session (%s) -- keeping it in train so the model can "
            "learn the class. TDD SS3's cross-session test constraint cannot be met with "
            "current data.",
            next(iter(no_helmet_sessions)),
        )

    for session, size in sorted(remaining.items(), key=lambda kv: -kv[1]):
        best_split = max(ratios, key=lambda s: targets[s] - current_counts[s])
        assigned[session] = best_split
        current_counts[best_split] += size

    return assigned


def write_split_lists(assignment: dict[str, str], session_images: dict[str, list[Path]], metadata_dir: Path) -> None:
    by_split: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    for session, split in assignment.items():
        by_split[split].extend(session_images[session])

    for split, paths in by_split.items():
        list_path = metadata_dir / f"{split}.txt"
        with list_path.open("w") as f:
            for path in sorted(paths):
                f.write(f"{path.as_posix()}\n")
        logger.info("Wrote %s (%d images)", list_path, len(paths))


def materialize_split(
    assignment: dict[str, str], session_images: dict[str, list[Path]], annotations_dir: Path, processed_dir: Path,
) -> None:
    for session, split in assignment.items():
        images_out = processed_dir / split / "images"
        labels_out = processed_dir / split / "labels"
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)
        for image_path in session_images[session]:
            shutil.copy2(image_path, images_out / image_path.name)
            label_path = annotations_dir / f"{image_path.stem}.txt"
            shutil.copy2(label_path, labels_out / label_path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    setup_logging()
    data_cfg = load_config("data")
    raw_images_dir = Path(data_cfg["raw_dir"]) / "images"
    cleaned_dir = Path(data_cfg["interim_dir"]) / "cleaned"
    annotations_dir = Path(data_cfg["raw_dir"]) / "annotations"
    metadata_dir = Path(data_cfg["metadata_dir"])
    processed_dir = Path(data_cfg["processed_dir"])
    class_names = [line.strip() for line in Path(data_cfg["classes_file"]).read_text().splitlines() if line.strip()]
    no_helmet_class_id = class_names.index("no_helmet")
    ratios = {
        "train": data_cfg["split"]["train"],
        "val": data_cfg["split"]["val"],
        "test": data_cfg["split"]["test"],
    }

    images = find_images(raw_images_dir, cleaned_dir)
    session_images, session_no_helmet, excluded = group_by_session(
        images, annotations_dir, cleaned_dir, no_helmet_class_id,
    )
    for path in excluded:
        logger.warning("Excluded from split (no label file): %s", path)

    session_sizes = {session: len(paths) for session, paths in session_images.items()}
    no_helmet_sessions = {session for session, count in session_no_helmet.items() if count > 0}
    logger.info("Sessions: %s", session_sizes)
    logger.info("no_helmet-bearing sessions: %s", no_helmet_sessions)

    assignment = assign_sessions(session_sizes, no_helmet_sessions, ratios)

    total = sum(session_sizes.values())
    split_totals = {"train": 0, "val": 0, "test": 0}
    for session, split in assignment.items():
        split_totals[split] += session_sizes[session]
    for split, count in split_totals.items():
        logger.info("%s: %d images (%.1f%%), sessions=%s", split, count, 100 * count / total,
                    sorted(s for s, sp in assignment.items() if sp == split))

    test_no_helmet_from_excluded_train = any(
        assignment[s] == "test" and s in no_helmet_sessions for s in assignment
    )
    logger.info(
        "TDD SS3 cross-session no_helmet test constraint: %s",
        "met" if test_no_helmet_from_excluded_train else "NOT MET",
    )

    write_split_lists(assignment, session_images, metadata_dir)
    materialize_split(assignment, session_images, annotations_dir, processed_dir)
    logger.info("Done. Processed splits written to %s", processed_dir)


if __name__ == "__main__":
    main()
