"""Validate the dataset as a whole: corrupt images, orphan images/labels, near-duplicate
images across sessions, resolution anomalies. Writes data/metadata/statistics.csv.

Scripts, not spot checks -- spot checks don't scale and don't reproduce (SKILL.md).

Usage:
    uv run python -m scripts.validation.check_dataset
"""
from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import imagehash
from PIL import Image, UnidentifiedImageError

from scripts.utilities.config import load_config
from scripts.utilities.logging_setup import setup_logging
from scripts.utilities.sessions import find_images, session_id_for

logger = logging.getLogger(__name__)

# Hamming distance below which two images are treated as near-duplicates dataset-wide (a
# stricter check than the per-video dedup in clean_frames.py -- this one runs across every
# session, catching accidental cross-source duplicates).
DUPLICATE_HAMMING_THRESHOLD = 5

# Box-size buckets in pixels, matching the PRD SS4 / TDD SS5 evaluation slices.
SIZE_BUCKETS = [(0, 32, "<32px"), (32, 96, "32-96px"), (96, float("inf"), ">96px")]


@dataclass
class ImageRecord:
    path: Path
    session_id: str
    width: int
    height: int


def find_corrupt(images: list[Path]) -> list[Path]:
    corrupt = []
    for path in images:
        try:
            with Image.open(path) as image:
                image.load()
        except (UnidentifiedImageError, OSError):
            corrupt.append(path)
    return corrupt


def find_near_duplicates(images: list[Path]) -> list[tuple[Path, Path]]:
    hashes: list[tuple[Path, imagehash.ImageHash]] = []
    duplicates = []
    for path in images:
        try:
            with Image.open(path) as image:
                current_hash = imagehash.phash(image)
        except (UnidentifiedImageError, OSError):
            continue
        for other_path, other_hash in hashes:
            if current_hash - other_hash <= DUPLICATE_HAMMING_THRESHOLD:
                duplicates.append((other_path, path))
        hashes.append((path, current_hash))
    return duplicates


def find_orphans(images: list[Path], annotations_dir: Path) -> tuple[list[Path], list[Path]]:
    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in annotations_dir.glob("*.txt") if p.stem != "classes"}
    orphan_images = sorted(p for p in images if p.stem not in label_stems)
    orphan_labels = sorted(annotations_dir / f"{stem}.txt" for stem in label_stems - image_stems)
    return orphan_images, orphan_labels


def size_bucket(size_px: float) -> str:
    for low, high, label in SIZE_BUCKETS:
        if low <= size_px < high:
            return label
    return SIZE_BUCKETS[-1][2]


def compute_statistics(
    images: list[Path], annotations_dir: Path, class_names: list[str], cleaned_dir: Path,
) -> list[dict]:
    rows = []
    class_counts = {name: 0 for name in class_names}
    session_image_counts: dict[str, int] = {}
    session_instance_counts: dict[str, int] = {}
    size_counts = {label: 0 for _, _, label in SIZE_BUCKETS}

    for path in images:
        session = session_id_for(path, cleaned_dir)
        session_image_counts[session] = session_image_counts.get(session, 0) + 1

        label_path = annotations_dir / f"{path.stem}.txt"
        if not label_path.exists():
            continue

        try:
            with Image.open(path) as image:
                img_w, img_h = image.size
        except (UnidentifiedImageError, OSError):
            continue

        for line in label_path.read_text().splitlines():
            tokens = line.split()
            if len(tokens) != 5:
                continue
            try:
                class_id = int(tokens[0])
                _, _, w, h = (float(t) for t in tokens[1:])
            except ValueError:
                continue
            if not (0 <= class_id < len(class_names)):
                continue
            class_counts[class_names[class_id]] += 1
            session_instance_counts[session] = session_instance_counts.get(session, 0) + 1
            box_px = min(w * img_w, h * img_h)
            size_counts[size_bucket(box_px)] += 1

    for name, count in class_counts.items():
        rows.append({"metric": "class_count", "key": name, "value": count})
    for session, count in sorted(session_image_counts.items()):
        rows.append({"metric": "session_image_count", "key": session, "value": count})
    for session, count in sorted(session_instance_counts.items()):
        rows.append({"metric": "session_instance_count", "key": session, "value": count})
    for label, count in size_counts.items():
        rows.append({"metric": "box_size_distribution", "key": label, "value": count})

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics-out", default=None)
    args = parser.parse_args()

    setup_logging()
    data_cfg = load_config("data")
    raw_images_dir = Path(data_cfg["raw_dir"]) / "images"
    cleaned_dir = Path(data_cfg["interim_dir"]) / "cleaned"
    annotations_dir = Path(data_cfg["raw_dir"]) / "annotations"
    classes_file = Path(data_cfg["classes_file"])
    class_names = [line.strip() for line in classes_file.read_text().splitlines() if line.strip()]
    stats_out = Path(args.statistics_out) if args.statistics_out else Path(data_cfg["metadata_dir"]) / "statistics.csv"

    images = find_images(raw_images_dir, cleaned_dir)
    logger.info("Found %d images", len(images))

    corrupt = find_corrupt(images)
    for path in corrupt:
        logger.warning("Corrupt/unreadable image: %s", path)

    orphan_images, orphan_labels = find_orphans(images, annotations_dir)
    for path in orphan_images:
        logger.warning("Image with no matching label: %s", path)
    for path in orphan_labels:
        logger.warning("Label with no matching image: %s", path)

    duplicates = find_near_duplicates([p for p in images if p not in corrupt])
    for a, b in duplicates:
        logger.warning("Near-duplicate images: %s <-> %s", a, b)

    resolutions: dict[tuple[int, int], int] = {}
    for path in images:
        if path in corrupt:
            continue
        with Image.open(path) as image:
            resolutions[image.size] = resolutions.get(image.size, 0) + 1
    logger.info("Resolutions found: %s", resolutions)

    rows = compute_statistics(images, annotations_dir, class_names, cleaned_dir)
    stats_out.parent.mkdir(parents=True, exist_ok=True)
    with stats_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "key", "value"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %s", stats_out)

    logger.info(
        "Summary: %d images, %d corrupt, %d orphan images, %d orphan labels, %d near-duplicate pairs",
        len(images), len(corrupt), len(orphan_images), len(orphan_labels), len(duplicates),
    )


if __name__ == "__main__":
    main()
