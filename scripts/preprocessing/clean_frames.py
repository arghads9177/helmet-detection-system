"""Clean extracted frames: drop corrupt/unreadable, near-blank, and blurry frames, dedup
near-duplicates via perceptual hash.

Reads a directory of frames written by scripts/preprocessing/extract_frames.py and writes the
surviving frames to data/interim/cleaned/<name>/. Disposable, regenerable from data/raw/.

Usage:
    uv run python -m scripts.preprocessing.clean_frames --source data/interim/extracted_frames/video_1
    uv run python -m scripts.preprocessing.clean_frames --source data/interim/extracted_frames
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image, UnidentifiedImageError

from scripts.utilities.config import load_config
from scripts.utilities.logging_setup import setup_logging

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

# Frames with pixel std-dev below this are treated as near-blank (solid color / lens-cap /
# corrupt-decode artifacts), not useful training signal.
BLANK_STD_THRESHOLD = 4.0

# Frames with Laplacian variance below this are treated as too blurred to annotate reliably.
# Variance-of-Laplacian is a standard cheap blur proxy: a sharp image has strong edges (high
# second-derivative variance); motion blur / defocus smooths edges out, lowering it.
BLUR_VARIANCE_THRESHOLD = 100.0

# Hamming distance below which two phash values are considered near-duplicates.
DEFAULT_DEDUP_HAMMING_THRESHOLD = 5


def is_blank(image: Image.Image, std_threshold: float = BLANK_STD_THRESHOLD) -> bool:
    arr = np.asarray(image.convert("L"), dtype=np.float32)
    return bool(arr.std() < std_threshold)


def is_blurry(image: Image.Image, variance_threshold: float = BLUR_VARIANCE_THRESHOLD) -> bool:
    arr = np.asarray(image.convert("L"), dtype=np.uint8)
    return bool(cv2.Laplacian(arr, cv2.CV_64F).var() < variance_threshold)


def load_image_safely(path: Path) -> Image.Image | None:
    try:
        image = Image.open(path)
        image.load()
        return image
    except (UnidentifiedImageError, OSError) as exc:
        logger.warning("Corrupt/unreadable frame, dropping: %s (%s)", path, exc)
        return None


def clean_frames(
    source_dir: Path,
    output_dir: Path,
    hamming_threshold: int = DEFAULT_DEDUP_HAMMING_THRESHOLD,
    blur_threshold: float = BLUR_VARIANCE_THRESHOLD,
) -> tuple[int, int]:
    """Clean a directory of frames. Returns (kept_count, total_count)."""
    frame_paths = sorted(p for p in source_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    output_dir.mkdir(parents=True, exist_ok=True)

    seen_hashes: list[imagehash.ImageHash] = []
    kept = 0
    for path in frame_paths:
        image = load_image_safely(path)
        if image is None:
            continue

        if is_blank(image):
            logger.warning("Near-blank frame, dropping: %s", path)
            continue

        if is_blurry(image, blur_threshold):
            logger.warning("Blurry frame, dropping: %s", path)
            continue

        frame_hash = imagehash.phash(image)
        if any(frame_hash - seen <= hamming_threshold for seen in seen_hashes):
            logger.debug("Near-duplicate frame, dropping: %s", path)
            continue

        seen_hashes.append(frame_hash)
        image.convert("RGB").save(output_dir / path.name)
        kept += 1

    logger.info(
        "Cleaned %s -> %s: kept %d / %d frames", source_dir, output_dir, kept, len(frame_paths),
    )
    return kept, len(frame_paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True,
        help="Directory of extracted frames, or a parent directory containing one subdir per video",
    )
    parser.add_argument(
        "--output-dir", default=None, help="Defaults to <interim_dir>/cleaned from configs/data.yaml",
    )
    parser.add_argument("--hamming-threshold", type=int, default=DEFAULT_DEDUP_HAMMING_THRESHOLD)
    parser.add_argument("--blur-threshold", type=float, default=BLUR_VARIANCE_THRESHOLD)
    args = parser.parse_args()

    setup_logging()
    data_cfg = load_config("data")
    output_root = Path(args.output_dir) if args.output_dir else Path(data_cfg["interim_dir"]) / "cleaned"

    source = Path(args.source)
    has_direct_images = any(p.suffix.lower() in IMAGE_SUFFIXES for p in source.iterdir())
    source_dirs = [source] if has_direct_images else sorted(p for p in source.iterdir() if p.is_dir())

    total_kept = total_frames = 0
    for src_dir in source_dirs:
        kept, total = clean_frames(
            src_dir, output_root / src_dir.name, args.hamming_threshold, args.blur_threshold,
        )
        total_kept += kept
        total_frames += total
    logger.info("Done: kept %d / %d frames overall", total_kept, total_frames)


if __name__ == "__main__":
    main()
