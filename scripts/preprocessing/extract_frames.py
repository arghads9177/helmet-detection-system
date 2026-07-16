"""Extract frames from raw video(s) at a fixed sampling rate.

Sampling at ~1-2 FPS (configs/data.yaml: frame_sample_fps), not every frame -- consecutive
frames at native FPS are near-duplicates that inflate the dataset and are the primary
vector for split leakage (see TDD SS3, R4).

Usage:
    uv run python -m scripts.preprocessing.extract_frames --source data/raw/videos/video_1.mp4
    uv run python -m scripts.preprocessing.extract_frames --source data/raw/videos
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2

from scripts.utilities.config import load_config
from scripts.utilities.logging_setup import setup_logging

logger = logging.getLogger(__name__)

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}


def sampling_stride(source_fps: float, target_fps: float) -> int:
    if target_fps <= 0:
        raise ValueError(f"target_fps must be > 0, got {target_fps}")
    return max(1, round(source_fps / target_fps))


def extract_frames(video_path: Path, output_dir: Path, target_fps: float) -> int:
    """Extract frames from a single video at target_fps. Returns the number of frames written."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Could not open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = sampling_stride(source_fps, target_fps)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            out_path = output_dir / f"{video_path.stem}_frame{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            written += 1
        frame_idx += 1
    cap.release()

    logger.info(
        "Extracted %d frames from %s (source_fps=%.2f, stride=%d) -> %s",
        written, video_path.name, source_fps, stride, output_dir,
    )
    return written


def iter_videos(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(p for p in source.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Video file or directory of videos")
    parser.add_argument(
        "--output-dir", default=None,
        help="Defaults to <interim_dir>/extracted_frames from configs/data.yaml",
    )
    parser.add_argument("--fps", type=float, default=None, help="Override frame_sample_fps")
    args = parser.parse_args()

    setup_logging()
    data_cfg = load_config("data")
    target_fps = args.fps or data_cfg.get("frame_sample_fps", 1.5)
    output_root = Path(args.output_dir) if args.output_dir else Path(data_cfg["interim_dir"]) / "extracted_frames"

    source = Path(args.source)
    videos = iter_videos(source)
    if not videos:
        logger.warning("No videos found under %s", source)
        return

    total = 0
    for video_path in videos:
        out_dir = output_root / video_path.stem
        total += extract_frames(video_path, out_dir, target_fps)
    logger.info("Done: %d frames written across %d video(s)", total, len(videos))


if __name__ == "__main__":
    main()
