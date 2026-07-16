"""Session-ID assignment shared by scripts/validation/ and scripts/dataset_split/.

Session mapping mirrors data/metadata/dataset_info.json's `sources` list: each source video
gets its own session (by containing folder under data/interim/cleaned/), and the two stills
pools are split by class-name prefix.
"""
from __future__ import annotations

from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def session_id_for(path: Path, cleaned_dir: Path) -> str:
    try:
        rel = path.relative_to(cleaned_dir)
        return rel.parts[0]
    except ValueError:
        pass
    if path.name.startswith("helmet_"):
        return "s04_stills_helmet"
    if path.name.startswith("no_helmet_"):
        return "s05_stills_no_helmet"
    return "unknown"


def find_images(raw_images_dir: Path, cleaned_dir: Path) -> list[Path]:
    images = []
    if raw_images_dir.exists():
        images += [p for p in raw_images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
    if cleaned_dir.exists():
        images += [p for p in cleaned_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES]
    return sorted(images)
