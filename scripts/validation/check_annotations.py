"""Validate YOLO-format label files: malformed lines, out-of-range class IDs, out-of-bounds
boxes, non-positive dimensions, duplicate boxes.

Scripts, not spot checks -- spot checks don't scale and don't reproduce (SKILL.md).

Usage:
    uv run python -m scripts.validation.check_annotations
    uv run python -m scripts.validation.check_annotations --annotations-dir data/raw/annotations
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

from scripts.utilities.config import load_config
from scripts.utilities.logging_setup import setup_logging

logger = logging.getLogger(__name__)

# Two boxes in the same file, same class, are treated as duplicates if every coordinate is
# within this tolerance -- catches accidental double-clicks/re-labels, not just exact repeats.
DUPLICATE_COORD_TOLERANCE = 1e-3

# Boxes are allowed to extend this far past [0, 1] before being flagged -- annotation tools
# sometimes emit values like 1.0000001 from floating-point rounding at the image edge.
BOUNDS_TOLERANCE = 1e-3


@dataclass
class FileReport:
    path: Path
    issues: list[str] = field(default_factory=list)
    box_count: int = 0

    @property
    def ok(self) -> bool:
        return not self.issues


def _parse_line(line: str) -> tuple[int, float, float, float, float] | None:
    tokens = line.split()
    if len(tokens) != 5:
        return None
    try:
        class_id = int(tokens[0])
        cx, cy, w, h = (float(t) for t in tokens[1:])
    except ValueError:
        return None
    return class_id, cx, cy, w, h


def check_label_file(path: Path, num_classes: int) -> FileReport:
    report = FileReport(path=path)
    seen: list[tuple[int, float, float, float, float]] = []

    for line_no, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        parsed = _parse_line(line)
        if parsed is None:
            report.issues.append(f"line {line_no}: malformed ({raw_line!r})")
            continue

        class_id, cx, cy, w, h = parsed

        if not (0 <= class_id < num_classes):
            report.issues.append(f"line {line_no}: class_id {class_id} out of range [0, {num_classes})")
            continue

        if w <= 0 or h <= 0:
            report.issues.append(f"line {line_no}: non-positive width/height (w={w}, h={h})")
            continue

        x_min, x_max = cx - w / 2, cx + w / 2
        y_min, y_max = cy - h / 2, cy + h / 2
        if x_min < -BOUNDS_TOLERANCE or x_max > 1 + BOUNDS_TOLERANCE or \
           y_min < -BOUNDS_TOLERANCE or y_max > 1 + BOUNDS_TOLERANCE:
            report.issues.append(f"line {line_no}: box extends outside image bounds ({cx=}, {cy=}, {w=}, {h=})")
            continue

        if any(
            other_class == class_id
            and abs(other_cx - cx) < DUPLICATE_COORD_TOLERANCE
            and abs(other_cy - cy) < DUPLICATE_COORD_TOLERANCE
            and abs(other_w - w) < DUPLICATE_COORD_TOLERANCE
            and abs(other_h - h) < DUPLICATE_COORD_TOLERANCE
            for other_class, other_cx, other_cy, other_w, other_h in seen
        ):
            report.issues.append(f"line {line_no}: duplicate box (class={class_id}, {cx=}, {cy=})")
            continue

        seen.append((class_id, cx, cy, w, h))
        report.box_count += 1

    return report


def check_annotations(annotations_dir: Path, num_classes: int) -> list[FileReport]:
    label_files = sorted(p for p in annotations_dir.glob("*.txt") if p.stem != "classes")
    return [check_label_file(p, num_classes) for p in label_files]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-dir", default=None)
    parser.add_argument("--classes-file", default=None)
    args = parser.parse_args()

    setup_logging()
    data_cfg = load_config("data")
    annotations_dir = Path(args.annotations_dir) if args.annotations_dir else Path(data_cfg["raw_dir"]) / "annotations"
    classes_file = Path(args.classes_file) if args.classes_file else Path(data_cfg["classes_file"])
    num_classes = len([line for line in classes_file.read_text().splitlines() if line.strip()])

    reports = check_annotations(annotations_dir, num_classes)
    total_issues = sum(len(r.issues) for r in reports)
    total_boxes = sum(r.box_count for r in reports)

    for report in reports:
        for issue in report.issues:
            logger.warning("%s: %s", report.path.name, issue)

    logger.info(
        "Checked %d label files, %d boxes, %d issues found", len(reports), total_boxes, total_issues,
    )
    if total_issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
