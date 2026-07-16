# Feature Spec — Phase 1, Slice 1.7: QC & Validation

**Status:** Implementing | **Source:** `docs/workflow/implementation_plan.md` Phase 1, Slice 1.7

## Context

Annotation (Slice 1.6) is done: 135 images (88 stills in `data/raw/images/`, 47 curated
video frames in `data/interim/cleaned/`), YOLO-format labels exported to
`data/raw/annotations/` (131 `.txt` files). A quick manual check already found 4 images with
no label file: `helmet_21`, `helmet_24`, `video_2_frame000360`, `video_2_frame000380`. This
slice builds the scripts to check this systematically instead of by hand, per the "scripts,
not spot checks" rule.

## `scripts/validation/check_annotations.py`

Per-label-file checks against `data/metadata/class_names.txt`:
- Malformed lines (wrong token count, non-numeric fields)
- Class ID out of range (must be 0 or 1)
- Box coordinates out of `[0, 1]` or extending past the image bounds (`cx +/- w/2`, `cy +/- h/2`)
- Non-positive width/height
- Duplicate boxes within a file (same class, near-identical coordinates)

Outputs a per-file issue list and a summary count. Exit non-zero if any issue found (so it
can gate CI/pre-training later), but this run is interactive/reporting.

## `scripts/validation/check_dataset.py`

- Corrupt/unreadable images (reuses the same PIL-load-and-catch pattern as
  `scripts/preprocessing/clean_frames.py`)
- Orphan images (image with no matching label file) and orphan labels (label with no
  matching image)
- Near-duplicate images across the whole dataset (perceptual hash, catches accidental
  duplicates missed by the earlier per-video dedup, e.g. across sessions)
- Resolution/size anomalies (summary of distinct resolutions found)
- Writes `data/metadata/statistics.csv` (committed): per-class instance counts, per-session
  image and instance counts, box-size distribution bucketed at <32px / 32-96px / >96px
  (matching the PRD §4 / TDD §5 evaluation slices)

Session assignment is by path/filename convention, matching `data/metadata/dataset_info.json`:
`data/raw/images/helmet_*` -> `s04_stills_helmet`, `data/raw/images/no_helmet_*` ->
`s05_stills_no_helmet`, `data/interim/cleaned/video_1/*` -> `s01_broadcast_tour`,
`video_2/*` -> `s02_workshop_handheld`, `video_3/*` -> `s03_selfie_hardhat`.

## Tests

`tests/integration/test_validation.py` using tiny synthetic fixtures (tmp_path), covering:
malformed line detection, out-of-bounds box detection, duplicate box detection, orphan
image/label detection, corrupt file detection.

## Out of scope here

Fixing any issues found (e.g. re-labeling the 4 missing images) is a follow-up decision for
the user, not silently auto-fixed by this slice.
