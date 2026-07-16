# Feature Spec — Phase 1, Slices 1.1–1.5

**Status:** Implementing | **Source:** `docs/workflow/implementation_plan.md` Phase 1

## Scope

Data engineering, up to and including frame extraction & cleaning. Annotation itself
(Slice 1.6) and everything after is out of scope for this slice batch.

Available raw data going in: `data/raw/videos/{video_1..4}.mp4` (4 short clips, two
resolutions: 640x360 and 360x640), `data/raw/images/` (46 `helmet_*` + 2 `no_helmet_*`
stills, already collected).

## 1.1 — Recon spike

- `scripts/preprocessing/extract_frames.py`: CLI, samples video(s) at
  `configs/data.yaml: frame_sample_fps` (1.5), writes JPEGs to
  `data/interim/extracted_frames/<video_stem>/`. Uses OpenCV `VideoCapture`, computes the
  frame-index stride from the source FPS (via `ffprobe`/`cv2.CAP_PROP_FPS`) rather than
  decoding every frame.
- Run it against all four clips.
- Measure resolution, and approximate head-size in px (manual/semi-manual — no helmet
  model exists yet, so this is a rough pixel measurement on the extracted frames, not an
  automated detector pass per the skill note that COCO-pretrained YOLO has no helmet
  concept).
- Write `docs/reports/recon_findings.md` with the decision table from the plan.
- Update PRD §4 "Assumed" rows with measured values.

## 1.2 — Dataset plan

- `docs/workflow/dataset_plan.md`: target counts, per-session breakdown, `no_helmet`
  floor and staging plan, negative samples, edge cases.
- Fill `data/metadata/dataset_info.json` (currently `{}`) to match.

## 1.3 — Annotation guidelines

- `docs/workflow/annotation_guidelines.md`, resolving the six cases from the plan table,
  before any labeling starts (labeling itself is Slice 1.6, not this batch).

## 1.4 — Collection (provenance only — footage already collected)

- Record source/date/camera/area for the existing videos and images into
  `data/metadata/dataset_info.json` under a `sources` list.
- No new capture in this batch; this slice is about documenting what's already in
  `data/raw/`.

## 1.5 — Frame extraction & cleaning

- `scripts/preprocessing/clean_frames.py`: reads a directory of extracted frames,
  drops corrupt/unreadable and near-blank frames (low-variance heuristic), dedups via
  perceptual hash (`imagehash.phash`, configurable Hamming-distance threshold), writes
  survivors to `data/interim/cleaned/<video_stem>/`.
- Both scripts: `logging` (no `print`), config via `scripts/utilities/config.py` +
  `configs/data.yaml`, thin CLI (`argparse`), reusable functions importable for tests.
- Tests: `tests/integration/test_preprocessing.py` using tiny synthetic fixtures
  (generated in-test, not committed binary fixtures) — stride math, corrupt-frame
  rejection, near-duplicate rejection via phash.

## Out of scope here

Slice 1.6 (annotation), 1.7 (QC/validation scripts), 1.8 (split). These need the
annotation tool output and come after this batch.
