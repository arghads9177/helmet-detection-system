# Feature Spec — Phase 2–3, Training & Evaluation (merged)

**Status:** Notebook ready to run | **Source:** `docs/workflow/implementation_plan.md` Phase 2–3
**Entry gate:** Phase 1 gate met (waived — see plan §Waivers) + OQ1, OQ3, OQ-M resolved.

## Scope

One Colab notebook (`notebooks/02_train_and_evaluate.ipynb`) covering dataset loading →
statistics gate → training config → training → training-curve gate → validation metrics +
threshold selection → test evaluation (run-once) → error analysis → save + promote → export.
No API/UI/inference-pipeline work — that's Phase 4+.

The notebook was drafted in Phase 0 (`b7c528c`) against a dataset that didn't exist yet. This
slice is: verify it against the *real* Phase 1 outputs now in `data/processed/`, fix what's
wrong, and make `configs/training.yaml` the actual source of truth it's supposed to be.

## What was found and fixed

1. **R4 leakage-check bug (cell "R4: split leakage gate").** `SESSION_OF` was
   `stem.split("_")[0]`, which collapses `video_1`/`video_2`/`video_3` all down to the string
   `"video"` — not the real session ids used everywhere else in the repo
   (`scripts/utilities/sessions.py::session_id_for`, `data/metadata/statistics.csv`). Fixed to
   mirror `session_id_for()` exactly: `video_\d+` prefix for video frames, `s04_stills_helmet`
   / `s05_stills_no_helmet` for the stills pools.

2. **`configs/training.yaml` incomplete (Slice 2.1).** The notebook's training-config cell
   already read `flipud`, `mixup`, `hsv_h/s/v`, `degrees`, `perspective`, `patience`, and
   `base_weights` — but with fallback defaults *in code*, because the YAML didn't have them.
   That's a config-as-code violation: the file that's supposed to be "the record of what
   produced a given result" wasn't actually recording those choices. Filled in explicitly:
   `base_weights: yolo11n.pt` (OQ-M — YOLO11n baseline), `flipud: 0.0` (documented disable —
   fixed camera orientation), `mixup: 0.1`, wide HSV jitter for lighting/helmet-color
   variation, `degrees: 10.0`, `patience: 20`.

3. **Notebook duplicated `scripts/utilities/config.py`.** The dataset-loading cell had a
   notebook-local `load_config()` mirror, commented "until ml/ exists" — but
   `scripts/utilities/config.py` already exists (built in Phase 1) and `scripts/` is on
   `sys.path` once the notebook `cd`s into the repo. Switched all three `load_config(...)`
   call sites to import and use the real shared helper instead of duplicating
   `yaml.safe_load`.

## Deliberately deferred: Slice 2.8 (promote to `ml/training/`, `ml/evaluation/`)

The plan calls for promoting the notebook's training/eval logic into `ml/training/train.py`
and `ml/evaluation/evaluate.py` with tests. **Decision (user, this session): defer until after
a real Colab run.** Reasons:

- Doing it properly needs `ultralytics` (pulls torch/CUDA wheels) added to `pyproject.toml`
  via `uv add` — a heavy install on this machine, for logic that hasn't been validated against
  a real GPU run yet.
- The plan's own phrasing: "Anything the notebook proves useful gets promoted into `ml/`" —
  promotion follows proof, not the other way around.

**Action item, not yet done:** once a training run completes on Colab and the notebook's
approach is confirmed to work end-to-end, promote per the table in the implementation plan
(§Slice 2.8) — `ml/training/train.py::train(config) -> RunResult`,
`ml/evaluation/evaluate.py::evaluate()` / `sweep_threshold()`, `ml/evaluation/errors.py` (the
IoU matcher currently inline in the error-analysis cell) — each with tests in `ml/tests/`.
This is a hard gate before Phase 4 starts (Phase 4 imports `ml/`, never the notebook).

## Verification done this session (no GPU, no Colab)

- `configs/training.yaml` parses as valid YAML with the new keys.
- Notebook JSON is well-formed after edits.
- `uv run pytest` — 23 passed (unaffected by these changes; no test coverage exists yet for
  the notebook logic itself, which is expected — see Slice 2.8 above).

## Not verified (needs an actual Colab run)

- Training itself: loss curves, convergence, PRD §2.1 numbers.
- The fixed `SESSION_OF` against a live run of the leakage-gate cell.
- Threshold sweep, error analysis, export/promotion cells.

## Out of scope for this slice

- Running the notebook on Colab (requires the user to actually execute it — GPU time,
  Drive/repo setup, OQ3 Colab-legality confirmation restated at the top of the notebook).
- `docs/reports/accuracy_report_<version>.md` — written only after a real run produces numbers.
- Slice 2.8 promotion (see above).
