# Implementation Plan — Helmet Detection System

**Author:** Argha Dey Sarkar | **Date:** 2026-07-15 | **Status:** Active
**Sources:** `docs/requirements/PRD.md` · `docs/architecture/TDD.md`

> This is the working document for building the POC. It expands the Level-1 POC/MVP workflow
> (Phases 0–8) into concrete, gated tasks with named file destinations.
> **The PRD defines *what* and *why*. The TDD defines *how*. This plan defines *in what order*,
> *with what verification*, and *what to do when it fails*.**

---

## How to use this document

1. **Never start a phase whose entry gate is unmet.** The gates are the whole point — every
   listed gate exists because skipping it produces a failure that surfaces late and costs a
   rebuild. If work is needed from a later phase, backfill the gate or record an explicit waiver
   in §Waivers.
2. **Work one vertical slice at a time.** Each slice gets a short spec in `.claude/feature-specs/`
   *before* implementation, and lands *with* its tests. Not tests "after".
3. **Check off the Definition of Done** at each phase end. An unchecked box is not "mostly done";
   it's the next thing to do.
4. **When a gate fails**, follow that phase's *If the gate fails* branch. Do not proceed and
   hope Phase 3 forgives it. It won't.
5. **Update the tracker below** as phases complete. This file is the source of truth for "where
   are we".

### Progress tracker

| Phase | Name | Status | Gate met? |
|---|---|---|---|
| 0 | Problem Definition | 🟢 Gate met — PRD ✅, TDD ✅, blocking OQs resolved, repo scaffolded | ✅ |
| 1 | Data Engineering | 🟢 Gate met with 2 explicit waivers (no_helmet floor, negative samples) — Slices 1.1–1.8 complete | ✅ (waived) |
| **2–3** | **Training & Evaluation** (merged — one notebook) | ⚪ Not started | ❌ |
| 4 | Inference Pipeline | ⚪ Not started | ❌ |
| 5 | REST API | ⚪ Not started | ❌ |
| 6 | Angular Dashboard | ⚪ Not started | ❌ |
| 7 | Testing | ⚪ Not started | ❌ |
| 8 | Deployment | ⚪ Not started | ❌ |

### Effort shape (rough, not commitments)

| Phase | Relative effort | Note |
|---|---|---|
| 0 | Days | Mostly waiting on decisions, not work |
| **1** | **Weeks — dominates the project** | Collection + annotation. This is the schedule |
| **2–3** | Days–weeks | One Colab notebook. Training YOLO11n is fast; the Slice 2.6 optimization loops are what add up |
| 4 | Days | |
| 5 | Days | |
| **6** | **1–2 weeks** | Angular, not Streamlit — see R7 |
| 7 | Days | |
| 8 | Days | Local Docker only |

The honest summary: **this project is a data project wearing a modeling project's clothes.**
The phases after it are well-trodden; Phase 1 is where it succeeds or fails.

---

## Phase 0 — Problem Definition & Requirement Analysis

**Status:** In progress. **Environment:** anywhere — this is writing and deciding, no code.

### Done

- [x] `docs/requirements/PRD.md` — numeric success criteria (§2.1/§2.2), full technical
      requirements table (§4), cost-of-errors (§7), scope (§8)
- [x] `docs/architecture/TDD.md` — model family, data pipeline, module design, API, deployment,
      risks

### 0.A — Resolve blocking decisions ⚠️ **DO THIS FIRST**

These are PRD §10 open questions. Each one, left open, invalidates work done downstream.
**They are ordered by cost-of-delay, not by difficulty.**

| # | Question | Blocks | Resolution (2026-07-15) |
|---|---|---|---|
| **OQ3** | Is factory footage cleared for use? Consent, on-prem-only storage, retention? | **Phase 1 start** | ✅ **Resolved.** Cleared for use; cloud storage/processing permitted (no on-prem-only restriction). Colab remains a legal venue for training. |
| **OQ4** | How many `no_helmet` instances can realistically be gathered? | **Phase 1 planning** | ✅ **Resolved (provisional).** Volume unknown — measure via **Slice 1.1** recon + **Slice 1.2** dataset plan. Default assumption: natural violations will be insufficient; **plan to stage** and confirm the floor after recon. |
| **OQ1** | Does the dev machine have a CUDA GPU? | Phase 2 sizing, PRD §2.2 | ✅ **Resolved.** No local GPU. Training moves to Colab/cloud GPU. Local inference is CPU-only — the ≥10 FPS target is **not achievable as scoped**; must be re-judged in Phase 7.3 or renegotiated (frame-skipping / smaller `img_size` / GPU inference hardware) before that gate. |
| **OQ2** | Real camera resolution / angle / worker distance → head size in px? | Phase 1 annotation | Still deferred — answered empirically by **Slice 1.1**, not by asking. |
| **OQ7** | Angular hard requirement, or is Streamlit acceptable? | Phase 6 scope | ✅ **Resolved.** Angular confirmed as a hard requirement. |
| **OQ5** | AGPL-3.0 — internal-only, or productized later? | Phase 8 / commercial | ✅ **Resolved.** Internal-only; not a blocker. |
| **OQ6** | Violation log retention + backend (SQLite OK?) | M5 | Deferred to Phase 5 (low cost to defer). |
| **OQ8** | Dedup window (proposed 30 s) | M5 | Deferred to Phase 5; 30 s default stands. |
| **OQ-M** | **YOLO11 vs. the templates' `yolo26s`** — which model family is the house standard? | Phase 2 | ✅ **Resolved.** YOLO11 (n baseline, s escalation) is the house standard. |

**Minimum to unblock Phase 1:** OQ3 ✅ and OQ4 ✅ (provisional — recon still required). **Minimum to unblock Phase 2:** OQ1 ✅ and OQ-M ✅.

Full resolutions recorded in `docs/requirements/PRD.md` §10.

### 0.B — Scaffold the repository

```bash
python <skill>/scripts/init_cv_project.py helmet-detection-system --path .. --python 3.11
```

⚠️ **This repo is not empty.** It has a committed `README.md` (your feature list) and a 218-line
`.gitignore`. The scaffold generates its own `README.md` skeleton, `.gitignore`, and
`pyproject.toml`. **Before running it:** confirm on a scratch copy whether it overwrites or
skips existing files. If it overwrites, scaffold to a temp directory and merge in, preserving:

- `README.md` — keep the feature list; add the scaffold's install/usage sections around it
- `.gitignore` — union both; verify `data/`, `ml/models/`, `outputs/`, `artifacts/`, `.env` are
  all ignored, and that `data/metadata/`, `artifacts/experiment_results/**/metadata.json`,
  `tests/fixtures/` are **not**
- `docs/` — already populated with PRD/TDD/this plan; must survive

**Verify:**

```bash
uv sync && uv run pytest          # scaffold's smoke test must pass
git status                        # no data/weights staged; docs/ intact
```

### 0.C — Encode conventions

- [x] `.claude/CLAUDE.md` written by the scaffold — verified: reflects uv/config/logging/layout/
      data/tests conventions; also documents the exFAT-drive venv workaround for this machine
- [ ] `docs/workflow/annotation_guidelines.md` — **drafted before Phase 1.3** (see Slice 1.3) —
      not yet written; correctly deferred to just before Slice 1.3, not part of Phase 0 gate

### 🚦 Gate → Phase 1

- [x] PRD + TDD approved (not just written — **approved**)
- [x] **OQ3 resolved** — cleared for use, cloud storage/processing permitted
- [x] **OQ4 answered (provisional)** — volume unknown; staging planned as default, to be
      confirmed by Slice 1.1 recon + Slice 1.2 dataset plan
- [x] Repo scaffolded; `uv sync && uv run pytest` green (2 passed); existing README/.gitignore
      preserved and merged with the scaffold's CV-specific rules
- [x] `uv export --no-dev -o requirements.txt` committed

**Note:** this working copy sits on an exFAT drive, which doesn't support symlinks that
`uv sync` needs for `.venv`. Set `UV_PROJECT_ENVIRONMENT=~/.venvs/helmet-detection-system`
before running `uv`/`uv run` — documented in `.claude/CLAUDE.md`.

**If the gate fails:** OQ3 unresolved → **do not collect footage.** OQ4 says violations are
unobtainable in useful numbers → stop and revisit scope with stakeholders; a helmet detector that
can't be taught what a bare head looks like isn't a POC, it's a `helmet`-class classifier.

---

## Phase 1 — Data Engineering ⭐ **The phase that decides the project**

**Environment:** local, **no GPU**. uv venv, OpenCV, Label Studio / CVAT / Roboflow.
**Entry gate:** Phase 0 gate met.

### Status (2026-07-16)

Slices 1.1–1.5 implemented against the in-hand data: 4 YouTube clips of the IISCO plant
(`data/raw/videos/`) + downloaded stock images (`data/raw/images/`). **Confirmed by the
user: there is no factory site access for this POC** — the entire dataset is public/
downloaded material, not first-party consented factory footage. This is recorded as an
explicit waiver above (see §Waivers), not an open question. **Before starting Slice 1.6
(annotation):**

- ✅ **Provenance resolved (as a scope decision, not a compliance clearance).** Every
  session in `data/metadata/dataset_info.json` is tagged `source_type: youtube_public` or
  `downloaded_stock`, internal-POC-use-only, no redistribution. This dataset is a proxy for
  feasibility testing, not evidence about the real deployment site.
- ⚠️ **Camera-representativeness gap stays open.** None of the 4 clips is fixed-CCTV footage
  at the real deployment distance/angle (see `docs/reports/recon_findings.md`).
  Background/crowd heads already measure below the 32 px floor — treat `img_size` 960 as a
  live option. Real site footage is still needed before Phase 3 results can be presented as
  evidence for the actual deployment, not just proof the pipeline works.
- `no_helmet` floor (150 instances, ≥3 sessions): **NOT MET — measured 2026-07-16 via
  `scripts/validation/check_dataset.py` against the real Slice 1.6 labels.** Actual count:
  **79** `no_helmet` instances (target 150), from only **2** sessions (`video_1`: 35,
  `s05_stills_no_helmet`: 44) — `video_2` and `video_3` contributed **zero** labeled
  `no_helmet` instances. Notably, 2 of the 4 orphan (unlabeled) images are
  `video_2_frame000360`/`000380` — possibly the bare-headed-subject frames recon (Slice 1.1)
  described in `video_2`, missed during labeling.
  **Decision 2026-07-16 (user):** proceed to Slice 1.8 without collecting more data now;
  mitigate via train-split oversampling of `no_helmet`-containing frames in Phase 2
  (Slice 2.1), not by collecting more or re-checking the orphan frames. **Caveat, stated
  plainly because it doesn't fully close the gap:** oversampling/augmentation reshapes the
  existing 79 instances, it cannot manufacture new sessions or new information — the
  cross-session generalization risk (TDD §3, only 2 sessions contain `no_helmet`) stays
  open regardless. Recorded as a waiver below. **Watch `no_helmet` recall specifically at
  the Phase 3 gate** (PRD §2.1 target ≥ 0.90) — if it underperforms, this is the first place
  to look.

### Slice 1.1 — Recon spike: validate the PRD §4 assumptions

**Do this before collecting at scale, and before annotating anything.** Every "Assumed" row in
PRD §4 is a guess; this slice turns them into measurements.

> **Note on what this spike can and cannot do.** COCO-pretrained YOLO has a `person` class and
> **no concept of helmets** — it cannot give a zero-shot helmet signal. Its use here is as a
> *measuring instrument*: run person detection on real frames to locate people, then measure the
> **head region size in pixels** at realistic working distances. That's the number that decides
> `img_size` and model size. A public-dataset helmet model may optionally be run for a rough
> domain-gap smell test, but its numbers are **not** evidence about our future model.

1. Obtain a short sample clip from each candidate camera/area (minutes, not hours)
2. `scripts/preprocessing/extract_frames.py` → `data/interim/extracted_frames/`
3. Measure: resolution, head size in px at near/mid/far, camera angle, lighting range, occlusion
   frequency, crowd density
4. Write findings to `docs/reports/recon_findings.md`

**Decision output:**

| Finding | Consequence |
|---|---|
| Heads ≥ 96 px | YOLO11n @ 640 is comfortable. Proceed as designed |
| Heads 32–96 px | Proceed at 640; watch the small-object eval slice (notebook §7–8) |
| **Heads < 32 px** | **Update TDD §2/§4 before annotating:** `img_size` → 960 first, then consider YOLO11s. Or move/add cameras — a resolution problem is cheaper to fix with a camera than with a model |
| Angle is top-down | Heads occlude helmets differently; revisit the class definition and guidelines |

**Update PRD §4** with measured values, replacing the **Assumed** markers. This closes OQ2.

### Slice 1.2 — Dataset plan

- `docs/workflow/dataset_plan.md` + `data/metadata/dataset_info.json`
- Must specify: target counts per class, **per-session** breakdown, sessions/cameras/areas,
  lighting conditions, **negative samples** (compliant + empty scenes), edge cases
  (occlusion, blur, crowding, backlight, helmet-like confusers)
- **Set an explicit `no_helmet` floor.** Per R2, this is the binding constraint. State the
  number, and state how staging will reach it.

### Slice 1.3 — Annotation guidelines ⚠️ **Before any labeling**

`docs/workflow/annotation_guidelines.md` must resolve, with example images:

| Case | Ruling (from PRD §3 / TDD §3) |
|---|---|
| Helmet carried, not worn | **Not** `helmet`. The class is *head wearing a helmet* |
| Cap / hairnet / bare head | `no_helmet` |
| Partially occluded head | Label if the head is identifiable |
| Head at frame edge | Label if > ~50% visible |
| Motion-blurred head | Label if classifiable by a human |
| Box extent | **Head region**, not whole person |

Ambiguity here produces label noise, and label noise is indistinguishable from model error in
Phase 2–3 — you will burn weeks tuning a model against your own inconsistent labels.

### Slice 1.4 — Collection

- Raw footage → `data/raw/videos/` — **immutable, read-only**
- Multi-session, multi-lighting, per the plan. **Stage `no_helmet` scenarios** if OQ4 says natural
  violations are too rare
- Record provenance (source, date, camera, area) in `data/metadata/dataset_info.json`
- ⚠️ Honor OQ3's storage ruling from the first byte — if on-prem-only, footage never touches a
  cloud bucket or W&B

### Slice 1.5 — Frame extraction & cleaning

- `scripts/preprocessing/extract_frames.py` — `frame_sample_fps: 1.5` from `configs/data.yaml`.
  **Not every frame:** 30 FPS neighbors are near-duplicates that inflate the dataset, add no
  information, and are the primary vector for split leakage (R4)
- `scripts/preprocessing/clean_frames.py` — dedup (perceptual hash), drop corrupt/blank frames
- Output → `data/interim/cleaned/`. Disposable, regenerable from `data/raw/`

### Slice 1.6 — Annotation

- Tool: Label Studio / CVAT / Roboflow. Export **YOLO format** → `data/raw/annotations/`
- Classes exactly: `helmet`, `no_helmet` → `data/metadata/class_names.txt`
- **Auto-label assist** only once a model exists (post-training loop), and then:
  - Human-review **every** auto-labeled frame
  - ⚠️ **Never auto-label the test split** (R9) — it bakes the model's blind spots into the
    yardstick, and a contaminated test set makes every Phase 3 number meaningless

### Slice 1.7 — QC & validation

Scripts, not spot checks — spot checks don't scale and don't reproduce.

- `scripts/validation/check_annotations.py` — wrong/missing labels, malformed boxes, duplicates,
  out-of-range coords, class-id sanity
- `scripts/validation/check_dataset.py` — corrupt files, duplicate images, orphan labels/images,
  size anomalies
- Output → `data/metadata/statistics.csv` (committed): per-class counts, per-session counts,
  box-size distribution

**Review the class balance here.** If `no_helmet` is a tiny fraction, that's R2 arriving — go back
to 1.4 and stage more. It is far cheaper now than after training.

### Slice 1.8 — Split

- `scripts/dataset_split/split_by_session.py` — 80/10/10, `strategy: by_session`
- **Never random-frame** (R4). Whole sessions go to exactly one split
- **Additional constraint (TDD §3):** the test split must contain `no_helmet` instances from ≥1
  session contributing **zero** frames to train. Otherwise the test set measures memorization of a
  scene, not generalization
- Split lists → `data/metadata/{train,val,test}.txt` — **committed**; they define the experiment
- Final data → `data/processed/{train,val,test}/`
- **Leakage test** → `tests/integration/test_split_integrity.py`: assert no session appears in two
  splits. This test is cheap and it guards the failure that would silently invalidate everything
  downstream

### 🚦 Gate → Phase 2–3

- [x] PRD §4 updated with **measured** values; `docs/reports/recon_findings.md` written
- [x] `docs/workflow/annotation_guidelines.md` written and followed
- [x] Validation scripts run **clean** (0 malformed/out-of-bounds boxes, 0 corrupt files, 0
      orphan labels; 4 images excluded from split for missing labels, 1 near-duplicate pair
      found and confirmed both in `train` — see `data/metadata/statistics.csv`)
- [x] `data/processed/{train,val,test}/` populated — 94 / 5 / 32 images
- [x] `data/metadata/` committed: `dataset_info.json`, `class_names.txt`, `statistics.csv`,
      split lists (`train.txt`, `val.txt`, `test.txt`)
- [x] `test_split_integrity.py` **passing** (no session spans two splits; every split non-empty)
- [ ] `no_helmet` count meets the Slice 1.2 floor, present across **multiple sessions** —
      **NOT MET, explicitly waived by the user 2026-07-16** (79 instances vs. 150 target, 2
      sessions not 3; see the Waivers table). Mitigation deferred to Phase 2 oversampling;
      `no_helmet` recall is the number to watch at the Phase 3 gate.
- [x] `data/processed/` regenerable from `data/raw/` by re-running
      `scripts/preprocessing/extract_frames.py` → `clean_frames.py` →
      `scripts/dataset_split/split_by_session.py` (annotations in `data/raw/annotations/`
      are the one non-regenerable input — they're the human labeling work)
- [ ] Negative samples present — **gap, not yet closed.** No dedicated empty/all-compliant
      scenes identified in the current dataset (see `docs/workflow/dataset_plan.md`); the
      PRD §2.1 false-positive-rate target (≤5%) has no dedicated data to measure against yet.

**Split result:** `video_1` → test (32 images, satisfies the TDD §3 cross-session
`no_helmet` constraint), `video_2` → val (5 images), `video_3` + both stills pools → train
(94 images). Achieved ratios (72% / 4% / 24%) deviate substantially from 80/10/10 — an
inherent consequence of only 5 capture sessions total, documented rather than forced.

**If the gate fails:** insufficient `no_helmet` → loop to 1.4 (stage more). Class-balance or
leakage failure → **fix here.** Both are invisible in the aggregate metrics the notebook reports
and will be misdiagnosed as model problems, costing far more than fixing them now.
**Status here:** proceeding to Phase 2 with the `no_helmet` floor and negative-samples gaps
explicitly accepted (not silently passed) — see the Waivers table.

---

## Phase 2–3 — Training & Evaluation (merged) ⭐ **The feasibility answer (M3)**

**Environment:** Google Colab (GPU runtime) — one notebook, run cell by cell.
**Notebook:** [`notebooks/02_train_and_evaluate.ipynb`](../../notebooks/02_train_and_evaluate.ipynb)
**Entry gate:** Phase 1 gate met **+ OQ1, OQ3 and OQ-M resolved**.

Phases 2 and 3 are merged: in a notebook workflow you train and evaluate in one GPU session, and
splitting them across two documents just means copying state between them. The notebook sections
map 1:1 onto the old phase tasks.

### ⚠️ What merging costs — and how the notebook pays it back

The old Phase 2→3 boundary was a **gate**, and it was doing two jobs. Merging removes the wall, so
the notebook has to enforce both by construction:

| What the gate did | How the merged notebook keeps it |
|---|---|
| **Forced a training-sanity check before evaluation** | §5 *Training metrics visualization* runs automated curve checks (divergence, overfitting signature) before any evaluation cell |
| **Kept the test split untouched until one deliberate moment** | §6 (val) is the **iteration surface**; §7 (test) sits behind an explicit `RUN_TEST_EVAL = False` flag, **meant to be run once** |

**The test-split risk is the real cost of merging, and it is yours to manage.** In a notebook you
re-run cells freely — that's the point. But every re-run of the test-evaluation cell *followed by a
change made in response to it* leaks test information into your decisions. Do it a handful of times
and the test split stops measuring generalization and starts measuring how well you tuned against
it. PRD §2.1 then becomes a number you cannot defend to anyone, and — worse — you won't know it,
because the numbers will look fine.

**Rule: iterate on val (§6). Touch test (§7) once, when you believe you are done.**

⚠️ **Colab vs. PRD OQ3.** This notebook uploads factory footage of identifiable workers to Google.
If OQ3 resolved to *on-prem-only*, **Colab is not a legal venue for this run** — move to a local or
on-prem GPU. The notebook re-states this at the top; it is not a formality, and deleting data
afterwards does not undo the upload. The same applies to W&B media logging (§3).

### Notebook section → task map

| § | Section | Was | Notes |
|---|---|---|---|
| 0 | Colab setup | — | GPU check, repo/dataset mount, deps, **git commit captured for provenance** |
| 1 | Dataset loading | 2.0 | Reads `configs/data.yaml`; **generates** `configs/yolo_dataset.yaml` (never hand-edit) |
| 2 | Dataset statistics | 1.7 recheck | **A gate, not decoration** — re-runs the R2 (imbalance) and R4 (leakage) checks on the GPU box |
| 3 | Training configuration | 2.1, 2.4 | All from `configs/training.yaml`; **no hardcoded hyperparameters** |
| 4 | Live epoch tracking | 2.3, 2.5 | Callback is `try/except`-wrapped — a plotting bug must never kill a long run |
| 5 | Training metrics visualization | 2.5 | Automated curve sanity checks — **the old Phase 2 gate** |
| 6 | Validation metrics | 3.3 | **Iteration surface** + threshold sweep on val |
| 7 | Model evaluation (test) | 3.1, 3.2, 3.5 | **`RUN_TEST_EVAL` — run once.** Judges PRD §2.1 number-by-number |
| 8 | Error analysis | 3.4 | FP/FN matcher, size breakdown, **look at the failing images** |
| 9 | Save `best.pt` | 2.4, 3.7 | Checkpoints + promotion + **committed `metadata.json`** |
| 10 | Export | — | ONNX + `.pt` → `artifacts/exported_models/`, with a load-and-agree check |

### Slice 2.1 — Augmentation policy → `configs/training.yaml`

Encode TDD §3's table. Key: `flipud: 0.0` (**documented disable** — fixed camera orientation;
upside-down heads never occur), `fliplr: 0.5`, `mosaic: 1.0`, `mixup: 0.1`, HSV wide for
lighting/helmet-color variation. Notebook §3 warns if `flipud != 0`.

Sanity-check augmented output visually before a long run — augmentation bugs (boxes not
transformed with the image) are silent and catastrophic.

### Slice 2.2 — Run the notebook

```
Colab → GPU runtime → open notebooks/02_train_and_evaluate.ipynb → run cells in order
```

Base: `yolo11n.pt` (COCO-pretrained). **Never from scratch** (Phase 2.3).

**Checkpoint after §5 (the old Phase 2 gate) — do not proceed to §6 if:**

| Symptom | Cause | Action |
|---|---|---|
| Loss diverges / NaN | LR too high | Lower `lr0`, check batch size |
| Val loss ↑ while train loss ↓ | Overfitting | More data / stronger augmentation |
| **Both classes ≈ 0 mAP** | **Data pipeline bug** — label format, class-id mapping, paths | **Not a model problem.** Verify by deliberately overfitting 10 images: if it can't memorize 10 images, it's wiring, not learning |

### Slice 2.3 — Persist artifacts ⚠️ **Run notebook §9 the moment training finishes**

Colab VMs are reclaimed without warning — this is a one-way door. §9 writes:

- `artifacts/checkpoints/<version>/` — `best.pt`, `last.pt`, `results.csv`
- `ml/models/best.pt` — the blessed model (only if this run wins)
- `artifacts/experiment_results/<version>/metadata.json` — **committed**, with the **git commit**
  of the training code, dataset version, hyperparameters, operating point, and metrics with the
  eval set named

### Slice 2.4 — Threshold selection (notebook §6) ⚠️ **On val, never test**

Sweep confidence on the **val** PR curve. Pick the **lowest** threshold that still holds
`no_helmet` precision ≥ 0.90, maximizing recall under that floor (PRD §7, TDD §5).

**Do not ship the F1-optimal point** — F1 treats a missed violation and a false alarm as equally
bad, and PRD §7 says they are not. The notebook computes F1 and reports it *only* to show what it
deliberately did not pick.

Output: `configs/inference.yaml` (`conf_threshold`, `alert_conf_threshold`) — **commit it**. The
sweep table goes to `docs/reports/`.

### Slice 2.5 — Judge against PRD §2.1 (notebook §7)

| Metric | Target | Actual | Pass? |
|---|---|---|---|
| mAP50 | ≥ 0.85 | | |
| mAP50-95 | ≥ 0.55 | | |
| **Precision (`no_helmet`)** | **≥ 0.90** | | |
| **Recall (`no_helmet`)** | **≥ 0.90** | | |
| FP rate on negatives | ≤ 5% | | |

**Gate on the per-class `no_helmet` rows, not just mAP50.** An aggregate mAP passes comfortably
while the alert-triggering class fails — that's precisely why PRD §2.1 scores them separately.

### Slice 2.6 — Optimization loop (only if §2.1 unmet)

**Strict leverage order** (Phase 3.2). Reaching for a bigger model first is the classic mistake —
least effective, most expensive:

1. **More data** — targeting the category notebook §8 identified as failing
2. **Better annotation** — if §8 shows label noise, fix guidelines and re-label
3. **Better augmentation** — if failures cluster in a condition (blur, lighting)
4. **`img_size` 640 → 960** — if small heads dominate the FNs (notebook §8 measures this)
5. **Hyperparameter tuning**
6. **Only then:** YOLO11n → 11s → 11m

Each iteration loops back through Phase 1 / notebook §3–6 for the *specific* failing category.
**Re-run §6 (val), not §7 (test).** Every run gets its own
`artifacts/experiment_results/<version>/metadata.json`.

### Slice 2.7 — Promote + report

Notebook §9 handles promotion; promotion must trace to exactly one
`artifacts/experiment_results/<version>/`. Write
`docs/reports/accuracy_report_<version>.md`: metrics vs. PRD §2.1, the threshold sweep, error
categories from §8, and known failure modes.

### Slice 2.8 — Promote logic out of the notebook ⚠️ **Convention, not optional**

**Notebooks never hold production logic.** Once the notebook proves it out, promote — with tests
in `ml/tests/`:

| Notebook § | → Module | Public surface |
|---|---|---|
| §3–4 (config → train) | `ml/training/train.py` | `train(config) -> RunResult`, `write_run_metadata(run, path)` |
| §6–7 (val/test metrics, sweep) | `ml/evaluation/evaluate.py` | `evaluate(model, split, slices) -> EvalReport`, `sweep_threshold(...)` |
| §1 (config loading) | `scripts/utilities/config.py` | `load_config(name) -> dict` |
| §8 (IoU matcher) | `ml/evaluation/errors.py` | reused by `ml/tests/` |

Phase 4 imports `ml/`, never the notebook. Skipping this slice means Phase 5's API has no library
to call and the logic gets re-implemented — the exact duplication that makes Phase 7.2's accuracy
validation undecidable.

### 🚦 Gate → Phase 4

- [ ] Notebook §2 checks green: no leakage (R4), `no_helmet` volume workable (R2)
- [ ] Loss curves sane; no overfitting signature (§5)
- [ ] `best.pt` + `last.pt` + logs saved **off** the Colab VM
- [ ] `metadata.json` committed, with the git commit of the training code
- [ ] Threshold selected **on val**, written to `configs/inference.yaml` and committed
- [ ] Error analysis complete; failure categories named (§8)
- [ ] **`RUN_TEST_EVAL` run — once**; PRD §2.1 judged number-by-number
- [ ] PRD §2.1 met — **including per-class `no_helmet`** — **or** shortfall explicitly accepted by
      the user and documented in `docs/reports/`
- [ ] `docs/reports/accuracy_report_<version>.md` written
- [ ] Model promoted to `ml/models/`
- [ ] **Slice 2.8 done** — `ml/training/`, `ml/evaluation/` exist with tests

**If the gate fails after the 2.6 loop:** the honest outcome is to **report the shortfall and
stop** — not to proceed to Phase 4 with a model nobody should trust. This is the POC working
correctly: it answered the feasibility question with "not yet, and here's precisely what's
missing." Building a dashboard on a model that misses one violation in four converts a technical
shortfall into a safety hazard, because it looks like it works.

---

## Phase 4 — Inference Pipeline

**Entry gate:** Phase 2–3 gate met (including Slice 2.8 — `ml/training/`, `ml/evaluation/` promoted).
Build in `ml/inference/` + `ml/postprocessing/` as **library code**. The API and UI import this —
never reimplement inference in a service (TDD §1).

### Slice 4.1 — Core types & detector

- `ml/types.py`: `Detection(cls, conf, xyxy)`, `Frame(image, ts, source_id)`,
  `ViolationEvent(ts, box, conf, frame_ref, source_id)`
- `ml/inference/predictor.py`: `HelmetDetector(config)` with `.predict(image) -> list[Detection]`
- **Load the model once** at construction, never per call
- Tests → `ml/tests/test_predictor.py`

### Slice 4.2 — Sources (all four input modes)

`ml/inference/sources.py`: `iter_frames(spec) -> Iterator[Frame]`, uniform across:
image · video · webcam · **mobile cam / RTSP**

RTSP needs reconnect-with-backoff and frame-drop accounting — a stream that dies at 3 AM must log
(`WARNING`) and recover, not wedge silently.

### Slice 4.3 — Filters

`ml/postprocessing/filters.py`: `apply(detections, config) -> list[Detection]` — conf threshold,
IoU/NMS, all driven by `configs/inference.yaml`. Tests → `ml/tests/test_filters.py`.

### Slice 4.4 — ViolationTracker ⭐ **Highest-value test target**

`ml/postprocessing/violations.py`: `ViolationTracker.update(detections, ts) -> list[ViolationEvent]`

Dedup (TDD §6): time-windowed deque of recent violation boxes; suppress a new `no_helmet` if
IoU ≥ `dedup_iou` (0.5) with one emitted within `dedup_window_seconds` (30).

**Known, accepted limitations** (v1 has no tracking) — document in `docs/reports/` **and** the
README, not discovered by a supervisor in month two:

- A worker who **moves across the frame** re-triggers (IoU drops)
- Two workers in **nearly the same spot** merge into one event

This module is pure logic, encodes the business rule, and its bugs are **invisible in aggregate
metrics** — which is exactly why it gets the most thorough unit tests in the project:
fires once inside the window · re-fires after it · suppresses at IoU ≥ 0.5 · emits separately
below it · handles empty input · handles clock jumps.

### Slice 4.5 — CLI entry points

`scripts/` — thin arg parsing only, all logic in `ml/`:

```bash
uv run python -m ml.inference.cli --source image --path X --config configs/inference.yaml
uv run python -m ml.inference.cli --source video  --path X
uv run python -m ml.inference.cli --source webcam
uv run python -m ml.inference.cli --source rtsp --url "$CAMERA_URL"
```

Annotated output → `outputs/inference/{images,videos,rtsp}/`.

### 🚦 Gate → Phase 5

- [ ] All four modes runnable from CLI against `ml/models/best.pt`
- [ ] Annotated outputs landing in `outputs/inference/`
- [ ] `ml/tests/` passing — **especially `ViolationTracker` dedup**
- [ ] Logging via `configs/logging.yaml`; **zero `print()`**
- [ ] `configs/inference.yaml`-driven; no hardcoded thresholds
- [ ] Tracker limitations documented

---

## Phase 5 — REST API

**Entry gate:** Phase 4 gate met. Build in `api-services/`; services orchestrate `ml.inference` →
`ml.postprocessing`; routes stay thin. Endpoints per TDD §8.

### Slice 5.1 — Skeleton, config, health

- `pydantic-settings` loader in `api-services/app/utils/config.py`; **never scattered `os.environ`**
- **Model loads once at startup**, not per request — per-request loading blows the PRD §2.2
  latency budget by an order of magnitude
- `GET /api/v1/health` → `{status, model_version, model_loaded, device, uptime_s}` — must report
  the **promoted model version** (this is the Phase 5 gate)

### Slice 5.2 — Image detection

`POST /api/v1/detect/image` (multipart) → `{detections, inference_ms, image_id}`;
`?annotated=true` → `image/jpeg` with boxes.

### Slice 5.3 — Video detection (async)

`POST /api/v1/detect/video` → `{job_id}`; `GET /api/v1/detect/video/{job_id}` → status/progress.
**Async is not optional:** a 2-minute clip on a laptop exceeds any sane HTTP timeout.
Bounded by `video_max_duration_seconds` / `max_upload_mb`.

### Slice 5.4 — Live stream

`POST /stream/start|stop` · `GET /stream/status` · `GET /stream/live.mjpg`
(`multipart/x-mixed-replace`, annotated).
MJPEG chosen over WebSocket: one-way, no extra protocol, trivial for Angular (TDD §8).

### Slice 5.5 — Violations: store + feed

- SQLite via `VIOLATION_DB_URL`; append-only `violations` table (id, ts, source_id, conf, box,
  frame_path)
- `GET /violations/feed` — **SSE**, pushes to the supervisor
- `GET /violations?from&to&page&page_size` — paginated audit log

### Slice 5.6 — Errors, CORS, hardening

400 unsupported/oversized · 422 schema · 503 model-not-loaded / source unreachable ·
404 unknown job/stream. `{error, detail}` shape.
CORS from `configs/api.yaml` (`http://localhost:4200`).

⚠️ **No auth in v1** (PRD §8). **Bind to localhost.** This must not be exposed to a network in
this state — an unauthenticated endpoint streaming factory camera footage is a serious problem,
and this warning belongs in `docs/deployment/local.md`.

### 🚦 Gate → Phase 6

- [ ] All endpoints pass `api-services/tests/` (detector **stubbed** for unit tests)
- [ ] `/health` reports the promoted model version
- [ ] `api-services/requirements.txt` = **scoped** `uv export`; no training-only deps
      (`albumentations`, W&B) dragged into the service image
- [ ] Latency for `/detect/image` within PRD §2.2

---

## Phase 6 — Angular Dashboard

**Entry gate:** Phase 5 gate met **+ OQ7 confirmed** (Angular vs. Streamlit).
The UI talks **only** to the API — it never loads the model.

⚠️ **Re-confirm OQ7 before starting.** This is R7: Angular is roughly an order of magnitude more
effort than Streamlit for the same POC evidence. The API contract (TDD §8) is UI-agnostic, so the
swap stays cheap — but **only until this phase starts.** After that, it's sunk cost.

| Slice | Feature | API surface |
|---|---|---|
| 6.1 | App shell, routing, API client, env config | — |
| 6.2 | Live camera view | `<img src=".../stream/live.mjpg">` + `/stream/start|stop|status` |
| 6.3 | Image upload + detection viz | `/detect/image?annotated=true` |
| 6.4 | Video upload + job polling | `/detect/video`, `/detect/video/{job_id}` |
| 6.5 | Violation feed (live alerts) | `EventSource` → `/violations/feed` |
| 6.6 | Violation log / audit view | `/violations` paginated |
| 6.7 | **"This was wrong" button** | Flags a false alarm → `outputs/inference/hard_cases/` |

**6.7 is small and pays for itself** (TDD §11): it turns supervisors into labelers and feeds the
next annotation round with precisely the examples the model is worst at.

### 🚦 Gate → Phase 7

- [ ] A user can upload an image/video and view a live feed with detections, end-to-end through
      the API
- [ ] Violations appear in the live feed within PRD §2.2's 3 s budget
- [ ] Audit log paginates
- [ ] `ui-services/tests/` passing

---

## Phase 7 — Testing

**Entry gate:** Phase 6 gate met. Four categories, all required.

### 7.1 Functional
`tests/end_to_end/` + service tests. Every endpoint and UI flow behaves per the PRD.
Real `ml/models/best.pt` + `tests/fixtures/` (one `no_helmet`, one compliant, one empty).
Assert **shape/type/range sanity**, not exact values.

⚠️ Fixture images of real workers inherit OQ3's constraints — use consented/synthetic frames or
blur non-target faces. **Committed fixtures are permanent in git history.**

### 7.2 Accuracy validation ⭐ **The one that catches what unit tests can't**
Re-verify PRD §2.1 **through the deployed pipeline** (API in front of the model), not the bare
model. **Preprocessing drift hides exactly here** — a resize or color-space mismatch between
training and serving silently degrades accuracy while every unit test stays green.

Compare number-to-number against the Phase 3 bare-model results. **Any material gap is a bug**,
not noise.

### 7.3 Performance
`tests/performance/`, TDD §5 protocol: 100 warmup + 500 timed inferences @ 640 on the **actual
target machine**; report **p50/p95/p99 end-to-end through the API**, plus sustained FPS over 60 s.
Judge against PRD §2.2. → `metadata.json` + `docs/reports/`.

⚠️ **If OQ1 resolved to CPU-only, the ≥10 FPS target will fail here.** That must have been
renegotiated in Phase 0 — arriving at this line to discover it is R1 having gone unmanaged.

### 7.4 Stress
Concurrent requests · long-running streams (hours) · **camera-disconnect recovery** ·
oversized/malformed uploads · disk growth from `frame_path` storage.

### 🚦 Gate → Phase 8

- [ ] All four categories executed; results in `docs/reports/`
- [ ] Accuracy through the API matches Phase 3 (no drift)
- [ ] Performance judged against PRD §2.2 (or documented + accepted)
- [ ] Failures fixed or explicitly accepted by the user

---

## Phase 8 — Deployment

**Entry gate:** Phase 7 gate met. Target: **local dev machine** (PRD §6).

- **8.1** `docker/Dockerfile.api` (+ GPU passthrough if OQ1 = GPU; slim CPU base otherwise),
  `Dockerfile.ui` (Angular build → nginx), `Dockerfile.ai` (CUDA, training only — **not** part of
  the demo stack)
- **8.2** `docker/docker-compose.yml` — `api` + `ui`; volumes for `ml/models/`, `outputs/`;
  `.env` wiring
- **8.3** `deployment/local/` — up/down scripts, model-fetch step
- **8.4** `docs/deployment/local.md` — prerequisites, GPU setup, model placement, env vars,
  smoke test, troubleshooting, **and the localhost-binding/no-auth warning**
- **8.5** Regenerate `requirements.txt` + `api-services/requirements.txt` via `uv export`
  (**never hand-edit**); final README pass

**Export path:** `.pt` served directly. ONNX archived per run for portability. **No TensorRT**
(that's a Jetson concern), **no quantization** (an edge optimization that buys little on a laptop
GPU and costs accuracy PRD §2.1 has no headroom for).

### 🚦 Done

- [ ] `docker compose up` → working system from a clean clone
- [ ] E2E tests pass against the containerized stack
- [ ] Deployment guide followed successfully **by someone who isn't you** — the only real test of
      a deployment doc

---

## POC Deliverables Checklist

The POC is **done** only when all six exist:

- [ ] **Trained model** — `ml/models/best.pt` + committed `artifacts/experiment_results/*/metadata.json`
- [ ] **REST API** — Phase 5 endpoints, tested
- [ ] **Dashboard** — Phase 6 UI
- [ ] **Live camera demo** — webcam or RTSP, end to end
- [ ] **Documentation** — README, PRD, TDD, this plan, deployment guide
- [ ] **Accuracy report** — `docs/reports/`, metrics vs. PRD §2.1

Mirror this checklist in the README and report status against it when asked "are we done?".

---

## Cross-cutting conventions (every phase, every slice)

| Rule | Why |
|---|---|
| Spec in `.claude/feature-specs/` **before** implementing a slice | Plan Claude Code can follow + record of what was built and why |
| Tests land **with** the slice | Not "after" — after never comes |
| `ml/` is the library; `scripts/`, `api-services/`, `ui-services/` **import** it | Prevents the duplicate-inference-pipeline failure that makes Phase 7.2 undecidable |
| No hardcoded config — YAML in `configs/`, secrets in `.env` | The record of what produced a result |
| `logging`, never `print()` | The only witness when a feed degrades at 3 AM |
| `uv add`, never `pip install`; `uv export` to regenerate requirements | Honest lockfile |
| Never commit data or weights; **do** commit `data/metadata/` + `metadata.json` | Reproducibility without repo bloat |
| After each slice: review for convention violations | Cheaper than a Phase 7 cleanup |

## Decision points / off-ramps

| Point | Trigger | Options |
|---|---|---|
| **Phase 0 gate** | OQ3 blocks footage use | Public dataset (accept domain gap) · resolve consent · stop |
| **Phase 1.1** | Heads < 32 px | `img_size` 960 · bigger model · **move/add cameras** (cheapest) |
| **Phase 1.7** | `no_helmet` too rare | Stage violations · public-dataset supplement · rescope |
| **Phase 2–3 start** | OQ3 = on-prem-only | **Colab is off the table** — local or on-prem GPU; notebook runs unchanged minus the Drive/W&B cells |
| **Notebook §7** ⭐ | §2.1 unmet after the 2.6 loop | Loop again on **val** · accept documented shortfall · **stop and report — the POC did its job** |
| **Phase 6 start** | Schedule pressure | Descope Angular → Streamlit (API contract makes this cheap **until 6.1 starts**) |
| **Phase 7.3** | CPU-only, FPS target missed | Frame-skipping + renegotiated target · GPU hardware |

## Waivers

Record any consciously skipped gate here — date, what was skipped, who approved, why, and the risk
accepted. An empty section is the goal.

| Date | Gate | Approved by | Rationale | Risk accepted |
|---|---|---|---|---|
| 2026-07-16 | Phase 1 data source (OQ3/OQ4 in practice) | Argha Dey Sarkar (user) | No factory site access for this POC. Dataset is built entirely from public sources: 4 YouTube clips of the IISCO steel plant (`s01`–`s03`) and downloaded stock images (`s04`, `s05`) — confirmed explicitly by the user 2026-07-16, not first-party consented factory footage as OQ3 originally assumed. | Model is trained/evaluated on public/downloaded data, not the real deployment camera. Phase 3 results measure feasibility on this proxy dataset only — must be re-validated against real site footage before being presented as evidence for the actual deployment (see `docs/reports/recon_findings.md`, `docs/workflow/dataset_plan.md`). Possible YouTube ToS/image-licensing exposure if this data were ever redistributed — internal POC use only, no redistribution (see `data/metadata/dataset_info.json`). |
| 2026-07-16 | Phase 1 gate — `no_helmet` floor (150 instances, ≥3 sessions) | Argha Dey Sarkar (user) | Measured count is 79 instances from 2 sessions after Slice 1.7 validation. User chose to proceed to Slice 1.8 (split) without collecting more data or re-checking the 2 unlabeled `video_2` frames, planning to mitigate via train-split oversampling/augmentation in Phase 2 instead. | Oversampling/augmentation cannot manufacture new sessions or new `no_helmet` information — the cross-session generalization risk (TDD §3) stays open. `no_helmet` recall (PRD §2.1, target ≥ 0.90) is the number to watch at the Phase 3 gate; if it underperforms, the fix is back here (more data / more sessions), not more training tricks. |
