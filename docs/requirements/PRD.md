# Product Requirements Document — Helmet Detection System

**Author:** Argha Dey Sarkar | **Date:** 2026-07-15 | **Status:** Draft

> Phase 0 artifact of the Level-1 POC/MVP workflow. The TDD (`docs/architecture/TDD.md`)
> derives its model-family and deployment decisions from this document. Section 2 is the
> contract Phase 3 evaluation is judged against.

---

## 1. Problem Statement

Workers in a factory are required to wear safety helmets in designated production areas.
Compliance is currently enforced by human observation — supervisors noticing violations on
the floor or after the fact via CCTV review. This is unreliable and unscalable: a
supervisor cannot watch every area continuously, violations are caught late or not at all,
and there is no defensible record of compliance over time for safety audits.

The pain is felt by:

- **Safety supervisors** — responsible for enforcement, currently working from spot checks.
- **Plant/EHS management** — accountable for audit evidence and incident liability, currently
  without trustworthy compliance data.
- **Workers** — bear the physical risk when a violation goes unnoticed.

This POC validates whether a computer vision model can detect helmet compliance from factory
camera footage accurately enough to justify building the production system.

## 2. Objective & Success Criteria

**Objective:** Detect whether workers in factory camera footage are wearing safety helmets,
surface no-helmet violations to a supervisor in near-real-time, and record them for audit
reporting.

### 2.1 Model accuracy criteria (measured on the held-out test split, Phase 3)

| Metric | Target | Notes |
|---|---|---|
| mAP50 (all classes) | ≥ 0.85 | Primary headline metric |
| mAP50-95 (all classes) | ≥ 0.55 | Localization quality guard |
| Precision (`no_helmet`) | ≥ 0.90 | Bounds alert fatigue |
| Recall (`no_helmet`) | ≥ 0.90 | Bounds missed violations — see §7 |
| False-positive rate on negative samples | ≤ 5% | Measured on helmet-compliant / no-person scenes |

`no_helmet` is the **decision-critical class** and is scored separately: an aggregate mAP
can pass while the class that actually triggers alerts fails. Both must clear the bar.

### 2.2 System performance criteria (measured through the API, Phase 7)

| Metric | Target | Notes |
|---|---|---|
| End-to-end latency (single image via API) | ≤ 200 ms | Laptop GPU, YOLO11n @ 640px |
| Live stream throughput | ≥ 10 FPS, 1 stream | Sustained on target hardware |
| Alert delivery latency (violation → dashboard) | ≤ 3 s | Near-real-time, not hard real-time |
| API availability during demo | ≥ 99% | POC-grade; not an SLA |

**Assumption (unconfirmed):** the dev machine has a CUDA-capable laptop GPU. On CPU-only,
YOLO11n @ 640px runs roughly 100–250 ms/frame, and the ≥10 FPS live-stream target is
**not achievable** — live mode would degrade to frame-skipping (process every Nth frame).
This must be confirmed before Phase 2; see §10.

## 3. CV Task Mapping

| Business ask | Canonical CV task | Output |
|---|---|---|
| "Is this worker wearing a helmet?" | Object detection (bounding box, multi-class) | Boxes + class + confidence per head |
| "Flag workers without helmets to a supervisor" | Detection + confidence thresholding + business rule | Violation event (timestamp, frame, box, confidence) |
| "Show compliance over time for audits" | Event persistence + aggregation | Violation log, counts per time window |

**Class list (v1):**

| Class | Definition |
|---|---|
| `helmet` | A head wearing a safety helmet |
| `no_helmet` | A visible human head without a safety helmet |

Detection is at the **head** level, not the person level — a person-level box makes
"wearing a helmet" ambiguous under occlusion and crowding. A `person` class is deliberately
excluded from v1 (see §8).

**Explicitly not v1:** tracking / re-identification. Without track IDs, one worker standing
in view for 30 seconds generates repeated violation events rather than one. v1 mitigates
this with time-window deduplication (§8), not tracking.

## 4. Users & Usage Context

| Consumer | What they get | Frequency |
|---|---|---|
| Safety supervisor | Angular dashboard: live camera view with boxes, violation feed | Continuous during shift |
| EHS / plant management | Violation log + counts for audit reporting | Weekly / per audit |
| ML engineer (this project) | Eval reports, error analysis | Per training iteration |

**Environment (to be confirmed against the actual site — see §10):**

| Requirement | Value | Confidence |
|---|---|---|
| Object classes | `helmet`, `no_helmet` | Decided |
| Camera type | Fixed CCTV assumed; mobile phone camera for demo | **Assumed** |
| Camera resolution | Unknown — assumed 1080p | **Assumed** |
| Camera angle | Unknown — assumed elevated/eye-level, not top-down | **Assumed** |
| Indoor / Outdoor | Indoor factory floor | Assumed |
| Day / Night | Daytime + artificial lighting; no IR handling in v1 | Assumed |
| Detection distance | Unknown — assumed 3–15 m | **Assumed** |
| Object size (px) | Unknown — heads assumed ≥ 32×32 px at max distance | **Assumed** |
| Expected FPS | ≥ 10 FPS, single stream | Decided |
| Latency budget | ≤ 200 ms/image; ≤ 3 s violation→dashboard | Decided |
| Accuracy target | See §2.1 | Decided |
| Number of streams | 1 (POC) | Decided |

Every row marked **Assumed** is a Phase 1 risk: if heads are smaller than ~32 px at the real
camera distance, YOLO11n @ 640px will miss them and the model choice and image size in the
TDD must change. Validate these against real footage before annotating.

## 5. Data Position

- **Available today:** none in the repository. Custom factory footage is the intended and
  only source.
- **To collect:** factory floor video from the target site(s), covering the areas where
  helmet compliance is enforced.
- **Coverage the dataset must include** (per Phase 1.1):
  - Both classes, with enough `no_helmet` instances — this is the minority class in a
    compliant factory and will be the binding constraint on recall.
  - **Negative samples**: compliant scenes and empty scenes, to bound the false-alarm rate.
  - Edge cases: occlusion (workers behind machinery), crowding, motion blur, backlighting /
    glare, varied helmet colors, and **helmet-like confusers** (caps, hairnets, bare heads
    with dark hair, hard hats carried not worn).
- **Labeling plan:** bounding boxes, YOLO format, via Label Studio / CVAT / Roboflow.
  Annotation guidelines must be written before labeling starts — "is a helmet carried in
  hand a `helmet`?" must have one answer (proposed: **no** — the class is *head wearing a
  helmet*).
- **Split discipline:** split by **source video / capture session**, never by random frame.
  Random-frame splits leak near-duplicate frames across train and test and will inflate every
  number in §2.1 into meaninglessness.
- **Privacy/compliance:** factory footage contains identifiable workers. Requires worker
  notice/consent per site policy, and likely on-prem-only storage. **Unresolved — see §10.**

**Schedule reality:** with no data in hand, Phase 1 (collection → annotation → QC → split)
dominates this project's timeline. Model training is days; getting a good labeled dataset is
weeks. Plan accordingly.

## 6. Constraints

- **Hardware / deployment target:** local dev machine (laptop GPU assumed, CPU fallback).
  Not a factory-deployed system in v1.
- **Latency / throughput:** §2.2.
- **Model family:** YOLO11, starting at YOLO11n (smallest that can plausibly hit the targets),
  scaling up the ladder only if Phase 3 error analysis demands it.
- **Licensing:** ⚠️ **Ultralytics YOLO11 is AGPL-3.0.** Acceptable for an internal POC. If this
  ships as a product or is deployed to a customer site without publishing source, it requires
  an Ultralytics commercial license or a differently-licensed model. **Flag for decision before
  Phase 8 — see §10.**
- **Stack:** Python + uv, FastAPI (API), Angular (dashboard), per README.
  *Deviation from convention:* the skill's Phase 6 specifies Streamlit for POC dashboards;
  the README specifies Angular. Angular is honored here as the explicit choice, at the cost of
  meaningfully more UI build time than a Streamlit POC dashboard.
- **Budget:** cloud GPU hours for training (Colab/Kaggle sufficient for YOLO11n); annotation
  effort is the dominant cost.

## 7. Cost of Errors

**A false negative** (a bare-headed worker classified as compliant, or missed entirely) is a
**missed safety violation** — the exact failure the system exists to prevent. It is silent:
nobody knows it happened. This is the expensive error.

**A false positive** (a compliant worker flagged) costs supervisor attention. Cheap once,
corrosive in bulk — a dashboard that cries wolf gets ignored, which converts every subsequent
true positive into a missed violation too.

**Therefore: bias toward recall on `no_helmet`.** Concretely:

- Set the operating confidence threshold **below** the F1-optimal point for `no_helmet`
  (start ~0.35, tune on the val PR curve in Phase 3 — do not hardcode; it lives in
  `configs/inference.yaml`).
- Precision ≥ 0.90 in §2.1 is a **floor**, not a target — it exists to keep alert fatigue
  bounded while recall is pushed up.
- The two thresholds are separable: the *alerting* threshold may be tuned independently of the
  *logging* threshold, so audit logs can be more conservative than live alerts.

## 8. Scope

**In scope (v1):**

- Fine-tuned YOLO11 model for `helmet` / `no_helmet` on custom factory footage
- Inference pipeline: image, video, webcam, and mobile-camera/RTSP input
- Configurable confidence threshold; bounding box visualization
- REST API (FastAPI): image detection, video detection, live stream start/stop/status, health check
- Angular dashboard: live camera view, image/video upload, detection visualization, violation feed
- Violation logging with time-window deduplication (one event per detection per N seconds)
- Accuracy report vs. §2.1 and deployment documentation

**Out of scope (v1):**

- Multi-object tracking / worker re-identification (so: no per-worker violation attribution,
  no "how long was this person non-compliant")
- Worker identity, face recognition, PPE beyond helmets (vests, goggles, gloves)
- Multi-camera / multi-stream concurrency
- Zone/ROI-restricted detection ("only enforce inside the press area")
- Night / IR camera handling
- Production deployment, authentication/RBAC on the dashboard, alerting to external channels
  (SMS/email/PLC)
- Retraining automation / data flywheel

**Scope note:** the three requested outputs (feasibility demo, live alerts, compliance
reporting) are staged as M3 → M4 → M5 in §9, not built at once. Compliance reporting adds a
datastore that a pure feasibility POC would not need; it is included but kept minimal
(append-only event log, not an analytics warehouse).

## 9. Milestones

| Milestone | Deliverable | Gate |
|---|---|---|
| **M0** | PRD + TDD approved; repo scaffolded | Phase 0 gate |
| **M1** | Feasibility baseline: pretrained YOLO on sample factory frames; head-size and camera assumptions in §4 validated against real footage | Assumptions confirmed or PRD revised |
| **M2** | Labeled dataset v1: collected, annotated, QC'd, source-split, `data/metadata/` committed | Phase 1 gate |
| **M3** | Trained YOLO11 model meeting §2.1 on the test split + accuracy report | Phase 3 gate — **the feasibility answer** |
| **M4** | Inference pipeline (4 modes) + REST API + Angular dashboard with live view | Phases 4–6 gates |
| **M5** | Violation logging + audit view; full test pass vs. §2.1/§2.2 through the API | Phase 7 gate |
| **M6** | Local deployment (Docker) + documentation complete | Phase 8 |

M3 is the decision point: if §2.1 is not met after the Phase 3 optimization loop
(more data → better annotation → better augmentation → hyperparameters → bigger model), the
honest outcome is to report the shortfall, not to proceed to M4 with a model nobody should
trust.

## 10. Open Questions

| # | Question | Blocks | Owner |
|---|---|---|---|
| 1 | Does the dev machine have a CUDA GPU? If CPU-only, the ≥10 FPS live target in §2.2 is unachievable and must be renegotiated. | Phase 2 model sizing | User |
| 2 | Real camera resolution, mounting angle, and worker distance — what is the actual head size in pixels? | Phase 1 annotation, imgsz choice | User / site |
| 3 | Is factory footage cleared for use? Worker consent, on-prem-only storage, retention period? | Phase 1 data collection | User / EHS |
| 4 | How much footage exists or can be captured, and how many `no_helmet` instances can realistically be gathered? Staged violations may be needed — a compliant factory yields few natural positives. | Phase 1 feasibility | User |
| 5 | AGPL-3.0 (Ultralytics) — internal-only forever, or productized later? Decide before Phase 8. | Phase 8 / commercialization | User |
| 6 | Violation log retention and storage backend (SQLite sufficient for POC?) | M5 | User |
| 7 | Angular vs. Streamlit for the POC dashboard — Angular is ~an order of magnitude more build effort for the same POC evidence. Confirm Angular is a hard requirement. | Phase 6 | User |
| 8 | Deduplication window for repeated violations of the same stationary worker (proposed: 30 s) | M5 | User |

---
_Saved to `docs/requirements/PRD.md`. The TDD (`docs/architecture/TDD.md`) references this file._
