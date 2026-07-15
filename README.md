# Helmet Detection System

A computer vision proof-of-concept that detects whether workers are wearing safety helmets on a
factory floor, from images, video, webcam, or an RTSP/mobile camera feed — surfacing violations to
a supervisor in near-real-time and logging them for compliance audits.

> **Status: Phase 0 (Problem Definition) — planning complete, blocking decisions resolved, repo
> scaffolded.** Code, data pipeline, and model training begin in Phase 1. See
> [Project status](#project-status) below.

## The problem

Helmet compliance on a factory floor is currently enforced by human observation — a supervisor
cannot watch every area continuously, so violations are caught late, or not at all, and there is
no defensible record for safety audits. This project evaluates whether a fine-tuned object
detection model can close that gap accurately enough to justify a production system.

Full context: [`docs/requirements/PRD.md`](docs/requirements/PRD.md).

## Planned features

- Head-level detection of two classes: `helmet` and `no_helmet`
- Image, video, webcam, and mobile-camera/RTSP inference
- Configurable confidence threshold, tuned separately for alerting vs. logging
- Bounding box visualization with per-detection confidence
- Fine-tuned YOLO11 model, trained on a custom industrial dataset
- REST API (FastAPI) — image/video detection, live stream control, health check
- Angular dashboard — live camera view, uploads, violation feed, audit log
- Time-windowed violation deduplication for near-real-time alerting

## Documentation

This project follows a phase-gated POC/MVP workflow. Each phase's rationale, decisions, and tasks
are written down before implementation, not discovered along the way:

| Document | Purpose |
|---|---|
| [`docs/requirements/PRD.md`](docs/requirements/PRD.md) | Problem statement, numeric success criteria, scope, cost of errors |
| [`docs/architecture/TDD.md`](docs/architecture/TDD.md) | Model selection, data pipeline, module design, API contract, deployment, risks |
| [`docs/workflow/implementation_plan.md`](docs/workflow/implementation_plan.md) | Phase-by-phase task breakdown, gates, and decision points |
| [`notebooks/02_train_and_evaluate.ipynb`](notebooks/02_train_and_evaluate.ipynb) | Training + evaluation notebook (Colab), Phase 2–3 |

## Success criteria

Measured on the held-out test split; the full rationale is in PRD §2.

| Metric | Target |
|---|---|
| mAP50 (all classes) | ≥ 0.85 |
| mAP50-95 (all classes) | ≥ 0.55 |
| Precision (`no_helmet`) | ≥ 0.90 |
| Recall (`no_helmet`) | ≥ 0.90 |
| False-positive rate on negative samples | ≤ 5% |
| End-to-end API latency (single image) | ≤ 200 ms |
| Live stream throughput | ≥ 10 FPS |

`no_helmet` is scored separately from the aggregate because it is the class that triggers alerts —
recall is prioritized over precision within the documented floor, since a missed violation is
silent while a false alarm merely costs attention.

## Project status

| Phase | What | Status |
|---|---|---|
| 0 | Problem definition (PRD + TDD, blocking decisions, scaffold) | ✅ Done |
| 1 | Data engineering (collection, annotation, split) | ⬜ Not started — ready to start |
| 2–3 | Training & evaluation (merged, one notebook) | ⬜ Not started |
| 4 | Inference pipeline (image/video/webcam/RTSP) | ⬜ Not started |
| 5 | REST API | ⬜ Not started |
| 6 | Angular dashboard | ⬜ Not started |
| 7 | Testing (functional, accuracy, performance, stress) | ⬜ Not started |
| 8 | Deployment (local Docker) | ⬜ Not started |

Live tracker with per-slice detail: [`docs/workflow/implementation_plan.md`](docs/workflow/implementation_plan.md#progress-tracker).

## Getting started

The repository is scaffolded (`uv` + `pyproject.toml`, `ml/`, `api-services/`, `ui-services/`,
`configs/`). Training, the inference pipeline, the API, and the dashboard land in Phases 1–6 —
until then this installs the environment and runs the (currently stub) test suite.

```bash
uv sync                                                # install dependencies
cp api-services/config/.env.example api-services/config/.env
cp ui-services/config/.env.example ui-services/config/.env
uv run pytest                                          # scaffold smoke test

# once later phases land:
uv run python -m ml.inference.cli --source image ...   # run inference (Phase 4+)
uv run uvicorn api-services.main:app                    # start the API (Phase 5+)
```

The PRD and implementation plan remain the entry point for anyone picking up this work —
`docs/workflow/implementation_plan.md` tracks exactly which phase is active.

## Known limitations (by design, v1)

- No multi-object tracking — violation deduplication uses a time-windowed IoU heuristic, not
  worker identity, and does not support per-worker attribution
- Detection is head-level, not full PPE (no vests, goggles, gloves)
- Single camera stream in v1; no multi-camera concurrency
- No zone/ROI restriction, no night/IR handling, no authentication on the dashboard
- Not licensed for productization as-is — YOLO11 (Ultralytics) is AGPL-3.0; see TDD §2 and §6

Full scope boundaries: PRD §8.

## License

Internal-only (confirmed, PRD §10 / OQ5). The model uses Ultralytics YOLO11, licensed AGPL-3.0 —
acceptable for internal use; revisit before any commercial or customer-facing deployment.
See `docs/architecture/TDD.md` §2 and §6.
