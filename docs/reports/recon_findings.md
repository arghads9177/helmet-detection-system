# Recon Findings — Slice 1.1

**Date:** 2026-07-16 | **Source:** `data/raw/videos/{video_1,video_2,video_3,video_4}.mp4`,
sampled via `scripts/preprocessing/extract_frames.py` at 1.5 FPS into
`data/interim/extracted_frames/`.

> Per the skill note: COCO-pretrained YOLO has no helmet concept, so this recon does not run
> a model — it is direct visual measurement on extracted frames (resolution, head size in
> px, angle, lighting), used to close PRD §4's "Assumed" rows.

## What's actually in the sample

| Clip | Resolution | Duration | Content |
|---|---|---|---|
| `video_1.mp4` | 640×360, 30 fps | 133 s | News-style broadcast of a VIP/official site visit to a steel plant: (a) a golf-cart tour past pipe racks with a mixed group of officials/workers, most wearing white hard hats, some bare-headed; (b) an indoor meeting-room scene, business attire, **no hard hats at all**. Picture-in-picture overlay of two public figures throughout. |
| `video_2.mp4` | 360×640, ~29.6 fps | 13 s | Handheld, heavily motion-blurred footage near hot-metal/press machinery. One person in a red shirt, **bare-headed** (no helmet), shot at very close range (<2 m). |
| `video_3.mp4` / `video_4.mp4` | 360×640, 30 fps | 4.7 s (identical content) | Vertical selfie-style clip, one person wearing a white hard hat, extreme close-up (head fills most of the frame), social-media watermark/emoji overlay visible. |

**None of the four clips is fixed-CCTV footage.** They are: a broadcast camera pan/handheld
mix (`video_1`), handheld amateur footage (`video_2`), and a phone selfie (`video_3`/`4`).
This is an important gap from the PRD §4 assumption of "fixed CCTV assumed."

## Measurements

| Finding | Observation |
|---|---|
| **Resolution** | 640×360 landscape (`video_1`) and 360×640 portrait (`video_2`–`4`) — well below the PRD's assumed 1080p. |
| **Head size, wide/mid shots** (`video_1` golf-cart scene, meeting room) | Foreground heads ≈ 40–55 px tall; background/crowd heads ≈ 15–25 px tall, at 640×360. |
| **Head size, close-range** (`video_2`) | Subject is cropped at the shoulders/chin at <2 m range — not a usable head-size data point for a working-distance estimate. |
| **Head size, selfie** (`video_3`/`4`) | ≈ 300–350 px tall at 360×640 — an extreme close-up, ~10× larger than any plausible fixed-camera working distance would produce. |
| **Camera angle** | Ranges from broadcast-camera eye-level pan, to handheld eye-level, to selfie — **no elevated/top-down CCTV angle present in the sample**. |
| **Lighting** | Highly variable: bright daylight outdoor (golf-cart scene), indoor fluorescent meeting room, dark furnace-glow workshop (`video_2`). Consistent with "varied factory lighting" but not with a single fixed camera's lighting profile. |
| **Occlusion / crowding** | `video_1` golf-cart scene has real crowding and partial occlusion (people behind each other) — useful for that edge case. `video_2`/`3`/`4` are single-subject, no occlusion. |
| **Helmet-like confusers** | None observed directly in this sample (no caps/hairnets seen), but sample is small. |

## Decision output (per implementation plan table)

| Finding | Consequence |
|---|---|
| Heads span ≈ 15 px (background crowd) to ≈ 350 px (selfie) in this sample — **not a single working distance** | The sample does not represent one consistent camera setup. **Cannot confirm `img_size` 640 is sufficient for the real deployment camera** from this data alone — the background-crowd end of this range (15–25 px) is below the PRD's assumed 32 px floor and below YOLO11n@640's comfortable zone. |
| No fixed-CCTV clip in the sample | The PRD §4 "Camera type: fixed CCTV assumed" is **not validated** by this sample — every clip is broadcast/handheld/selfie. A real recon clip from the actual deployment camera(s) is still needed before this row can be marked confirmed. |
| No top-down angle observed | N/A yet — no data point either way. |

## PRD §4 update

Rows below are updated from **Assumed** to **Partially observed (sample non-representative)** —
not to **Decided**, because none of the four clips is footage from a fixed monitoring camera
at the actual deployment distance/angle:

| Requirement | Prior (PRD §4) | Updated |
|---|---|---|
| Camera resolution | Assumed 1080p | **Observed: 360p-equivalent (640×360 / 360×640) across all 4 sample clips.** If the real deployment camera is also this resolution, `img_size` 640 gives little headroom for small/background heads — recommend confirming actual camera resolution before Phase 2. |
| Object size (px) | Assumed ≥ 32×32 px | **Observed range 15–350 px across the sample, driven by wildly different shot types, not by one working distance.** Background/crowd heads (15–25 px) are **below** the 32 px floor. Treat this as a live risk (TDD §2 small-object row), not resolved. |
| Camera angle | Assumed elevated/eye-level | **Observed: eye-level broadcast pan, handheld eye-level, and extreme close-up selfie. No CCTV-elevated angle observed.** |
| Detection distance | Assumed 3–15 m | **Observed: <2 m (video_2, video_3/4) to ~5–8 m (video_1 golf-cart, meeting room). No far-range (>10 m) example in the sample.** |

## Recommendation

Per the plan's decision table, **heads < 32 px appear in this sample** (background/crowd in
`video_1`). Per the workflow: **update `img_size` → 960 before committing to 640**, or —
cheaper — get a recon clip from the *actual* fixed camera(s) that will run in production,
since this sample is domain-mismatched (broadcast/handheld/selfie, not CCTV) on top of the
small-head-size signal. Recommend treating this batch of clips as **useful for annotation
practice and pipeline testing**, but **not sufficient alone** to close OQ2 or the PRD §4
camera-type/angle assumptions — a short clip from the real deployment camera is still the
cheapest way to remove this risk before Phase 2.
