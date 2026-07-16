# Dataset Plan — Helmet Detection POC

**Date:** 2026-07-16 | **Source:** PRD §5, TDD §3, `docs/reports/recon_findings.md`

## Classes

Exactly two, per `data/metadata/class_names.txt`:

- `helmet` — a head wearing a safety helmet
- `no_helmet` — a visible human head without a safety helmet

## Sessions / sources (current inventory)

| Session ID | Source file(s) | Type | Notes |
|---|---|---|---|
| `s01_broadcast_tour` | `video_1.mp4` | Broadcast/news pan | Golf-cart site tour (mixed helmet/bare-head, crowding) + indoor meeting room (bare-head only) |
| `s02_workshop_handheld` | `video_2.mp4` | Handheld | Near hot-metal machinery, one bare-headed subject, close range |
| `s03_selfie_hardhat` | `video_3.mp4`, `video_4.mp4` | Selfie (duplicate content — **treat as one session**, not two) | Single subject, white hard hat, extreme close-up |
| `s04_stills_helmet` | `data/raw/images/helmet_1..46.jpeg` | Static images | 46 `helmet`-class stills |
| `s05_stills_no_helmet` | `data/raw/images/no_helmet_1..42.jpg/.jpeg` | Static images | 42 `no_helmet` stills (expanded from 2 on 2026-07-16 specifically for session/source diversity — see floor below) |

`video_3.mp4` and `video_4.mp4` are identical content (confirmed by matching file size and
frame count in Slice 1.1) — they count as **one** capture session for split purposes, not
two, to avoid the exact-duplicate leakage TDD §3 warns about.

## Target counts (v1, working estimate)

| Class | Current available (rough) | Target for a workable v1 | Gap |
|---|---|---|---|
| `helmet` | ~46 stills + frames from `s01`/`s03` | ≥ 300 labeled instances | Extract more frames from `s01`, stage/collect more |
| `no_helmet` | 2 stills + `s02` handheld subject + `s01` meeting-room bystanders | **≥ 150 labeled instances** (see floor below) | **Large — binding constraint, see R2** |

These are working targets, not commitments — Phase 1 is explicitly the phase that may need
to loop back to collection (Slice 1.4) once labeling volume is visible.

## `no_helmet` floor — the binding constraint (R2)

**Explicit floor: 150 labeled `no_helmet` instances, spread across ≥ 3 sessions**, before
Phase 1 gate is considered met. Rationale: PRD §7 makes `no_helmet` recall the
decision-critical metric, and a single-session minority class cannot support a test split
that measures generalization (TDD §3's cross-session `no_helmet` requirement).

**Status as of 2026-07-16, revised after manual frame curation.** The user manually reviewed
`data/interim/cleaned/` and removed redundant/blurry frames: `s01` went from 180 → 32 frames
(kept mostly the delegation-tour segment with bare-headed dignitaries/escorts), `s02` from
17 → 7, and `video_4` was dropped entirely from `s03` as exact-duplicate content of
`video_3` (correctly collapsing it to one session, per this plan's guidance). At roughly
4–6 `no_helmet` instances per remaining `s01` frame, the projected count is ~130–190 —
**close to the 150 floor, no longer comfortably clear of it** as it was before curation.
Session/source diversity (`s01`, `s02`, `s05`) still holds. `s05`'s 42 stock images remain
the cheapest lever if the real labeled count from `s01`/`s02` falls short — verify with
actual box counts once Slice 1.6 labeling is underway, via
`scripts/validation/check_dataset.py`/`check_annotations.py` in Slice 1.7.

**Revised 2026-07-16: there is no site access for this POC.** All data in hand is public —
YouTube clips of the IISCO plant (`s01`–`s03`) and downloaded stock images (`s04`, `s05`).
"Stage more footage" (the original first-choice mitigation) is not available. Leverage order,
adjusted for this constraint:

1. **Maximize yield from `s01` first — this is the highest-value, already-in-hand source.**
   The delegation-tour segment shows 5–8 bare-headed dignitaries/escorts per frame across
   ~180 cleaned frames; labeling this segment thoroughly plausibly clears the 150-instance
   count on its own. Caveat: it's largely the same ~10–15 individuals repeating across
   frames, so it adds count, not identity diversity — label it fully, but don't treat the
   count alone as proof the floor is *qualitatively* met.
2. **Add session/source diversity via more public downloads**, since staging isn't possible.
   Concretely: download a handful more `no_helmet` stock images or a short public clip from a
   **different** plant/site than IISCO (not just a different IISCO upload) and add it to
   `data/raw/images/` or `data/raw/videos/` as a new tagged source in `dataset_info.json`.
   This is the cheapest lever actually available and directly addresses the "≥3 sessions"
   and cross-session-test requirements without needing site access.
3. If still short after (1) and (2), a public-dataset supplement (e.g. Roboflow Universe
   helmet datasets) for `no_helmet` only, clearly tagged by source, accepting the domain-gap
   risk documented in `docs/reports/recon_findings.md`.

**Standing limitation, not currently fixable:** because this is a public/downloaded dataset
rather than first-party site footage, no combination of the above produces data from the
*actual deployment camera*. That gap stays open until real site footage becomes available
(see the Waivers entry in `docs/workflow/implementation_plan.md`) and must be re-checked
before the Phase 3 accuracy numbers are presented as evidence for the real deployment.

## Negative samples

Include empty/no-person scenes and helmet-compliant-only scenes so the false-alarm rate
(PRD §2.1, ≤ 5%) is measurable. None currently identified in the sample — flag as a gap to
fill during Slice 1.4 collection follow-up.

## Edge cases to cover (per PRD §5 / TDD §3)

| Edge case | Present in current sample? |
|---|---|
| Occlusion (behind machinery/people) | Yes — `s01` golf-cart crowd scene |
| Crowding | Yes — `s01` golf-cart scene |
| Motion blur | Yes — `s02` |
| Backlighting / glare | Not yet observed — gap |
| Varied helmet colors | Only white observed so far — gap |
| Helmet-like confusers (caps, hairnets, carried helmets) | Not yet observed — gap |
| Extreme close-range (selfie-scale) | Yes (`s03`) — **note:** this shot type is not representative of the deployment distance per recon findings; keep but don't over-weight in the split |

## Lighting conditions to cover

Observed so far: bright outdoor daylight, indoor fluorescent, dark furnace-glow indoor. Per
recon findings this is not from one fixed camera, so lighting diversity here is incidental,
not systematic coverage of the real deployment site's lighting — still worth keeping as
augmentation-relevant variety.

## Open gap carried from Slice 1.1

Per `docs/reports/recon_findings.md`, none of the current sessions is fixed-CCTV footage at
the real deployment distance/angle. This dataset plan proceeds with what's available for
pipeline development and annotation practice, but **the `no_helmet` floor and edge-case
coverage above should be re-checked once real deployment-camera footage is collected** —
that footage is still the highest-priority Slice 1.4 follow-up.
