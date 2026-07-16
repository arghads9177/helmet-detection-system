# Annotation Guidelines — Helmet Detection POC

**Read this before labeling anything (Slice 1.6).** Ambiguity here produces label noise,
and label noise is indistinguishable from model error in Phase 2–3 — inconsistent labels
cost weeks of misdirected tuning, not minutes of relabeling.

## Classes

Exactly two, defined at the **head level**, not the person level:

| Class | Definition |
|---|---|
| `helmet` | A head wearing a safety helmet |
| `no_helmet` | A visible human head without a safety helmet |

A `person` class is deliberately excluded (PRD §3). If a head is not identifiable at all
(fully occluded, out of frame), **do not label it** — don't guess.

## Box extent

The box covers the **head region** (approximately crown to chin, ear to ear), not the whole
person and not just the helmet shell. This keeps the box meaningful for both classes
consistently — a `no_helmet` box has no helmet shell to anchor to, so anchoring to the head
itself is the only rule that works for both classes.

## Rulings on ambiguous cases

| Case | Ruling | Rationale |
|---|---|---|
| Helmet carried in hand, not worn | **Not** `helmet`. If the head is bare, label `no_helmet`. | The class is *head wearing a helmet*, not *helmet present in frame* (PRD §3). |
| Cap, hairnet, or bare head | `no_helmet` | None of these are safety helmets. |
| Partially occluded head (behind machinery, another person) | Label if the head is **identifiable as a head** — enough visible to judge helmet/no-helmet | Matches TDD §3; don't label unidentifiable fragments. |
| Head at frame edge | Label if **> ~50% visible** | Below that, the box would be mostly guesswork. |
| Motion-blurred head | Label if a human annotator can still classify it (helmet vs. no helmet) | Blur alone isn't disqualifying if the class is still visually decidable. |
| Extreme close-up (e.g. selfie-style clips, `s03_selfie_hardhat`) | Label normally, but flag the session in notes — see caveat below | Box drawing rules are the same; the *representativeness* of the shot is a dataset-composition concern, not a labeling one. |
| Two heads overlapping / crowding | Label each identifiable head separately | Matches the crowding edge case in the dataset plan. |
| Helmet color / style variation (hard hat, bump cap, etc.) | All count as `helmet` if it's a rigid safety helmet | The class is about protection category, not a specific product. |

## Session-specific notes (from Slice 1.1 recon)

- `s01_broadcast_tour` (`video_1.mp4`): mixed crowd scene — expect several `helmet` and
  `no_helmet` instances per frame, with real occlusion. The meeting-room segment is
  **entirely `no_helmet`** (business attire, no hard hats) — good negative-for-helmet /
  positive-for-no_helmet source, but note it's an indoor office setting, not a factory floor,
  so don't over-represent it relative to floor scenes in the final split.
- `s02_workshop_handheld` (`video_2.mp4`): single bare-headed subject at close range with
  heavy motion blur — a genuine hard case for the motion-blur ruling above.
- `s03_selfie_hardhat` (`video_3.mp4`/`video_4.mp4`, duplicate content — label only one, or
  label both but keep them in the same split): extreme close-up is not representative of the
  real deployment distance (see `docs/reports/recon_findings.md`) — fine to label, but do not
  let this session dominate the `helmet` class numerically since it teaches a very different
  head scale than a real fixed-camera shot would.

## Format

Export **YOLO format** (`class cx cy w h`, normalized) to `data/raw/annotations/`. Class IDs
must match `data/metadata/class_names.txt` exactly (`0 = helmet`, `1 = no_helmet`).

## Auto-label assist (deferred)

Not applicable yet — no trained model exists. Once Phase 2's M1 baseline exists, auto-label
assist may be used per TDD §3: human-review **every** auto-labeled frame, and **never**
auto-label the test split (R9) — it would bake the model's blind spots into the yardstick
used to judge it.
