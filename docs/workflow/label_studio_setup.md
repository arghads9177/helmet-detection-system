# Label Studio — Local Setup (Slice 1.6)

Installed as an isolated `uv tool` (not a project dependency — it's a one-time annotation
utility, not something `ml/`/`api-services/` import):

```bash
uv tool install label-studio
```

## Restarting the server

The project (`helmet_detection_poc`) and its local-files config already exist under
`~/.local/share/label-studio/`. To restart after a reboot:

```bash
cd /media/argha-ds/F9AD-D6C3/softmeets/cv-poc/helmet-detection-system
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$(pwd)"
label-studio start helmet_detection_poc --port 8080 --no-browser --enable-legacy-api-token
```

- **URL:** http://localhost:8080
- **Login:** `admin@helmet-poc.local` / `helmetpoc-admin-2026`

## Project configuration

- Labeling interface: bounding boxes, classes `helmet` (green) / `no_helmet` (red), matching
  `data/metadata/class_names.txt`. Config source: `docs/workflow/label_studio_config.xml`.
- Two local-file storage sources synced as tasks (136 total):
  - `raw_images` → `data/raw/images/` (88 stills)
  - `cleaned_video_frames` → `data/interim/cleaned/` (47 curated frames, sub-folders scanned)
- Re-sync either source (Project > Settings > Cloud Storage > Sync Storage) after adding new
  images/frames to those directories.

Read `docs/workflow/annotation_guidelines.md` before labeling — it resolves the ambiguous
cases (carried helmets, occlusion, frame-edge heads, motion blur, box extent).

## Exporting annotations (Slice 1.6 output)

Project menu > Export > YOLO format, into `data/raw/annotations/` (per TDD §3). Do this once
labeling is complete, or incrementally if you want intermediate QC passes (Slice 1.7) sooner.
