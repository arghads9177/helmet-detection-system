# CLAUDE.md — helmet-detection-system

This repo follows the **cv-project-engineering** conventions. Key rules:

- **Environment**: uv only, at the repo root. Add deps with `uv add`, never bare pip.
  After dependency changes run `uv export --no-dev -o requirements.txt` and, if
  `api-services/` deps changed, re-export its scoped `requirements.txt` too.
- **Config**: pipeline config (paths, hyperparameters, thresholds) lives in
  `configs/*.yaml`, loaded via `scripts/utilities/config.py`. Per-service secrets and
  runtime env (`.env`) live in `api-services/config/` and `ui-services/config/`, loaded
  via each service's pydantic-settings module. Never hardcode these values, never read
  `os.environ` directly outside a settings object.
- **Logging**: no print() outside notebooks. `logger = logging.getLogger(__name__)`;
  entry points call `setup_logging()` from `scripts/utilities/logging_setup.py`
  (driven by `configs/logging.yaml`). Logs land in `outputs/logs/`.
- **Layout**: `ml/` is the importable package (training/ inference/ evaluation/
  postprocessing/) — the shared library `scripts/`, `api-services/`, and `ui-services/`
  all import from. `scripts/` are focused entry points, not where reusable logic lives.
  Notebooks are exploration only.
- **Data**: `data/raw` is immutable; `data/interim` is disposable scratch space;
  `data/processed` must be regenerable by `scripts/preprocessing/`,
  `scripts/augmentation/`, and `scripts/dataset_split/`. Datasets and weights are
  gitignored; `data/metadata/` and `artifacts/experiment_results/*/metadata.json`
  ARE committed and must accompany every trained model.
- **Tests**: every feature lands with tests. `ml/tests/` = unit tests for ml/ logic,
  service `tests/` dirs = component tests, `tests/integration|performance|end_to_end`
  = cross-cutting. Run: `uv run pytest`.
- **Docs**: keep README, `docs/requirements/PRD.md`, `docs/architecture/TDD.md` current
  as the design evolves. New vertical slices get a short spec in `.claude/feature-specs/`
  before implementation.

## Local environment quirk: exFAT working directory

This repo's working copy lives on an exFAT-formatted drive, which does not support
symlinks. `uv sync` fails by default (`failed to symlink ... Operation not permitted`)
because uv links the Python interpreter into `.venv/bin`. Fix: point the venv outside
the exFAT filesystem before running `uv sync` or `uv run`:

```bash
export UV_PROJECT_ENVIRONMENT=~/.venvs/helmet-detection-system
```

Set this in your shell profile or `direnv` for this directory rather than repeating it
per command.
