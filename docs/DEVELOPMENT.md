# Development Guide

## Prerequisites

- Python **3.10+** (tested on 3.13, 3.14)
- `make` (optional but recommended)
- ~150 MB free disk for dependencies

---

## Local setup

```bash
git clone https://github.com/AneekHait/titanic-data-analysis.git
cd titanic-data-analysis

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev,reports]"    # editable install + dev + report extras

make download                       # fetch titanic5.csv
make test                           # 49 tests, ~5 s
```

`pip install -e ".[dev,reports]"` installs:

- The `src/` package in editable mode (so changes are picked up without reinstalling).
- `dev` extras: pytest, jupyter, ruff, pre-commit.
- `reports` extras: fpdf2 (classical PDF), python-docx (analyst DOCX).

---

## Pre-commit hooks

Set up once after cloning:

```bash
pre-commit install
```

Every `git commit` then runs:

- **ruff** with `--fix` (lint + auto-fix)
- **ruff-format** (formatter)
- **pytest** (full suite)

Configuration: [.pre-commit-config.yaml](../.pre-commit-config.yaml) and the `[tool.ruff]` section of [pyproject.toml](../pyproject.toml).

---

## Building the outputs

| Target | What it builds | Time |
|---|---|---|
| `make eda` | Console summary + 8 PNG charts in `outputs/figures/` | ~10 s |
| `make dashboard` | `dashboard/index.html` | ~3 s |
| `make pdf` | `dashboard/titanic_eda_report.pdf` (classical, fpdf2) | ~15 s |
| `make report` | `reports/Titanic_Survival_Analyst_Report.docx` | ~3 s |
| `make all` | All of the above | ~30 s |
| `make validate` | `python reports/validate_dataset.py` — reconcile counts | ~2 s |
| `make clean` | Remove generated figures, caches | ~1 s |

---

## Running tests

```bash
make test                            # all 49 tests
pytest tests/ -v                     # same, more verbose
pytest tests/test_processing.py      # one file
pytest -k "test_engineer_features"   # by name pattern
```

CI runs `pytest tests/ -v` on every push via [.github/workflows/ci.yml](../.github/workflows/ci.yml).

---

## Project layout (for contributors)

| Path | Purpose |
|---|---|
| `src/data/` | Raw data IO + cleaning + feature engineering. Pure functions; no side effects beyond filesystem reads. |
| `src/analysis/` | Statistical summaries (`eda.py`), formal tests (`statistics.py`), and inference helpers (`inference.py`). |
| `src/visualization/` | Reusable Matplotlib plot functions. |
| `src/config.py` | Project paths and constants — single source of truth for filesystem layout. |
| `scripts/` | CLI entry points (click-powered). Scripts inject `sys.path` so they run from anywhere. |
| `dashboard/` | Output generators (HTML, classical PDF) + their generated outputs. |
| `reports/` | Author-written analyst report + its builder + validation script. |
| `tests/` | pytest suite. Shared fixtures in `conftest.py` (raw_df, clean_df, engineered_df). |

---

## Conventions

- **Functional, not object-oriented.** Data flows `load → clean → engineer → analyse → render`. Each function takes a DataFrame, returns a DataFrame or dict.
- **The engineered DataFrame is the source of truth.** Don't compute on `raw` after `clean_data` runs — totals won't reconcile (the 2 missing Embarked rows are the usual culprit).
- **One unit test per public function.** Tests live in `tests/test_<module>.py` mirroring `src/<module>.py`.
- **Line length 100.** Enforced by ruff.
- **Charts are deterministic.** Random seeds are set where any sampling happens.

---

## Common tasks

### Adding a new statistical test

1. Add the implementation to `src/analysis/statistics.py` (return a dict with keys: statistic name, value, p-value, effect-size value, interpretation string).
2. Add a unit test in `tests/test_statistics.py`.
3. Wire it into `dashboard/generate.py` and `reports/build_analyst_report.py` if it deserves user-facing surfacing.

### Adding a new engineered feature

1. Implement in `src/data/processing.py`, called from `engineer_features`.
2. Add a row to the engineered-features table in [docs/DATA.md](DATA.md).
3. Add a unit test in `tests/test_processing.py`.
4. Run `python reports/validate_dataset.py` to make sure totals still reconcile.

### Bumping the version

1. Update `version` in [pyproject.toml](../pyproject.toml).
2. Update `__version__` in [src/__init__.py](../src/__init__.py).
3. Add a section to [CHANGELOG.md](../CHANGELOG.md).

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `ModuleNotFoundError: src` when running scripts | Run from project root, or use `pip install -e .` |
| `FileNotFoundError: data/raw/titanic5.csv` | Run `make download` first |
| `make dashboard` produces empty / broken HTML | The Chart.js CDN is offline. The page degrades gracefully but charts won't render until network returns. |
| Counts don't add to 1,309 in a new analysis | You're computing on the raw DataFrame instead of the engineered one. See [docs/DATA.md](DATA.md#counts-that-should-reconcile). |
| `fpdf2` font errors on the classical PDF | Latin-1 only — strip non-ASCII before writing |
