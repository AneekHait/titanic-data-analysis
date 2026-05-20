# AGENTS.md

## Dataset

- **titanic5** (1,309 passengers) — NOT the Kaggle train set (891 rows)
- Source: `https://hbiostat.org/data/repo/titanic5.csv` (Encyclopedia Titanica / Vanderbilt)
- Local file: `data/raw/titanic5.csv` (gitignored — run `make download` to fetch)
- Columns: PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Embarked, Occupation, BoatBody, NameId
- Missing: Age (51 / 3.9%), Occupation (621 / 47.4%)

## Key Commands

```bash
make install        # pip install -r requirements.txt
make download       # fetch titanic5.csv from hbiostat.org
make eda            # run scripts/run_eda.py (console + 8 charts)
make test           # pytest tests/ -v (49 tests)
make clean          # remove generated figures and caches
```

Scripts must be run from the project root. They inject `sys.path` internally.

## Architecture

- `src/` — installable package (setuptools, `where = ["src"]`)
  - `src/data/` — loader.py (download/load), processing.py (clean + feature engineering)
  - `src/analysis/` — eda.py (stats, rates, correlations), statistics.py (chi-square, t-test, ANOVA, effect sizes)
  - `src/visualization/` — plots.py (6 chart functions)
- `scripts/` — CLI entry points (click), self-add project root to `sys.path`
- `dashboard/` — self-contained generators (HTML + PDF)
- `tests/` — pytest suite (49 tests across 4 files)
- `outputs/figures/` — generated PNGs (gitignored)
- `dashboard/_pdf_charts/` — PDF-specific charts (gitignored)

## Feature Engineering (`src/data/processing.py`)

- `clean_data(df)` — fills Embarked (mode), Fare (median by Pclass)
- `engineer_features(df, impute_age=True)` — creates:
  - Title, FamilySize, IsAlone, AgeGroup, FareGroup
  - FarePerPerson, HasCabin, Lifeboat, BodyRecovered
  - Age imputation via median by Sex+Pclass

## Statistical Tests (`src/analysis/statistics.py`)

- `chi_square_test(df, col)` — independence test with Cramer's V
- `t_test_survival(df, col)` — Welch's t-test with Cohen's d
- `anova_survival(df, group_col)` — one-way ANOVA
- `effect_sizes(df)` — Cramer's V (categorical) + point-biserial r (numerical)

## Quirks

- **Seaborn 0.13+**: `palette={0: ..., 1: ...}` dicts fail — use list palettes (`["#red", "#green"]`) and set `hue` explicitly on boxplots
- **fpdf2**: Latin-1 encoding only — no bullet (`•`), em-dash, or non-ASCII in PDF text
- **Chart.js datalabels**: requires `Chart.register(ChartDataLabels)` before creating charts
- **Title extraction**: names use `, Title Firstname` format (no period after title) — regex is `r", ([A-Za-z]+) "`
- **Fare parsing**: British pre-decimal format (`£211 60s 9d`) — `_parse_price()` converts to float
- **Embarked mapping**: includes Belfast (`B`) for crew — 10 passengers
- **Numpy booleans**: `scipy_stats` returns `np.bool_`, use `== True` not `is True` in tests

## Testing

- 49 tests across 4 files: `test_loader.py`, `test_processing.py`, `test_eda.py`, `test_statistics.py`
- Shared fixtures in `tests/conftest.py` (raw_df, clean_df, engineered_df)
- Run: `pytest tests/ -v`
- CI: `.github/workflows/ci.yml` runs tests on push/PR

## Linting & Formatting

- **Ruff** — configured in `pyproject.toml` (line-length 100, py310 target)
- Rules: E, F, I, N, W, UP
- Pre-commit: `.pre-commit-config.yaml` runs ruff + pytest

## Dependencies

Core: pandas, numpy, matplotlib, seaborn, scipy, click
Dev: pytest, jupyter, ruff
PDF: fpdf2

## Docker

```bash
docker build -t titanic-eda .
docker run titanic-eda
```
