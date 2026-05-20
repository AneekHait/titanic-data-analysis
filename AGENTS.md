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
make test           # pytest tests/ -v
make clean          # remove generated figures and caches
```

Scripts must be run from the project root. They inject `sys.path` internally.

## Architecture

- `src/` — installable package (setuptools, `where = ["src"]`)
- `scripts/` — CLI entry points (click), self-add project root to `sys.path`
- `dashboard/` — self-contained generators (HTML + PDF), no `src/` imports needed beyond loader
- `outputs/figures/` — generated PNGs (gitignored)
- `dashboard/_pdf_charts/` — PDF-specific charts (gitignored)

## Quirks

- **Seaborn 0.13+**: `palette={0: ..., 1: ...}` dicts fail — use list palettes (`["#red", "#green"]`) and set `hue` explicitly on boxplots
- **fpdf2**: Latin-1 encoding only — no bullet (`•`), em-dash, or non-ASCII in PDF text
- **Chart.js datalabels**: requires `Chart.register(ChartDataLabels)` before creating charts
- **Title extraction**: names use `, Title Firstname` format (no period after title) — regex is `r", ([A-Za-z]+) "`
- **Fare parsing**: British pre-decimal format (`£211 60s 9d`) — `_parse_price()` converts to float
- **Embarked mapping**: includes Belfast (`B`) for crew — 10 passengers

## Stale Files (need updates)

- `README.md` — all stats reference old Kaggle dataset (891 passengers). Correct titanic5 stats:
  - Women: 72.8%, Men: 19.1%
  - Class 1: 62.0%, Class 2: 42.8%, Class 3: 25.5%
  - Children (0-16): 49.0%
  - Cherbourg: 56.6%
- `notebooks/01_exploratory_analysis.ipynb` — same stale stats, references 891 rows

## Testing

- Only `tests/test_loader.py` exists (4 tests — all for data loading)
- No tests for `src/analysis/eda.py`, `src/visualization/plots.py`, or dashboard generators
- Run: `pytest tests/ -v`

## Dependencies

Core: pandas, numpy, matplotlib, seaborn, scipy, click
Dev: pytest, jupyter
PDF: fpdf2 (installed separately, not in requirements.txt)
