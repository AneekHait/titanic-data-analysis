# Titanic Survival Analysis

A complete exploratory analysis of the **titanic5** passenger dataset (1,309 passengers, 14 features). The project quantifies who survived and why, validates every claim with formal statistical tests, and ships three production-quality deliverables: an **interactive HTML dashboard**, a **comprehensive analyst report** (DOCX + PDF), and a **classical EDA report PDF**.

**Author:** [Aneek Hait](https://aneekhait.github.io)
**License:** [MIT](LICENSE)
**Dataset:** [titanic5](https://hbiostat.org/data/repo/titanic5.csv) — Encyclopedia Titanica / Vanderbilt Biostatistics

---

## What you'll find here

### 📊 Four deliverables

| Output | Audience | Path |
|---|---|---|
| **📖 Analyst report — rendered Markdown** | **Reads inline on GitHub** | [**REPORT.md**](REPORT.md) ← start here |
| **Interactive dashboard** (Chart.js, dark + light themes, scroll-spy nav) | Anyone exploring the data | [dashboard/index.html](dashboard/index.html) |
| **Analyst report — DOCX (editable)** | Stakeholders, hiring managers, peer review | [reports/Titanic_Survival_Analyst_Report.docx](reports/Titanic_Survival_Analyst_Report.docx) |
| **Analyst report — PDF** | Same content, fixed-layout for distribution | [reports/Titanic_Survival_Analyst_Report.pdf](reports/Titanic_Survival_Analyst_Report.pdf) |
| **Classical EDA report** (generated via fpdf2) | Legacy / chart-heavy view | [dashboard/titanic_eda_report.pdf](dashboard/titanic_eda_report.pdf) |

### 🧰 Reusable Python package
A clean `src/` layout: data loading, cleaning, feature engineering, EDA, statistics, plotting — all unit-tested.

---

## Quick start

```bash
git clone https://github.com/AneekHait/titanic-data-analysis.git
cd titanic-data-analysis
make install       # install dependencies
make download      # fetch the titanic5 CSV (~1 MB)
make eda           # run the full EDA pipeline + write 8 charts
make dashboard     # build the interactive HTML dashboard
make report        # build the analyst DOCX + PDF
make test          # 49 pytest tests
```

Outputs land in `dashboard/`, `reports/`, and `outputs/figures/`.

---

## Headline findings

| Factor | Finding |
|---|---|
| **Overall** | 38.2% of 1,309 passengers survived (500 lived, 809 perished) |
| **Sex** | Women 72.8% vs men 19.1% — odds ratio ≈ **11.3×** |
| **Class** | 1st 62.0%, 2nd 42.8%, 3rd 25.5% (Cramer's V ≈ 0.31) |
| **Class × Sex** | 1st-class women 96.5% vs 3rd-class men 15.2% — an 81-pp gap |
| **Age** | Children (≤16) survived at 49.0%; "children first" was real but small (OR ≈ 1.7×) |
| **Fare** | Strongest numeric predictor (r = 0.247), but mostly a proxy for class |
| **Embarked** | Cherbourg 56.6% — confounded with class composition |
| **Lifeboat (mechanism)** | 98.6% of recorded boat occupants survived vs 2.6% without a boat record |

Effect-size ranking (single comparable scale, |r| or Cramer's V):

```
Sex      ▰▰▰▰▰▰▰▰▰▰▰▰▰  0.527  (Large)
Pclass   ▰▰▰▰▰▰▰         0.313  (Medium)
Fare     ▰▰▰▰▰           0.247  (Small)
Embarked ▰▰▰▰            0.204  (Small)
Parch    ▰▰              0.083  (Negligible)
Age      ▰              -0.031  (Negligible)
SibSp    ▰              -0.028  (Negligible)
```

---

## Project structure

```
titanic-data-analysis/
├── src/                          # Python package (installable, tested)
│   ├── config.py                 # Paths and constants
│   ├── data/
│   │   ├── loader.py             # Download & load CSV
│   │   └── processing.py         # clean_data + engineer_features
│   ├── analysis/
│   │   ├── eda.py                # Survival rates, missing values
│   │   ├── statistics.py         # Chi², t-test, ANOVA, Cohen's d, Cramer's V
│   │   └── inference.py          # Wilson CIs, odds ratios, joint tables
│   └── visualization/
│       └── plots.py              # Reusable Matplotlib plot functions
├── scripts/
│   ├── download_data.py          # Fetch titanic5.csv
│   └── run_eda.py                # CLI EDA entry point (click)
├── dashboard/                    # Interactive Chart.js dashboard
│   ├── generate.py               # Builds index.html
│   ├── generate_pdf.py           # Builds the classical EDA PDF
│   ├── index.html                # ← published dashboard
│   └── titanic_eda_report.pdf    # ← published classical PDF
├── reports/                      # Author-written analyst report
│   ├── build_analyst_report.py
│   ├── validate_dataset.py       # Cross-stage count reconciliation
│   ├── Titanic_Survival_Analyst_Report.docx
│   └── Titanic_Survival_Analyst_Report.pdf
├── notebooks/
│   └── 01_exploratory_analysis.ipynb
├── tests/                        # 49 pytest tests across 5 files
├── docs/                         # Methodology, data dictionary, dev guide
│   ├── METHODOLOGY.md
│   ├── DATA.md
│   └── DEVELOPMENT.md
├── data/raw/                     # titanic5.csv (gitignored)
├── Dockerfile
├── Makefile
├── pyproject.toml
├── requirements.txt
├── README.md                     # ← you are here
├── CHANGELOG.md
├── CONTRIBUTING.md
├── ROADMAP.md                    # What's still aspirational
├── LICENSE                       # MIT
└── AGENTS.md                     # Notes for AI coding agents
```

---

## Documentation

| Doc | What's inside |
|---|---|
| [docs/METHODOLOGY.md](docs/METHODOLOGY.md) | Analytical approach, statistical machinery, why each test was chosen |
| [docs/DATA.md](docs/DATA.md) | Data dictionary, source, cleaning rules, engineered features |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Local setup, running the suite, adding a feature |
| [reports/README.md](reports/README.md) | Index of generated reports and how to rebuild them |
| [dashboard/README.md](dashboard/README.md) | What's in the dashboard, how to regenerate, theme notes |
| [AGENTS.md](AGENTS.md) | Repo-specific notes for AI agents (Claude Code, Cursor, etc.) |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to propose changes |
| [CHANGELOG.md](CHANGELOG.md) | Recent improvements |
| [ROADMAP.md](ROADMAP.md) | Outstanding ideas (ML modelling, Streamlit app, MLflow, etc.) |

---

## Tech stack

**Core:** pandas, numpy, scipy, matplotlib, seaborn, click
**Reports:** fpdf2 (classical PDF), python-docx (analyst DOCX), Chart.js (interactive dashboard)
**Tooling:** pytest, ruff, pre-commit, GitHub Actions, Docker

---

## Reproducibility

- **Dataset is gitignored.** `make download` fetches it from `hbiostat.org` on demand.
- **All numbers in every output trace to the same engineered DataFrame.** Run `python reports/validate_dataset.py` to verify counts reconcile across raw → cleaned → engineered.
- **CI runs the full pytest suite on every push** ([.github/workflows/ci.yml](.github/workflows/ci.yml)).
- **Docker:** `docker build -t titanic-eda . && docker run titanic-eda`.

---

## A note on this project

The classical PDF report ([dashboard/titanic_eda_report.pdf](dashboard/titanic_eda_report.pdf)) is the original output — generated programmatically by `generate_pdf.py`. It's chart-heavy and section-by-section.

The analyst report ([reports/Titanic_Survival_Analyst_Report.docx](reports/Titanic_Survival_Analyst_Report.docx)) is what I'd hand to a stakeholder — written as a narrative, with an executive summary, a glossary, and a "What this means in plain English" callout under every statistical finding. The DOCX is editable; the PDF is fixed-layout. The dashboard is the same content as an interactive page with hover tooltips, CIs as error bars, scroll-spy nav, and dark/light themes.

If you only have time for one: open the dashboard.

---

## Acknowledgements

- The titanic5 dataset is curated by [Encyclopedia Titanica](https://www.encyclopedia-titanica.org/) and hosted by [Vanderbilt Biostatistics](https://hbiostat.org/data/).
- Statistical methodology references: Wilson 1927 (score interval), Cohen 1988 (effect sizes), Cramer 1946 (Cramer's V).

Built by **[Aneek Hait](https://aneekhait.github.io)**.
