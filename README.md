# Titanic EDA

Exploratory Data Analysis of the Titanic passenger dataset — a structured Python project with CLI tools, interactive dashboard, and detailed PDF report.

## Quick Start

```bash
git clone https://github.com/AneekHait/titanic-data-analysis.git
cd titanic-data-analysis
make install       # install dependencies
make download      # fetch the dataset (891 passengers)
make eda           # run full analysis + generate 8 charts
```

## Project Structure

```
titanic-eda/
├── src/                          # Reusable Python package
│   ├── config.py                 # Paths, constants, column definitions
│   ├── data/
│   │   └── loader.py             # Download & load Titanic CSV
│   ├── analysis/
│   │   └── eda.py                # Stats, survival rates, correlations
│   └── visualization/
│       └── plots.py              # 8 reusable chart functions
├── scripts/
│   ├── download_data.py          # One-shot data download
│   └── run_eda.py                # CLI entry point (click)
├── dashboard/
│   ├── generate.py               # Interactive HTML dashboard generator
│   ├── index.html                # Generated Chart.js dashboard
│   ├── generate_pdf.py           # PDF report generator
│   └── titanic_eda_report.pdf    # Generated 14-page report
├── notebooks/
│   └── 01_exploratory_analysis.ipynb  # Jupyter walkthrough
├── tests/
│   └── test_loader.py            # pytest test suite
├── data/raw/                     # Raw dataset (gitignored)
├── outputs/figures/              # Generated PNG charts
├── requirements.txt
├── pyproject.toml
└── Makefile
```

## Features

### CLI Analysis

```bash
python scripts/run_eda.py              # full report + all charts
python scripts/run_eda.py --no-plots   # stats only, skip chart generation
python scripts/run_eda.py --data path/to/custom.csv  # use your own data
```

Produces:
- Data overview (shape, dtypes, summary statistics)
- Missing value analysis
- Survival rate by Sex, Pclass, Embarked
- Survival by Age and Fare (binned)
- Correlation matrix
- 8 PNG charts in `outputs/figures/`
- Key insights summary

### Interactive Dashboard

```bash
python dashboard/generate.py   # generates dashboard/index.html
```

Features:
- Sidebar navigation with smooth scrolling
- 10 interactive Chart.js charts
- Animated stat counters
- Dark/Light theme toggle
- Mobile responsive
- Correlation heatmap
- Missing values with progress bars

### PDF Report

```bash
python dashboard/generate_pdf.py   # generates dashboard/titanic_eda_report.pdf
```

A 14-page professional report with:
- Cover page and table of contents
- 11 embedded charts (survival, age, fare, class, gender, correlation, etc.)
- Data tables with formatted statistics
- 8 key findings with detailed analysis
- Conclusion section

## Key Findings

| Factor | Finding |
|---|---|
| **Overall** | Only 38.4% of passengers survived |
| **Gender** | Women: 74.2% survival vs Men: 18.9% |
| **Class** | 1st: 63.0%, 2nd: 47.3%, 3rd: 24.2% |
| **Age** | Children (0-16): 55.0% survival |
| **Fare** | Top quintile: 64.7-100% survival |
| **Port** | Cherbourg: 55.4% (highest) |
| **Family** | Small families (2-4) had best odds |
| **Title** | Mrs: 79.5%, Mr: 15.7% |

## Tech Stack

- **pandas** — data wrangling
- **numpy** — numerical operations
- **matplotlib + seaborn** — static visualizations
- **Chart.js** — interactive dashboard charts
- **fpdf2** — PDF report generation
- **click** — CLI argument parsing
- **pytest** — testing framework

## Commands

| Command | Description |
|---|---|
| `make install` | Install all dependencies |
| `make download` | Download the Titanic dataset |
| `make eda` | Run full EDA analysis |
| `make test` | Run pytest test suite |
| `make clean` | Remove generated files |

## Dataset

Source: [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data)

- **891 passengers**, 12 features
- Features: PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
- Missing values: Cabin (77.1%), Age (19.9%), Embarked (0.2%)

## License

MIT
