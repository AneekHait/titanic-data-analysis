#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import click

from src.config import DATA_RAW, CAT_COLS
from src.data.loader import load_titanic
from src.analysis.eda import (
    data_overview,
    missing_summary,
    survival_rate,
    survival_by_categorical,
    survival_by_numerical,
    correlation_analysis,
)
from src.visualization.plots import (
    set_style,
    plot_survival_rate,
    plot_categorical_survival,
    plot_age_distribution,
    plot_fare_distribution,
    plot_correlation_heatmap,
    plot_pairgrid,
)


@click.command()
@click.option("--data", default=None, help="Path to CSV file")
@click.option("--no-plots", is_flag=True, help="Skip generating plots")
def main(data: str | None, no_plots: bool) -> None:
    path = DATA_RAW / "titanic5.csv" if data is None else data
    df = load_titanic(path)

    sep = "─" * 56

    print(f"\n{sep}")
    print("  TITANIC — EXPLORATORY DATA ANALYSIS (titanic5)")
    print(f"{sep}\n")

    overview = data_overview(df)
    print(f"Dataset shape: {overview['shape'][0]} rows × {overview['shape'][1]} cols\n")

    print("Columns and dtypes:")
    for col, dtype in overview["dtypes"].items():
        print(f"  {col:20s} {str(dtype):15s}")
    print()

    print("Summary statistics (numeric):")
    print(df.describe().round(2).to_string())
    print()

    print(f"{sep}")
    print("  MISSING VALUES")
    print(f"{sep}")
    missing = missing_summary(df)
    if missing.empty:
        print("  No missing values found.\n")
    else:
        print(missing.to_string())
        print()

    print(f"{sep}")
    print("  SURVIVAL RATE")
    print(f"{sep}")
    sr = survival_rate(df)
    print(f"  Perished: {sr.get(0, 0):.1f}%")
    print(f"  Survived: {sr.get(1, 0):.1f}%")
    print()

    if not no_plots:
        set_style()
        print("  Generating plots...")

        for col in CAT_COLS:
            path = plot_categorical_survival(df, col)
            print(f"    ✓ {path.name}")

        for func in [plot_survival_rate, plot_age_distribution,
                     plot_fare_distribution, plot_correlation_heatmap,
                     plot_pairgrid]:
            path = func(df)
            print(f"    ✓ {path.name}")

        print()

    print(f"{sep}")
    print("  SURVIVAL BY CATEGORICAL FEATURES")
    print(f"{sep}")
    for col in CAT_COLS:
        print(f"\n  [{col}]")
        print(survival_by_categorical(df, col).to_string())
    print()

    print(f"{sep}")
    print("  SURVIVAL BY NUMERICAL FEATURES (binned)")
    print(f"{sep}")
    for col in ["Age", "Fare"]:
        print(f"\n  [{col}]")
        print(survival_by_numerical(df, col).to_string())
    print()

    print(f"{sep}")
    print("  CORRELATION MATRIX (numerical features)")
    print(f"{sep}")
    print(correlation_analysis(df).to_string())
    print()

    print(f"{sep}")
    print("  KEY INSIGHTS")
    print(f"{sep}")
    print("""
  1. Only about 38% of passengers survived the Titanic disaster.
  2. Women had a much higher survival rate than men.
  3. First-class passengers survived at a much higher rate than third-class.
  4. Children (age < 10) had higher survival rates.
  5. Passengers who paid higher fares were more likely to survive.
  6. Passengers embarking from Cherbourg (C) had higher survival rates.
  7. Age has only 51 missing values (3.9%) — far better than the Kaggle version.
  8. Dataset includes 1,309 passengers (47% more than Kaggle train set).
  """)


if __name__ == "__main__":
    main()
