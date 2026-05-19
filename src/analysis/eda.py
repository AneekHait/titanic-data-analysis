import pandas as pd
import numpy as np
from src.config import TARGET, NUM_COLS, CAT_COLS


def data_overview(df: pd.DataFrame) -> dict:
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "head": df.head(3),
        "describe": df.describe(include="all"),
    }


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    summary = pd.DataFrame({"missing": missing, "percent": pct})
    return summary[summary["missing"] > 0].sort_values("percent", ascending=False)


def survival_rate(df: pd.DataFrame) -> pd.Series:
    return df[TARGET].value_counts(normalize=True).mul(100)


def survival_by_categorical(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return (
        df.groupby(col)[TARGET]
        .agg(["count", "mean"])
        .rename(columns={"count": "passengers", "mean": "survival_rate"})
        .assign(survival_rate=lambda x: x["survival_rate"].mul(100))
        .round(2)
    )


def survival_by_numerical(df: pd.DataFrame, col: str, bins: int = 5) -> pd.DataFrame:
    df = df.copy()
    binned = pd.cut(df[col], bins=bins, precision=0)
    return (
        df.groupby(binned, observed=False)[TARGET]
        .agg(["count", "mean"])
        .rename(columns={"count": "passengers", "mean": "survival_rate"})
        .assign(survival_rate=lambda x: x["survival_rate"].mul(100))
        .round(2)
    )


def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df[NUM_COLS + [TARGET]].copy()
    return numeric_df.corr(numeric_only=True).round(3)
