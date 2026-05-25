"""Confidence intervals and odds ratios for survival analysis."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def wilson_ci(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    More accurate than the normal approximation for small n or proportions near 0/1.
    Returns (lower, upper) as percentages (0-100).
    """
    if n == 0:
        return 0.0, 0.0
    z = scipy_stats.norm.ppf(1 - (1 - confidence) / 2)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, (center - half) * 100), min(100.0, (center + half) * 100)


def survival_rates_with_ci(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Survival rate plus Wilson 95% CI for each level of a categorical column."""
    df_clean = df.dropna(subset=[col])
    rows = []
    for level, sub in df_clean.groupby(col, observed=True):
        n = len(sub)
        s = int(sub["Survived"].sum())
        rate = s / n * 100
        lo, hi = wilson_ci(s, n)
        rows.append({
            "level": level,
            "n": n,
            "survived": s,
            "rate": round(rate, 2),
            "ci_low": round(lo, 2),
            "ci_high": round(hi, 2),
            "ci_half": round((hi - lo) / 2, 2),
        })
    return pd.DataFrame(rows)


def odds_ratio(df: pd.DataFrame, mask_exposed: pd.Series, label: str) -> dict:
    """Compute odds ratio of survival for an exposed vs unexposed group.

    Uses Haldane-Anscombe correction (add 0.5 to all cells) if any cell is zero.
    Returns dict with OR, 95% CI, exposed/unexposed survival rates, lift, p-value.
    """
    sub = df.dropna(subset=["Survived"]).copy()
    sub["_exposed"] = mask_exposed.reindex(sub.index).fillna(False).astype(bool)

    a = int(((sub["_exposed"]) & (sub["Survived"] == 1)).sum())  # exposed + survived
    b = int(((sub["_exposed"]) & (sub["Survived"] == 0)).sum())  # exposed + died
    c = int(((~sub["_exposed"]) & (sub["Survived"] == 1)).sum())  # unexposed + survived
    d = int(((~sub["_exposed"]) & (sub["Survived"] == 0)).sum())  # unexposed + died

    if min(a, b, c, d) == 0:
        a_, b_, c_, d_ = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    else:
        a_, b_, c_, d_ = a, b, c, d

    or_val = (a_ * d_) / (b_ * c_)
    se_log_or = math.sqrt(1 / a_ + 1 / b_ + 1 / c_ + 1 / d_)
    log_or = math.log(or_val)
    z = scipy_stats.norm.ppf(0.975)
    ci_low = math.exp(log_or - z * se_log_or)
    ci_high = math.exp(log_or + z * se_log_or)

    n_exposed = a + b
    n_unexposed = c + d
    rate_exposed = a / n_exposed * 100 if n_exposed else 0.0
    rate_unexposed = c / n_unexposed * 100 if n_unexposed else 0.0
    lift = rate_exposed - rate_unexposed

    table = np.array([[a, b], [c, d]])
    try:
        _, p_value = scipy_stats.fisher_exact(table)
    except ValueError:
        p_value = float("nan")

    return {
        "label": label,
        "odds_ratio": round(or_val, 3),
        "ci_low": round(ci_low, 3),
        "ci_high": round(ci_high, 3),
        "rate_exposed": round(rate_exposed, 2),
        "rate_unexposed": round(rate_unexposed, 2),
        "lift": round(lift, 2),
        "n_exposed": n_exposed,
        "n_unexposed": n_unexposed,
        "p_value": p_value,
    }


def key_odds_ratios(df: pd.DataFrame) -> list[dict]:
    """Standard battery of odds ratios for headline survival contrasts."""
    contrasts = [
        ("Female vs Male", df["Sex"] == "female"),
        ("1st Class vs 2nd/3rd", df["Pclass"] == 1),
        ("3rd Class vs 1st/2nd", df["Pclass"] == 3),
        ("Child (<=16) vs Adult", df["Age"] <= 16),
        ("Cherbourg vs Other Ports", df["Embarked"] == "C"),
        ("Top Fare Quartile vs Rest", df["Fare"] >= df["Fare"].quantile(0.75)),
        ("Bottom Fare Quartile vs Rest", df["Fare"] <= df["Fare"].quantile(0.25)),
        ("Has Family Aboard vs Alone", (df["SibSp"] + df["Parch"]) > 0),
    ]
    return [odds_ratio(df, mask, label) for label, mask in contrasts]


def joint_survival(df: pd.DataFrame, row_col: str, col_col: str) -> pd.DataFrame:
    """Joint survival-rate table with counts for a row x column breakdown."""
    rate = df.groupby([row_col, col_col], observed=True)["Survived"].mean().mul(100).round(1)
    count = df.groupby([row_col, col_col], observed=True)["Survived"].count()
    return pd.DataFrame({"rate": rate, "count": count}).reset_index()
