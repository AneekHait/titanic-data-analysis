"""Statistical tests for Titanic survival analysis."""

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats


def chi_square_test(df: pd.DataFrame, col: str, target: str = "Survived") -> dict:
    """Run chi-square test of independence between a categorical column and survival.

    Args:
        df: DataFrame with the column and target.
        col: Categorical column name.
        target: Target column name (default: Survived).

    Returns:
        Dict with chi2 statistic, p-value, degrees of freedom, and interpretation.
    """
    df_clean = df.dropna(subset=[col])
    ct = pd.crosstab(df_clean[col], df_clean[target])
    chi2, p, dof, _ = scipy_stats.chi2_contingency(ct)

    strength = "Very Strong" if p < 1e-10 else "Strong" if p < 0.001 else "Moderate" if p < 0.01 else "Weak" if p < 0.05 else "None"

    return {
        "chi2": round(chi2, 3),
        "p_value": p,
        "dof": dof,
        "significant": p < 0.05,
        "strength": strength,
        "contingency_table": ct,
    }


def t_test_survival(df: pd.DataFrame, col: str) -> dict:
    """Run Welch's t-test comparing a numerical column between survived and perished.

    Args:
        df: DataFrame with the column and Survived.
        col: Numerical column name.

    Returns:
        Dict with t-statistic, p-value, means, and interpretation.
    """
    survived = df[df["Survived"] == 1][col].dropna()
    perished = df[df["Survived"] == 0][col].dropna()

    t_stat, p_value = scipy_stats.ttest_ind(survived, perished, equal_var=False)

    cohens_d = (survived.mean() - perished.mean()) / np.sqrt(
        (survived.std() ** 2 + perished.std() ** 2) / 2
    )

    effect = "Large" if abs(cohens_d) > 0.8 else "Medium" if abs(cohens_d) > 0.5 else "Small" if abs(cohens_d) > 0.2 else "Negligible"

    return {
        "t_statistic": round(t_stat, 3),
        "p_value": p_value,
        "mean_survived": round(survived.mean(), 2),
        "mean_perished": round(perished.mean(), 2),
        "cohens_d": round(cohens_d, 3),
        "effect_size": effect,
        "significant": p_value < 0.05,
        "n_survived": len(survived),
        "n_perished": len(perished),
    }


def anova_survival(df: pd.DataFrame, group_col: str, value_col: str = "Survived") -> dict:
    """Run one-way ANOVA comparing survival rates across groups.

    Args:
        df: DataFrame with group and value columns.
        group_col: Categorical grouping column.
        value_col: Numerical value column (default: Survived).

    Returns:
        Dict with F-statistic, p-value, and group means.
    """
    df_clean = df.dropna(subset=[group_col, value_col])
    groups = [g[value_col].values for _, g in df_clean.groupby(group_col, observed=True)]

    f_stat, p_value = scipy_stats.f_oneway(*groups)

    group_means = df_clean.groupby(group_col, observed=True)[value_col].mean()

    return {
        "f_statistic": round(f_stat, 3),
        "p_value": p_value,
        "significant": p_value < 0.05,
        "group_means": group_means.round(3),
    }


def effect_sizes(df: pd.DataFrame) -> pd.DataFrame:
    """Compute effect sizes for all features against survival.

    For categorical features: Cramer's V.
    For numerical features: Point-biserial correlation.

    Args:
        df: DataFrame with features and Survived column.

    Returns:
        DataFrame with feature, type, effect size, and interpretation.
    """
    results = []

    cat_cols = ["Sex", "Pclass", "Embarked"]
    num_cols = ["Age", "Fare", "SibSp", "Parch"]

    for col in cat_cols:
        df_clean = df.dropna(subset=[col])
        ct = pd.crosstab(df_clean[col], df_clean["Survived"])
        chi2, _, dof, _ = scipy_stats.chi2_contingency(ct)
        n = ct.sum().sum()
        cramers_v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))

        strength = "Large" if cramers_v > 0.5 else "Medium" if cramers_v > 0.3 else "Small" if cramers_v > 0.1 else "Negligible"

        results.append({
            "feature": col,
            "type": "categorical",
            "effect_size": round(cramers_v, 3),
            "metric": "Cramer's V",
            "strength": strength,
        })

    for col in num_cols:
        df_clean = df.dropna(subset=[col])
        r, p = scipy_stats.pointbiserialr(df_clean["Survived"], df_clean[col])

        strength = "Large" if abs(r) > 0.5 else "Medium" if abs(r) > 0.3 else "Small" if abs(r) > 0.1 else "Negligible"

        results.append({
            "feature": col,
            "type": "numerical",
            "effect_size": round(r, 3),
            "metric": "Point-biserial r",
            "strength": strength,
        })

    return pd.DataFrame(results).sort_values("effect_size", ascending=False, key=abs)
