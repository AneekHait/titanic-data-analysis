import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from src.config import TARGET, NUM_COLS, CAT_COLS, OUTPUTS_FIGURES


def set_style() -> None:
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


def _save(fig: plt.Figure, name: str) -> Path:
    OUTPUTS_FIGURES.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS_FIGURES / name
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return path


def plot_survival_rate(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df[TARGET].value_counts()
    bars = ax.bar(["Perished", "Survived"], counts.values, color=["#e74c3c", "#2ecc71"])
    ax.bar_label(bars, fmt="%d", padding=3)
    ax.set_ylabel("Passenger count")
    ax.set_title("Overall Survival Rate")
    total = len(df)
    for bar, label in zip(bars, ["Perished", "Survived"]):
        pct = bar.get_height() / total * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                f"{pct:.1f}%", ha="center", va="center", fontweight="bold", color="white")
    return _save(fig, "survival_rate.png")


def plot_categorical_survival(df: pd.DataFrame, col: str) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    ctab = pd.crosstab(df[col], df[TARGET], normalize="index").mul(100)
    ctab.plot(kind="bar", ax=ax, color=["#e74c3c", "#2ecc71"],
              legend=False)
    ax.set_ylabel("Survival rate (%)")
    ax.set_title(f"Survival Rate by {col}")
    ax.set_xlabel(col)
    ax.legend(["Perished", "Survived"])
    fig.tight_layout()
    return _save(fig, f"survival_by_{col.lower()}.png")


PALETTE = ["#e74c3c", "#2ecc71"]


def plot_age_distribution(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(df, x="Age", hue=TARGET, ax=axes[0], kde=True,
                 palette=PALETTE, edgecolor="white")
    axes[0].set_title("Age Distribution by Survival")
    sns.boxplot(df, x=TARGET, y="Age", hue=TARGET, ax=axes[1],
                palette=PALETTE, legend=False)
    axes[1].set_title("Age Boxplot by Survival")
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["Perished", "Survived"])
    fig.tight_layout()
    return _save(fig, "age_distribution.png")


def plot_fare_distribution(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(df, x="Fare", hue=TARGET, ax=axes[0], kde=True,
                 palette=PALETTE, edgecolor="white")
    axes[0].set_title("Fare Distribution by Survival")
    sns.boxplot(df, x=TARGET, y="Fare", hue=TARGET, ax=axes[1],
                palette=PALETTE, legend=False)
    axes[1].set_title("Fare Boxplot by Survival")
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["Perished", "Survived"])
    fig.tight_layout()
    return _save(fig, "fare_distribution.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> Path:
    numeric_df = df[NUM_COLS + [TARGET]].select_dtypes(include="number")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, ax=ax, cbar_kws={"shrink": 0.75})
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    return _save(fig, "correlation_heatmap.png")


def plot_pairgrid(df: pd.DataFrame) -> Path:
    sample = df.sample(min(500, len(df)), random_state=42)
    g = sns.PairGrid(sample, vars=NUM_COLS, hue=TARGET,
                     palette=PALETTE)
    g.map_diag(sns.histplot, edgecolor="white", kde=True)
    g.map_lower(sns.scatterplot, alpha=0.6)
    g.add_legend()
    g.fig.suptitle("Pairwise Relationships", y=1.02)
    path = OUTPUTS_FIGURES / "pairgrid.png"
    OUTPUTS_FIGURES.mkdir(parents=True, exist_ok=True)
    g.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(g.fig)
    return path
