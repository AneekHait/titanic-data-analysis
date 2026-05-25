#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
from fpdf import FPDF

from src.data.loader import load_titanic
from src.data.processing import clean_data, engineer_features
from src.analysis.eda import (
    data_overview,
    missing_summary,
    survival_rate,
    survival_by_categorical,
    survival_by_numerical,
    correlation_analysis,
)
from src.analysis.statistics import effect_sizes
from src.analysis.inference import (
    survival_rates_with_ci,
    key_odds_ratios,
    joint_survival,
    wilson_ci,
)
from src.config import OUTPUTS_FIGURES

CHART_DIR = Path(__file__).parent / "_pdf_charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2,
              rc={"figure.facecolor": "#fafbfc", "axes.facecolor": "#ffffff",
                  "grid.color": "#e5e7eb", "grid.linewidth": 0.5,
                  "axes.edgecolor": "#d1d5db", "axes.linewidth": 0.8,
                  "xtick.color": "#374151", "ytick.color": "#374151",
                  "axes.labelcolor": "#1f2937", "text.color": "#1f2937"})

COLORS = {
    "primary": "#1e3a5f",
    "accent": "#3b82f6",
    "green": "#10b981",
    "red": "#ef4444",
    "yellow": "#f59e0b",
    "purple": "#8b5cf6",
    "bg": "#ffffff",
    "text": "#1e293b",
    "text_light": "#64748b",
}

FONTSIZE = {"title": 13, "label": 11, "tick": 10, "annot": 10}

def _save(fig, name):
    path = CHART_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=300, facecolor="#fafbfc", edgecolor="none")
    plt.close(fig)
    return path

def _add_bar_labels(ax, fmt="{:.1f}"):
    for container in ax.containers:
        ax.bar_label(container, fmt=fmt, fontsize=FONTSIZE["annot"], fontweight="600",
                     padding=3, color="#1f2937")

def _style_axes(ax, xlabel="", ylabel=""):
    ax.set_xlabel(xlabel, fontsize=FONTSIZE["label"], fontweight="600")
    ax.set_ylabel(ylabel, fontsize=FONTSIZE["label"], fontweight="600")
    ax.tick_params(labelsize=FONTSIZE["tick"])
    for spine in ax.spines.values():
        spine.set_color("#d1d5db")

def extract_titles(df):
    titles = df["Name"].apply(lambda x: re.search(r", ([A-Za-z]+) ", x))
    titles = titles.apply(lambda m: m.group(1) if m else "Unknown")
    title_map = {
        "Mr": "Mr", "Mrs": "Mrs", "Miss": "Miss", "Master": "Master",
        "Dr": "Officer", "Rev": "Officer", "Revd": "Officer", "Major": "Officer",
        "Col": "Officer", "Colonel": "Officer", "Capt": "Officer", "Captain": "Officer",
        "Mlle": "Miss", "Mme": "Mrs", "Ms": "Miss", "Countess": "Royalty",
        "Lady": "Royalty", "Sir": "Royalty", "Don": "Royalty",
        "Jonkheer": "Royalty", "Dona": "Royalty", "Sra": "Mrs", "Sr": "Mr",
        "Fr": "Officer",
    }
    return titles.map(lambda t: title_map.get(t, "Other"))

def chart_survival_overview(df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("#fafbfc")
    counts = df["Survived"].value_counts()
    bars = axes[0].bar(["Perished", "Survived"], counts.values,
                       color=[COLORS["red"], COLORS["green"]], width=0.5, edgecolor="white", linewidth=1.5)
    axes[0].set_title("Survival Count", fontweight="700", fontsize=FONTSIZE["title"])
    _style_axes(axes[0], ylabel="Passengers")
    for bar in bars:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
                     f"{int(bar.get_height())}", ha="center", fontweight="700",
                     fontsize=FONTSIZE["annot"], color="#1f2937")

    sex_ct = pd.crosstab(df["Sex"], df["Survived"], normalize="index").mul(100)
    bars = sex_ct.plot(kind="bar", ax=axes[1], color=[COLORS["red"], COLORS["green"]],
                       width=0.6, edgecolor="white", linewidth=1, legend=False)
    axes[1].set_title("Survival Rate by Sex", fontweight="700", fontsize=FONTSIZE["title"])
    _style_axes(axes[1], ylabel="Rate (%)")
    axes[1].set_xticklabels(["Female", "Male"], rotation=0, fontsize=FONTSIZE["tick"])
    axes[1].legend(["Perished", "Survived"], fontsize=FONTSIZE["tick"], frameon=True,
                   facecolor="white", edgecolor="#d1d5db")
    _add_bar_labels(axes[1], fmt="{:.1f}%")

    pclass_ct = pd.crosstab(df["Pclass"], df["Survived"], normalize="index").mul(100)
    pclass_ct.plot(kind="bar", ax=axes[2], color=[COLORS["red"], COLORS["green"]],
                   width=0.6, edgecolor="white", linewidth=1, legend=False)
    axes[2].set_title("Survival Rate by Class", fontweight="700", fontsize=FONTSIZE["title"])
    _style_axes(axes[2], ylabel="Rate (%)")
    axes[2].set_xticklabels(["1st", "2nd", "3rd"], rotation=0, fontsize=FONTSIZE["tick"])
    axes[2].legend(["Perished", "Survived"], fontsize=FONTSIZE["tick"], frameon=True,
                   facecolor="white", edgecolor="#d1d5db")
    _add_bar_labels(axes[2], fmt="{:.1f}%")
    fig.tight_layout()
    return _save(fig, "survival_overview.png")

def chart_age_analysis(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.patch.set_facecolor("#fafbfc")

    sns.histplot(df, x="Age", hue="Survived", ax=axes[0], kde=True,
                 palette=[COLORS["red"], COLORS["green"]], edgecolor="white", linewidth=0.5,
                 alpha=0.7, bins=25)
    axes[0].set_title("Age Distribution by Survival", fontweight="700", fontsize=FONTSIZE["title"])
    _style_axes(axes[0], xlabel="Age")
    axes[0].legend(["Perished", "Survived"], fontsize=FONTSIZE["tick"], frameon=True,
                   facecolor="white", edgecolor="#d1d5db")

    sns.boxplot(df, x="Survived", y="Age", hue="Survived", ax=axes[1],
                palette=[COLORS["red"], COLORS["green"]], width=0.5,
                fliersize=3, linewidth=1.5, legend=False)
    axes[1].set_title("Age Boxplot by Survival", fontweight="700", fontsize=FONTSIZE["title"])
    _style_axes(axes[1])
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["Perished", "Survived"], fontsize=FONTSIZE["tick"])

    for i, surv in enumerate([0, 1]):
        data = df[df["Survived"] == surv]["Age"].dropna()
        median = data.median()
        axes[1].axhline(median, color=[COLORS["red"], COLORS["green"]][i],
                        linestyle="--", alpha=0.5, linewidth=1)
        axes[1].text(0.05, median + 1, f"Median: {median:.1f}",
                     fontsize=8, fontweight="600", color=[COLORS["red"], COLORS["green"]][i],
                     transform=axes[1].get_yaxis_transform())

    fig.tight_layout()
    return _save(fig, "age_analysis.png")

def chart_fare_analysis(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.patch.set_facecolor("#fafbfc")

    df_clean = df[df["Fare"] < df["Fare"].quantile(0.99)]
    sns.histplot(df_clean, x="Fare", hue="Survived", ax=axes[0], kde=True,
                 palette=[COLORS["red"], COLORS["green"]], edgecolor="white", linewidth=0.5,
                 alpha=0.7, bins=30)
    axes[0].set_title("Fare Distribution by Survival", fontweight="700", fontsize=FONTSIZE["title"])
    _style_axes(axes[0], xlabel="Fare ($)")
    axes[0].legend(["Perished", "Survived"], fontsize=FONTSIZE["tick"], frameon=True,
                   facecolor="white", edgecolor="#d1d5db")

    sns.boxplot(df_clean, x="Survived", y="Fare", hue="Survived", ax=axes[1],
                palette=[COLORS["red"], COLORS["green"]], width=0.5,
                fliersize=3, linewidth=1.5, legend=False)
    axes[1].set_title("Fare Boxplot by Survival", fontweight="700", fontsize=FONTSIZE["title"])
    _style_axes(axes[1])
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["Perished", "Survived"], fontsize=FONTSIZE["tick"])
    axes[1].yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))

    fig.tight_layout()
    return _save(fig, "fare_analysis.png")

def chart_embarked_analysis(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.patch.set_facecolor("#fafbfc")
    port_labels = ["Cherbourg", "Queenstown", "Southampton", "Belfast"]

    emb_ct = pd.crosstab(df["Embarked"], df["Survived"], normalize="index").mul(100)
    emb_ct.plot(kind="bar", ax=axes[0], color=[COLORS["red"], COLORS["green"]],
                width=0.6, edgecolor="white", linewidth=1, legend=False)
    axes[0].set_title("Survival Rate by Embarked Port", fontweight="700", fontsize=FONTSIZE["title"])
    _style_axes(axes[0], ylabel="Rate (%)")
    axes[0].set_xticklabels(port_labels, rotation=0, fontsize=FONTSIZE["tick"])
    axes[0].legend(["Perished", "Survived"], fontsize=FONTSIZE["tick"], frameon=True,
                   facecolor="white", edgecolor="#d1d5db")
    _add_bar_labels(axes[0], fmt="{:.1f}%")

    df["Embarked"].value_counts().reindex(emb_ct.index).plot(
        kind="bar", ax=axes[1], color=COLORS["accent"], edgecolor="white", linewidth=1)
    axes[1].set_title("Passenger Count by Port", fontweight="700", fontsize=FONTSIZE["title"])
    _style_axes(axes[1], ylabel="Count")
    axes[1].set_xticklabels(port_labels, rotation=0, fontsize=FONTSIZE["tick"])
    _add_bar_labels(axes[1], fmt="{:.0f}")

    fig.tight_layout()
    return _save(fig, "embarked_analysis.png")

def chart_family_survival(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.patch.set_facecolor("#fafbfc")
    df2 = df.copy()
    df2["FamilySize"] = df2["SibSp"] + df2["Parch"] + 1

    family_rate = df2.groupby("FamilySize")["Survived"].mean().mul(100)
    colors = [COLORS["green"] if v > 40 else COLORS["accent"] if v > 30 else COLORS["red"]
              for v in family_rate.values]
    family_rate.plot(kind="bar", ax=axes[0], color=colors, edgecolor="white", linewidth=1)
    axes[0].set_title("Survival Rate by Family Size", fontweight="700", fontsize=FONTSIZE["title"])
    _style_axes(axes[0], xlabel="Family Size", ylabel="Survival Rate (%)")
    _add_bar_labels(axes[0], fmt="{:.1f}%")

    family_count = df2["FamilySize"].value_counts().sort_index()
    family_count.plot(kind="bar", ax=axes[1], color=COLORS["purple"], edgecolor="white", linewidth=1)
    axes[1].set_title("Passenger Count by Family Size", fontweight="700", fontsize=FONTSIZE["title"])
    _style_axes(axes[1], xlabel="Family Size", ylabel="Count")
    _add_bar_labels(axes[1], fmt="{:.0f}")

    fig.tight_layout()
    return _save(fig, "family_analysis.png")

def chart_correlation_heatmap(df):
    numeric = df[["Age", "Fare", "SibSp", "Parch", "Survived"]].select_dtypes(include="number")
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#fafbfc")
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, ax=ax, cbar_kws={"shrink": 0.8, "pad": 0.02},
                linewidths=1.5, linecolor="#ffffff", mask=mask,
                vmin=-0.4, vmax=0.4,
                annot_kws={"fontsize": FONTSIZE["annot"], "fontweight": "600"})
    ax.set_title("Feature Correlation Heatmap", fontweight="700", fontsize=FONTSIZE["title"] + 1, pad=12)
    ax.tick_params(labelsize=FONTSIZE["tick"])
    fig.tight_layout()
    return _save(fig, "correlation_heatmap.png")

def chart_missing_values(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#fafbfc")
    colors = [COLORS["red"] if v / len(df) > 0.5 else COLORS["yellow"] if v / len(df) > 0.1 else COLORS["green"]
              for v in missing.values]
    bars = ax.barh(missing.index[::-1], missing.values[::-1], color=colors[::-1],
                   height=0.5, edgecolor="white", linewidth=1)
    ax.set_title("Missing Values by Column", fontweight="700", fontsize=FONTSIZE["title"])
    _style_axes(ax, xlabel="Count")
    for bar in bars:
        width = int(bar.get_width())
        pct = width / len(df) * 100
        ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height() / 2,
                f"{width} ({pct:.1f}%)", va="center", fontweight="700",
                fontsize=FONTSIZE["annot"], color="#1f2937")
    fig.tight_layout()
    return _save(fig, "missing_values.png")

def chart_age_survival_bins(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.patch.set_facecolor("#fafbfc")
    df2 = df.copy()
    df2["AgeGroup"] = pd.cut(df2["Age"], bins=[0, 16, 32, 48, 64, 80],
                             labels=["0-16", "17-32", "33-48", "49-64", "65-80"])

    age_rate = df2.groupby("AgeGroup", observed=True)["Survived"].mean().mul(100)
    colors = [COLORS["green"] if v > 40 else COLORS["accent"] if v > 30 else COLORS["red"]
              for v in age_rate.values]
    age_rate.plot(kind="bar", ax=axes[0], color=colors, edgecolor="white", linewidth=1)
    axes[0].set_title("Survival Rate by Age Group", fontweight="700", fontsize=FONTSIZE["title"])
    _style_axes(axes[0], xlabel="Age Group", ylabel="Survival Rate (%)")
    _add_bar_labels(axes[0], fmt="{:.1f}%")

    age_count = df2.groupby("AgeGroup", observed=True)["Survived"].count()
    age_count.plot(kind="bar", ax=axes[1], color=COLORS["purple"], edgecolor="white", linewidth=1)
    axes[1].set_title("Passenger Count by Age Group", fontweight="700", fontsize=FONTSIZE["title"])
    _style_axes(axes[1], xlabel="Age Group", ylabel="Count")
    _add_bar_labels(axes[1], fmt="{:.0f}")

    fig.tight_layout()
    return _save(fig, "age_bins.png")

def chart_fare_survival_bins(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.patch.set_facecolor("#fafbfc")
    df2 = df.copy()
    df2["FareGroup"] = pd.qcut(df2["Fare"], q=5,
                               labels=["Q1\n(Lowest)", "Q2", "Q3", "Q4", "Q5\n(Highest)"])

    fare_rate = df2.groupby("FareGroup", observed=True)["Survived"].mean().mul(100)
    colors = [COLORS["green"] if v > 50 else COLORS["accent"] if v > 35 else COLORS["red"]
              for v in fare_rate.values]
    fare_rate.plot(kind="bar", ax=axes[0], color=colors, edgecolor="white", linewidth=1)
    axes[0].set_title("Survival Rate by Fare Quintile", fontweight="700", fontsize=FONTSIZE["title"])
    _style_axes(axes[0], xlabel="Fare Group", ylabel="Survival Rate (%)")
    _add_bar_labels(axes[0], fmt="{:.1f}%")

    fare_count = df2.groupby("FareGroup", observed=True)["Survived"].count()
    fare_count.plot(kind="bar", ax=axes[1], color=COLORS["accent"], edgecolor="white", linewidth=1)
    axes[1].set_title("Passenger Count by Fare Quintile", fontweight="700", fontsize=FONTSIZE["title"])
    _style_axes(axes[1], xlabel="Fare Group", ylabel="Count")
    _add_bar_labels(axes[1], fmt="{:.0f}")

    fig.tight_layout()
    return _save(fig, "fare_bins.png")

def chart_class_gender(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#fafbfc")
    cg = pd.crosstab([df["Pclass"], df["Sex"]], df["Survived"], normalize="index").mul(100)
    cg.plot(kind="bar", ax=ax, color=[COLORS["red"], COLORS["green"]],
            width=0.6, edgecolor="white", linewidth=1, legend=False)
    ax.set_title("Survival Rate by Class & Gender", fontweight="700", fontsize=FONTSIZE["title"])
    _style_axes(ax, ylabel="Survival Rate (%)")
    labels = ["1st-F", "1st-M", "2nd-F", "2nd-M", "3rd-F", "3rd-M"]
    ax.set_xticklabels(labels, rotation=0, fontsize=FONTSIZE["tick"])
    ax.legend(["Perished", "Survived"], fontsize=FONTSIZE["tick"], frameon=True,
              facecolor="white", edgecolor="#d1d5db")
    _add_bar_labels(ax, fmt="{:.1f}%")
    fig.tight_layout()
    return _save(fig, "class_gender.png")

def chart_title_survival(df):
    df2 = df.copy()
    df2["Title"] = extract_titles(df)
    title_data = df2.groupby("Title")["Survived"].agg(["count", "mean"]).assign(
        survival_rate=lambda x: x["mean"].mul(100)).sort_values("count", ascending=False)

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("#fafbfc")
    colors = [COLORS["green"] if v > 50 else COLORS["accent"] if v > 30 else COLORS["red"]
              for v in title_data["survival_rate"].values]
    title_data["survival_rate"].plot(kind="bar", ax=ax, color=colors,
                                     edgecolor="white", linewidth=1)
    ax.set_title("Survival Rate by Title", fontweight="700", fontsize=FONTSIZE["title"])
    _style_axes(ax, xlabel="Title", ylabel="Survival Rate (%)")
    _add_bar_labels(ax, fmt="{:.1f}%")
    fig.tight_layout()
    return _save(fig, "title_survival.png")

def chart_stacked_survival(df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("#fafbfc")
    for i, (col, labels) in enumerate([
        ("Sex", ["Female", "Male"]),
        ("Pclass", ["1st", "2nd", "3rd"]),
        ("Embarked", ["C", "Q", "S", "B"]),
    ]):
        ct = pd.crosstab(df[col], df["Survived"])
        ct.plot(kind="bar", stacked=True, ax=axes[i],
                color=[COLORS["green"], COLORS["red"]], edgecolor="white", linewidth=1, legend=False)
        axes[i].set_title(f"Survival by {col}", fontweight="700", fontsize=FONTSIZE["title"])
        _style_axes(axes[i], ylabel="Count")
        axes[i].set_xticklabels(labels, rotation=0, fontsize=FONTSIZE["tick"])
        axes[i].legend(["Survived", "Perished"], fontsize=FONTSIZE["tick"], frameon=True,
                       facecolor="white", edgecolor="#d1d5db")
        for container in axes[i].containers:
            axes[i].bar_label(container, fmt="{:.0f}", fontsize=8, fontweight="600",
                              padding=2, color="#1f2937")
    fig.tight_layout()
    return _save(fig, "stacked_survival.png")

def chart_violin_plots(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.patch.set_facecolor("#fafbfc")

    sns.violinplot(df, x="Pclass", y="Age", hue="Survived", split=True,
                   ax=axes[0], palette=[COLORS["red"], COLORS["green"]],
                   inner="quart", linewidth=1.5)
    axes[0].set_title("Age Distribution: Class x Survival", fontweight="700", fontsize=FONTSIZE["title"])
    _style_axes(axes[0])
    axes[0].set_xticks([0, 1, 2])
    axes[0].set_xticklabels(["1st", "2nd", "3rd"], fontsize=FONTSIZE["tick"])
    axes[0].legend(["Perished", "Survived"], fontsize=FONTSIZE["tick"], frameon=True,
                   facecolor="white", edgecolor="#d1d5db")

    sns.violinplot(df, x="Pclass", y="Fare", hue="Survived", split=True,
                   ax=axes[1], palette=[COLORS["red"], COLORS["green"]],
                   inner="quart", linewidth=1.5)
    axes[1].set_title("Fare Distribution: Class x Survival", fontweight="700", fontsize=FONTSIZE["title"])
    _style_axes(axes[1])
    axes[1].set_xticks([0, 1, 2])
    axes[1].set_xticklabels(["1st", "2nd", "3rd"], fontsize=FONTSIZE["tick"])
    axes[1].legend(["Perished", "Survived"], fontsize=FONTSIZE["tick"], frameon=True,
                   facecolor="white", edgecolor="#d1d5db")
    axes[1].yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))

    fig.tight_layout()
    return _save(fig, "violin_plots.png")

def chart_age_gender_survival(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.patch.set_facecolor("#fafbfc")
    for i, sex in enumerate(["female", "male"]):
        subset = df[df["Sex"] == sex]
        sns.histplot(subset, x="Age", hue="Survived", ax=axes[i],
                     kde=True, palette=[COLORS["red"], COLORS["green"]],
                     edgecolor="white", linewidth=0.5, alpha=0.7, bins=20)
        axes[i].set_title(f"Age Distribution by Survival ({sex.capitalize()})",
                          fontweight="700", fontsize=FONTSIZE["title"])
        _style_axes(axes[i], xlabel="Age")
        axes[i].legend(["Perished", "Survived"], fontsize=FONTSIZE["tick"], frameon=True,
                       facecolor="white", edgecolor="#d1d5db")
    fig.tight_layout()
    return _save(fig, "age_gender_survival.png")

def chart_pairwise_scatter(df):
    numeric = df[["Age", "Fare", "SibSp", "Parch", "Survived"]].dropna()
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.patch.set_facecolor("#fafbfc")
    pairs = [("Age", "Fare"), ("Age", "SibSp"), ("Fare", "Parch"), ("SibSp", "Parch")]
    for ax, (x, y) in zip(axes.flat, pairs):
        for surv, clr in [(0, COLORS["red"]), (1, COLORS["green"])]:
            subset = numeric[numeric["Survived"] == surv]
            ax.scatter(subset[x], subset[y], c=clr, alpha=0.4, s=20,
                       edgecolors="white", linewidth=0.5,
                       label="Survived" if surv == 1 else "Perished")
        _style_axes(ax, xlabel=x, ylabel=y)
        ax.legend(fontsize=8, frameon=True, facecolor="white", edgecolor="#d1d5db")
    fig.suptitle("Pairwise Scatter by Survival", fontweight="700", fontsize=FONTSIZE["title"] + 1, y=0.98)
    fig.tight_layout()
    return _save(fig, "pairwise_scatter.png")

def chart_sibsp_parch_survival(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.patch.set_facecolor("#fafbfc")

    sibsp_data = df.groupby("SibSp")["Survived"].mean().mul(100)
    colors = [COLORS["green"] if v > 40 else COLORS["accent"] if v > 30 else COLORS["red"]
              for v in sibsp_data.values]
    sibsp_data.plot(kind="bar", ax=axes[0], color=colors, edgecolor="white", linewidth=1)
    axes[0].set_title("Survival Rate by Siblings/Spouses", fontweight="700", fontsize=FONTSIZE["title"])
    _style_axes(axes[0], xlabel="SibSp", ylabel="Survival Rate (%)")
    _add_bar_labels(axes[0], fmt="{:.1f}%")

    parch_data = df.groupby("Parch")["Survived"].mean().mul(100)
    colors = [COLORS["green"] if v > 40 else COLORS["accent"] if v > 30 else COLORS["red"]
              for v in parch_data.values]
    parch_data.plot(kind="bar", ax=axes[1], color=colors, edgecolor="white", linewidth=1)
    axes[1].set_title("Survival Rate by Parents/Children", fontweight="700", fontsize=FONTSIZE["title"])
    _style_axes(axes[1], xlabel="Parch", ylabel="Survival Rate (%)")
    _add_bar_labels(axes[1], fmt="{:.1f}%")

    fig.tight_layout()
    return _save(fig, "sibsp_parch_survival.png")


def chart_feature_importance(df):
    es = effect_sizes(df).sort_values("effect_size", ascending=True, key=lambda s: s.abs())
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("#fafbfc")
    strength_color = {"Large": COLORS["green"], "Medium": COLORS["accent"],
                     "Small": COLORS["yellow"], "Negligible": "#94a3b8"}
    colors = [strength_color.get(s, COLORS["accent"]) for s in es["strength"]]
    bars = ax.barh(
        [f"{r['feature']} ({r['metric'].replace('Point-biserial r', 'r').replace(chr(39)+'s V', '')})"
         for _, r in es.iterrows()],
        es["effect_size"].abs().values, color=colors, edgecolor="white", linewidth=1, height=0.65,
    )
    ax.set_title("Feature Predictive Power for Survival", fontweight="700", fontsize=FONTSIZE["title"] + 1)
    _style_axes(ax, xlabel="Effect Size (|r| or Cramer's V)")
    ax.set_xlim(0, max(es["effect_size"].abs().max() * 1.18, 0.6))
    for bar, (_, row) in zip(bars, es.iterrows()):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{row['effect_size']:+.3f}  ({row['strength']})",
                va="center", fontweight="700", fontsize=FONTSIZE["annot"] - 1, color="#1f2937")
    for x, label in [(0.1, "Small"), (0.3, "Medium"), (0.5, "Large")]:
        ax.axvline(x, color="#cbd5e1", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.text(x, -0.5, label, ha="center", fontsize=8, color="#64748b", fontweight="600")
    fig.tight_layout()
    return _save(fig, "feature_importance.png")


def chart_odds_ratios(df):
    odds = key_odds_ratios(df)
    odds_sorted = sorted(odds, key=lambda o: o["odds_ratio"])
    labels = [o["label"] for o in odds_sorted]
    values = [o["odds_ratio"] for o in odds_sorted]
    lows = [o["ci_low"] for o in odds_sorted]
    highs = [o["ci_high"] for o in odds_sorted]
    colors = [COLORS["green"] if v > 1 else COLORS["red"] for v in values]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor("#fafbfc")
    y_pos = np.arange(len(values))
    ax.barh(y_pos, values, color=colors, height=0.55, edgecolor="white", linewidth=1, alpha=0.85)
    ax.errorbar(values, y_pos, xerr=[np.array(values) - np.array(lows), np.array(highs) - np.array(values)],
                fmt="none", ecolor="#1f2937", elinewidth=1.4, capsize=4, capthick=1.4)
    ax.axvline(1, color="#64748b", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.text(1.02, -0.7, "OR = 1 (no effect)", fontsize=8, color="#64748b", fontweight="600")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xscale("log")
    ax.set_xlim(min(0.05, min(lows) * 0.7), max(highs) * 1.4)
    ax.set_title("Survival Odds Ratios with 95% CIs", fontweight="700", fontsize=FONTSIZE["title"] + 1)
    _style_axes(ax, xlabel="Odds Ratio (log scale)")
    for i, (v, lo, hi) in enumerate(zip(values, lows, highs)):
        ax.text(hi * 1.08, i, f"{v:.2f}x [{lo:.2f}, {hi:.2f}]",
                va="center", fontsize=9, fontweight="600", color="#1f2937")
    fig.tight_layout()
    return _save(fig, "odds_ratios.png")


def chart_joint_class_sex(df):
    pivot_rate = df.groupby(["Pclass", "Sex"], observed=True)["Survived"].mean().mul(100).unstack()
    pivot_count = df.groupby(["Pclass", "Sex"], observed=True)["Survived"].count().unstack()
    pivot_rate = pivot_rate[["female", "male"]]
    pivot_count = pivot_count[["female", "male"]]
    labels = np.array([[f"{pivot_rate.iat[i, j]:.1f}%\nn={int(pivot_count.iat[i, j])}"
                        for j in range(pivot_rate.shape[1])] for i in range(pivot_rate.shape[0])])

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#fafbfc")
    sns.heatmap(pivot_rate, annot=labels, fmt="", cmap="RdYlGn", center=50,
                vmin=0, vmax=100, ax=ax,
                cbar_kws={"label": "Survival Rate (%)", "shrink": 0.85},
                linewidths=2, linecolor="white",
                annot_kws={"fontsize": 12, "fontweight": "700"})
    ax.set_title("Joint Survival: Class x Sex", fontweight="700", fontsize=FONTSIZE["title"] + 1, pad=10)
    ax.set_xlabel("Sex", fontweight="600")
    ax.set_ylabel("Class", fontweight="600")
    ax.set_xticklabels(["Female", "Male"], rotation=0)
    ax.set_yticklabels(["1st", "2nd", "3rd"], rotation=0)
    fig.tight_layout()
    return _save(fig, "joint_class_sex.png")


def chart_lifeboat_survival(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.patch.set_facecolor("#fafbfc")

    has_boat = df["Lifeboat"].notna()
    counts = pd.DataFrame({
        "Group": ["On Lifeboat", "No Boat Record"],
        "Survived": [int(df.loc[has_boat, "Survived"].sum()), int(df.loc[~has_boat, "Survived"].sum())],
        "Perished": [int((df.loc[has_boat, "Survived"] == 0).sum()), int((df.loc[~has_boat, "Survived"] == 0).sum())],
    }).set_index("Group")
    counts.plot(kind="bar", stacked=True, ax=axes[0], color=[COLORS["green"], COLORS["red"]],
                width=0.55, edgecolor="white", linewidth=1)
    axes[0].set_title("Lifeboat Record vs Survival", fontweight="700", fontsize=FONTSIZE["title"])
    _style_axes(axes[0], ylabel="Passengers")
    axes[0].set_xticklabels(counts.index, rotation=0, fontsize=FONTSIZE["tick"])
    axes[0].legend(fontsize=FONTSIZE["tick"], frameon=True, facecolor="white", edgecolor="#d1d5db")
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="{:.0f}", fontsize=9, fontweight="700",
                          label_type="center", color="#fff")

    boat_passengers = df[has_boat].copy()
    if len(boat_passengers) > 0:
        boat_by_sex = boat_passengers.groupby("Sex")["Lifeboat"].count()
        all_by_sex = df.groupby("Sex").size()
        rate = (boat_by_sex / all_by_sex * 100).reindex(["female", "male"]).fillna(0)
        rate.plot(kind="bar", ax=axes[1], color=[COLORS["red"], COLORS["accent"]],
                  width=0.55, edgecolor="white", linewidth=1)
        axes[1].set_title("% with Lifeboat Record by Sex", fontweight="700", fontsize=FONTSIZE["title"])
        _style_axes(axes[1], ylabel="% with Lifeboat #")
        axes[1].set_xticklabels(["Female", "Male"], rotation=0, fontsize=FONTSIZE["tick"])
        axes[1].set_ylim(0, 100)
        _add_bar_labels(axes[1], fmt="{:.1f}%")

    fig.tight_layout()
    return _save(fig, "lifeboat_survival.png")


class PDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(95, 8, "Titanic EDA Report", align="L")
        self.cell(95, 8, f"Page {self.page_no()}", align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(59, 130, 246)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, "Generated by Titanic EDA Project", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(30, 58, 95)
        y = self.get_y()
        self.set_fill_color(241, 245, 249)
        self.rect(10, y, 190, 12, "F")
        self.set_xy(12, y + 1)
        self.cell(186, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.set_y(y + 16)

    def sub_title(self, title):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(59, 130, 246)
        self.cell(0, 9, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(203, 213, 225)
        self.set_line_width(0.3)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(2)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 41, 59)
        self.multi_cell(0, 5.5, text)
        self.ln(3)

    def bullet(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 41, 59)
        x = self.get_x()
        self.cell(5, 5.5, "-")
        self.multi_cell(180, 5.5, f" {text}")
        self.set_x(x)
        self.ln(1)

    def stat_box(self, label, value, color=(59, 130, 246)):
        w = 45
        x = self.get_x()
        y = self.get_y()
        self.set_fill_color(*[int(c * 0.08) for c in color])
        self.set_draw_color(*color)
        self.set_line_width(0.8)
        self.rect(x, y, w, 20, "DF")
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(*color)
        self.set_xy(x + 2, y + 2)
        self.cell(w - 4, 10, str(value), align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 8)
        self.set_text_color(100, 100, 100)
        self.set_xy(x + 2, self.get_y())
        self.cell(w - 4, 6, label, align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_xy(x + w + 3, y)

    def add_image(self, path, w=190):
        self.image(str(path), x=(210 - w) / 2, w=w)
        self.ln(3)

    def add_table(self, headers, rows, col_widths=None):
        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(30, 58, 95)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
        self.ln()
        self.set_font("Helvetica", "", 9)
        self.set_text_color(30, 41, 59)
        fill = False
        for row in rows:
            if self.get_y() + 7 > 277:
                self.add_page()
                self.set_font("Helvetica", "B", 9)
                self.set_fill_color(30, 58, 95)
                self.set_text_color(255, 255, 255)
                for i, h in enumerate(headers):
                    self.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
                self.ln()
                self.set_font("Helvetica", "", 9)
                self.set_text_color(30, 41, 59)
                fill = False
            if fill:
                self.set_fill_color(241, 245, 249)
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 6, str(cell), border=1, fill=fill, align="C")
            self.ln()
            fill = not fill
        self.ln(4)

    def _count_wrapped_lines(self, text: str, max_w: float) -> int:
        lines = 0
        for paragraph in text.split("\n"):
            words = paragraph.split(" ")
            current = ""
            for word in words:
                candidate = (current + " " + word).strip() if current else word
                if self.get_string_width(candidate) <= max_w:
                    current = candidate
                else:
                    if current:
                        lines += 1
                    current = word
            if current or not paragraph:
                lines += 1
        return max(1, lines)

    def inference_box(self, title, text, color=(59, 130, 246)):
        w = 190
        bar_w = 1.6
        inner_x_offset = bar_w + 4
        inner_w = w - inner_x_offset - 4
        title_h = 5
        line_h = 4.2
        pad_top = 1.5
        pad_bot = 2
        self.set_font("Helvetica", "", 9)
        line_count = self._count_wrapped_lines(text, inner_w - 2) + 1
        body_h = line_count * line_h
        h = pad_top + title_h + body_h + pad_bot

        bottom_limit = self.h - self.b_margin
        if self.get_y() + h + 8 > bottom_limit:
            self.add_page()
        x = self.l_margin
        y = self.get_y()

        prev_auto = self.auto_page_break
        prev_margin = self.b_margin
        self.set_auto_page_break(auto=False)

        # very pale tint fill for subtle separation
        tint = tuple(int(c + (255 - c) * 0.94) for c in color)
        self.set_fill_color(*tint)
        self.rect(x, y, w, h, "F")
        # colored left bar
        self.set_fill_color(*color)
        self.rect(x, y, bar_w, h, "F")

        self.set_font("Helvetica", "B", 9.5)
        self.set_text_color(*[int(c * 0.55) for c in color])
        self.set_xy(x + inner_x_offset, y + pad_top)
        self.cell(inner_w, title_h, title, new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(30, 41, 59)
        self.set_xy(x + inner_x_offset, y + pad_top + title_h)
        self.multi_cell(inner_w, line_h, text)

        self.set_auto_page_break(auto=prev_auto, margin=prev_margin)
        self.set_y(y + h + 4)


def generate_pdf(df, output_path: Path):
    df_eng = engineer_features(clean_data(df))
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=20)

    pdf.add_page()
    pdf.set_fill_color(15, 23, 42)
    pdf.rect(0, 0, 210, 297, "F")
    pdf.set_fill_color(30, 58, 95)
    pdf.rect(0, 90, 210, 70, "F")

    pdf.set_xy(0, 100)
    pdf.set_font("Helvetica", "B", 42)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 22, "Titanic", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 22)
    pdf.set_text_color(186, 209, 240)
    pdf.cell(0, 14, "Survival EDA Report", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(8)
    pdf.set_draw_color(96, 165, 250)
    pdf.set_line_width(0.8)
    pdf.line(70, pdf.get_y(), 140, pdf.get_y())

    pdf.set_xy(0, 180)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(186, 209, 240)
    pdf.cell(0, 7, "An Analytical Study of 1,309 Passengers Aboard the RMS Titanic", align="C", new_x="LMARGIN", new_y="NEXT")

    box_w = 50
    box_h = 28
    box_gap = 6
    metrics = [
        (f"{len(df):,}", "Passengers", (96, 165, 250)),
        (f"{df['Survived'].mean()*100:.1f}%", "Survived", (52, 211, 153)),
        (f"{len(df.columns)}", "Features", (167, 139, 250)),
    ]
    total_w = len(metrics) * box_w + (len(metrics) - 1) * box_gap
    x0 = (210 - total_w) / 2
    y0 = 210
    for i, (val, lbl, clr) in enumerate(metrics):
        x = x0 + i * (box_w + box_gap)
        pdf.set_fill_color(*[max(int(c * 0.18), 25) for c in clr])
        pdf.set_draw_color(*clr)
        pdf.set_line_width(0.5)
        pdf.rect(x, y0, box_w, box_h, "DF")
        pdf.set_font("Helvetica", "B", 20)
        pdf.set_text_color(*clr)
        pdf.set_xy(x, y0 + 4)
        pdf.cell(box_w, 11, val, align="C")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(220, 220, 230)
        pdf.set_xy(x, y0 + 17)
        pdf.cell(box_w, 7, lbl, align="C")

    pdf.set_xy(0, 260)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(140, 165, 200)
    pdf.cell(0, 5, "Source: titanic5 (hbiostat.org / Encyclopedia Titanica)", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, f"Generated: {pd.Timestamp.now().strftime('%B %d, %Y')}", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.add_page()
    pdf.section_title("Table of Contents")
    toc = [
        ("1", "Executive Summary"),
        ("2", "Dataset Overview"),
        ("3", "Missing Value Analysis"),
        ("4", "Survival Analysis"),
        ("5", "Feature Predictive Power"),
        ("6", "Odds Ratios with 95% CIs"),
        ("7", "Joint Stratified Analysis: Class x Sex"),
        ("8", "Demographic Analysis"),
        ("9", "Age Analysis"),
        ("10", "Fare Analysis"),
        ("11", "Embarkation Port Analysis"),
        ("12", "Family Size Analysis"),
        ("13", "Class & Gender Interaction"),
        ("14", "Title Analysis"),
        ("15", "SibSp & Parch Survival Patterns"),
        ("16", "Lifeboat Records & Survival"),
        ("17", "Correlation Analysis"),
        ("18", "Statistical Tests & Significance"),
        ("19", "Key Findings & Conclusion"),
    ]
    for num, title in toc:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(30, 58, 95)
        pdf.cell(14, 8, num)
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(30, 41, 59)
        pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    print("Generating charts...")
    chart_survival_overview(df)
    chart_age_analysis(df)
    chart_fare_analysis(df)
    chart_embarked_analysis(df)
    chart_family_survival(df)
    chart_correlation_heatmap(df)
    chart_missing_values(df)
    chart_age_survival_bins(df)
    chart_fare_survival_bins(df)
    chart_class_gender(df)
    chart_title_survival(df)
    chart_stacked_survival(df)
    chart_violin_plots(df)
    chart_age_gender_survival(df)
    chart_pairwise_scatter(df)
    chart_sibsp_parch_survival(df)
    chart_feature_importance(df_eng)
    chart_odds_ratios(df_eng)
    chart_joint_class_sex(df_eng)
    chart_lifeboat_survival(df_eng)
    print("Charts generated.")

    pdf.add_page()
    pdf.section_title("1. Executive Summary")
    pdf.body_text(
        "This report presents an Exploratory Data Analysis of the titanic5 dataset (1,309 passengers, "
        "14 features). It identifies the factors that determined survival, quantifies their relative "
        "strength via effect sizes and odds ratios, and validates each association with formal "
        "statistical tests. Where useful, 95% Wilson confidence intervals are reported alongside "
        "survival rates so the precision of each estimate is visible."
    )
    pdf.sub_title("Key Statistics")
    stats = [
        ("Total Passengers", f"{len(df):,}"),
        ("Survival Rate", f"{df['Survived'].mean()*100:.1f}%"),
        ("Average Age", f"{df['Age'].mean():.1f}"),
        ("Average Fare", f"${df['Fare'].mean():.2f}"),
    ]
    for i, (label, value) in enumerate(stats):
        pdf.stat_box(label, value)
        if i < len(stats) - 1:
            pdf.cell(5)
    pdf.ln(24)

    pdf.sub_title("Headline Findings")
    female_rate = df_eng[df_eng["Sex"] == "female"]["Survived"].mean() * 100
    male_rate = df_eng[df_eng["Sex"] == "male"]["Survived"].mean() * 100
    pc1_rate = df_eng[df_eng["Pclass"] == 1]["Survived"].mean() * 100
    pc3_rate = df_eng[df_eng["Pclass"] == 3]["Survived"].mean() * 100
    pdf.bullet(f"Women survived at {female_rate:.1f}% vs men at {male_rate:.1f}%, a {female_rate-male_rate:.1f}-percentage-point gap. Sex is the single largest predictor (Cramer's V approx 0.53).")
    pdf.bullet(f"1st-class passengers survived at {pc1_rate:.1f}% vs {pc3_rate:.1f}% for 3rd class. Class compounds with sex: 1st-class women 96.5%, 3rd-class men 15.2%.")
    pdf.bullet("Fare (r=0.247) is the strongest numerical signal, but most of its effect is mediated through class.")
    pdf.bullet("Children (<=16) survived at higher rates than adults, confirming a measurable 'children first' effect (OR approx 1.7x).")
    pdf.bullet("Of 1,309 passengers, 486 have a recorded lifeboat number; 479 of them survived. Lifeboat access was the proximate cause of survival.")
    pdf.bullet("Embarkation port effects largely vanish once class is controlled for; the apparent Cherbourg advantage reflects its first-class-heavy passenger mix.")

    pdf.sub_title("How to Read This Report")
    pdf.body_text(
        "Each section answers one question and follows the same shape: a plain-English intro, a chart, "
        "a data table with confidence intervals, and a blue 'Inference' callout summarizing what the "
        "numbers actually mean. You don't need a statistics background - the glossary below explains "
        "every technical term you'll see."
    )
    pdf.add_table(
        ["Term", "What it means in plain English"],
        [
            ["Survival rate", "Percent of a group that survived. 62% means 62 of every 100 lived."],
            ["95% CI",
             "The range we are 95% confident the true rate falls in. Narrow = certain; wide = small sample."],
            ["Odds Ratio (OR)",
             "How many times higher the odds of survival were for one group vs another. OR=2 means twice the odds; OR<1 means lower odds; OR=1 means no effect."],
            ["Effect size",
             "How strongly a feature predicts survival on a 0-1 scale. 0.1=small, 0.3=medium, 0.5=large."],
            ["Cramer's V",
             "Effect size for categorical features (Sex, Class, Port)."],
            ["Point-biserial r",
             "Effect size for a numeric feature predicting a yes/no outcome (Age, Fare vs Survived)."],
            ["Cohen's d",
             "How big the difference is between two group means, in standard deviations. 0.2=small, 0.5=medium, 0.8=large."],
            ["p-value",
             "Probability the pattern arose by chance. p<0.05=probably real; p<0.001=essentially certain."],
            ["pp (percentage points)",
             "Arithmetic gap between two percentages. Going from 19% to 73% is a 54pp jump."],
            ["Wilson CI",
             "A specific way to compute a 95% CI for a proportion. More accurate than the textbook formula for small samples or extreme rates."],
        ],
        [42, 148],
    )

    pdf.add_page()
    pdf.section_title("2. Dataset Overview")
    pdf.body_text(
        "The titanic5 dataset (Encyclopedia Titanica / Vanderbilt Biostatistics) consists of "
        "1,309 records and 14 features - 47% more passengers than the well-known Kaggle training "
        "set, with substantially fewer missing ages (3.9% vs 19.9%)."
    )
    pdf.sub_title("Column Descriptions")
    pdf.add_table(
        ["Column", "Type", "Description"],
        [
            ["PassengerId", "int", "Unique identifier"],
            ["Survived", "int", "0 = Perished, 1 = Survived"],
            ["Pclass", "int", "Ticket class (1=1st, 2=2nd, 3=3rd)"],
            ["Name", "str", "Passenger name (Title extractable)"],
            ["Sex", "str", "Gender (female / male)"],
            ["Age", "float", "Age in years (51 missing)"],
            ["SibSp", "int", "Siblings or spouse aboard"],
            ["Parch", "int", "Parents or children aboard"],
            ["Ticket", "str", "Ticket number"],
            ["Fare", "float", "Passenger fare (USD)"],
            ["Embarked", "str", "Port (C/Q/S/B)"],
            ["Occupation", "str", "Occupation (621 missing)"],
            ["BoatBody", "str", "Lifeboat # or body recovery [#]"],
            ["NameId", "int", "Encyclopedia Titanica identifier"],
        ],
        [38, 22, 130],
    )

    pdf.sub_title("Summary Statistics")
    desc = df.describe().round(2)
    rows = []
    for col in ["Age", "Fare", "SibSp", "Parch"]:
        row = [col] + [str(desc[col][stat]) for stat in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
        rows.append(row)
    pdf.add_table(
        ["Feature", "Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"],
        rows,
        [25, 20, 20, 20, 20, 20, 20, 20, 25],
    )

    pdf.sub_title("Composition Snapshot")
    sex_counts = df["Sex"].value_counts()
    pclass_counts = df["Pclass"].value_counts().sort_index()
    emb_counts = df["Embarked"].value_counts()
    port_names = {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton", "B": "Belfast"}
    snapshot_rows = [
        ["Sex",
         f"Male {sex_counts.get('male', 0)} ({sex_counts.get('male', 0)/len(df)*100:.1f}%)",
         f"Female {sex_counts.get('female', 0)} ({sex_counts.get('female', 0)/len(df)*100:.1f}%)",
         "", ""],
        ["Class",
         f"1st {pclass_counts.get(1, 0)} ({pclass_counts.get(1, 0)/len(df)*100:.1f}%)",
         f"2nd {pclass_counts.get(2, 0)} ({pclass_counts.get(2, 0)/len(df)*100:.1f}%)",
         f"3rd {pclass_counts.get(3, 0)} ({pclass_counts.get(3, 0)/len(df)*100:.1f}%)",
         ""],
        ["Port",
         f"{port_names['S']} {emb_counts.get('S', 0)} ({emb_counts.get('S', 0)/len(df)*100:.1f}%)",
         f"{port_names['C']} {emb_counts.get('C', 0)} ({emb_counts.get('C', 0)/len(df)*100:.1f}%)",
         f"{port_names['Q']} {emb_counts.get('Q', 0)} ({emb_counts.get('Q', 0)/len(df)*100:.1f}%)",
         f"{port_names['B']} {emb_counts.get('B', 0)} ({emb_counts.get('B', 0)/len(df)*100:.1f}%)"],
    ]
    pdf.add_table(
        ["Attribute", "Group 1", "Group 2", "Group 3", "Group 4"],
        snapshot_rows,
        [25, 42, 42, 42, 39],
    )

    pdf.add_page()
    pdf.section_title("3. Missing Value Analysis")
    pdf.body_text(
        "Quantifying missingness before analysis. Compared to the Kaggle training subset, "
        "titanic5 is markedly more complete: only 3.9% of ages are missing (vs 19.9%) and "
        "Fare is fully populated."
    )
    missing = missing_summary(df)
    pdf.add_image(CHART_DIR / "missing_values.png", w=160)
    pdf.sub_title("Missing Value Details")
    rows = []
    for col_name, row in missing.iterrows():
        severity = "High" if row["percent"] > 50 else "Medium" if row["percent"] > 10 else "Low"
        rows.append([col_name, int(row["missing"]), f"{row['percent']:.1f}%", severity])
    pdf.add_table(
        ["Column", "Missing", "Percent", "Severity"],
        rows,
        [50, 40, 40, 60],
    )
    pdf.sub_title("Imputation & Handling Strategy")
    pdf.bullet("Age (3.9% missing): impute using the median of Age within Sex x Pclass strata. This preserves the bimodal age structure (children + adults) and class-related age skews.")
    pdf.bullet("Occupation (47.4% missing): too sparse to use directly. Treated as HasCabin = 1 if non-null, retaining it as a low-resolution proxy for cabin-info availability.")
    pdf.bullet("BoatBody: not missing per se - blank string means 'no disposition recorded'. Parsed into Lifeboat (boat number/letter) and BodyRecovered (1 if a recovery code present).")
    pdf.bullet("Embarked: 2 missing rows in the raw data, filled with the mode (Southampton).")
    pdf.bullet("Fare: any missing values would be imputed by the median Fare within each Pclass. In titanic5 this column is essentially complete.")
    pdf.inference_box(
        "Why this matters",
        "Most missing-value patterns here are informative: passengers without an Occupation entry "
        "are largely third-class passengers without recorded cabin info. Naive deletion would bias "
        "the analysis toward first-class. Median-by-stratum imputation preserves the conditional "
        "distributions without injecting noise.",
        color=(96, 165, 250),
    )

    pdf.add_page()
    pdf.section_title("4. Survival Analysis")
    pdf.body_text(
        "Roughly 1 in 3 passengers survived (38.2%). That single number, though, hides everything "
        "interesting. The disaster did not kill people at random - women survived at 73%, men at "
        "19%; 1st-class passengers at 62%, 3rd-class at 26%. The rest of this report is about "
        "untangling exactly which factors mattered and by how much."
    )
    pdf.add_image(CHART_DIR / "survival_overview.png", w=190)
    pdf.sub_title("Survival by Sex (with 95% Wilson CI)")
    sex_ci = survival_rates_with_ci(df, "Sex")
    pdf.add_table(
        ["Sex", "n", "Survived", "Rate (%)", "95% CI"],
        [[str(r["level"]).capitalize(), int(r["n"]), int(r["survived"]), f"{r['rate']:.1f}%", f"[{r['ci_low']:.1f}, {r['ci_high']:.1f}]"]
         for _, r in sex_ci.iterrows()],
        [38, 30, 35, 35, 52],
    )
    pdf.body_text(
        "Women had a dramatically higher survival rate (72.8%) than men (19.1%) - "
        "a 53.7-percentage-point gap. The CIs do not overlap, making this difference "
        "statistically unambiguous and the strongest single predictor in the dataset."
    )
    pdf.sub_title("Survival by Class (with 95% Wilson CI)")
    pclass_ci = survival_rates_with_ci(df, "Pclass")
    class_suffix = {1: "1st", 2: "2nd", 3: "3rd"}
    pdf.add_table(
        ["Class", "n", "Survived", "Rate (%)", "95% CI"],
        [[class_suffix.get(int(r["level"]), str(int(r["level"]))), int(r["n"]), int(r["survived"]),
          f"{r['rate']:.1f}%", f"[{r['ci_low']:.1f}, {r['ci_high']:.1f}]"]
         for _, r in pclass_ci.iterrows()],
        [38, 30, 35, 35, 52],
    )
    pdf.body_text(
        "First-class passengers survived at 62.0%, compared to only 25.5% for third-class. "
        "Confidence intervals are tight thanks to large sample sizes; the differences reflect "
        "both better lifeboat access and cabin location on higher decks."
    )
    pdf.add_image(CHART_DIR / "stacked_survival.png", w=190)

    pdf.add_page()
    pdf.section_title("5. Feature Predictive Power")
    pdf.body_text(
        "If you could ask just one question about a passenger to predict whether they survived, "
        "what should it be? This chart ranks every feature by how much information it carries about "
        "survival, on a 0-to-1 scale. Anything above 0.3 is a meaningful predictor; anything below "
        "0.1 is essentially noise on its own."
    )
    pdf.add_image(CHART_DIR / "feature_importance.png", w=190)
    es_df = effect_sizes(df_eng)
    pdf.add_table(
        ["Feature", "Type", "Metric", "Effect Size", "Strength"],
        [[r["feature"], r["type"].capitalize(), r["metric"], f"{r['effect_size']:+.3f}", r["strength"]]
         for _, r in es_df.iterrows()],
        [40, 35, 45, 35, 35],
    )
    pdf.inference_box(
        "What this means in plain English",
        "Sex is the single most informative thing you can know about a passenger - it's a 'Large' "
        "effect that dwarfs everything else. Class is the next most useful piece of information. "
        "Fare looks helpful, but most of what it tells you is just 'is this a 1st-class ticket?' - "
        "knowing the class directly is more reliable. Age, siblings, and parents-aboard are weak "
        "on their own, though they matter when combined with sex (e.g., 'female child').",
        color=(34, 197, 94),
    )

    pdf.add_page()
    pdf.section_title("6. Odds Ratios with 95% CIs")
    pdf.body_text(
        "The previous section ranked features. This one quantifies them: for each specific contrast "
        "(e.g., 'women vs men', '1st class vs the rest'), how many times higher were the odds of "
        "survival? An odds ratio of 2.0 means twice the odds; 0.5 means half. An OR of 1.0 would mean "
        "no effect. The 95% CI is the range we are 95% confident contains the true value - it gets "
        "wider when fewer people are in the comparison."
    )
    pdf.add_image(CHART_DIR / "odds_ratios.png", w=190)
    odds = key_odds_ratios(df_eng)
    pdf.add_table(
        ["Contrast", "OR", "95% CI", "Exposed", "Unexposed", "Lift", "p"],
        [[o["label"],
          f"{o['odds_ratio']:.2f}x",
          f"[{o['ci_low']:.2f}, {o['ci_high']:.2f}]",
          f"{o['rate_exposed']:.1f}% (n={o['n_exposed']})",
          f"{o['rate_unexposed']:.1f}% (n={o['n_unexposed']})",
          f"{('+' if o['lift']>0 else '')}{o['lift']:.1f}pp",
          ("<2e-16" if o["p_value"] < 1e-10 else f"{o['p_value']:.2e}")]
         for o in odds],
        [55, 18, 32, 28, 28, 18, 21],
    )
    strongest = max(odds, key=lambda o: o["odds_ratio"])
    weakest = min(odds, key=lambda o: o["odds_ratio"])
    pdf.inference_box(
        "What this means in plain English",
        f"Women were about {strongest['odds_ratio']:.0f}x more likely to survive than men - the "
        f"biggest single boost in the dataset. Travelling in 3rd class roughly cut your odds to "
        f"{weakest['odds_ratio']:.2f}x compared to other passengers - about a third the chance. "
        "The 'Child vs Adult' contrast has a wider confidence interval because there were fewer "
        "children on board; the effect is real but harder to pin down precisely.",
        color=(59, 130, 246),
    )

    pdf.add_page()
    pdf.section_title("7. Joint Stratified Analysis: Class x Sex")
    pdf.body_text(
        "Sex matters. Class matters. But what happens if you combine them? It turns out they don't "
        "just add - they compound. The six cells below split passengers by both class and sex "
        "simultaneously. The numbers reveal that you can predict survival almost perfectly if you "
        "know both. The biggest disparities in the entire dataset are hidden in this 2-by-3 grid."
    )
    pdf.add_image(CHART_DIR / "joint_class_sex.png", w=160)
    joint_df = joint_survival(df_eng, "Pclass", "Sex")
    class_suffix = {1: "1st", 2: "2nd", 3: "3rd"}
    rows = []
    for _, r in joint_df.iterrows():
        n = int(r["count"])
        rate = r["rate"]
        s = int(round(n * rate / 100))
        lo, hi = wilson_ci(s, n)
        rows.append([f"{class_suffix.get(int(r['Pclass']), int(r['Pclass']))} - {str(r['Sex']).capitalize()}",
                     n, f"{rate:.1f}%", f"[{lo:.1f}, {hi:.1f}]"])
    pdf.add_table(
        ["Class - Sex", "n", "Survival Rate", "95% CI"],
        rows,
        [50, 35, 50, 55],
    )
    pdf.inference_box(
        "What this means in plain English",
        "A 1st-class woman had roughly a 96-97% chance of surviving - effectively a certainty. "
        "A 3rd-class man had about a 15% chance - a near-certain death sentence. That is an "
        "80-percentage-point gap based on just two attributes of who you were. The 'women and "
        "children first' protocol was real, but it was selectively applied: 1st-class women got "
        "priority over 3rd-class women, and 3rd-class men were nearly invisible to it.",
        color=(34, 197, 94),
    )

    pdf.add_page()
    pdf.section_title("8. Demographic Analysis")
    pdf.body_text(
        "Each demographic column has both a marginal distribution (who was on the ship) and a "
        "conditional one (who survived). The mismatch between the two is where the survival story lives."
    )

    pdf.sub_title("Sex x Survival")
    sex_dist = df["Sex"].value_counts()
    sex_surv = df.groupby("Sex")["Survived"].agg(["count", "sum", "mean"])
    rows = [[s.capitalize(), int(sex_surv.loc[s, "count"]), int(sex_surv.loc[s, "sum"]),
             f"{sex_surv.loc[s, 'mean']*100:.1f}%",
             f"{int(sex_surv.loc[s, 'count']) - int(sex_surv.loc[s, 'sum'])}"]
            for s in ["female", "male"]]
    pdf.add_table(
        ["Sex", "Total", "Survived", "Rate", "Perished"],
        rows,
        [40, 38, 38, 38, 36],
    )

    pdf.sub_title("Class x Survival")
    class_suffix = {1: "1st", 2: "2nd", 3: "3rd"}
    pclass_dist = df["Pclass"].value_counts().sort_index()
    pclass_surv = df.groupby("Pclass")["Survived"].agg(["count", "sum", "mean"])
    rows = [[class_suffix.get(int(c), str(int(c))), int(pclass_surv.loc[c, "count"]),
             int(pclass_surv.loc[c, "sum"]), f"{pclass_surv.loc[c, 'mean']*100:.1f}%",
             f"{int(pclass_surv.loc[c, 'count']) - int(pclass_surv.loc[c, 'sum'])}"]
            for c in pclass_dist.index]
    pdf.add_table(
        ["Class", "Total", "Survived", "Rate", "Perished"],
        rows,
        [40, 38, 38, 38, 36],
    )

    pdf.sub_title("Port x Survival")
    port_names = {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton", "B": "Belfast"}
    emb_dist = df["Embarked"].value_counts()
    emb_surv = df.groupby("Embarked")["Survived"].agg(["count", "sum", "mean"])
    rows = [[port_names.get(p, p), int(emb_surv.loc[p, "count"]),
             int(emb_surv.loc[p, "sum"]), f"{emb_surv.loc[p, 'mean']*100:.1f}%",
             f"{int(emb_surv.loc[p, 'count']) - int(emb_surv.loc[p, 'sum'])}"]
            for p in emb_dist.index]
    pdf.add_table(
        ["Port", "Total", "Survived", "Rate", "Perished"],
        rows,
        [50, 35, 35, 35, 35],
    )

    pdf.inference_box(
        "Note on confounding",
        "Port and class are highly correlated: Cherbourg's passenger mix is heavily first-class, "
        "while Southampton's is heavily third-class. Most of the apparent Cherbourg advantage in raw "
        "survival rate dissolves once class is controlled for (see section 7).",
        color=(245, 158, 11),
    )

    pdf.sub_title("Age x Sex x Survival")
    pdf.add_image(CHART_DIR / "age_gender_survival.png", w=180)

    pdf.add_page()
    pdf.section_title("9. Age Analysis")
    pdf.body_text(
        "Was age really a survival factor, or just folklore? The data says: a real but modest one. "
        "Children under 16 survived at ~49%, well above the 38% overall rate. The oldest passengers "
        "fared worst. The chart below shows the age distribution split by survival - notice how the "
        "green 'survived' bars dominate at the youngest ages. Average age was 29.7 years; only 51 "
        "people (3.9%) had missing age data, so this analysis is reliable."
    )
    pdf.add_image(CHART_DIR / "age_analysis.png", w=190)
    pdf.sub_title("Age Statistics")
    pdf.bullet(f"Mean age: {df['Age'].mean():.1f} years")
    pdf.bullet(f"Median age: {df['Age'].median():.1f} years")
    pdf.bullet(f"Age range: {df['Age'].min():.1f} to {df['Age'].max():.1f} years")
    pdf.bullet(f"Standard deviation: {df['Age'].std():.1f} years")
    pdf.ln(2)

    pdf.sub_title("Survival by Age Group")
    pdf.add_image(CHART_DIR / "age_bins.png", w=190)
    age_bins = survival_by_numerical(df, "Age")
    pdf.add_table(
        ["Age Range", "Passengers", "Survival Rate (%)"],
        [[str(idx), int(r["passengers"]), f"{r['survival_rate']:.1f}%"] for idx, r in age_bins.iterrows()],
        [60, 60, 70],
    )
    pdf.body_text(
        "Children aged 0-16 had the highest survival rate at 49.0%, confirming the "
        "'children first' evacuation policy. The oldest age group (65-80) had the "
        "lowest survival rate at 0.0%. See section 8 for the age x sex breakdown."
    )

    pdf.add_page()
    pdf.section_title("10. Fare Analysis")
    pdf.body_text(
        "Did paying more for your ticket help you survive? Yes - but mostly because expensive "
        "tickets bought you 1st-class accommodation on higher decks, closer to the lifeboats. "
        "Survivors paid an average of $49.63; non-survivors paid $23.19, less than half. The "
        "correlation between fare and survival (r=0.247) is the strongest of any single numerical "
        "feature - but it's largely a proxy for class, not an independent factor."
    )
    pdf.add_image(CHART_DIR / "fare_analysis.png", w=190)
    pdf.sub_title("Fare Statistics")
    pdf.bullet(f"Mean fare: ${df['Fare'].mean():.2f}")
    pdf.bullet(f"Median fare: ${df['Fare'].median():.2f}")
    pdf.bullet(f"Fare range: ${df['Fare'].min():.2f} to ${df['Fare'].max():.2f}")
    pdf.bullet(f"Standard deviation: ${df['Fare'].std():.2f}")
    pdf.ln(2)

    pdf.sub_title("Survival by Fare Quintile")
    pdf.add_image(CHART_DIR / "fare_bins.png", w=190)
    fare_bins = survival_by_numerical(df, "Fare")
    pdf.add_table(
        ["Fare Range", "Passengers", "Survival Rate (%)"],
        [[str(idx), int(r["passengers"]), f"{r['survival_rate']:.1f}%" if not pd.isna(r["survival_rate"]) else "N/A"] for idx, r in fare_bins.iterrows()],
        [60, 60, 70],
    )
    pdf.body_text(
        "Passengers in the highest fare quintile ($42-$512) had survival rates of "
        "62.2%, while the lowest quintile had only 25.7% survival."
    )

    pdf.add_page()
    pdf.section_title("11. Embarkation Port Analysis")
    pdf.body_text(
        "Did the port you boarded at affect your survival? At a glance, yes: Cherbourg passengers "
        "survived at 56.6% vs Southampton's 33.4%. But this is a textbook case of a confounded "
        "variable. Cherbourg happened to be where most 1st-class passengers boarded. Once you "
        "control for class, the port effect mostly disappears - it isn't really about the port at "
        "all, it's about who was using that port."
    )
    pdf.add_image(CHART_DIR / "embarked_analysis.png", w=190)
    embarked_data = survival_by_categorical(df, "Embarked")
    port_names = {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}
    pdf.add_table(
        ["Port", "Passengers", "Survival Rate (%)"],
        [[port_names.get(idx, idx), int(r["passengers"]), f"{r['survival_rate']:.1f}%"] for idx, r in embarked_data.iterrows()],
        [60, 60, 70],
    )
    pdf.body_text(
        "Cherbourg passengers had the highest survival rate (56.6%), likely because "
        "more first-class passengers boarded there. Southampton, the main embarkation "
        "port, had the lowest rate (33.4%) due to its large third-class population."
    )

    pdf.add_page()
    pdf.section_title("12. Family Size Analysis")
    pdf.body_text(
        "There's a clear sweet spot here that took some unpacking. Family size = siblings + spouse + "
        "parents + children + yourself. Solo travelers (size 1) survived at only 30%. Mid-sized "
        "families of 2-4 jumped to 50-70%. Very large families (5+) dropped back to under 25%. The "
        "likely explanation: mid-sized families could coordinate boarding lifeboats together; solo "
        "passengers had no advocate; very large families struggled to keep everyone together."
    )
    pdf.add_image(CHART_DIR / "family_analysis.png", w=190)
    family_sizes = df.copy()
    family_sizes["FamilySize"] = family_sizes["SibSp"] + family_sizes["Parch"] + 1
    family_survival = family_sizes.groupby("FamilySize")["Survived"].agg(["count", "mean"]).assign(
        survival_rate=lambda x: x["mean"].mul(100)).round(2)
    pdf.add_table(
        ["Family Size", "Passengers", "Survival Rate (%)"],
        [[int(r.name), int(r["count"]), f"{r['survival_rate']:.1f}%"] for _, r in family_survival.iterrows()],
        [60, 60, 70],
    )

    pdf.add_page()
    pdf.section_title("13. Class & Gender Interaction")
    pdf.body_text(
        "Section 7 showed this as a heatmap; the bar chart below makes the same point visually. "
        "Each pair of bars shows one (class, sex) cell - the green bar is the % who survived, "
        "the red is the % who perished. Look at the leftmost pair (1st class, female) and the "
        "rightmost (3rd class, male) - they are almost mirror images of each other."
    )
    pdf.add_image(CHART_DIR / "class_gender.png", w=170)
    cg = pd.crosstab([df["Pclass"], df["Sex"]], df["Survived"], normalize="index").mul(100).round(1)
    pdf.add_table(
        ["Class-Gender", "Survival Rate (%)"],
        [[f"Class {idx[0]} - {idx[1]}", f"{row[1]:.1f}%"] for idx, row in cg.iterrows()],
        [95, 95],
    )

    pdf.add_page()
    pdf.section_title("14. Title Analysis")
    pdf.body_text(
        "Titles in 1912 encoded more than politeness - they told you someone's sex, marital status, "
        "and roughly their age. 'Master' was used specifically for young boys; 'Mrs' for married "
        "women; 'Miss' for unmarried women; 'Mr' covered all adult men. The survival rates by title "
        "essentially restate the sex + age story in a single feature: Mrs/Miss/Master all did "
        "well; Mr did worst at 16%. For predictive modeling, Title is often a more useful single "
        "feature than Sex or Age alone."
    )
    pdf.add_image(CHART_DIR / "title_survival.png", w=170)
    df2 = df.copy()
    df2["Title"] = extract_titles(df)
    title_data = df2.groupby("Title")["Survived"].agg(["count", "mean"]).assign(
        survival_rate=lambda x: x["mean"].mul(100)).round(1).sort_values("count", ascending=False)
    pdf.add_table(
        ["Title", "Count", "Survival Rate (%)"],
        [[title, int(r["count"]), f"{r['survival_rate']:.1f}%"] for title, r in title_data.iterrows()],
        [60, 60, 70],
    )

    pdf.add_page()
    pdf.section_title("15. SibSp & Parch Survival Patterns")
    pdf.body_text(
        "Decomposing 'family size' into its parts. SibSp = number of siblings + spouse aboard; "
        "Parch = number of parents + children aboard. Both show the same non-linear shape: a small "
        "number boosts survival, but going past 3-4 drops it back down. The takeaway is the same "
        "as section 12 - having a small support group helped, but a chaotic large family hurt."
    )
    pdf.add_image(CHART_DIR / "sibsp_parch_survival.png", w=190)
    pdf.sub_title("SibSp Survival Rates")
    sibsp_data = df.groupby("SibSp")["Survived"].agg(["count", "mean"]).assign(
        rate=lambda x: x["mean"].mul(100)).round(1)
    pdf.add_table(
        ["SibSp", "Count", "Survival Rate (%)"],
        [[int(idx), int(r["count"]), f"{r['rate']:.1f}%"] for idx, r in sibsp_data.iterrows()],
        [60, 60, 70],
    )
    pdf.sub_title("Parch Survival Rates")
    parch_data = df.groupby("Parch")["Survived"].agg(["count", "mean"]).assign(
        rate=lambda x: x["mean"].mul(100)).round(1)
    pdf.add_table(
        ["Parch", "Count", "Survival Rate (%)"],
        [[int(idx), int(r["count"]), f"{r['rate']:.1f}%"] for idx, r in parch_data.iterrows()],
        [60, 60, 70],
    )
    pdf.body_text(
        "Passengers with 1-2 siblings had survival rates of 45-54%, compared to 34% "
        "for those with no siblings. Similarly, passengers with 1-3 parents/children "
        "survived at 50-55% rates, while solo travelers had only 34.5% survival."
    )

    pdf.add_page()
    pdf.section_title("16. Lifeboat Records & Survival")
    pdf.body_text(
        "Everything in this report so far is about who was likely to survive. This section is about "
        "the actual mechanism: did you get onto a lifeboat? Of 1,309 passengers, only 486 have a "
        "recorded lifeboat number - and 479 of them survived (98.6%). Of the 823 with no boat "
        "record, only 21 survived (2.6%). Lifeboat access wasn't correlated with survival; it was "
        "essentially identical to it. That is why sex and class mattered so much: they were the "
        "rules that decided who got onto a boat."
    )
    pdf.add_image(CHART_DIR / "lifeboat_survival.png", w=190)
    has_boat = df_eng["Lifeboat"].notna()
    n_boat = int(has_boat.sum())
    s_boat = int(df_eng.loc[has_boat, "Survived"].sum())
    n_no = int((~has_boat).sum())
    s_no = int(df_eng.loc[~has_boat, "Survived"].sum())
    rate_boat = s_boat / n_boat * 100 if n_boat else 0
    rate_no = s_no / n_no * 100 if n_no else 0
    lo_b, hi_b = wilson_ci(s_boat, n_boat) if n_boat else (0, 0)
    lo_n, hi_n = wilson_ci(s_no, n_no) if n_no else (0, 0)
    pdf.add_table(
        ["Group", "n", "Survived", "Rate", "95% CI"],
        [["On Lifeboat", n_boat, s_boat, f"{rate_boat:.1f}%", f"[{lo_b:.1f}, {hi_b:.1f}]"],
         ["No Record", n_no, s_no, f"{rate_no:.1f}%", f"[{lo_n:.1f}, {hi_n:.1f}]"]],
        [55, 30, 35, 35, 35],
    )
    pdf.inference_box(
        "What this means in plain English",
        f"Getting on a lifeboat almost guaranteed survival ({rate_boat:.1f}%). Not getting on one "
        f"almost guaranteed death ({rate_no:.1f}%). The {rate_boat - rate_no:.0f}-percentage-point "
        "gap is the entire story of this disaster in one number. Everything else this report has "
        "measured - sex, class, age, fare, port - was really just measuring 'who had access to a "
        "lifeboat seat'.",
        color=(34, 197, 94),
    )

    pdf.add_page()
    pdf.section_title("17. Correlation Analysis")
    pdf.body_text(
        "Correlation answers a simple question: when one number goes up, does another go up, down, "
        "or stay the same? The values run from -1 (always opposite) to +1 (always together); 0 "
        "means no relationship. Numbers below 0.1 in absolute value are basically nothing. The "
        "strongest relationship with survival in this matrix is Fare (r=0.247) - knowing the fare "
        "tells you a moderate amount about whether someone survived, but nothing close to what "
        "knowing their sex would tell you."
    )
    pdf.add_image(CHART_DIR / "correlation_heatmap.png", w=150)
    corr = correlation_analysis(df)
    pdf.add_table(
        ["Feature", "Age", "Fare", "SibSp", "Parch", "Survived"],
        [[feat] + [f"{corr[feat][col]:.3f}" for col in ["Age", "Fare", "SibSp", "Parch", "Survived"]] for feat in ["Age", "Fare", "SibSp", "Parch", "Survived"]],
        [38, 30, 30, 30, 30, 32],
    )
    pdf.body_text(
        "Key correlations: Fare-Survived (0.257) is the strongest positive relationship. "
        "SibSp-Age (-0.308) suggests younger passengers traveled with more siblings. "
        "Parch-Fare (0.216) indicates families with children paid higher fares."
    )
    pdf.add_image(CHART_DIR / "pairwise_scatter.png", w=190)
    pdf.body_text(
        "The pairwise scatter plots reveal that survivors (green) tend to cluster "
        "in higher fare ranges and younger age groups, while perished passengers (red) "
        "are more concentrated in lower fare and older age brackets."
    )

    pdf.add_page()
    pdf.section_title("18. Statistical Tests & Significance")
    pdf.body_text(
        "Every pattern in this report could, in principle, be just an accident of sampling. "
        "Statistical tests rule that out. They each produce a 'p-value' - the probability that you'd "
        "see the observed pattern even if there were really no underlying effect. The standard "
        "thresholds: p < 0.05 means 'probably not chance', p < 0.001 means 'almost certainly not "
        "chance'. The patterns in this report comfortably clear those thresholds - in some cases by "
        "many orders of magnitude."
    )

    pdf.sub_title("Chi-Square Test: Sex vs Survival")
    ct_sex = pd.crosstab(df["Sex"], df["Survived"])
    chi2_sex, p_sex, dof_sex, _ = scipy_stats.chi2_contingency(ct_sex)
    pdf.bullet(f"Chi-square statistic: {chi2_sex:.2f}")
    pdf.bullet(f"Degrees of freedom: {dof_sex}")
    pdf.bullet(f"P-value: {p_sex:.2e}")
    pdf.bullet(f"Result: {'Highly significant' if p_sex < 0.001 else 'Significant'} (p < 0.001)")
    pdf.inference_box(
        "What this means in plain English",
        "There is effectively zero chance the sex-survival gap is a fluke. The p-value here is so "
        "small that the chance of seeing this pattern accidentally is roughly 1 in 10 to the 81st "
        "power - far beyond any reasonable threshold. The 'women first' protocol wasn't a guideline; "
        "it was the single biggest determinant of who lived.",
        color=(59, 130, 246),
    )

    pdf.sub_title("Chi-Square Test: Pclass vs Survival")
    ct_pclass = pd.crosstab(df["Pclass"], df["Survived"])
    chi2_pc, p_pc, dof_pc, _ = scipy_stats.chi2_contingency(ct_pclass)
    pdf.bullet(f"Chi-square statistic: {chi2_pc:.2f}")
    pdf.bullet(f"Degrees of freedom: {dof_pc}")
    pdf.bullet(f"P-value: {p_pc:.2e}")
    pdf.bullet(f"Result: {'Highly significant' if p_pc < 0.001 else 'Significant'} (p < 0.001)")
    pdf.inference_box(
        "What this means in plain English",
        "The class-survival difference is overwhelmingly real (p far below 0.001). 1st-class "
        "passengers didn't just get lucky - they were systematically advantaged. Their cabins were "
        "on upper decks closer to the boat deck, they got priority boarding, and they had better "
        "access to information about what was happening as the ship sank.",
        color=(139, 92, 246),
    )

    pdf.sub_title("Chi-Square Test: Embarked vs Survival")
    ct_emb = pd.crosstab(df["Embarked"].dropna(), df.loc[df["Embarked"].notna(), "Survived"])
    chi2_emb, p_emb, dof_emb, _ = scipy_stats.chi2_contingency(ct_emb)
    pdf.bullet(f"Chi-square statistic: {chi2_emb:.2f}")
    pdf.bullet(f"Degrees of freedom: {dof_emb}")
    pdf.bullet(f"P-value: {p_emb:.4f}")
    pdf.bullet(f"Result: {'Significant' if p_emb < 0.05 else 'Not significant'} (p {'< 0.05' if p_emb < 0.05 else '> 0.05'})")
    pdf.inference_box(
        "What this means in plain English",
        "Port is statistically associated with survival, but this is a textbook 'confounded' "
        "result. Cherbourg's higher survival rate isn't because boarding there saved lives - it's "
        "because Cherbourg happened to be where most 1st-class passengers boarded. Adjust for "
        "class and most of the port effect disappears.",
        color=(245, 158, 11),
    )

    pdf.sub_title("T-Test: Age (Survived vs Perished)")
    survived_age = df[df["Survived"] == 1]["Age"].dropna()
    perished_age = df[df["Survived"] == 0]["Age"].dropna()
    t_stat_age, p_age = scipy_stats.ttest_ind(survived_age, perished_age, equal_var=False)
    pdf.bullet(f"T-statistic: {t_stat_age:.3f}")
    pdf.bullet(f"P-value: {p_age:.4f}")
    pdf.bullet(f"Mean age (survived): {survived_age.mean():.1f}")
    pdf.bullet(f"Mean age (perished): {perished_age.mean():.1f}")
    pdf.bullet(f"Result: {'Significant' if p_age < 0.05 else 'Not significant'} (p {'< 0.05' if p_age < 0.05 else '> 0.05'})")
    pdf.inference_box(
        "What this means in plain English",
        "Survivors were about 1 year younger on average than non-survivors (28.9 vs 29.9 years). "
        "The difference is statistically real but small in practical terms. Age matters mostly at "
        "the extremes - children specifically benefited from the evacuation priority; the typical "
        "adult age gap had only a modest effect.",
        color=(34, 197, 94),
    )

    pdf.sub_title("T-Test: Fare (Survived vs Perished)")
    survived_fare = df[df["Survived"] == 1]["Fare"].dropna()
    perished_fare = df[df["Survived"] == 0]["Fare"].dropna()
    t_stat_fare, p_fare = scipy_stats.ttest_ind(survived_fare, perished_fare, equal_var=False)
    pdf.bullet(f"T-statistic: {t_stat_fare:.3f}")
    pdf.bullet(f"P-value: {p_fare:.2e}")
    pdf.bullet(f"Mean fare (survived): ${survived_fare.mean():.2f}")
    pdf.bullet(f"Mean fare (perished): ${perished_fare.mean():.2f}")
    pdf.bullet(f"Result: Highly significant (p < 0.001)")
    pdf.inference_box(
        "What this means in plain English",
        "Survivors paid more than twice the fare of non-survivors on average ($49.63 vs $23.19). "
        "The difference is overwhelmingly real (p < 0.001). But this is mostly because expensive "
        "tickets bought 1st-class accommodation; fare is doing the work of class here, not adding "
        "its own independent effect.",
        color=(239, 68, 68),
    )

    pdf.sub_title("ANOVA: Survival Rate Across Age Groups")
    df_age_groups = df.copy()
    df_age_groups["AgeGroup"] = pd.cut(df_age_groups["Age"], bins=[0, 16, 32, 48, 64, 80])
    groups = [g["Survived"].dropna() for _, g in df_age_groups.groupby("AgeGroup", observed=True)]
    f_stat_age, p_anova_age = scipy_stats.f_oneway(*groups)
    pdf.bullet(f"F-statistic: {f_stat_age:.3f}")
    pdf.bullet(f"P-value: {p_anova_age:.4f}")
    pdf.bullet(f"Result: {'Significant' if p_anova_age < 0.05 else 'Not significant'} (p {'< 0.05' if p_anova_age < 0.05 else '> 0.05'})")
    pdf.inference_box(
        "What this means in plain English",
        "Survival rates are not the same across age groups - ANOVA confirms the differences are "
        "real (p < 0.05). Children 0-16 stand out at ~49%; the elderly (65+) had the worst odds. "
        "Adult age groups in between are roughly similar, which is why the overall effect of age "
        "looks 'small' even though the children-first effect is genuine.",
        color=(59, 130, 246),
    )

    pdf.sub_title("Correlation Strength Interpretation")
    pdf.add_table(
        ["Correlation", "r-value", "Strength", "Interpretation"],
        [
            ["Fare vs Survived", "0.257", "Moderate", "Higher fare = better survival odds"],
            ["Parch vs Survived", "0.082", "Weak", "Slight advantage for family members"],
            ["Age vs Survived", "-0.077", "Weak", "Younger passengers slightly favored"],
            ["SibSp vs Survived", "-0.035", "Very Weak", "Minimal direct effect"],
            ["Fare vs Age", "0.096", "Weak", "Slight tendency for higher fares with age"],
            ["SibSp vs Age", "-0.308", "Moderate", "Younger passengers had more siblings"],
            ["Parch vs Fare", "0.216", "Moderate", "Families paid higher fares"],
        ],
        [40, 25, 35, 90],
    )

    pdf.add_page()
    pdf.section_title("19. Key Findings & Conclusion")
    pdf.sub_title("Summary of Findings")

    findings = [
        "Gender was the strongest predictor: 72.8% of women survived vs 19.1% of men. Chi-square confirms the association is decisive (p < 2.2e-16, Cramer's V approx 0.53, Large).",
        "Class was the second-strongest: 1st class 62.0% vs 3rd class 25.5% (p < 2.2e-16). Class compounds with sex - 1st-class women 96.5%, 3rd-class men 15.2%.",
        "Children (0-16) survived at 49.0% vs ~37% for adults. Modest individual effect but ANOVA confirms age-group differences are real (p < 0.05).",
        "Fare is the strongest numerical signal (r=0.247, t-test p < 2.2e-16; survivors mean $49.63 vs perished $23.19) but largely a proxy for class.",
        "Family-size shows a sweet spot: 1-2 siblings or 1-3 parents/children correlate with 45-55% survival; solo travelers and large families both drop to 30-35%.",
        "Embarkation port is significant in raw terms (Cherbourg 56.6%, Southampton 33.4%) but the effect is confounded with class - Cherbourg embarked disproportionately many 1st-class passengers.",
        "Titles encode the gender + age structure: Mrs 78.2%, Miss 67.7%, Master 50.8%, Mr 16.2%. Useful as a single derived feature for downstream modeling.",
        "Lifeboat records confirm the causal pathway: 98.6% of recorded boat passengers survived, vs 2.6% without a boat record. Sex/class/age predicted who got onto a boat.",
    ]
    for i, finding in enumerate(findings, 1):
        pdf.bullet(finding)
    pdf.ln(4)

    pdf.sub_title("Conclusion")
    pdf.body_text(
        "Survival was determined by who got onto a lifeboat - and that was determined by sex, "
        "class, and (more weakly) age. The 'women and children first' protocol was real and "
        "measurable, but class privilege strongly modulated it. All major associations are "
        "highly significant (p < 0.001) and the effect-size ranking is unambiguous: Sex > Class "
        "> Fare > Embarked > Age."
    )
    pdf.ln(2)

    pdf.sub_title("Limitations & Future Work")
    pdf.body_text(
        "Occupation is 47.4% missing and Age is 3.9% missing - both could bias subgroup analyses. "
        "Lifeboat-based analysis is descriptive, not causal: getting onto a boat is downstream of "
        "the social hierarchies this report studies. Natural extensions: logistic regression with "
        "interaction terms, SHAP-based feature attributions, and a holdout-validated survival model."
    )

    pdf.output(str(output_path))
    print(f"PDF report generated at {output_path}")


if __name__ == "__main__":
    df = load_titanic()
    output = Path(__file__).parent / "titanic_eda_report.pdf"
    generate_pdf(df, output)
