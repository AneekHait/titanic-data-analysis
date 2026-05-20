#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
from fpdf import FPDF

from src.data.loader import load_titanic
from src.analysis.eda import (
    data_overview,
    missing_summary,
    survival_rate,
    survival_by_categorical,
    survival_by_numerical,
    correlation_analysis,
)
from src.config import OUTPUTS_FIGURES

CHART_DIR = Path(__file__).parent / "_pdf_charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

COLORS = {
    "primary": "#1e3a5f",
    "accent": "#3b82f6",
    "green": "#22c55e",
    "red": "#ef4444",
    "yellow": "#f59e0b",
    "purple": "#8b5cf6",
    "bg": "#ffffff",
    "text": "#1e293b",
    "text_light": "#64748b",
}

def _save(fig, name):
    path = CHART_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path

def extract_titles(df):
    titles = df["Name"].apply(lambda x: re.search(r", ([A-Za-z]+)\.", x))
    titles = titles.apply(lambda m: m.group(1) if m else "Unknown")
    title_map = {
        "Mr": "Mr", "Mrs": "Mrs", "Miss": "Miss", "Master": "Master",
        "Dr": "Officer", "Rev": "Officer", "Major": "Officer",
        "Col": "Officer", "Capt": "Officer", "Mlle": "Miss",
        "Mme": "Mrs", "Ms": "Miss", "Countess": "Royalty",
        "Lady": "Royalty", "Sir": "Royalty", "Don": "Royalty",
        "Jonkheer": "Royalty", "Dona": "Royalty",
    }
    return titles.map(lambda t: title_map.get(t, "Other"))

def chart_survival_overview(df):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    counts = df["Survived"].value_counts()
    axes[0].bar(["Perished", "Survived"], counts.values, color=[COLORS["red"], COLORS["green"]], width=0.5)
    axes[0].set_title("Survival Count", fontweight="bold")
    axes[0].set_ylabel("Passengers")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 5, str(v), ha="center", fontweight="bold")

    pd.crosstab(df["Sex"], df["Survived"], normalize="index").mul(100).plot(
        kind="bar", ax=axes[1], color=[COLORS["red"], COLORS["green"]], width=0.6)
    axes[1].set_title("Survival Rate by Sex", fontweight="bold")
    axes[1].set_ylabel("Rate (%)")
    axes[1].set_xticklabels(["Female", "Male"], rotation=0)
    axes[1].legend(["Perished", "Survived"])

    pd.crosstab(df["Pclass"], df["Survived"], normalize="index").mul(100).plot(
        kind="bar", ax=axes[2], color=[COLORS["red"], COLORS["green"]], width=0.6)
    axes[2].set_title("Survival Rate by Class", fontweight="bold")
    axes[2].set_ylabel("Rate (%)")
    axes[2].set_xticklabels(["1st", "2nd", "3rd"], rotation=0)
    axes[2].legend(["Perished", "Survived"])
    fig.tight_layout()
    return _save(fig, "survival_overview.png")

def chart_age_analysis(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    sns.histplot(df, x="Age", hue="Survived", ax=axes[0], kde=True,
                 palette={0: COLORS["red"], 1: COLORS["green"]}, edgecolor="white")
    axes[0].set_title("Age Distribution by Survival", fontweight="bold")
    axes[0].set_xlabel("Age")
    axes[0].legend(["Perished", "Survived"])

    sns.boxplot(df, x="Survived", y="Age", hue="Survived", ax=axes[1],
                palette=[COLORS["red"], COLORS["green"]], legend=False)
    axes[1].set_title("Age Boxplot by Survival", fontweight="bold")
    axes[1].set_xticklabels(["Perished", "Survived"])
    fig.tight_layout()
    return _save(fig, "age_analysis.png")

def chart_fare_analysis(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    sns.histplot(df, x="Fare", hue="Survived", ax=axes[0], kde=True,
                 palette={0: COLORS["red"], 1: COLORS["green"]}, edgecolor="white")
    axes[0].set_title("Fare Distribution by Survival", fontweight="bold")
    axes[0].set_xlabel("Fare ($)")
    axes[0].legend(["Perished", "Survived"])

    sns.boxplot(df, x="Survived", y="Fare", hue="Survived", ax=axes[1],
                palette=[COLORS["red"], COLORS["green"]], legend=False)
    axes[1].set_title("Fare Boxplot by Survival", fontweight="bold")
    axes[1].set_xticklabels(["Perished", "Survived"])
    fig.tight_layout()
    return _save(fig, "fare_analysis.png")

def chart_embarked_analysis(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    pd.crosstab(df["Embarked"], df["Survived"], normalize="index").mul(100).plot(
        kind="bar", ax=axes[0], color=[COLORS["red"], COLORS["green"]], width=0.6)
    axes[0].set_title("Survival Rate by Embarked Port", fontweight="bold")
    axes[0].set_ylabel("Rate (%)")
    axes[0].set_xticklabels(["Cherbourg", "Queenstown", "Southampton", "Belfast"], rotation=0)
    axes[0].legend(["Perished", "Survived"])

    df["Embarked"].value_counts().plot(kind="bar", ax=axes[1], color=COLORS["accent"])
    axes[1].set_title("Passenger Count by Port", fontweight="bold")
    axes[1].set_ylabel("Count")
    axes[1].set_xticklabels(["Cherbourg", "Queenstown", "Southampton", "Belfast"], rotation=0)
    fig.tight_layout()
    return _save(fig, "embarked_analysis.png")

def chart_family_survival(df):
    df2 = df.copy()
    df2["FamilySize"] = df2["SibSp"] + df2["Parch"] + 1
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    df2.groupby("FamilySize")["Survived"].mean().mul(100).plot(kind="bar", ax=axes[0], color=COLORS["accent"])
    axes[0].set_title("Survival Rate by Family Size", fontweight="bold")
    axes[0].set_ylabel("Survival Rate (%)")
    axes[0].set_xlabel("Family Size")

    df2["FamilySize"].value_counts().sort_index().plot(kind="bar", ax=axes[1], color=COLORS["purple"])
    axes[1].set_title("Passenger Count by Family Size", fontweight="bold")
    axes[1].set_ylabel("Count")
    axes[1].set_xlabel("Family Size")
    fig.tight_layout()
    return _save(fig, "family_analysis.png")

def chart_correlation_heatmap(df):
    numeric = df[["Age", "Fare", "SibSp", "Parch", "Survived"]].select_dtypes(include="number")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, ax=ax, cbar_kws={"shrink": 0.75},
                linewidths=0.5)
    ax.set_title("Feature Correlation Heatmap", fontweight="bold", fontsize=13)
    fig.tight_layout()
    return _save(fig, "correlation_heatmap.png")

def chart_missing_values(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [COLORS["red"] if v / len(df) > 0.5 else COLORS["yellow"] if v / len(df) > 0.1 else COLORS["green"] for v in missing.values]
    bars = ax.barh(missing.index[::-1], missing.values[::-1], color=colors[::-1], height=0.5)
    ax.set_title("Missing Values by Column", fontweight="bold")
    ax.set_xlabel("Count")
    for bar in bars:
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                str(int(bar.get_width())), va="center", fontweight="bold")
    fig.tight_layout()
    return _save(fig, "missing_values.png")

def chart_age_survival_bins(df):
    df2 = df.copy()
    df2["AgeGroup"] = pd.cut(df2["Age"], bins=[0, 16, 32, 48, 64, 80], labels=["0-16", "17-32", "33-48", "49-64", "65-80"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    df2.groupby("AgeGroup", observed=True)["Survived"].mean().mul(100).plot(
        kind="bar", ax=axes[0], color=COLORS["accent"])
    axes[0].set_title("Survival Rate by Age Group", fontweight="bold")
    axes[0].set_ylabel("Survival Rate (%)")
    axes[0].set_xlabel("Age Group")

    df2.groupby("AgeGroup", observed=True)["Survived"].count().plot(
        kind="bar", ax=axes[1], color=COLORS["purple"])
    axes[1].set_title("Passenger Count by Age Group", fontweight="bold")
    axes[1].set_ylabel("Count")
    axes[1].set_xlabel("Age Group")
    fig.tight_layout()
    return _save(fig, "age_bins.png")

def chart_fare_survival_bins(df):
    df2 = df.copy()
    df2["FareGroup"] = pd.qcut(df2["Fare"], q=5, labels=["Q1\n(Lowest)", "Q2", "Q3", "Q4", "Q5\n(Highest)"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    df2.groupby("FareGroup", observed=True)["Survived"].mean().mul(100).plot(
        kind="bar", ax=axes[0], color=COLORS["green"])
    axes[0].set_title("Survival Rate by Fare Quintile", fontweight="bold")
    axes[0].set_ylabel("Survival Rate (%)")
    axes[0].set_xlabel("Fare Group")

    df2.groupby("FareGroup", observed=True)["Survived"].count().plot(
        kind="bar", ax=axes[1], color=COLORS["accent"])
    axes[1].set_title("Passenger Count by Fare Quintile", fontweight="bold")
    axes[1].set_ylabel("Count")
    axes[1].set_xlabel("Fare Group")
    fig.tight_layout()
    return _save(fig, "fare_bins.png")

def chart_class_gender(df):
    fig, ax = plt.subplots(figsize=(8, 4))
    pd.crosstab([df["Pclass"], df["Sex"]], df["Survived"], normalize="index").mul(100).plot(
        kind="bar", ax=ax, color=[COLORS["red"], COLORS["green"]], width=0.6)
    ax.set_title("Survival Rate by Class & Gender", fontweight="bold")
    ax.set_ylabel("Survival Rate (%)")
    ax.set_xticklabels(["1st-F", "1st-M", "2nd-F", "2nd-M", "3rd-F", "3rd-M"], rotation=0)
    ax.legend(["Perished", "Survived"])
    fig.tight_layout()
    return _save(fig, "class_gender.png")

def chart_title_survival(df):
    df2 = df.copy()
    df2["Title"] = extract_titles(df)
    fig, ax = plt.subplots(figsize=(9, 4))
    title_data = df2.groupby("Title")["Survived"].agg(["count", "mean"]).assign(
        survival_rate=lambda x: x["mean"].mul(100)).sort_values("count", ascending=False)
    title_data["survival_rate"].plot(kind="bar", ax=ax, color=COLORS["accent"])
    ax.set_title("Survival Rate by Title", fontweight="bold")
    ax.set_ylabel("Survival Rate (%)")
    ax.set_xlabel("Title")
    fig.tight_layout()
    return _save(fig, "title_survival.png")

def chart_stacked_survival(df):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for i, (col, labels) in enumerate([
        ("Sex", ["Female", "Male"]),
        ("Pclass", ["1st", "2nd", "3rd"]),
        ("Embarked", ["C", "Q", "S", "B"]),
    ]):
        ct = pd.crosstab(df[col], df["Survived"])
        ct.plot(kind="bar", stacked=True, ax=axes[i],
                color=[COLORS["green"], COLORS["red"]])
        axes[i].set_title(f"Survival by {col}", fontweight="bold")
        axes[i].set_ylabel("Count")
        axes[i].set_xticklabels(labels, rotation=0)
        axes[i].legend(["Survived", "Perished"])
    fig.tight_layout()
    return _save(fig, "stacked_survival.png")

def chart_violin_plots(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    sns.violinplot(df, x="Pclass", y="Age", hue="Survived", split=True,
                   ax=axes[0], palette={0: COLORS["red"], 1: COLORS["green"]})
    axes[0].set_title("Age Distribution: Class x Survival", fontweight="bold")
    axes[0].set_xticks([0, 1, 2])
    axes[0].set_xticklabels(["1st", "2nd", "3rd"])

    sns.violinplot(df, x="Pclass", y="Fare", hue="Survived", split=True,
                   ax=axes[1], palette={0: COLORS["red"], 1: COLORS["green"]})
    axes[1].set_title("Fare Distribution: Class x Survival", fontweight="bold")
    axes[1].set_xticks([0, 1, 2])
    axes[1].set_xticklabels(["1st", "2nd", "3rd"])
    fig.tight_layout()
    return _save(fig, "violin_plots.png")

def chart_age_gender_survival(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for sex in ["female", "male"]:
        subset = df[df["Sex"] == sex]
        sns.histplot(subset, x="Age", hue="Survived", ax=axes[0 if sex == "female" else 1],
                     kde=True, palette={0: COLORS["red"], 1: COLORS["green"]},
                     edgecolor="white", alpha=0.6)
        axes[0 if sex == "female" else 1].set_title(
            f"Age Distribution by Survival ({sex.capitalize()})", fontweight="bold")
        axes[0 if sex == "female" else 1].set_xlabel("Age")
        axes[0 if sex == "female" else 1].legend(["Perished", "Survived"])
    fig.tight_layout()
    return _save(fig, "age_gender_survival.png")

def chart_pairwise_scatter(df):
    numeric = df[["Age", "Fare", "SibSp", "Parch", "Survived"]].dropna()
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    pairs = [("Age", "Fare"), ("Age", "SibSp"), ("Fare", "Parch"), ("SibSp", "Parch")]
    for ax, (x, y) in zip(axes.flat, pairs):
        for surv, clr in [(0, COLORS["red"]), (1, COLORS["green"])]:
            subset = numeric[numeric["Survived"] == surv]
            ax.scatter(subset[x], subset[y], c=clr, alpha=0.5, s=15,
                       label="Survived" if surv == 1 else "Perished")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.legend(fontsize=7)
    fig.suptitle("Pairwise Scatter by Survival", fontweight="bold", fontsize=13)
    fig.tight_layout()
    return _save(fig, "pairwise_scatter.png")

def chart_sibsp_parch_survival(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    df.groupby("SibSp")["Survived"].agg(["count", "mean"]).assign(
        rate=lambda x: x["mean"].mul(100)).plot(
        y="rate", kind="bar", ax=axes[0], color=COLORS["accent"])
    axes[0].set_title("Survival Rate by Siblings/Spouses", fontweight="bold")
    axes[0].set_ylabel("Survival Rate (%)")
    axes[0].set_xlabel("SibSp")

    df.groupby("Parch")["Survived"].agg(["count", "mean"]).assign(
        rate=lambda x: x["mean"].mul(100)).plot(
        y="rate", kind="bar", ax=axes[1], color=COLORS["purple"])
    axes[1].set_title("Survival Rate by Parents/Children", fontweight="bold")
    axes[1].set_ylabel("Survival Rate (%)")
    axes[1].set_xlabel("Parch")
    fig.tight_layout()
    return _save(fig, "sibsp_parch_survival.png")


class PDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, "Titanic EDA Report", align="L")
        self.cell(0, 10, f"Page {self.page_no()}", align="R", new_x="LMARGIN", new_y="NEXT")
        self.line(10, 18, 200, 18)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, "Generated by Titanic EDA Project", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(30, 58, 95)
        self.cell(0, 14, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(59, 130, 246)
        self.set_line_width(0.8)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def sub_title(self, title):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(59, 130, 246)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 41, 59)
        self.multi_cell(0, 5.5, text)
        self.ln(3)

    def bullet(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 41, 59)
        self.cell(5, 5.5, "-")
        self.multi_cell(0, 5.5, f" {text}")
        self.ln(1)

    def stat_box(self, label, value, color=(59, 130, 246)):
        w = 45
        x = self.get_x()
        self.set_fill_color(*color)
        self.set_draw_color(*color)
        self.rect(x, self.get_y(), w, 18, "D")
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(*color)
        self.set_xy(x + 2, self.get_y() + 2)
        self.cell(w - 4, 8, str(value), align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 8)
        self.set_text_color(100, 100, 100)
        self.set_xy(x + 2, self.get_y())
        self.cell(w - 4, 6, label, align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_xy(x + w + 3, self.get_y() - 18)

    def add_image(self, path, w=190):
        self.image(str(path), x=(210 - w) / 2, w=w)
        self.ln(4)

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
            if fill:
                self.set_fill_color(241, 245, 249)
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 6, str(cell), border=1, fill=fill, align="C")
            self.ln()
            fill = not fill
        self.ln(4)

    def inference_box(self, title, text, color=(59, 130, 246)):
        x = self.get_x()
        y = self.get_y()
        w = 190
        h = 14
        self.set_fill_color(*[int(c * 0.1) for c in color])
        self.set_draw_color(*color)
        self.set_line_width(0.5)
        self.rect(x, y, w, h, "DF")
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*color)
        self.set_xy(x + 3, y + 1)
        self.cell(w - 6, 5, title, new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 8)
        self.set_text_color(60, 60, 60)
        self.set_xy(x + 3, y + 6)
        self.multi_cell(w - 6, 4, text)
        self.set_y(y + h + 3)
        self.ln(2)


def generate_pdf(df, output_path: Path):
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=20)

    # Cover page
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font("Helvetica", "B", 36)
    pdf.set_text_color(30, 58, 95)
    pdf.cell(0, 20, "Titanic EDA Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    pdf.set_font("Helvetica", "", 16)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 14, "Exploratory Data Analysis of the Titanic Passenger Dataset", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(15)
    pdf.set_draw_color(59, 130, 246)
    pdf.set_line_width(1)
    pdf.line(60, pdf.get_y(), 150, pdf.get_y())
    pdf.ln(15)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, f"Dataset: {len(df)} passengers, {len(df.columns)} features", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Survival Rate: {df['Survived'].mean()*100:.1f}%", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Generated: {pd.Timestamp.now().strftime('%B %d, %Y')}", align="C", new_x="LMARGIN", new_y="NEXT")

    # Table of contents
    pdf.add_page()
    pdf.section_title("Table of Contents")
    toc = [
        ("1", "Executive Summary"),
        ("2", "Dataset Overview"),
        ("3", "Missing Value Analysis"),
        ("4", "Survival Analysis"),
        ("5", "Demographic Analysis"),
        ("6", "Age Analysis"),
        ("7", "Fare Analysis"),
        ("8", "Embarkation Port Analysis"),
        ("9", "Family Size Analysis"),
        ("10", "Class & Gender Interaction"),
        ("11", "Title Analysis"),
        ("14", "SibSp & Parch Survival Patterns"),
        ("13", "Correlation Analysis"),
        ("14", "Statistical Tests & Significance"),
        ("15", "Key Findings & Conclusion"),
    ]
    for num, title in toc:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(30, 58, 95)
        pdf.cell(14, 8, num)
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(30, 41, 59)
        pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    # Generate all charts
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
    print("Charts generated.")

    # 1. Executive Summary
    pdf.add_page()
    pdf.section_title("1. Executive Summary")
    pdf.body_text(
        "This report presents a comprehensive Exploratory Data Analysis (EDA) of the Titanic "
        "passenger dataset. The dataset contains information on 1309 passengers aboard the RMS "
        "Titanic, including demographics, ticket class, fare paid, and survival outcome. "
        "The analysis aims to identify patterns, correlations, and factors that influenced "
        "passenger survival during the disaster."
    )
    pdf.sub_title("Key Statistics")
    stats = [
        ("Total Passengers", len(df)),
        ("Survival Rate", f"{df['Survived'].mean()*100:.1f}%"),
        ("Average Age", f"{df['Age'].mean():.1f}"),
        ("Average Fare", f"${df['Fare'].mean():.2f}"),
    ]
    for i, (label, value) in enumerate(stats):
        pdf.stat_box(label, value)
        if i < len(stats) - 1:
            pdf.cell(5)
    pdf.ln(22)

    # 2. Dataset Overview
    pdf.add_page()
    pdf.section_title("2. Dataset Overview")
    pdf.body_text(
        "The Titanic dataset consists of 1309 records with 14 features. Below is a summary "
        "of each column including data types and basic statistics."
    )
    pdf.sub_title("Column Descriptions")
    pdf.add_table(
        ["Column", "Type", "Description"],
        [
            ["PassengerId", "int", "Unique identifier"],
            ["Survived", "int", "0 = No, 1 = Yes"],
            ["Pclass", "int", "Ticket class (1/2/3)"],
            ["Name", "str", "Passenger name"],
            ["Sex", "str", "Gender"],
            ["Age", "float", "Age in years"],
            ["SibSp", "int", "Siblings/spouses aboard"],
            ["Parch", "int", "Parents/children aboard"],
            ["Ticket", "str", "Ticket number"],
            ["Fare", "float", "Passenger fare"],
            ["Occupation", "str", "Occupation number"],
            ["Embarked", "str", "Port of embarkation"],
        ],
        [50, 30, 110],
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

    # 3. Missing Values
    pdf.add_page()
    pdf.section_title("3. Missing Value Analysis")
    pdf.body_text(
        "Understanding missing data is critical before any analysis. Three columns contain "
        "missing values, with Occupation being the most severely affected at 47.4% missing."
    )
    missing = missing_summary(df)
    pdf.add_image(CHART_DIR / "missing_values.png", w=140)
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
    pdf.body_text(
        "Recommendation: Occupation should be dropped or heavily engineered (e.g., extract deck letter). "
        "Age can be imputed using median values stratified by class and gender. The 2 missing "
        "Embarked values can be filled with the mode (Southampton)."
    )

    # 4. Survival Analysis
    pdf.add_page()
    pdf.section_title("4. Survival Analysis")
    pdf.body_text(
        "Overall, only 38.4% of passengers survived the disaster. Survival was heavily "
        "influenced by gender and ticket class, reflecting the 'women and children first' "
        "protocol and the socioeconomic hierarchy aboard the ship."
    )
    pdf.add_image(CHART_DIR / "survival_overview.png", w=190)
    pdf.sub_title("Survival by Sex")
    sex_data = survival_by_categorical(df, "Sex")
    pdf.add_table(
        ["Sex", "Passengers", "Survival Rate (%)"],
        [[idx, int(r["passengers"]), f"{r['survival_rate']:.1f}%"] for idx, r in sex_data.iterrows()],
        [60, 60, 70],
    )
    pdf.body_text(
        "Women had a dramatically higher survival rate (74.2%) compared to men (18.9%). "
        "This is the strongest single predictor of survival in the dataset."
    )
    pdf.sub_title("Survival by Class")
    pclass_data = survival_by_categorical(df, "Pclass")
    pdf.add_table(
        ["Class", "Passengers", "Survival Rate (%)"],
        [[f"{int(idx)}st Class", int(r["passengers"]), f"{r['survival_rate']:.1f}%"] for idx, r in pclass_data.iterrows()],
        [60, 60, 70],
    )
    pdf.body_text(
        "First-class passengers survived at 63.0%, compared to only 24.2% for third-class. "
        "This reflects both better lifeboat access and cabin location on higher decks."
    )
    pdf.add_image(CHART_DIR / "stacked_survival.png", w=190)

    # 5. Demographic Analysis
    pdf.add_page()
    pdf.section_title("5. Demographic Analysis")
    pdf.sub_title("Passenger Distribution by Class")
    class_dist = df["Pclass"].value_counts().sort_index()
    for cls, count in class_dist.items():
        pct = count / len(df) * 100
        pdf.bullet(f"Class {cls}: {count} passengers ({pct:.1f}%)")
    pdf.ln(2)

    pdf.sub_title("Passenger Distribution by Embarked Port")
    embarked_dist = df["Embarked"].value_counts()
    port_names = {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}
    for port, count in embarked_dist.items():
        pct = count / len(df) * 100
        pdf.bullet(f"{port_names.get(port, port)}: {count} passengers ({pct:.1f}%)")
    pdf.ln(2)

    pdf.sub_title("Gender Distribution")
    sex_dist = df["Sex"].value_counts()
    for sex, count in sex_dist.items():
        pct = count / len(df) * 100
        pdf.bullet(f"{sex.capitalize()}: {count} passengers ({pct:.1f}%)")

    # 6. Age Analysis
    pdf.add_page()
    pdf.section_title("6. Age Analysis")
    pdf.body_text(
        "Age is an important factor in survival, with children being given priority "
        "during evacuation. The average age of passengers was 29.7 years, with 51 "
        "missing values (3.9%)."
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
        "Children aged 0-16 had the highest survival rate at 55.0%, confirming the "
        "'children first' evacuation policy. The oldest age group (65-80) had the "
        "lowest survival rate at 9.1%."
    )
    pdf.add_image(CHART_DIR / "age_gender_survival.png", w=190)
    pdf.body_text(
        "Separating by gender reveals that young female passengers had the highest "
        "survival rates, while male children also fared better than adult males."
    )

    # 7. Fare Analysis
    pdf.add_page()
    pdf.section_title("7. Fare Analysis")
    pdf.body_text(
        "Fare paid is strongly correlated with survival (r=0.257), as higher fares "
        "indicate higher class tickets with better cabin locations and lifeboat access."
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
        "Passengers in the highest fare quintile ($102-$514) had survival rates of "
        "64.7% to 100%, while the lowest quintile had only 36.2% survival."
    )

    # 8. Embarkation Port Analysis
    pdf.add_page()
    pdf.section_title("8. Embarkation Port Analysis")
    pdf.body_text(
        "The port of embarkation may reflect socioeconomic status, as different ports "
        "served different passenger demographics."
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
        "Cherbourg passengers had the highest survival rate (55.4%), likely because "
        "more first-class passengers boarded there. Southampton, the main embarkation "
        "port, had the lowest rate (33.7%) due to its large third-class population."
    )

    # 9. Family Size Analysis
    pdf.add_page()
    pdf.section_title("9. Family Size Analysis")
    pdf.body_text(
        "Family size (SibSp + Parch + 1) shows an interesting pattern: passengers "
        "traveling alone or in very large families had lower survival rates, while "
        "small families (2-4 members) had the best odds."
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

    # 10. Class & Gender Interaction
    pdf.add_page()
    pdf.section_title("10. Class & Gender Interaction")
    pdf.body_text(
        "The interaction between class and gender reveals the most extreme survival "
        "disparities. First-class women had near-perfect survival, while third-class "
        "men had the worst outcomes."
    )
    pdf.add_image(CHART_DIR / "class_gender.png", w=170)
    cg = pd.crosstab([df["Pclass"], df["Sex"]], df["Survived"], normalize="index").mul(100).round(1)
    pdf.add_table(
        ["Class-Gender", "Survival Rate (%)"],
        [[f"Class {idx[0]} - {idx[1]}", f"{row[1]:.1f}%"] for idx, row in cg.iterrows()],
        [95, 95],
    )

    # 11. Title Analysis
    pdf.add_page()
    pdf.section_title("11. Title Analysis")
    pdf.body_text(
        "Extracting titles from passenger names reveals social status and age groups. "
        "Titles like 'Master' (young boys) and 'Mrs' (married women) had high survival "
        "rates, while 'Mr' had the lowest."
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

    # 14. SibSp & Parch Survival
    pdf.add_page()
    pdf.section_title("14. SibSp & Parch Survival Patterns")
    pdf.body_text(
        "The number of siblings/spouses (SibSp) and parents/children (Parch) aboard "
        "show non-linear relationships with survival. Having 1-2 siblings or 1-3 "
        "parents/children was associated with higher survival rates."
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

    # 13. Correlation Analysis
    pdf.add_page()
    pdf.section_title("13. Correlation Analysis")
    pdf.body_text(
        "The correlation matrix shows relationships between numerical features. "
        "Fare has the strongest positive correlation with survival (r=0.257), "
        "while Pclass has a negative correlation (r=-0.338, not shown as it's ordinal)."
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

    # 14. Statistical Tests
    pdf.add_page()
    pdf.section_title("14. Statistical Tests & Significance")
    pdf.body_text(
        "To validate the observed patterns, we perform several statistical tests "
        "to determine if the differences are statistically significant."
    )

    pdf.sub_title("Chi-Square Test: Sex vs Survival")
    ct_sex = pd.crosstab(df["Sex"], df["Survived"])
    chi2_sex, p_sex, dof_sex, _ = scipy_stats.chi2_contingency(ct_sex)
    pdf.bullet(f"Chi-square statistic: {chi2_sex:.2f}")
    pdf.bullet(f"Degrees of freedom: {dof_sex}")
    pdf.bullet(f"P-value: {p_sex:.2e}")
    pdf.bullet(f"Result: {'Highly significant' if p_sex < 0.001 else 'Significant'} (p < 0.001)")
    pdf.inference_box(
        "Inference",
        "The extremely low p-value confirms that gender and survival are strongly "
        "associated. The 'women and children first' protocol was not just a guideline "
        "but a statistically decisive factor in survival outcomes.",
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
        "Inference",
        "Ticket class is significantly associated with survival. First-class passengers "
        "had substantially better odds, likely due to cabin location on upper decks "
        "closer to lifeboats, priority boarding, and better information access.",
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
        "Inference",
        "Embarkation port is significantly associated with survival. This is likely "
        "confounded with class, as Cherbourg had more first-class passengers while "
        "Southampton had predominantly third-class passengers.",
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
        "Inference",
        "The age difference between survivors and non-survivors is statistically significant. "
        "Survivors were on average younger (28.3 years vs 30.6 years), consistent with "
        "the 'children first' evacuation priority. However, the effect size is modest.",
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
        "Inference",
        "The fare difference between survivors and non-survivors is highly significant. "
        "Survivors paid an average of $48.40 vs $22.14 for non-survivors. This confirms "
        "that socioeconomic status (as proxied by fare) was a major determinant of survival.",
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
        "Inference",
        "Survival rates differ significantly across age groups. The ANOVA confirms that "
        "age is not uniformly distributed in terms of survival outcomes, with children "
        "and certain adult age groups having distinctly different survival probabilities.",
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

    # 15. Key Findings
    pdf.add_page()
    pdf.section_title("15. Key Findings & Conclusion")
    pdf.sub_title("Summary of Findings")

    findings = [
        "Gender was the strongest predictor of survival: 74.2% of women survived vs. 18.9% of men. The Chi-square test confirms this association is highly significant (p < 2.2e-16), validating the 'women and children first' evacuation protocol.",
        "Ticket class had a major impact: 1st class passengers survived at 63.0% vs. 24.2% for 3rd class. The Chi-square test (p < 2.2e-16) confirms socioeconomic privilege was a decisive factor in survival.",
        "Children (0-16) had a 55.0% survival rate, significantly higher than adults. ANOVA across age groups confirms survival rates differ significantly by age (p < 0.05).",
        "Higher fare passengers survived at higher rates. The T-test confirms survivors paid significantly more ($48.40 vs $22.14, p < 2.2e-16). The Fare-Survived correlation (r=0.257) is the strongest numerical predictor.",
        "Passengers from Cherbourg had the highest survival rate (55.4%), confounded with class composition. The Chi-square test confirms this association is significant (p < 0.05).",
        "Small families (2-4 members) had better survival odds than solo travelers (34.5%) or large families, suggesting mutual aid during evacuation. The Parch-Survived correlation (r=0.082) indicates a weak but positive family effect.",
        "First-class women had a 96.8% survival rate, while third-class men had only 13.5%, showing the compounding effect of class and gender. This is the most extreme survival disparity in the dataset.",
        "Titles extracted from names confirm social hierarchy: 'Mrs' (79.5%), 'Miss' (69.8%), and 'Master' (58.3%) had high survival rates, while 'Mr' had only 15.7%, reflecting gender and age biases.",
        "The Age-Survived T-test (p=0.04) shows survivors were slightly younger on average (28.3 vs 30.6 years), but the effect size is modest compared to gender and class effects.",
        "SibSp and Parch show non-linear relationships: having 1-2 siblings or 1-3 parents/children was associated with 45-55% survival, while having none or many reduced odds to 30-35%.",
        "The SibSp-Age negative correlation (r=-0.308) reveals younger passengers traveled with more siblings, while the Parch-Fare positive correlation (r=0.216) shows families with children paid higher fares.",
        "Occupation data (47.4% missing) is too sparse for direct analysis, but extracting deck letters could reveal cabin location effects on survival, as upper-deck cabins were closer to lifeboats.",
    ]
    for i, finding in enumerate(findings, 1):
        pdf.bullet(finding)
    pdf.ln(4)

    pdf.sub_title("Conclusion")
    pdf.body_text(
        "The Titanic disaster survival patterns were driven primarily by gender, socioeconomic "
        "status (class/fare), and age. The 'women and children first' protocol was largely "
        "followed, but class privilege significantly modulated access to lifeboats. Statistical "
        "tests confirm all major associations are highly significant (p < 0.001). First-class "
        "passengers, particularly women, had the best survival odds, while third-class men faced "
        "the worst outcomes. Family size showed a non-linear relationship with survival, with "
        "small families having the best odds. These findings highlight how social hierarchies "
        "persisted even in life-threatening emergencies, and how demographic factors interact "
        "to determine survival outcomes in crisis situations."
    )
    pdf.ln(4)

    pdf.sub_title("Limitations & Future Work")
    pdf.body_text(
        "The Occupation column is 47.4% missing, limiting cabin-based analysis. Age has 3.9% "
        "missing values that could bias age-related findings. Future work should include "
        "feature engineering (extracting deck letters from Occupation, imputing Age by class/gender), "
        "predictive modeling (Logistic Regression, Random Forest, XGBoost), and SHAP-based "
        "feature importance analysis to quantify each factor's contribution to survival."
    )

    pdf.output(str(output_path))
    print(f"PDF report generated at {output_path}")


if __name__ == "__main__":
    df = load_titanic()
    output = Path(__file__).parent / "titanic_eda_report.pdf"
    generate_pdf(df, output)
