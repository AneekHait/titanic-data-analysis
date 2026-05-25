"""Cross-stage row-count and survival-breakdown reconciliation for titanic5.

Run from project root:

    python reports/validate_dataset.py
    # or
    make validate

This script is the canonical sanity check for the dataset. Every analytical
output in the project (dashboard, classical PDF, analyst report) should produce
numbers consistent with what this script prints.

What it verifies:
  * Row counts at every pipeline stage (raw -> clean -> engineered) all equal 1,309
  * Survival breakdowns by Sex, Pclass, Embarked, Title, FamilySize, AgeGroup
  * The joint Class x Sex 6-cell table sums to 1,309
  * The Lifeboat / No-record split sums to 1,309
  * Reports the canonical missing-value counts (Occupation: 621, Age: 51)

If any subtotal disagrees with a number shown in the dashboard or the report,
the most common cause is mixing the raw DataFrame with the engineered one in
the same analysis. See docs/DATA.md#counts-that-should-reconcile.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.data.loader import load_titanic
from src.data.processing import clean_data, engineer_features

raw = load_titanic()
clean = clean_data(raw)
df = engineer_features(clean)

print("=" * 60)
print("ROW COUNTS AT EACH STAGE")
print("=" * 60)
print(f"raw:        {len(raw)} rows")
print(f"clean_data: {len(clean)} rows")
print(f"engineered: {len(df)} rows")
print(f"Match: {len(raw) == len(clean) == len(df)}")
print()

print("=" * 60)
print("OVERALL SURVIVAL")
print("=" * 60)
print(f"Survived: {int(df['Survived'].sum())} / {len(df)} = {df['Survived'].mean()*100:.2f}%")
print()

print("=" * 60)
print("BY SEX")
print("=" * 60)
print(df.groupby("Sex")["Survived"].agg(["count", "sum", "mean"]).round(4))
print()

print("=" * 60)
print("BY PCLASS")
print("=" * 60)
print(df.groupby("Pclass")["Survived"].agg(["count", "sum", "mean"]).round(4))
print()

print("=" * 60)
print("BY EMBARKED (after clean_data fills mode)")
print("=" * 60)
print(df.groupby("Embarked")["Survived"].agg(["count", "sum", "mean"]).round(4))
print()

print("=" * 60)
print("RAW MISSING VALUES")
print("=" * 60)
miss = raw.isnull().sum()
print(miss[miss > 0])
print()

print("=" * 60)
print("JOINT Pclass x Sex")
print("=" * 60)
joint = df.groupby(["Pclass", "Sex"])["Survived"].agg(["count", "sum", "mean"])
joint["rate_pct"] = (joint["mean"] * 100).round(2)
print(joint)
print()
print(f"Total in joint: {joint['count'].sum()} (should be {len(df)})")
print()

print("=" * 60)
print("LIFEBOAT")
print("=" * 60)
has_boat = df["Lifeboat"].notna()
print(f"Has Lifeboat: {int(has_boat.sum())}, of which survived {int(df.loc[has_boat, 'Survived'].sum())}")
print(f"No record:    {int((~has_boat).sum())}, of which survived {int(df.loc[~has_boat, 'Survived'].sum())}")
print(f"Total: {int(has_boat.sum()) + int((~has_boat).sum())} (should be {len(df)})")
print()

print("=" * 60)
print("FAMILY SIZE")
print("=" * 60)
fam = df.groupby("FamilySize")["Survived"].agg(["count", "sum", "mean"])
fam["rate_pct"] = (fam["mean"] * 100).round(2)
print(fam)
print(f"Total: {fam['count'].sum()}")
print()

print("=" * 60)
print("AGE GROUP (engineered bins 0,16,25,50,100)")
print("=" * 60)
ag = df.groupby("AgeGroup", observed=True)["Survived"].agg(["count", "sum", "mean"])
ag["rate_pct"] = (ag["mean"] * 100).round(2)
print(ag)
print(f"Total: {ag['count'].sum()}")
print(f"NaN age count: {df['AgeGroup'].isna().sum()}")
print()

print("=" * 60)
print("AGE BINS USED IN REPORT (0,16,32,48,64,100)")
print("=" * 60)
df2 = df.copy()
df2["RA"] = pd.cut(df2["Age"], bins=[0, 16, 32, 48, 64, 100],
                   labels=["Child (0-16)", "Young Adult (17-32)", "Adult (33-48)",
                           "Older Adult (49-64)", "Senior (65+)"])
rep = df2.groupby("RA", observed=True)["Survived"].agg(["count", "sum", "mean"])
rep["rate_pct"] = (rep["mean"] * 100).round(2)
print(rep)
print(f"Total: {rep['count'].sum()}")
print()

print("=" * 60)
print("TITLE")
print("=" * 60)
ti = df.groupby("Title")["Survived"].agg(["count", "sum", "mean"])
ti["rate_pct"] = (ti["mean"] * 100).round(2)
print(ti.sort_values("count", ascending=False))
print(f"Total: {ti['count'].sum()}")
