#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import math
import re
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "REPORT.md"
GH_BLOB = "https://github.com/AneekHait/titanic-data-analysis/blob/main/"


def _slugify(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return s or "section"


def _strip_leading_number(title: str) -> str:
    return re.sub(r"^\d+\.\s*", "", title).strip()


def _rewrite_report_links(md_text: str) -> str:
    """Make REPORT.md links work when embedded in the dashboard (local + Pages)."""
    md_text = re.sub(r"\]\(dashboard/_pdf_charts/", "](_pdf_charts/", md_text)
    md_text = md_text.replace("](dashboard/index.html)", "](#)")
    md_text = re.sub(r"\]\((reports/[^)]+)\)", rf"]({GH_BLOB}\1)", md_text)
    md_text = re.sub(r"\]\((docs/[^)]+)\)", rf"]({GH_BLOB}\1)", md_text)
    return md_text


def build_report_blocks():
    """Read REPORT.md, split at H2 headings, return (nav_html, sections_html)."""
    try:
        import markdown as md
    except ImportError:
        return "", ""
    if not REPORT_PATH.exists():
        return "", ""

    text = REPORT_PATH.read_text(encoding="utf-8")
    text = _rewrite_report_links(text)

    # Drop everything before the first level-2 heading (title + front-matter).
    m = re.search(r"(?m)^## ", text)
    body = text[m.start():] if m else text

    chunks = [c for c in re.split(r"(?m)^## ", body) if c.strip()]

    md_renderer = md.Markdown(extensions=["extra", "tables", "sane_lists"])
    nav_links: list[str] = []
    sections: list[str] = []

    for chunk in chunks:
        nl = chunk.find("\n")
        title = chunk[:nl].strip() if nl != -1 else chunk.strip()
        body_md = chunk[nl + 1:] if nl != -1 else ""
        rendered = md_renderer.convert(f"## {title}\n{body_md}")
        md_renderer.reset()

        sid = f"report-{_slugify(title)}"
        nav_label = _strip_leading_number(title)
        nav_links.append(
            f'    <a href="#{sid}" class="sub"><span class="icon">›</span> {nav_label}</a>'
        )
        sections.append(
            f'  <section id="{sid}" class="md-section">\n'
            f'    <div class="md-content">{rendered}</div>\n'
            f'  </section>'
        )

    nav_html = (
        '    <div class="group-label">\U0001F4D6 Analyst Report</div>\n'
        + "\n".join(nav_links)
    )
    return nav_html, "\n\n".join(sections)

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
from src.analysis.statistics import effect_sizes, chi_square_test, t_test_survival
from src.analysis.inference import (
    survival_rates_with_ci,
    key_odds_ratios,
    joint_survival,
    wilson_ci,
)


def safe_num(val):
    """Return rounded value or None (so Chart.js renders a gap, not a misleading 0)."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    return round(val, 1)


def build_data(df_raw):
    df = engineer_features(clean_data(df_raw))

    overview = data_overview(df)
    missing = missing_summary(df_raw)
    sr = survival_rate(df)

    sex_ci = survival_rates_with_ci(df, "Sex")
    pclass_ci = survival_rates_with_ci(df, "Pclass")
    emb_ci = survival_rates_with_ci(df, "Embarked")

    def _records_with_nullable(table: pd.DataFrame) -> list[dict]:
        # to_dict turns NaN into Python float('nan'); convert to None so JSON
        # emits valid `null`, which Chart.js renders as a gap instead of 0.
        records = table.reset_index().to_dict(orient="records")
        for rec in records:
            for k, v in rec.items():
                if isinstance(v, float) and math.isnan(v):
                    rec[k] = None
        return records

    survival_by_age = survival_by_numerical(df, "Age")
    survival_by_age.index = survival_by_age.index.astype(str)
    survival_by_fare = survival_by_numerical(df, "Fare")
    survival_by_fare.index = survival_by_fare.index.astype(str)

    corr = correlation_analysis(df)

    title_data = (
        df.groupby("Title")["Survived"].agg(["count", "mean"])
        .rename(columns={"count": "passengers", "mean": "survival_rate"})
        .assign(survival_rate=lambda x: x["survival_rate"].mul(100))
        .round(2)
        .sort_values("passengers", ascending=False)
        .reset_index()
        .to_dict(orient="records")
    )

    family_data = (
        df.groupby("FamilySize")["Survived"].agg(["count", "mean"])
        .rename(columns={"count": "passengers", "mean": "survival_rate"})
        .assign(survival_rate=lambda x: x["survival_rate"].mul(100))
        .round(2)
        .reset_index()
        .to_dict(orient="records")
    )

    age_bins = pd.cut(df["Age"], bins=10, precision=0)
    age_dist = df.groupby(age_bins, observed=False).size().reset_index(name="count")
    age_dist.columns = ["bin", "count"]
    age_dist["bin"] = age_dist["bin"].astype(str)

    fare_clip = df[df["Fare"] < df["Fare"].quantile(0.99)]
    fare_bins = pd.cut(fare_clip["Fare"], bins=10, precision=0)
    fare_dist = fare_clip.groupby(fare_bins, observed=False).size().reset_index(name="count")
    fare_dist.columns = ["bin", "count"]
    fare_dist["bin"] = fare_dist["bin"].astype(str)

    corr_features = ["Age", "Fare", "SibSp", "Parch", "Survived"]
    corr_matrix = [[round(corr[feat][col], 3) for col in corr_features] for feat in corr_features]

    es = effect_sizes(df).to_dict(orient="records")

    odds = key_odds_ratios(df)

    joint_cs = joint_survival(df, "Pclass", "Sex").to_dict(orient="records")

    lifeboat_n = int(df["Lifeboat"].notna().sum())
    lifeboat_survival = int(df[df["Lifeboat"].notna()]["Survived"].sum())
    no_boat_n = int(df["Lifeboat"].isna().sum())
    no_boat_survival = int(df[df["Lifeboat"].isna()]["Survived"].sum())

    chi_sex = chi_square_test(df, "Sex")
    chi_pc = chi_square_test(df, "Pclass")
    chi_emb = chi_square_test(df, "Embarked")
    t_age = t_test_survival(df, "Age")
    t_fare = t_test_survival(df, "Fare")

    return {
        "shape": list(overview["shape"]),
        "missing": missing.reset_index().rename(columns={"index": "column"}).to_dict(orient="records"),
        "survival_rate": {
            "perished": float(sr.get(0, 0)),
            "survived": float(sr.get(1, 0)),
            "perished_count": int((df["Survived"] == 0).sum()),
            "survived_count": int((df["Survived"] == 1).sum()),
        },
        "sex_ci": sex_ci.to_dict(orient="records"),
        "pclass_ci": pclass_ci.to_dict(orient="records"),
        "embarked_ci": emb_ci.to_dict(orient="records"),
        "survival_by_age": _records_with_nullable(survival_by_age),
        "survival_by_fare": _records_with_nullable(survival_by_fare),
        "title_survival": title_data,
        "family_survival": family_data,
        "age_dist": age_dist.to_dict(orient="records"),
        "fare_dist": fare_dist.to_dict(orient="records"),
        "corr_features": corr_features,
        "corr_matrix": corr_matrix,
        "effect_sizes": es,
        "odds_ratios": odds,
        "joint_class_sex": joint_cs,
        "lifeboat": {
            "with_boat": {"n": lifeboat_n, "survived": lifeboat_survival},
            "without_boat": {"n": no_boat_n, "survived": no_boat_survival},
        },
        "tests": {
            "chi_sex": {"chi2": chi_sex["chi2"], "p": chi_sex["p_value"], "strength": chi_sex["strength"]},
            "chi_pclass": {"chi2": chi_pc["chi2"], "p": chi_pc["p_value"], "strength": chi_pc["strength"]},
            "chi_embarked": {"chi2": chi_emb["chi2"], "p": chi_emb["p_value"], "strength": chi_emb["strength"]},
            "t_age": {"t": t_age["t_statistic"], "p": t_age["p_value"], "d": t_age["cohens_d"], "effect": t_age["effect_size"]},
            "t_fare": {"t": t_fare["t_statistic"], "p": t_fare["p_value"], "d": t_fare["cohens_d"], "effect": t_fare["effect_size"]},
        },
        "passenger_stats": {
            "total": len(df),
            "avg_age": round(df["Age"].mean(), 1),
            "median_age": round(df["Age"].median(), 1),
            "avg_fare": round(df["Fare"].mean(), 2),
            "median_fare": round(df["Fare"].median(), 2),
            "max_siblings": int(df["SibSp"].max()),
            "max_parch": int(df["Parch"].max()),
            "missing_ages": int(df_raw["Age"].isnull().sum()),
        },
    }


def generate_dashboard(df_raw, output_path: Path):
    data = build_data(df_raw)
    js_data = json.dumps(data)
    report_nav_html, report_sections_html = build_report_blocks()

    sex_data = data["sex_ci"]
    pclass_data = data["pclass_ci"]
    emb_data = data["embarked_ci"]
    female_rate = next((r["rate"] for r in sex_data if r["level"] == "female"), 0)
    male_rate = next((r["rate"] for r in sex_data if r["level"] == "male"), 0)
    first_class_rate = next((r["rate"] for r in pclass_data if r["level"] == 1), 0)
    third_class_rate = next((r["rate"] for r in pclass_data if r["level"] == 3), 0)
    cherbourg_rate = next((r["rate"] for r in emb_data if r["level"] == "C"), 0)

    top_effect = data["effect_sizes"][0]
    strongest_or = max(data["odds_ratios"], key=lambda x: x["odds_ratio"])

    rendered_missing_rows = []
    for r in data["missing"]:
        if r["percent"] > 40:
            tag, severity, bar_color = "tag-red", "High", "#f87171"
        elif r["percent"] > 10:
            tag, severity, bar_color = "tag-yellow", "Medium", "#fbbf24"
        else:
            tag, severity, bar_color = "tag-green", "Low", "#34d399"
        rendered_missing_rows.append(
            f"""<tr><td>{r['column']}</td><td>{r['missing']}</td><td>{r['percent']:.1f}%</td>"""
            f"""<td><span class="tag {tag}">{severity}</span></td></tr>"""
            f"""<tr class="bar-row"><td colspan="4"><div class="progress-bar"><div class="progress-bar-fill" style="width:{r['percent']}%;background:{bar_color}"></div></div></td></tr>"""
        )
    missing_html = "".join(rendered_missing_rows)

    chi_sex = data["tests"]["chi_sex"]
    chi_pc = data["tests"]["chi_pclass"]
    chi_emb = data["tests"]["chi_embarked"]
    t_age = data["tests"]["t_age"]
    t_fare = data["tests"]["t_fare"]

    def fmt_p(p):
        if p < 1e-10:
            return "&lt;2e-16"
        return f"{p:.2e}" if p < 0.01 else f"{p:.3f}"

    odds_rows_html = []
    for o in data["odds_ratios"]:
        lift_tag = "tag-green" if o["lift"] > 0 else "tag-red"
        lift_sign = "+" if o["lift"] > 0 else ""
        odds_rows_html.append(
            f"<tr><td><strong>{o['label']}</strong></td>"
            f"<td><code>{o['odds_ratio']:.2f}×</code></td>"
            f"<td>[{o['ci_low']:.2f}, {o['ci_high']:.2f}]</td>"
            f"<td>{o['rate_exposed']:.1f}% (n={o['n_exposed']})</td>"
            f"<td>{o['rate_unexposed']:.1f}% (n={o['n_unexposed']})</td>"
            f"<td><span class=\"tag {lift_tag}\">{lift_sign}{o['lift']:.1f}pp</span></td>"
            f"<td><code>{fmt_p(o['p_value'])}</code></td></tr>"
        )
    odds_html = "".join(odds_rows_html)

    test_rows = [
        ("Sex → Survived", "Chi-square", f"{chi_sex['chi2']:.1f}",
         "<2e-16" if chi_sex["p"] < 1e-10 else f"{chi_sex['p']:.2e}",
         chi_sex["strength"], "tag-green"),
        ("Pclass → Survived", "Chi-square", f"{chi_pc['chi2']:.1f}",
         "<2e-16" if chi_pc["p"] < 1e-10 else f"{chi_pc['p']:.2e}",
         chi_pc["strength"], "tag-green"),
        ("Embarked → Survived", "Chi-square", f"{chi_emb['chi2']:.1f}",
         f"{chi_emb['p']:.2e}", chi_emb["strength"], "tag-blue"),
        ("Fare (survived vs perished)", "Welch t-test", f"{t_fare['t']:.2f}",
         f"{t_fare['p']:.2e}", f"d={t_fare['d']:.2f} ({t_fare['effect']})", "tag-blue"),
        ("Age (survived vs perished)", "Welch t-test", f"{t_age['t']:.2f}",
         f"{t_age['p']:.3f}", f"d={t_age['d']:.2f} ({t_age['effect']})", "tag-yellow"),
    ]
    tests_html = "".join(
        f"<tr><td>{name}</td><td>{test}</td><td>{stat}</td><td><code>{p}</code></td>"
        f"<td><span class=\"tag {tag}\">{strength}</span></td></tr>"
        for (name, test, stat, p, strength, tag) in test_rows
    )

    lb = data["lifeboat"]
    boat_n = lb["with_boat"]["n"]
    boat_s = lb["with_boat"]["survived"]
    no_boat_n = lb["without_boat"]["n"]
    no_boat_s = lb["without_boat"]["survived"]
    boat_rate = (boat_s / boat_n * 100) if boat_n else 0
    no_boat_rate = (no_boat_s / no_boat_n * 100) if no_boat_n else 0
    boat_lo, boat_hi = wilson_ci(boat_s, boat_n) if boat_n else (0, 0)
    no_boat_lo, no_boat_hi = wilson_ci(no_boat_s, no_boat_n) if no_boat_n else (0, 0)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Titanic Survival Analysis — Interactive EDA Dashboard</title>
<meta name="description" content="Interactive exploratory data analysis of the Titanic dataset (1,309 passengers): survival rates by sex, class, age, fare, and embarkation, with 95% Wilson confidence intervals, odds ratios, chi-square tests, and effect sizes.">
<meta name="keywords" content="Titanic, Titanic dataset, Titanic survival, exploratory data analysis, EDA, data analysis, data science, statistics, odds ratio, chi-square, Wilson confidence interval, Cramer's V, Cohen's d, Kaggle, machine learning, Python, pandas, dashboard">
<meta name="author" content="Aneek Hait">
<meta name="robots" content="index, follow">
<link rel="canonical" href="https://aneekhait.github.io/titanic-data-analysis/">
<meta property="og:type" content="website">
<meta property="og:title" content="Titanic Survival Analysis — Interactive EDA Dashboard">
<meta property="og:description" content="Quantitative analysis of who survived the Titanic and why. 1,309 passengers, 14 features, 95% CIs, odds ratios, effect sizes, dark + light themes.">
<meta property="og:url" content="https://aneekhait.github.io/titanic-data-analysis/">
<meta property="og:site_name" content="Titanic Survival Analysis">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="Titanic Survival Analysis — Interactive EDA Dashboard">
<meta name="twitter:description" content="Who survived the Titanic and why — quantified with 95% CIs, odds ratios, and effect sizes.">
<script type="application/ld+json">
{{"@context":"https://schema.org","@type":"Dataset","name":"Titanic Survival Analysis","description":"Exploratory data analysis of the titanic5 dataset (1,309 passengers, 14 features) with statistical inference: 95% Wilson confidence intervals, odds ratios, chi-square tests, Cohen's d, and Cramer's V.","keywords":["Titanic","survival analysis","exploratory data analysis","Kaggle","statistics"],"creator":{{"@type":"Person","name":"Aneek Hait","url":"https://aneekhait.github.io"}},"license":"https://opensource.org/licenses/MIT","url":"https://github.com/AneekHait/titanic-data-analysis"}}
</script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://unpkg.com/chartjs-plugin-datalabels@2.2.0/dist/chartjs-plugin-datalabels.min.js"></script>
<style>
:root {{
  --bg-primary: #0b1220; --bg-secondary: #111a2e; --bg-card: #131d33; --bg-hover: #1c2840;
  --bg-elev: #0f172a; --border: #1f2c47; --border-strong: #2a3a5e;
  --text-primary: #e6edf7; --text-secondary: #94a3b8; --text-muted: #64748b;
  --accent: #60a5fa; --accent-strong: #3b82f6; --accent-purple: #a78bfa;
  --green: #34d399; --green-strong: #10b981; --red: #f87171; --red-strong: #ef4444;
  --yellow: #fbbf24; --pink: #f472b6;
  --sidebar-width: 260px;
  --shadow-card: 0 1px 0 rgba(255,255,255,0.04), 0 8px 24px rgba(0,0,0,0.25);
}}
.light-theme {{
  --bg-primary: #f5f7fb; --bg-secondary: #ffffff; --bg-card: #ffffff; --bg-hover: #eef2f7;
  --bg-elev: #f3f4f8; --border: #e4e9f0; --border-strong: #cbd5e1;
  --text-primary: #0f172a; --text-secondary: #334155; --text-muted: #64748b;
  --accent: #2563eb; --accent-strong: #1d4ed8; --accent-purple: #7c3aed;
  --green: #059669; --green-strong: #047857;
  --red: #dc2626; --red-strong: #b91c1c;
  --yellow: #b45309; --pink: #be185d;
  --shadow-card: 0 1px 2px rgba(15,23,42,0.06), 0 4px 16px rgba(15,23,42,0.08);
}}
/* Tags need more contrast on white backgrounds */
.light-theme .tag-green {{ background: rgba(5, 150, 105, 0.12); color: #047857; }}
.light-theme .tag-red {{ background: rgba(220, 38, 38, 0.12); color: #b91c1c; }}
.light-theme .tag-blue {{ background: rgba(37, 99, 235, 0.12); color: #1d4ed8; }}
.light-theme .tag-yellow {{ background: rgba(180, 83, 9, 0.14); color: #92400e; }}
.light-theme .tag-purple {{ background: rgba(124, 58, 237, 0.12); color: #6d28d9; }}
.light-theme code {{ background: var(--bg-elev); color: var(--accent-strong); border: 1px solid var(--border); }}
.light-theme .takeaway {{ background: rgba(37, 99, 235, 0.08); }}
.light-theme .howto {{ background: var(--bg-card); }}
.light-theme .howto-grid dt {{ color: var(--accent-strong); }}
.light-theme .insights {{ border-left-color: var(--accent-strong); }}
.light-theme .insights li::before {{ color: var(--accent-strong); }}
.light-theme .kpi::before {{ opacity: 1; }}
.light-theme .section-head h2 {{ color: var(--text-primary); }}
.light-theme th {{ color: var(--accent-strong); border-bottom: 2px solid var(--border-strong); }}
.light-theme .compare-card h4,
.light-theme .kpi .label {{ color: var(--text-muted); }}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
html {{ scroll-behavior: smooth; }}
body {{
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: var(--bg-primary); color: var(--text-primary);
  line-height: 1.6; transition: background .25s, color .25s;
  font-feature-settings: 'cv11', 'ss01';
}}
.sidebar {{
  position: fixed; top: 0; left: 0; width: var(--sidebar-width); height: 100vh;
  background: var(--bg-secondary); border-right: 1px solid var(--border);
  padding: 1.5rem 1rem; overflow-y: auto; z-index: 100;
  transition: transform 0.3s;
}}
.sidebar-logo {{
  font-size: 1.4rem; font-weight: 800; letter-spacing: -0.02em;
  background: linear-gradient(90deg, var(--accent), var(--accent-purple));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  margin-bottom: 0.25rem; display: flex; align-items: center; gap: 0.5rem;
}}
.sidebar-logo span {{ font-size: 1.6rem; -webkit-text-fill-color: initial; }}
.sidebar-sub {{ color: var(--text-muted); font-size: 0.75rem; margin-bottom: 1.5rem;
  text-transform: uppercase; letter-spacing: 0.08em; }}
.sidebar-nav a {{
  display: flex; align-items: center; gap: 0.6rem; padding: 0.55rem 0.75rem;
  color: var(--text-secondary); text-decoration: none; border-radius: 8px;
  margin-bottom: 0.15rem; transition: all 0.15s; font-size: 0.9rem; font-weight: 500;
  border-left: 2px solid transparent;
}}
.sidebar-nav a:hover {{ background: var(--bg-hover); color: var(--text-primary); }}
.sidebar-nav a.active {{ background: var(--bg-hover); color: var(--accent);
  border-left-color: var(--accent); }}
.sidebar-nav a .icon {{ width: 18px; text-align: center; font-size: 0.95rem; opacity: 0.9; }}
.sidebar-nav .group-label {{
  font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em;
  color: var(--text-muted); padding: 0.85rem 0.75rem 0.4rem; font-weight: 700;
  border-top: 1px solid var(--border); margin-top: 0.5rem;
  display: flex; align-items: center; gap: 0.4rem;
}}
.sidebar-nav a.sub {{ font-size: 0.82rem; padding-left: 1.25rem; }}
.sidebar-nav a.sub .icon {{ font-size: 0.85rem; opacity: 0.55; }}

.md-divider {{
  margin: 2.5rem 0 1.25rem; padding: 1.1rem 1.4rem;
  background: linear-gradient(90deg, var(--bg-card), var(--bg-secondary));
  border: 1px solid var(--border); border-left: 3px solid var(--accent);
  border-radius: 12px; box-shadow: var(--shadow-card);
}}
.md-divider-label {{
  font-size: 1.05rem; font-weight: 800; letter-spacing: -0.01em;
  color: var(--text-primary); margin-bottom: 0.3rem;
}}
.md-divider-desc {{
  color: var(--text-secondary); font-size: 0.9rem; max-width: 80ch; line-height: 1.55;
}}

.md-section {{ scroll-margin-top: 1.5rem; }}
.md-section .md-content {{
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: 12px; padding: 1.4rem 1.6rem;
  box-shadow: var(--shadow-card); margin-bottom: 1.25rem;
  color: var(--text-secondary); font-size: 0.95rem; line-height: 1.62;
}}
.md-section .md-content > *:first-child {{ margin-top: 0; }}
.md-section .md-content h2 {{
  font-size: 1.18rem; font-weight: 800; letter-spacing: -0.01em;
  color: var(--text-primary); margin-bottom: 0.65rem;
  padding-bottom: 0.5rem; border-bottom: 1px solid var(--border);
}}
.md-section .md-content h3 {{
  font-size: 1rem; font-weight: 700; color: var(--text-primary);
  margin: 1.4rem 0 0.45rem;
}}
.md-section .md-content h4 {{
  font-size: 0.9rem; font-weight: 700; color: var(--text-primary);
  margin: 1.1rem 0 0.35rem;
}}
.md-section .md-content p {{ margin: 0 0 0.7rem; }}
.md-section .md-content strong {{ color: var(--text-primary); font-weight: 700; }}
.md-section .md-content em {{ color: var(--text-primary); }}
.md-section .md-content a {{ color: var(--accent); text-decoration: none; }}
.md-section .md-content a:hover {{ text-decoration: underline; }}
.md-section .md-content ul,
.md-section .md-content ol {{ padding-left: 1.35rem; margin: 0 0 0.85rem; }}
.md-section .md-content li {{ margin-bottom: 0.32rem; }}
.md-section .md-content blockquote {{
  margin: 0.85rem 0; padding: 0.7rem 1rem;
  background: rgba(96, 165, 250, 0.08); border-left: 3px solid var(--accent);
  border-radius: 4px;
}}
.md-section .md-content blockquote p {{ margin-bottom: 0; }}
.md-section .md-content code {{
  font-family: 'JetBrains Mono', 'SF Mono', Consolas, monospace;
  background: var(--bg-elev); color: var(--accent);
  padding: 0.1em 0.4em; border-radius: 4px; font-size: 0.85em;
  border: 1px solid var(--border);
}}
.md-section .md-content pre {{
  background: var(--bg-elev); border: 1px solid var(--border);
  border-radius: 8px; padding: 1rem; overflow-x: auto;
  margin: 0.6rem 0 1rem; font-size: 0.85rem;
}}
.md-section .md-content pre code {{
  background: transparent; border: none; padding: 0; color: var(--text-primary);
}}
.md-section .md-content hr {{
  border: none; border-top: 1px solid var(--border); margin: 1.4rem 0;
}}
.md-section .md-content table {{ margin: 0.4rem 0 1rem; font-size: 0.88rem; }}
.md-section .md-content img {{
  display: block; max-width: 100%; height: auto;
  border-radius: 8px; border: 1px solid var(--border);
  margin: 0.85rem auto; box-shadow: var(--shadow-card);
}}
.light-theme .md-section .md-content blockquote {{ background: rgba(37, 99, 235, 0.08); }}
.light-theme .md-section .md-content code {{ color: var(--accent-strong); }}
.main {{ margin-left: var(--sidebar-width); padding: 2rem 2.5rem; min-height: 100vh; max-width: 1400px; }}
.top-bar {{
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 1.5rem; padding-bottom: 1rem; border-bottom: 1px solid var(--border);
}}
.top-bar h1 {{ font-size: 1.6rem; font-weight: 800; letter-spacing: -0.02em; }}
.top-bar h1 .sub {{ color: var(--text-muted); font-weight: 500; font-size: 0.9rem; margin-left: 0.5rem; }}
.theme-toggle {{
  background: var(--bg-card); border: 1px solid var(--border); color: var(--text-primary);
  padding: 0.45rem 0.85rem; border-radius: 8px; cursor: pointer; font-size: 0.9rem;
  transition: all 0.15s; font-weight: 500;
}}
.theme-toggle:hover {{ border-color: var(--accent); color: var(--accent); }}

.section-head {{ margin: 2rem 0 0.75rem; }}
.section-head h2 {{ font-size: 1.15rem; font-weight: 700; letter-spacing: -0.01em;
  display: flex; align-items: center; gap: 0.5rem; }}
.section-head h2 .icon {{ font-size: 1.25rem; }}
.section-head .desc {{ color: var(--text-secondary); font-size: 0.92rem; margin-top: 0.35rem; max-width: 80ch; }}
.section-head .desc strong {{ color: var(--text-primary); }}

.howto {{ background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px;
  padding: 1.1rem 1.3rem; margin-bottom: 1.5rem; box-shadow: var(--shadow-card); }}
.howto h3 {{ font-size: 1rem; font-weight: 700; margin-bottom: 0.6rem;
  display: flex; align-items: center; gap: 0.4rem; color: var(--text-primary); }}
.howto h3 .icon {{ font-size: 1.1rem; }}
.howto p {{ font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 0.7rem; }}
.howto-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 0.75rem 1rem; }}
.howto-grid dt {{ font-weight: 700; color: var(--accent); font-size: 0.82rem;
  font-family: 'JetBrains Mono', 'SF Mono', Consolas, monospace; }}
.howto-grid dd {{ color: var(--text-secondary); font-size: 0.82rem; margin-top: 0.1rem;
  margin-bottom: 0.6rem; line-height: 1.45; }}

.takeaway {{ margin-top: 0.7rem; padding: 0.55rem 0.8rem;
  background: rgba(96, 165, 250, 0.08); border-left: 3px solid var(--accent);
  border-radius: 4px; font-size: 0.85rem; color: var(--text-secondary); }}
.takeaway strong {{ color: var(--text-primary); }}

.kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 0.85rem; margin-bottom: 1.5rem; }}
.kpi {{
  background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px;
  padding: 1rem 1.1rem; box-shadow: var(--shadow-card); position: relative; overflow: hidden;
}}
.kpi::before {{ content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: var(--accent); opacity: 0.6; }}
.kpi.green::before {{ background: var(--green); }}
.kpi.red::before {{ background: var(--red); }}
.kpi.purple::before {{ background: var(--accent-purple); }}
.kpi.yellow::before {{ background: var(--yellow); }}
.kpi .label {{ color: var(--text-muted); font-size: 0.72rem; text-transform: uppercase;
  letter-spacing: 0.06em; font-weight: 600; }}
.kpi .value {{ font-size: 1.7rem; font-weight: 800; margin: 0.15rem 0; letter-spacing: -0.02em; }}
.kpi.green .value {{ color: var(--green); }}
.kpi.red .value {{ color: var(--red); }}
.kpi.purple .value {{ color: var(--accent-purple); }}
.kpi.yellow .value {{ color: var(--yellow); }}
.kpi .delta {{ font-size: 0.78rem; color: var(--text-secondary); }}
.kpi .delta .up {{ color: var(--green); font-weight: 700; }}
.kpi .delta .down {{ color: var(--red); font-weight: 700; }}

.compare {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0.85rem;
  margin-bottom: 1.5rem; }}
.compare-card {{
  background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px;
  padding: 1.1rem 1.2rem; box-shadow: var(--shadow-card);
}}
.compare-card h4 {{ color: var(--text-muted); font-size: 0.72rem; text-transform: uppercase;
  letter-spacing: 0.06em; font-weight: 600; margin-bottom: 0.65rem; }}
.compare-row {{ display: flex; justify-content: space-between; align-items: center;
  padding: 0.35rem 0; font-size: 0.92rem; }}
.compare-row .name {{ color: var(--text-secondary); }}
.compare-row .pct {{ font-weight: 700; font-family: 'JetBrains Mono', monospace; font-size: 0.95rem; }}
.compare-row .pct.high {{ color: var(--green); }}
.compare-row .pct.low {{ color: var(--red); }}
.compare-row .pct.mid {{ color: var(--accent); }}
.compare-delta {{ margin-top: 0.5rem; padding-top: 0.6rem; border-top: 1px dashed var(--border);
  font-size: 0.82rem; color: var(--text-secondary); }}
.compare-delta strong {{ color: var(--text-primary); }}

.card {{
  background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px;
  padding: 1.2rem 1.3rem; margin-bottom: 1.25rem; box-shadow: var(--shadow-card);
}}
.charts-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(440px, 1fr));
  gap: 1.25rem; margin-bottom: 1.5rem; }}
.chart-card {{
  background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px;
  padding: 1rem 1.1rem; box-shadow: var(--shadow-card);
}}
.chart-card h3 {{ font-size: 0.95rem; font-weight: 600; color: var(--text-primary);
  margin-bottom: 0.3rem; }}
.chart-card .chart-note {{ color: var(--text-muted); font-size: 0.78rem; margin-bottom: 0.7rem; }}
.chart-wrapper {{ position: relative; height: 280px; }}
.chart-wrapper.tall {{ height: 340px; }}
.chart-wrapper.short {{ height: 220px; }}

table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
th, td {{ padding: 0.55rem 0.85rem; text-align: left; border-bottom: 1px solid var(--border); }}
th {{ background: var(--bg-elev); color: var(--text-secondary); font-weight: 600;
  text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.05em; }}
tr.bar-row td {{ padding-top: 0; padding-bottom: 0.4rem; border-bottom: none; }}
tr:hover {{ background: var(--bg-hover); }}
code {{ font-family: 'JetBrains Mono', 'SF Mono', Consolas, monospace; font-size: 0.82rem;
  background: var(--bg-elev); padding: 0.1rem 0.4rem; border-radius: 4px; color: var(--accent); }}
.progress-bar {{ height: 6px; background: var(--border); border-radius: 4px; overflow: hidden; }}
.progress-bar-fill {{ height: 100%; border-radius: 4px; transition: width 1s ease; }}

.heatmap {{ display: grid; gap: 3px; margin-top: 0.75rem; }}
.heatmap-cell, .heatmap-label {{ aspect-ratio: 2.4; display: flex; flex-direction: column;
  align-items: center; justify-content: center; border-radius: 6px; font-size: 0.85rem; padding: 0.25rem; }}
.heatmap-cell {{ font-weight: 700; transition: transform .15s; cursor: default; }}
.heatmap-cell:hover {{ transform: scale(1.05); z-index: 2; }}
.heatmap-cell .sub {{ font-size: 0.65rem; font-weight: 500; opacity: 0.85; }}
.heatmap-label {{ font-weight: 600; font-size: 0.78rem; color: var(--text-secondary);
  background: var(--bg-elev); }}

.insights {{ background: var(--bg-card); border-left: 4px solid var(--accent);
  border-radius: 12px; padding: 1.2rem 1.4rem; box-shadow: var(--shadow-card); }}
.insights ul {{ list-style: none; }}
.insights li {{ padding: 0.6rem 0; border-bottom: 1px solid var(--border);
  display: flex; align-items: flex-start; gap: 0.5rem; font-size: 0.92rem; }}
.insights li:last-child {{ border-bottom: none; }}
.insights li::before {{ content: "▸"; color: var(--accent); font-size: 1rem; flex-shrink: 0; }}

.tag {{ display: inline-block; padding: 0.18rem 0.5rem; border-radius: 4px;
  font-size: 0.72rem; font-weight: 600; }}
.tag-green {{ background: rgba(52, 211, 153, 0.15); color: var(--green); }}
.tag-red {{ background: rgba(248, 113, 113, 0.15); color: var(--red); }}
.tag-blue {{ background: rgba(96, 165, 250, 0.15); color: var(--accent); }}
.tag-yellow {{ background: rgba(251, 191, 36, 0.15); color: var(--yellow); }}
.tag-purple {{ background: rgba(167, 139, 250, 0.15); color: var(--accent-purple); }}

.footer {{ text-align: center; padding: 2rem; color: var(--text-muted); font-size: 0.82rem;
  border-top: 1px solid var(--border); margin-top: 2rem; }}
.mobile-toggle {{ display: none; position: fixed; top: 1rem; left: 1rem; z-index: 200;
  background: var(--bg-secondary); border: 1px solid var(--border); color: var(--text-primary);
  padding: 0.45rem 0.6rem; border-radius: 8px; cursor: pointer; font-size: 1.15rem; }}
@media (max-width: 900px) {{
  .compare {{ grid-template-columns: 1fr; }}
  .charts-grid {{ grid-template-columns: 1fr; }}
}}
@media (max-width: 768px) {{
  .sidebar {{ transform: translateX(-100%); }}
  .sidebar.open {{ transform: translateX(0); }}
  .main {{ margin-left: 0; padding: 1rem; padding-top: 4rem; }}
  .kpis {{ grid-template-columns: repeat(2, 1fr); }}
  .mobile-toggle {{ display: block; }}
}}
</style>
</head>
<body>
<button class="mobile-toggle" onclick="document.querySelector('.sidebar').classList.toggle('open')">☰</button>
<aside class="sidebar">
  <div class="sidebar-logo"><span>\U0001F6A2</span> Titanic EDA</div>
  <div class="sidebar-sub">titanic5 · 1,309 passengers</div>
  <div style="margin-bottom:1.25rem;font-size:0.78rem;color:var(--text-muted);line-height:1.45;">
    by <a href="https://aneekhait.github.io" target="_blank" rel="noopener" style="color:var(--accent);text-decoration:none;font-weight:600;">Aneek Hait</a>
  </div>
  <nav class="sidebar-nav">
    <a href="#overview"><span class="icon">\U0001F4CA</span> Overview</a>
    <a href="#power"><span class="icon">\U0001F3AF</span> Feature Power</a>
    <a href="#odds"><span class="icon">⚖️</span> Odds Ratios</a>
    <a href="#survival"><span class="icon">\U0001F49A</span> Survival Rates</a>
    <a href="#joint"><span class="icon">\U0001F9E9</span> Class × Sex</a>
    <a href="#demographics"><span class="icon">\U0001F465</span> Demographics</a>
    <a href="#distributions"><span class="icon">\U0001F4C8</span> Distributions</a>
    <a href="#lifeboat"><span class="icon">\U0001F6F6</span> Lifeboats</a>
    <a href="#tests"><span class="icon">\U0001F9EA</span> Statistical Tests</a>
    <a href="#correlation"><span class="icon">\U0001F517</span> Correlations</a>
    <a href="#missing"><span class="icon">⚠️</span> Missing Values</a>
    <a href="#insights"><span class="icon">\U0001F4A1</span> Key Insights</a>
__REPORT_NAV__
  </nav>
</aside>
<main class="main">
  <div class="top-bar">
    <h1>Titanic Survival Dashboard<span class="sub">interactive EDA</span></h1>
    <button class="theme-toggle" onclick="toggleTheme()" id="themeBtn">☾ Theme</button>
  </div>

  <section id="overview">
    <div class="section-head">
      <h2><span class="icon">\U0001F4CA</span>Headline Numbers</h2>
      <div class="desc">Of 1,309 people aboard the Titanic, only <strong>38.2%</strong> made it out alive. But that average hides everything that matters &mdash; survival was wildly uneven across <strong>sex</strong>, <strong>passenger class</strong>, and to a lesser extent <strong>age</strong>. This dashboard breaks down exactly who survived and quantifies how much each factor mattered.</div>
    </div>
    <div class="kpis">
      <div class="kpi"><div class="label">Total Passengers</div>
        <div class="value">{data['passenger_stats']['total']:,}</div>
        <div class="delta">14 features, {data['passenger_stats']['missing_ages']} missing ages</div></div>
      <div class="kpi green"><div class="label">Survived</div>
        <div class="value">{data['survival_rate']['survived_count']}</div>
        <div class="delta"><span class="up">{data['survival_rate']['survived']:.1f}%</span> of passengers</div></div>
      <div class="kpi red"><div class="label">Perished</div>
        <div class="value">{data['survival_rate']['perished_count']}</div>
        <div class="delta"><span class="down">{data['survival_rate']['perished']:.1f}%</span> of passengers</div></div>
      <div class="kpi purple"><div class="label">Women Survival</div>
        <div class="value">{female_rate:.1f}%</div>
        <div class="delta">vs men <strong>{male_rate:.1f}%</strong> &middot; <span class="up">+{female_rate - male_rate:.1f}pp</span></div></div>
      <div class="kpi yellow"><div class="label">1st Class Survival</div>
        <div class="value">{first_class_rate:.1f}%</div>
        <div class="delta">vs 3rd class <strong>{third_class_rate:.1f}%</strong> &middot; <span class="up">+{first_class_rate - third_class_rate:.1f}pp</span></div></div>
      <div class="kpi"><div class="label">Strongest Odds Ratio</div>
        <div class="value">{strongest_or['odds_ratio']:.1f}×</div>
        <div class="delta">{strongest_or['label']}</div></div>
    </div>

    <div class="compare">
      <div class="compare-card">
        <h4>Survival by Sex</h4>
        {''.join(f'<div class="compare-row"><span class="name">{r["level"].capitalize()} (n={r["n"]})</span><span class="pct {"high" if r["rate"] > 50 else "low" if r["rate"] < 25 else "mid"}">{r["rate"]:.1f}%</span></div>' for r in data['sex_ci'])}
        <div class="compare-delta">Difference: <strong>{female_rate - male_rate:.1f} percentage points</strong></div>
      </div>
      <div class="compare-card">
        <h4>Survival by Class</h4>
        {''.join(f'<div class="compare-row"><span class="name">Class {r["level"]} (n={r["n"]})</span><span class="pct {"high" if r["rate"] > 50 else "low" if r["rate"] < 30 else "mid"}">{r["rate"]:.1f}%</span></div>' for r in data['pclass_ci'])}
        <div class="compare-delta">1st vs 3rd: <strong>{first_class_rate - third_class_rate:.1f}pp gap</strong></div>
      </div>
      <div class="compare-card">
        <h4>Survival by Port</h4>
        {''.join(f'<div class="compare-row"><span class="name">{ {"C":"Cherbourg","Q":"Queenstown","S":"Southampton","B":"Belfast"}.get(r["level"], r["level"]) } (n={r["n"]})</span><span class="pct {"high" if r["rate"] > 50 else "low" if r["rate"] < 30 else "mid"}">{r["rate"]:.1f}%</span></div>' for r in data['embarked_ci'])}
        <div class="compare-delta">Cherbourg edge: <strong>+{cherbourg_rate - 33.4:.1f}pp vs Southampton</strong></div>
      </div>
    </div>

    <div class="howto">
      <h3><span class="icon">\U0001F4D6</span>How to read this dashboard</h3>
      <p>You'll see a few statistical terms repeat throughout. Here's what they mean in plain English:</p>
      <dl class="howto-grid">
        <dt>Survival rate</dt>
        <dd>The percentage of a group that survived. "<strong>62.0%</strong>" means 62 of every 100 people in that group survived.</dd>

        <dt>95% Confidence Interval (CI)</dt>
        <dd>The range the true rate is very likely to fall in. A tight CI = certain estimate; a wide CI = small sample, less certain.</dd>

        <dt>Odds Ratio (OR)</dt>
        <dd>How many times more likely one group is to survive vs another. <strong>OR = 11.3x</strong> means 11 times the odds; <strong>OR &lt; 1</strong> means lower odds.</dd>

        <dt>Effect size (Cramer's V / r)</dt>
        <dd>How strongly a feature predicts survival on a 0&ndash;1 scale. Rule of thumb: <strong>0.1</strong> = small, <strong>0.3</strong> = medium, <strong>0.5</strong> = large.</dd>

        <dt>p-value</dt>
        <dd>The probability the pattern is just coincidence. <strong>p &lt; 0.05</strong> = unlikely a fluke; <strong>p &lt; 0.001</strong> = essentially impossible.</dd>

        <dt>pp (percentage points)</dt>
        <dd>The arithmetic gap between two percentages. Going from 19% to 73% is a <strong>54pp</strong> jump, not "54%".</dd>
      </dl>
    </div>
  </section>

  <section id="power">
    <div class="section-head">
      <h2><span class="icon">\U0001F3AF</span>Which factors mattered most?</h2>
      <div class="desc">If you had to bet someone's chance of surviving knowing only one thing about them, what should that one thing be? This chart ranks every feature by how strongly it predicts survival on a 0&ndash;1 scale (small / medium / large).</div>
    </div>
    <div class="chart-card">
      <h3>Predictive power, ranked</h3>
      <div class="chart-note">Bigger bar = stronger predictor. The dashed lines mark the conventional thresholds (small / medium / large).</div>
      <div class="chart-wrapper tall"><canvas id="powerChart"></canvas></div>
      <div class="takeaway"><strong>Takeaway:</strong> <strong>Sex</strong> is in a league of its own (large effect, ~0.53). <strong>Class</strong> is a clear medium-strength predictor. Everything else &mdash; age, family size, port &mdash; is small or negligible on its own (though they matter in combination).</div>
    </div>
  </section>

  <section id="odds">
    <div class="section-head">
      <h2><span class="icon">⚖️</span>How much did each factor change your odds?</h2>
      <div class="desc">An <strong>odds ratio</strong> answers a simple question: if you compare two groups, how many times more (or less) likely was one to survive? <strong>OR = 2.0</strong> means twice the odds. <strong>OR = 0.5</strong> means half. The dashed line at <strong>1.0</strong> is "no effect at all". Error bars show the 95% range we're confident the true value lies in.</div>
    </div>
    <div class="chart-card">
      <h3>How much each factor boosted (or shrank) the odds of survival</h3>
      <div class="chart-note">Log scale &mdash; each gridline is 10&times;. Green bars are above 1 (helped survival), red bars below 1 (hurt survival). Bars to the right of <strong>1</strong> = better odds; to the left = worse odds.</div>
      <div class="chart-wrapper tall"><canvas id="oddsChart"></canvas></div>
      <div class="takeaway"><strong>Takeaway:</strong> Being a woman gave you roughly <strong>{next((o["odds_ratio"] for o in data["odds_ratios"] if o["label"]=="Female vs Male"), 0):.0f}&times; the survival odds</strong> of a man. Being in 3rd class cut your odds to about <strong>{next((o["odds_ratio"] for o in data["odds_ratios"] if o["label"]=="3rd Class vs 1st/2nd"), 0):.2f}&times;</strong> &mdash; roughly a third of the rest. These are the two biggest levers; everything else is much smaller.</div>
    </div>
    <div class="card">
      <table>
        <thead><tr><th>Contrast</th><th>Odds Ratio</th><th>95% CI</th><th>Exposed</th><th>Unexposed</th><th>Lift</th><th>p</th></tr></thead>
        <tbody>
        {odds_html}
        </tbody>
      </table>
    </div>
  </section>

  <section id="survival">
    <div class="section-head">
      <h2><span class="icon">\U0001F49A</span>Survival rate by group</h2>
      <div class="desc">The percentage of each group that made it out alive. The little vertical lines on each bar are <strong>95% confidence intervals</strong> — they show how certain we are about the number. When two bars' intervals don't overlap, the difference between them is almost certainly real, not a fluke.</div>
    </div>
    <div class="charts-grid">
      <div class="chart-card"><h3>Overall Survival</h3>
        <div class="chart-wrapper"><canvas id="survivalChart"></canvas></div>
        <div class="takeaway"><strong>{data['survival_rate']['survived_count']} of {data['passenger_stats']['total']:,}</strong> passengers survived. Roughly 1 in 3.</div></div>
      <div class="chart-card"><h3>By Sex</h3>
        <div class="chart-note">Hover for sample size and 95% CI.</div>
        <div class="chart-wrapper"><canvas id="sexChart"></canvas></div>
        <div class="takeaway">Women survived at <strong>{female_rate:.0f}%</strong>, men at <strong>{male_rate:.0f}%</strong>. That's the single biggest gap in the entire dataset.</div></div>
      <div class="chart-card"><h3>By Passenger Class</h3>
        <div class="chart-wrapper"><canvas id="pclassChart"></canvas></div>
        <div class="takeaway">A 1st-class ticket gave you over <strong>2× the survival rate</strong> of a 3rd-class one (62% vs 25%) — cabins on upper decks, closer to the lifeboats.</div></div>
      <div class="chart-card"><h3>By Embarkation Port</h3>
        <div class="chart-wrapper"><canvas id="embarkedChart"></canvas></div>
        <div class="takeaway">Cherbourg looks better at a glance, but most of that is just because more 1st-class passengers boarded there. Port itself isn't really doing the work.</div></div>
    </div>
  </section>

  <section id="joint">
    <div class="section-head">
      <h2><span class="icon">\U0001F9E9</span>Class &times; Sex: what happens when you combine them</h2>
      <div class="desc">Sex and class don't add up — they <strong>multiply</strong>. Knowing just one of them gives you a guess; knowing both gives you a near-certain prediction. Each cell shows the survival rate for that exact combination and how many people were in it (n).</div>
    </div>
    <div class="card">
      <div id="jointHeatmap" class="heatmap"></div>
      <div class="takeaway"><strong>The most extreme contrast in the dataset:</strong> a 1st-class woman had a <strong>96.5%</strong> chance of survival. A 3rd-class man had a <strong>15.2%</strong> chance. Same ship, same iceberg — an 81-percentage-point gap based purely on what ticket you held and what sex you were.</div>
    </div>
  </section>

  <section id="demographics">
    <div class="section-head">
      <h2><span class="icon">\U0001F465</span>Demographics: who else was favored?</h2>
      <div class="desc">Beyond sex and class, four more attributes shifted the odds a little: <strong>age</strong> (children first), <strong>fare paid</strong> (mostly a proxy for class), <strong>title</strong> (encodes sex + age + status), and <strong>family size</strong> (a sweet spot at 2–4).</div>
    </div>
    <div class="charts-grid">
      <div class="chart-card"><h3>By Age Group</h3>
        <div class="chart-wrapper"><canvas id="ageChart"></canvas></div>
        <div class="takeaway">Youngest passengers fared best — clear evidence the "children first" protocol was real. Survival drops steadily with age.</div></div>
      <div class="chart-card"><h3>By Fare Range</h3>
        <div class="chart-wrapper"><canvas id="fareChart"></canvas></div>
        <div class="takeaway">Higher fares paid = higher survival. But this is largely just class repackaged: a 1st-class ticket cost more <em>and</em> got you onto a higher deck.</div></div>
      <div class="chart-card"><h3>By Title</h3>
        <div class="chart-wrapper"><canvas id="titleChart"></canvas></div>
        <div class="takeaway">"Master" (young boys) and "Mrs" (married women) had the best odds. "Mr" (adult men) had by far the worst at 16% — the same story sex+age tells, just labeled differently.</div></div>
      <div class="chart-card"><h3>By Family Size</h3>
        <div class="chart-wrapper"><canvas id="familyChart"></canvas></div>
        <div class="takeaway">Sweet spot at <strong>2–4 family members</strong>. Solo travelers and very large families (5+) both did worse — possibly because mid-size families coordinated boarding lifeboats together.</div></div>
    </div>
  </section>

  <section id="distributions">
    <div class="section-head">
      <h2><span class="icon">\U0001F4C8</span>Distributions: what did the passengers actually look like?</h2>
      <div class="desc">Background on the population itself — without context for who was on the ship, the survival numbers above are hard to interpret. Most passengers were in their 20s–30s, and the fare distribution is heavily skewed: most paid a little, a few paid a lot. We clip the top 1% of fares for readability.</div>
    </div>
    <div class="charts-grid">
      <div class="chart-card"><h3>Age Distribution</h3>
        <div class="chart-wrapper"><canvas id="ageDistChart"></canvas></div></div>
      <div class="chart-card"><h3>Fare Distribution (≤0.99 quantile)</h3>
        <div class="chart-wrapper"><canvas id="fareDistChart"></canvas></div></div>
    </div>
  </section>

  <section id="lifeboat">
    <div class="section-head">
      <h2><span class="icon">\U0001F6F6</span>Lifeboat reality check — the proximate cause</h2>
      <div class="desc">Everything above is about who was <em>likely</em> to survive. This is about the mechanism: <strong>did you get onto a lifeboat?</strong> The dataset records lifeboat numbers for confirmed boat occupants. The numbers below explain why sex and class mattered: they determined who got onto a boat.</div>
    </div>
    <div class="compare">
      <div class="compare-card">
        <h4>Recorded on a Lifeboat</h4>
        <div class="compare-row"><span class="name">Passengers</span><span class="pct mid">{boat_n}</span></div>
        <div class="compare-row"><span class="name">Survived</span><span class="pct high">{boat_s} ({boat_rate:.1f}%)</span></div>
        <div class="compare-delta">95% CI: [{boat_lo:.1f}%, {boat_hi:.1f}%]</div>
      </div>
      <div class="compare-card">
        <h4>No Lifeboat Record</h4>
        <div class="compare-row"><span class="name">Passengers</span><span class="pct mid">{no_boat_n}</span></div>
        <div class="compare-row"><span class="name">Survived</span><span class="pct low">{no_boat_s} ({no_boat_rate:.1f}%)</span></div>
        <div class="compare-delta">95% CI: [{no_boat_lo:.1f}%, {no_boat_hi:.1f}%]</div>
      </div>
      <div class="compare-card">
        <h4>Lift</h4>
        <div class="compare-row"><span class="name">Δ Survival</span><span class="pct high">+{boat_rate - no_boat_rate:.1f}pp</span></div>
        <div class="compare-row"><span class="name">Coverage</span><span class="pct mid">{boat_n/(boat_n+no_boat_n)*100:.1f}%</span></div>
        <div class="compare-delta">Lifeboat record is near-deterministic of survival, as expected.</div>
      </div>
    </div>
  </section>

  <section id="tests">
    <div class="section-head">
      <h2><span class="icon">\U0001F9EA</span>Could these patterns just be random luck?</h2>
      <div class="desc">Statistical tests answer one question: <strong>how likely is it that these patterns appeared by chance?</strong> The "p-value" is the probability of seeing the observed difference if there were really no underlying effect. <strong>p &lt; 0.05</strong> is the standard threshold for "probably not coincidence". <strong>p &lt; 2e-16</strong> means it's effectively impossible to be a fluke.</div>
    </div>
    <div class="card">
      <table>
        <thead><tr><th>Relationship</th><th>Test</th><th>Statistic</th><th>p-value</th><th>Effect</th></tr></thead>
        <tbody>{tests_html}</tbody>
      </table>
    </div>
  </section>

  <section id="correlation">
    <div class="section-head">
      <h2><span class="icon">\U0001F517</span>Which numerical features move together?</h2>
      <div class="desc">Correlation values run from <strong>-1</strong> (move in opposite directions) to <strong>+1</strong> (move together perfectly). <strong>0</strong> = no relationship. Hover any cell to see the exact value. Reds = positive, blues = negative. Most cells here are pale — only a few of the numerical features are strongly related.</div>
    </div>
    <div class="card">
      <div id="corrHeatmap" class="heatmap"></div>
    </div>
  </section>

  <section id="missing">
    <div class="section-head">
      <h2><span class="icon">⚠️</span>How much of the data is actually missing?</h2>
      <div class="desc">Before drawing conclusions, you need to know what's missing. Two columns have gaps: <strong>Occupation</strong> (47% missing — too sparse to use directly) and <strong>Age</strong> (only 3.9% missing, far better than the popular Kaggle subset's 19.9%, so age-based analysis here is reliable).</div>
    </div>
    <div class="card">
      <table>
        <thead><tr><th>Column</th><th>Missing</th><th>Percent</th><th>Severity</th></tr></thead>
        <tbody>{missing_html}</tbody>
      </table>
    </div>
  </section>

  <section id="insights">
    <div class="section-head">
      <h2><span class="icon">\U0001F4A1</span>Key insights, in plain English</h2>
      <div class="desc">The story the numbers tell, summarized.</div>
    </div>
    <div class="insights">
      <ul>
        <li><strong>Sex was by far the biggest factor.</strong> Women survived at {female_rate:.0f}%, men at {male_rate:.0f}% — a {female_rate - male_rate:.0f}-percentage-point gap. In odds terms, women were about <strong>{next((o["odds_ratio"] for o in data["odds_ratios"] if o["label"]=="Female vs Male"), 0):.0f}× more likely to survive</strong> than men.</li>
        <li><strong>Class made things worse if you were already disadvantaged.</strong> Combine 3rd class with male, and survival drops to 15%. Combine 1st class with female and it jumps to 96%. The two factors don't add — they multiply.</li>
        <li><strong>The "children first" rule was real but small.</strong> Kids under 16 had about <strong>{next((o["odds_ratio"] for o in data["odds_ratios"] if o["label"]=="Child (<=16) vs Adult"), 0):.1f}× the survival odds</strong> of adults. A genuine effect, but nothing like sex or class.</li>
        <li><strong>Higher fares helped — but mostly because they bought a 1st-class ticket.</strong> Top fare quartile survived at {next((o["rate_exposed"] for o in data["odds_ratios"] if o["label"]=="Top Fare Quartile vs Rest"), 0):.0f}% vs {next((o["rate_unexposed"] for o in data["odds_ratios"] if o["label"]=="Top Fare Quartile vs Rest"), 0):.0f}% for everyone else. Fare isn't doing independent work; it's class in disguise.</li>
        <li><strong>"Where you boarded" is a red herring.</strong> Cherbourg embarkees had a higher survival rate, but that's because lots of 1st-class passengers happened to board there. Once you control for class, port effects mostly disappear.</li>
        <li><strong>Family size has a sweet spot at 2–4.</strong> Solo travelers and 5+ families both did worse. Mid-sized families may have coordinated boarding together; solo passengers may have lacked someone advocating for them.</li>
        <li><strong>Lifeboat access is the actual mechanism.</strong> 98.6% of passengers with a recorded boat survived; 2.6% without one did. Everything else — sex, class, age — was really just predicting <em>who got onto a boat</em>.</li>
        <li><strong>The data is unusually complete.</strong> Only {data['passenger_stats']['missing_ages']} ages ({data['passenger_stats']['missing_ages'] / data['passenger_stats']['total'] * 100:.1f}%) are missing here vs ~20% in the well-known Kaggle subset, so age-based analyses are reliable.</li>
      </ul>
    </div>
  </section>

  <div class="md-divider">
    <div class="md-divider-label">\U0001F4D6 Full Analyst Report</div>
    <div class="md-divider-desc">The same narrative that ships as DOCX + PDF, rendered inline. Sections below are scroll-spy targets and link to the dashboard sub-sections above when they cover the same ground.</div>
  </div>

__REPORT_SECTIONS__

  <div class="footer">
    <div style="margin-bottom:0.5rem;">Prepared by <strong style="color:var(--accent);">Aneek Hait</strong> &middot;
      <a href="https://aneekhait.github.io" target="_blank" rel="noopener" style="color:var(--accent);text-decoration:none;">aneekhait.github.io</a>
    </div>
    <div>
      Dataset: <a href="https://hbiostat.org/data/repo/titanic5.csv" target="_blank" rel="noopener" style="color:var(--accent);text-decoration:none;">titanic5</a>
      (Encyclopedia Titanica / Vanderbilt Biostatistics) &middot; {data['passenger_stats']['total']:,} passengers, {data['shape'][1]} features
    </div>
  </div>
</main>

<script>
const D = {js_data};
const DL = typeof ChartDataLabels !== 'undefined' ? ChartDataLabels : null;
if (DL) Chart.register(DL);
const DL_PLUGIN = DL ? [DL] : [];

function themeColors() {{
  // .light-theme is applied to <body>, so CSS variables must be read from
  // body (not documentElement) to pick up theme-specific overrides.
  const s = getComputedStyle(document.body);
  return {{
    text: s.getPropertyValue('--text-secondary').trim(),
    textStrong: s.getPropertyValue('--text-primary').trim(),
    textMuted: s.getPropertyValue('--text-muted').trim(),
    border: s.getPropertyValue('--border').trim(),
    bg: s.getPropertyValue('--bg-primary').trim(),
    elev: s.getPropertyValue('--bg-elev').trim(),
    accent: s.getPropertyValue('--accent').trim(),
    green: s.getPropertyValue('--green').trim(),
    red: s.getPropertyValue('--red').trim(),
  }};
}}

function baseOpts() {{
  const c = themeColors();
  return {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{
      legend: {{ labels: {{ color: c.text, font: {{ size: 11, weight: '500' }} }} }},
      tooltip: {{ backgroundColor: c.elev, titleColor: c.textStrong, bodyColor: c.text,
        borderColor: c.border, borderWidth: 1, padding: 10, cornerRadius: 6 }},
    }},
    scales: {{
      x: {{ ticks: {{ color: c.text, font: {{ size: 10 }} }}, grid: {{ color: c.border, drawBorder: false }} }},
      y: {{ ticks: {{ color: c.text, font: {{ size: 10 }} }}, grid: {{ color: c.border, drawBorder: false }} }},
    }},
  }};
}}

function pctYScale() {{
  return {{ max: 100, ticks: {{ callback: v => v + '%' }}, title: {{ display: true, text: 'Survival Rate (%)', color: themeColors().text, font: {{ weight: '600' }} }} }};
}}

const errorBarPlugin = {{
  id: 'errorBars',
  afterDatasetsDraw(chart) {{
    const {{ ctx, chartArea, scales: {{ y }}, data }} = chart;
    const ds = data.datasets[0];
    if (!ds || !ds.errorBars) return;
    ctx.save();
    ctx.strokeStyle = themeColors().textStrong;
    ctx.lineWidth = 1.5;
    chart.getDatasetMeta(0).data.forEach((bar, i) => {{
      const eb = ds.errorBars[i];
      if (!eb) return;
      const x = bar.x;
      const yLo = y.getPixelForValue(eb.low);
      const yHi = y.getPixelForValue(eb.high);
      ctx.beginPath();
      ctx.moveTo(x, yLo); ctx.lineTo(x, yHi);
      ctx.moveTo(x - 6, yLo); ctx.lineTo(x + 6, yLo);
      ctx.moveTo(x - 6, yHi); ctx.lineTo(x + 6, yHi);
      ctx.stroke();
    }});
    ctx.restore();
  }}
}};
Chart.register(errorBarPlugin);

new Chart(document.getElementById('survivalChart'), {{
  plugins: DL_PLUGIN,
  type: 'doughnut',
  data: {{ labels: ['Perished', 'Survived'],
    datasets: [{{ data: [D.survival_rate.perished_count, D.survival_rate.survived_count],
      backgroundColor: ['#f87171', '#34d399'], borderWidth: 3, borderColor: themeColors().bg }}] }},
  options: {{ responsive: true, maintainAspectRatio: false, cutout: '68%',
    plugins: {{
      legend: {{ position: 'bottom', labels: {{ color: themeColors().text, font: {{ size: 11, weight: '500' }} }} }},
      tooltip: {{
        backgroundColor: themeColors().elev, titleColor: themeColors().textStrong,
        bodyColor: themeColors().text, borderColor: themeColors().border, borderWidth: 1,
        padding: 10, cornerRadius: 6,
        callbacks: {{
          label: ctx => {{
            const v = ctx.raw;
            const pct = (v / D.passenger_stats.total * 100).toFixed(1);
            return ` ${{ctx.label}}: ${{v.toLocaleString()}} passengers (${{pct}}%)`;
          }}
        }}
      }},
      datalabels: {{ color: '#fff', font: {{ weight: 'bold', size: 13 }},
        formatter: v => (v / D.passenger_stats.total * 100).toFixed(1) + '%' }}
    }}
  }}
}});

function renderRateChart(canvasId, items, levelKey, levelLabels, colors) {{
  const labels = items.map(r => levelLabels ? levelLabels[r[levelKey]] || r[levelKey] : r[levelKey]);
  const values = items.map(r => r.rate);
  const errors = items.map(r => ({{ low: r.ci_low, high: r.ci_high }}));
  new Chart(document.getElementById(canvasId), {{
    plugins: DL_PLUGIN,
    type: 'bar',
    data: {{ labels, datasets: [{{ label: 'Survival Rate (%)', data: values, errorBars: errors,
      backgroundColor: colors, borderRadius: 6, barPercentage: 0.6 }}] }},
    options: {{
      ...baseOpts(),
      plugins: {{
        ...baseOpts().plugins, legend: {{ display: false }},
        tooltip: {{ ...baseOpts().plugins.tooltip,
          callbacks: {{ afterLabel: ctx => `n = ${{items[ctx.dataIndex].n}} | 95% CI [${{items[ctx.dataIndex].ci_low.toFixed(1)}}, ${{items[ctx.dataIndex].ci_high.toFixed(1)}}]` }} }},
        datalabels: {{ anchor: 'end', align: 'end', offset: 8, color: () => themeColors().textStrong,
          font: {{ weight: '700', size: 11 }}, formatter: v => v.toFixed(1) + '%' }}
      }},
      scales: {{ ...baseOpts().scales, y: {{ ...baseOpts().scales.y, ...pctYScale() }} }}
    }}
  }});
}}

renderRateChart('sexChart', D.sex_ci, 'level', {{ female: 'Female', male: 'Male' }}, ['#f472b6', '#60a5fa']);
renderRateChart('pclassChart', D.pclass_ci, 'level', {{ 1: '1st Class', 2: '2nd Class', 3: '3rd Class' }}, ['#34d399', '#fbbf24', '#f87171']);
renderRateChart('embarkedChart', D.embarked_ci, 'level', {{ C: 'Cherbourg', Q: 'Queenstown', S: 'Southampton', B: 'Belfast' }}, ['#34d399', '#fbbf24', '#f87171', '#a78bfa']);

function binTooltipCallbacks(records) {{
  return {{
    label: ctx => {{
      const r = records[ctx.dataIndex];
      const n = (r && (r.passengers != null)) ? r.passengers : null;
      if (ctx.raw == null) return ` No data (n=${{n != null ? n : 0}})`;
      return ` Survival rate: ${{ctx.raw.toFixed(1)}}% (n=${{n != null ? n : '?'}})`;
    }}
  }};
}}

const ageLabels = D.survival_by_age.map(r => r.Age);
new Chart(document.getElementById('ageChart'), {{
  plugins: DL_PLUGIN, type: 'bar',
  data: {{ labels: ageLabels, datasets: [{{ data: D.survival_by_age.map(r => r.survival_rate),
    backgroundColor: ['#34d399', '#60a5fa', '#a78bfa', '#f59e0b', '#f87171'],
    borderRadius: 6, barPercentage: 0.7 }}] }},
  options: {{ ...baseOpts(), plugins: {{ ...baseOpts().plugins, legend: {{ display: false }},
    tooltip: {{ ...baseOpts().plugins.tooltip, callbacks: binTooltipCallbacks(D.survival_by_age) }},
    datalabels: {{ anchor: 'end', align: 'end', offset: 4, color: () => themeColors().textStrong,
      font: {{ weight: '700', size: 11 }},
      formatter: (v, ctx) => v == null ? 'N/A' : (v > 0 ? v.toFixed(1) + '%' : '') }} }},
    scales: {{ ...baseOpts().scales, y: {{ ...baseOpts().scales.y, ...pctYScale() }} }} }}
}});

const fareLabels = D.survival_by_fare.map(r => r.Fare);
new Chart(document.getElementById('fareChart'), {{
  plugins: DL_PLUGIN, type: 'bar',
  data: {{ labels: fareLabels, datasets: [{{ data: D.survival_by_fare.map(r => r.survival_rate),
    backgroundColor: ['#f87171', '#f59e0b', '#a78bfa', '#60a5fa', '#34d399'],
    borderRadius: 6, barPercentage: 0.7 }}] }},
  options: {{ ...baseOpts(), plugins: {{ ...baseOpts().plugins, legend: {{ display: false }},
    tooltip: {{ ...baseOpts().plugins.tooltip, callbacks: binTooltipCallbacks(D.survival_by_fare) }},
    datalabels: {{ anchor: 'end', align: 'end', offset: 4, color: () => themeColors().textStrong,
      font: {{ weight: '700', size: 11 }},
      formatter: (v, ctx) => v == null ? 'N/A' : (v > 0 ? v.toFixed(1) + '%' : '') }} }},
    scales: {{ ...baseOpts().scales, y: {{ ...baseOpts().scales.y, ...pctYScale() }} }} }}
}});

new Chart(document.getElementById('titleChart'), {{
  plugins: DL_PLUGIN, type: 'bar',
  data: {{ labels: D.title_survival.map(r => r.Title),
    datasets: [{{ data: D.title_survival.map(r => r.survival_rate),
      backgroundColor: D.title_survival.map(r => r.survival_rate > 50 ? '#34d399' : r.survival_rate > 30 ? '#60a5fa' : '#f87171'),
      borderRadius: 6, barPercentage: 0.7 }}] }},
  options: {{ ...baseOpts(), plugins: {{ ...baseOpts().plugins, legend: {{ display: false }},
    tooltip: {{ ...baseOpts().plugins.tooltip,
      callbacks: {{ afterLabel: ctx => `n = ${{D.title_survival[ctx.dataIndex].passengers}}` }} }},
    datalabels: {{ anchor: 'end', align: 'end', offset: 4, color: () => themeColors().textStrong,
      font: {{ weight: '700', size: 11 }}, formatter: v => v.toFixed(1) + '%' }} }},
    scales: {{ ...baseOpts().scales, y: {{ ...baseOpts().scales.y, ...pctYScale() }} }} }}
}});

new Chart(document.getElementById('familyChart'), {{
  plugins: DL_PLUGIN, type: 'bar',
  data: {{ labels: D.family_survival.map(r => String(r.FamilySize)),
    datasets: [{{ data: D.family_survival.map(r => r.survival_rate),
      backgroundColor: D.family_survival.map(r => r.survival_rate > 50 ? '#34d399' : r.survival_rate > 30 ? '#60a5fa' : '#f87171'),
      borderRadius: 6, barPercentage: 0.7 }}] }},
  options: {{ ...baseOpts(), plugins: {{ ...baseOpts().plugins, legend: {{ display: false }},
    tooltip: {{ ...baseOpts().plugins.tooltip,
      callbacks: {{ afterLabel: ctx => `n = ${{D.family_survival[ctx.dataIndex].passengers}}` }} }},
    datalabels: {{ anchor: 'end', align: 'end', offset: 4, color: () => themeColors().textStrong,
      font: {{ weight: '700', size: 11 }}, formatter: v => v.toFixed(1) + '%' }} }},
    scales: {{ ...baseOpts().scales, y: {{ ...baseOpts().scales.y, ...pctYScale() }},
      x: {{ ...baseOpts().scales.x, title: {{ display: true, text: 'Family Size', color: themeColors().text, font: {{ weight: '600' }} }} }} }} }}
}});

new Chart(document.getElementById('ageDistChart'), {{
  type: 'bar',
  data: {{ labels: D.age_dist.map(r => r.bin),
    datasets: [{{ data: D.age_dist.map(r => r.count),
      backgroundColor: 'rgba(96, 165, 250, 0.7)', borderColor: '#60a5fa',
      borderWidth: 1, borderRadius: 3, barPercentage: 1, categoryPercentage: 0.95 }}] }},
  options: {{ ...baseOpts(), plugins: {{ ...baseOpts().plugins, legend: {{ display: false }} }},
    scales: {{ ...baseOpts().scales, y: {{ ...baseOpts().scales.y,
      title: {{ display: true, text: 'Passenger Count', color: themeColors().text, font: {{ weight: '600' }} }} }} }} }}
}});

new Chart(document.getElementById('fareDistChart'), {{
  type: 'bar',
  data: {{ labels: D.fare_dist.map(r => r.bin),
    datasets: [{{ data: D.fare_dist.map(r => r.count),
      backgroundColor: 'rgba(167, 139, 250, 0.7)', borderColor: '#a78bfa',
      borderWidth: 1, borderRadius: 3, barPercentage: 1, categoryPercentage: 0.95 }}] }},
  options: {{ ...baseOpts(), plugins: {{ ...baseOpts().plugins, legend: {{ display: false }} }},
    scales: {{ ...baseOpts().scales, y: {{ ...baseOpts().scales.y,
      title: {{ display: true, text: 'Passenger Count', color: themeColors().text, font: {{ weight: '600' }} }} }} }} }}
}});

const powerColors = D.effect_sizes.map(e => {{
  const s = e.strength;
  return s === 'Large' ? '#34d399' : s === 'Medium' ? '#60a5fa' : s === 'Small' ? '#fbbf24' : '#94a3b8';
}});
new Chart(document.getElementById('powerChart'), {{
  plugins: DL_PLUGIN, type: 'bar',
  data: {{ labels: D.effect_sizes.map(e => `${{e.feature}} (${{e.metric}})`),
    datasets: [{{ data: D.effect_sizes.map(e => Math.abs(e.effect_size)),
      backgroundColor: powerColors, borderRadius: 6 }}] }},
  options: {{ ...baseOpts(), indexAxis: 'y',
    plugins: {{ ...baseOpts().plugins, legend: {{ display: false }},
      tooltip: {{ ...baseOpts().plugins.tooltip,
        callbacks: {{ label: ctx => `${{D.effect_sizes[ctx.dataIndex].metric}}: ${{D.effect_sizes[ctx.dataIndex].effect_size}} (${{D.effect_sizes[ctx.dataIndex].strength}})` }} }},
      datalabels: {{ anchor: 'end', align: 'end', offset: 4, color: () => themeColors().textStrong,
        font: {{ weight: '700', size: 11 }},
        formatter: (v, ctx) => `${{D.effect_sizes[ctx.dataIndex].effect_size}} (${{D.effect_sizes[ctx.dataIndex].strength}})` }} }},
    scales: {{ x: {{ ...baseOpts().scales.x, max: 0.6,
      title: {{ display: true, text: 'Effect Size (|r| or Cramer\\'s V)', color: themeColors().text, font: {{ weight: '600' }} }} }},
      y: {{ ...baseOpts().scales.y }} }} }}
}});

const oddsData = D.odds_ratios;
new Chart(document.getElementById('oddsChart'), {{
  plugins: DL_PLUGIN, type: 'bar',
  data: {{ labels: oddsData.map(o => o.label),
    datasets: [{{ data: oddsData.map(o => o.odds_ratio),
      errorBars: oddsData.map(o => ({{ low: o.ci_low, high: o.ci_high }})),
      backgroundColor: oddsData.map(o => o.odds_ratio > 1 ? '#34d399' : '#f87171'),
      borderRadius: 6, barPercentage: 0.7 }}] }},
  options: {{ ...baseOpts(), indexAxis: 'y',
    plugins: {{ ...baseOpts().plugins, legend: {{ display: false }},
      tooltip: {{ ...baseOpts().plugins.tooltip,
        callbacks: {{ afterLabel: ctx => {{
          const o = oddsData[ctx.dataIndex];
          return `95% CI [${{o.ci_low}}, ${{o.ci_high}}] | exposed ${{o.rate_exposed}}% vs ${{o.rate_unexposed}}%`;
        }} }} }},
      datalabels: {{ anchor: 'end', align: 'end', offset: 4, color: () => themeColors().textStrong,
        font: {{ weight: '700', size: 11 }}, formatter: v => v.toFixed(2) + 'x' }} }},
    scales: {{ x: {{ ...baseOpts().scales.x, type: 'logarithmic',
      title: {{ display: true, text: 'Odds Ratio (log scale)', color: themeColors().text, font: {{ weight: '600' }} }},
      ticks: {{ color: themeColors().text, callback: v => v }} }},
      y: {{ ...baseOpts().scales.y }} }} }}
}});

// RdYlGn diverging palette (matches matplotlib/seaborn). Anchors: 0% red, 50% pale, 100% green.
const RDYLGN_STOPS = [
  [0.00, [165, 0, 38]],     [0.10, [215, 48, 39]],   [0.20, [244, 109, 67]],
  [0.30, [253, 174, 97]],   [0.40, [254, 224, 139]], [0.50, [255, 255, 191]],
  [0.60, [217, 239, 139]],  [0.70, [166, 217, 106]], [0.80, [102, 189, 99]],
  [0.90, [26, 152, 80]],    [1.00, [0, 104, 55]],
];
function rdylgn(t) {{
  t = Math.max(0, Math.min(1, t));
  for (let i = 0; i < RDYLGN_STOPS.length - 1; i++) {{
    const [t0, c0] = RDYLGN_STOPS[i], [t1, c1] = RDYLGN_STOPS[i + 1];
    if (t <= t1) {{
      const k = (t - t0) / (t1 - t0);
      return c0.map((v, j) => Math.round(v + k * (c1[j] - v)));
    }}
  }}
  return RDYLGN_STOPS[RDYLGN_STOPS.length - 1][1];
}}
function relLuminance([r, g, b]) {{
  const f = v => {{ v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); }};
  return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b);
}}

const joint = D.joint_class_sex;
const classes = [...new Set(joint.map(r => r.Pclass))].sort();
const sexes = ['female', 'male'];
const heatEl = document.getElementById('jointHeatmap');

window.renderJointHeatmap = function() {{
  heatEl.style.gridTemplateColumns = `120px repeat(${{sexes.length}}, 1fr)`;
  heatEl.innerHTML = '';
  const corner = document.createElement('div'); corner.className = 'heatmap-label'; corner.textContent = 'Class \\\\ Sex';
  heatEl.appendChild(corner);
  sexes.forEach(s => {{ const d = document.createElement('div'); d.className = 'heatmap-label'; d.textContent = s[0].toUpperCase() + s.slice(1); heatEl.appendChild(d); }});
  classes.forEach(c => {{
    const lab = document.createElement('div'); lab.className = 'heatmap-label';
    lab.textContent = `Class ${{c}}`; heatEl.appendChild(lab);
    sexes.forEach(s => {{
      const cell = joint.find(r => r.Pclass === c && r.Sex === s);
      const div = document.createElement('div'); div.className = 'heatmap-cell';
      if (cell && Number.isFinite(cell.rate)) {{
        const v = cell.rate;
        const rgb = rdylgn(v / 100);
        div.style.background = `rgb(${{rgb[0]}},${{rgb[1]}},${{rgb[2]}})`;
        div.style.color = relLuminance(rgb) > 0.55 ? '#1f2937' : '#ffffff';
        div.innerHTML = `${{v.toFixed(1)}}%<div class="sub">n=${{cell.count}}</div>`;
        div.title = `Class ${{c}}, ${{s}}: ${{v}}% survived (n=${{cell.count}})`;
      }} else {{
        div.style.background = themeColors().elev;
        div.style.color = themeColors().textMuted || themeColors().text;
        div.innerHTML = 'N/A<div class="sub">no data</div>';
      }}
      heatEl.appendChild(div);
    }});
  }});
}};
window.renderJointHeatmap();

const corrFeatures = D.corr_features;
const corrMatrix = D.corr_matrix;
const corrEl = document.getElementById('corrHeatmap');

window.renderCorrHeatmap = function() {{
  corrEl.style.gridTemplateColumns = `100px repeat(${{corrFeatures.length}}, 1fr)`;
  corrEl.innerHTML = '';
  const corner2 = document.createElement('div'); corner2.className = 'heatmap-label'; corner2.textContent = ''; corrEl.appendChild(corner2);
  corrFeatures.forEach(f => {{ const d = document.createElement('div'); d.className = 'heatmap-label'; d.textContent = f; corrEl.appendChild(d); }});
  corrMatrix.forEach((row, i) => {{
    const lab = document.createElement('div'); lab.className = 'heatmap-label'; lab.textContent = corrFeatures[i]; corrEl.appendChild(lab);
    row.forEach((val, j) => {{
      const cell = document.createElement('div'); cell.className = 'heatmap-cell';
      const t = Math.max(-1, Math.min(1, val));
      let r, g, b;
      if (t >= 0) {{ r = Math.round(255 - t * (255 - 16)); g = Math.round(255 - t * (255 - 124)); b = Math.round(255 - t * (255 - 181)); }}
      else {{ r = Math.round(255 + t * (255 - 220)); g = Math.round(255 + t * (255 - 38)); b = Math.round(255 + t * (255 - 38)); }}
      cell.style.background = `rgb(${{r}},${{g}},${{b}})`;
      cell.style.color = Math.abs(t) > 0.4 ? '#ffffff' : '#1f2937';
      cell.textContent = val.toFixed(2);
      cell.title = `${{corrFeatures[i]}} × ${{corrFeatures[j]}}: ${{val}}`;
      corrEl.appendChild(cell);
    }});
  }});
}};
window.renderCorrHeatmap();

function applyChartTheme() {{
  const c = themeColors();
  for (const id in Chart.instances) {{
    const ch = Chart.instances[id];
    const o = ch.options;
    if (o.scales) {{
      for (const k of ['x', 'y']) {{
        if (!o.scales[k]) continue;
        if (o.scales[k].ticks) o.scales[k].ticks.color = c.text;
        if (o.scales[k].grid) o.scales[k].grid.color = c.border;
        if (o.scales[k].title) o.scales[k].title.color = c.text;
      }}
    }}
    if (o.plugins) {{
      if (o.plugins.legend && o.plugins.legend.labels)
        o.plugins.legend.labels.color = c.text;
      if (o.plugins.tooltip) {{
        o.plugins.tooltip.backgroundColor = c.elev;
        o.plugins.tooltip.titleColor = c.textStrong;
        o.plugins.tooltip.bodyColor = c.text;
        o.plugins.tooltip.borderColor = c.border;
      }}
      if (o.plugins.datalabels) o.plugins.datalabels.color = c.textStrong;
    }}
    // Doughnut needs the dividing border to follow the page background
    if (ch.config.type === 'doughnut' && ch.data.datasets[0]) {{
      ch.data.datasets[0].borderColor = c.bg;
    }}
    ch.update('none');
  }}
  // Repaint the joint and correlation heatmaps (their cells are static HTML)
  if (typeof window.renderJointHeatmap === 'function') window.renderJointHeatmap();
  if (typeof window.renderCorrHeatmap === 'function') window.renderCorrHeatmap();
}}

function toggleTheme() {{
  document.body.classList.toggle('light-theme');
  const btn = document.getElementById('themeBtn');
  btn.textContent = document.body.classList.contains('light-theme') ? '☀ Theme' : '☾ Theme';
  applyChartTheme();
}}

const observer = new IntersectionObserver((entries) => {{
  entries.forEach(entry => {{
    if (entry.isIntersecting) {{
      const id = entry.target.id;
      document.querySelectorAll('.sidebar-nav a').forEach(a => {{
        a.classList.toggle('active', a.getAttribute('href') === '#' + id);
      }});
    }}
  }});
}}, {{ threshold: 0.2, rootMargin: '-15% 0px -50% 0px' }});
document.querySelectorAll('section[id]').forEach(s => observer.observe(s));

document.querySelectorAll('.sidebar-nav a').forEach(link => {{
  link.addEventListener('click', () => {{
    if (window.innerWidth <= 768) document.querySelector('.sidebar').classList.remove('open');
  }});
}});
</script>
</body>
</html>"""

    html = html.replace("__REPORT_NAV__", report_nav_html)
    html = html.replace("__REPORT_SECTIONS__", report_sections_html)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Dashboard generated at {output_path}")


if __name__ == "__main__":
    df = load_titanic()
    output = Path(__file__).parent / "index.html"
    generate_dashboard(df, output)
