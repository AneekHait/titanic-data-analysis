#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import math
import re
import pandas as pd
from collections import Counter

from src.data.loader import load_titanic
from src.analysis.eda import (
    data_overview,
    missing_summary,
    survival_rate,
    survival_by_categorical,
    survival_by_numerical,
    correlation_analysis,
)

def safe_num(val):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return 0
    return round(val, 1)

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
    titles = titles.map(lambda t: title_map.get(t, "Other"))
    return titles

def compute_family_size(df):
    return df["SibSp"] + df["Parch"] + 1

def generate_dashboard(df, output_path: Path):
    overview = data_overview(df)
    missing = missing_summary(df)
    sr = survival_rate(df)

    survival_by_sex = survival_by_categorical(df, "Sex")
    survival_by_pclass = survival_by_categorical(df, "Pclass")
    survival_by_embarked = survival_by_categorical(df, "Embarked")

    survival_by_age = survival_by_numerical(df, "Age")
    survival_by_age.index = survival_by_age.index.astype(str)

    survival_by_fare = survival_by_numerical(df, "Fare")
    survival_by_fare.index = survival_by_fare.index.astype(str)

    corr = correlation_analysis(df)

    titles = extract_titles(df)
    title_survival = df.copy()
    title_survival["Title"] = titles
    title_survival_data = title_survival.groupby("Title")["Survived"].agg(["count", "mean"]).rename(columns={"count": "passengers", "mean": "survival_rate"}).assign(survival_rate=lambda x: x["survival_rate"].mul(100)).round(2).reset_index().to_dict(orient="records")

    family_sizes = compute_family_size(df)
    family_survival = df.copy()
    family_survival["FamilySize"] = family_sizes
    family_survival_data = family_survival.groupby("FamilySize")["Survived"].agg(["count", "mean"]).rename(columns={"count": "passengers", "mean": "survival_rate"}).assign(survival_rate=lambda x: x["survival_rate"].mul(100)).round(2).reset_index().to_dict(orient="records")

    age_bins = pd.cut(df["Age"], bins=10, precision=0)
    age_dist = df.groupby(age_bins, observed=False).size().reset_index(name="count")
    age_dist.columns = ["bin", "count"]
    age_dist["bin"] = age_dist["bin"].astype(str)
    age_dist_data = age_dist.to_dict(orient="records")

    fare_bins = pd.cut(df["Fare"], bins=10, precision=0)
    fare_dist = df.groupby(fare_bins, observed=False).size().reset_index(name="count")
    fare_dist.columns = ["bin", "count"]
    fare_dist["bin"] = fare_dist["bin"].astype(str)
    fare_dist_data = fare_dist.to_dict(orient="records")

    survived_age = df[df["Survived"] == 1]["Age"].dropna()
    perished_age = df[df["Survived"] == 0]["Age"].dropna()
    survived_fare = df[df["Survived"] == 1]["Fare"].dropna()
    perished_fare = df[df["Survived"] == 0]["Fare"].dropna()

    corr_features = ["Age", "Fare", "SibSp", "Parch", "Survived"]
    corr_matrix = [[round(corr[feat][col], 3) for col in corr_features] for feat in corr_features]

    data = {
        "shape": overview["shape"],
        "missing": missing.reset_index().rename(columns={"index": "column"}).to_dict(orient="records"),
        "survival_rate": {
            "perished": float(sr.get(0, 0)),
            "survived": float(sr.get(1, 0)),
            "perished_count": int((df["Survived"] == 0).sum()),
            "survived_count": int((df["Survived"] == 1).sum()),
        },
        "survival_by_sex": survival_by_sex.reset_index().to_dict(orient="records"),
        "survival_by_pclass": survival_by_pclass.reset_index().to_dict(orient="records"),
        "survival_by_embarked": survival_by_embarked.reset_index().to_dict(orient="records"),
        "survival_by_age": survival_by_age.reset_index().to_dict(orient="records"),
        "survival_by_fare": survival_by_fare.reset_index().to_dict(orient="records"),
        "title_survival": title_survival_data,
        "family_survival": family_survival_data,
        "age_dist": age_dist_data,
        "fare_dist": fare_dist_data,
        "corr_features": corr_features,
        "corr_matrix": corr_matrix,
        "age_stats": {
            "mean": round(df["Age"].mean(), 1),
            "median": round(df["Age"].median(), 1),
            "min": round(df["Age"].min(), 1),
            "max": round(df["Age"].max(), 1),
            "std": round(df["Age"].std(), 1),
            "missing": int(df["Age"].isnull().sum()),
        },
        "fare_stats": {
            "mean": round(df["Fare"].mean(), 2),
            "median": round(df["Fare"].median(), 2),
            "min": round(df["Fare"].min(), 2),
            "max": round(df["Fare"].max(), 2),
            "std": round(df["Fare"].std(), 2),
        },
        "passenger_stats": {
            "total": len(df),
            "avg_age": round(df["Age"].mean(), 1),
            "avg_fare": round(df["Fare"].mean(), 2),
            "max_siblings": int(df["SibSp"].max()),
            "max_parents": int(df["Parch"].max()),
        },
        "class_distribution": {str(k): int(v) for k, v in df["Pclass"].value_counts().sort_index().items()},
        "embarked_distribution": {str(k): int(v) for k, v in df["Embarked"].value_counts().items()},
        "survived_age_stats": {
            "mean": round(survived_age.mean(), 1) if len(survived_age) > 0 else 0,
            "median": round(survived_age.median(), 1) if len(survived_age) > 0 else 0,
        },
        "perished_age_stats": {
            "mean": round(perished_age.mean(), 1) if len(perished_age) > 0 else 0,
            "median": round(perished_age.median(), 1) if len(perished_age) > 0 else 0,
        },
    }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Titanic EDA Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2.2.0/dist/chartjs-plugin-datalabels.min.js"></script>
    <style>
        :root {{
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-card: #1e293b;
            --bg-hover: #2d3a4f;
            --border: #334155;
            --text-primary: #e2e8f0;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --accent: #60a5fa;
            --accent-purple: #a78bfa;
            --green: #34d399;
            --red: #f87171;
            --yellow: #fbbf24;
            --sidebar-width: 260px;
        }}
        .light-theme {{
            --bg-primary: #f8fafc;
            --bg-secondary: #ffffff;
            --bg-card: #ffffff;
            --bg-hover: #f1f5f9;
            --border: #e2e8f0;
            --text-primary: #1e293b;
            --text-secondary: #475569;
            --text-muted: #94a3b8;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        html {{ scroll-behavior: smooth; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            transition: background 0.3s, color 0.3s;
        }}
        .sidebar {{
            position: fixed;
            top: 0;
            left: 0;
            width: var(--sidebar-width);
            height: 100vh;
            background: var(--bg-secondary);
            border-right: 1px solid var(--border);
            padding: 1.5rem;
            overflow-y: auto;
            z-index: 100;
            transition: transform 0.3s;
        }}
        .sidebar-logo {{
            font-size: 1.5rem;
            font-weight: 700;
            background: linear-gradient(90deg, var(--accent), var(--accent-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 2rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .sidebar-logo span {{ font-size: 1.75rem; -webkit-text-fill-color: initial; }}
        .sidebar-nav a {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.75rem 1rem;
            color: var(--text-secondary);
            text-decoration: none;
            border-radius: 8px;
            margin-bottom: 0.25rem;
            transition: all 0.2s;
            font-size: 0.95rem;
        }}
        .sidebar-nav a:hover, .sidebar-nav a.active {{
            background: var(--bg-hover);
            color: var(--accent);
        }}
        .sidebar-nav a .icon {{ width: 20px; text-align: center; }}
        .main {{
            margin-left: var(--sidebar-width);
            padding: 2rem;
            min-height: 100vh;
        }}
        .top-bar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border);
        }}
        .top-bar h1 {{ font-size: 1.75rem; font-weight: 700; }}
        .top-bar h1 span {{ color: var(--text-muted); font-weight: 400; font-size: 1rem; }}
        .theme-toggle {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 0.5rem 1rem;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1rem;
            transition: all 0.2s;
        }}
        .theme-toggle:hover {{ border-color: var(--accent); }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .stat-card {{
            background: var(--bg-card);
            border-radius: 12px;
            padding: 1.25rem;
            border: 1px solid var(--border);
            text-align: center;
            transition: transform 0.2s, box-shadow 0.2s;
            position: relative;
            overflow: hidden;
        }}
        .stat-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }}
        .stat-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
        }}
        .stat-card.survived::before {{ background: var(--green); }}
        .stat-card.perished::before {{ background: var(--red); }}
        .stat-card.info::before {{ background: var(--accent); }}
        .stat-card.warning::before {{ background: var(--yellow); }}
        .stat-card .value {{
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }}
        .stat-card .label {{ color: var(--text-secondary); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; }}
        .stat-card.survived .value {{ color: var(--green); }}
        .stat-card.perished .value {{ color: var(--red); }}
        .stat-card.info .value {{ color: var(--accent); }}
        .stat-card.warning .value {{ color: var(--yellow); }}
        .section {{
            background: var(--bg-card);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            border: 1px solid var(--border);
        }}
        .section h2 {{
            font-size: 1.25rem;
            margin-bottom: 1rem;
            color: var(--accent);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}
        .chart-container {{
            background: var(--bg-primary);
            border-radius: 10px;
            padding: 1.25rem;
            border: 1px solid var(--border);
        }}
        .chart-container h3 {{
            text-align: center;
            margin-bottom: 1rem;
            color: var(--text-secondary);
            font-size: 1rem;
        }}
        .chart-wrapper {{ position: relative; height: 280px; }}
        .chart-wrapper.small {{ height: 220px; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }}
        th, td {{
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        th {{
            background: var(--bg-primary);
            color: var(--accent);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.8rem;
            letter-spacing: 0.05em;
        }}
        tr:hover {{ background: var(--bg-hover); }}
        .progress-bar {{
            height: 8px;
            background: var(--border);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 0.5rem;
        }}
        .progress-bar-fill {{
            height: 100%;
            border-radius: 4px;
            transition: width 1s ease;
        }}
        .heatmap {{
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 4px;
            margin-top: 1rem;
        }}
        .heatmap-cell {{
            aspect-ratio: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 6px;
            font-size: 0.85rem;
            font-weight: 600;
            transition: transform 0.2s;
        }}
        .heatmap-cell:hover {{ transform: scale(1.1); }}
        .heatmap-label {{
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.8rem;
            color: var(--text-secondary);
            font-weight: 600;
        }}
        .insights {{
            background: var(--bg-card);
            border-radius: 12px;
            padding: 1.5rem;
            border-left: 4px solid var(--accent);
            margin-bottom: 2rem;
        }}
        .insights h2 {{ color: var(--accent); margin-bottom: 1rem; }}
        .insights ul {{ list-style: none; }}
        .insights li {{
            padding: 0.75rem 0;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: flex-start;
        }}
        .insights li:last-child {{ border-bottom: none; }}
        .insights li::before {{
            content: "▸";
            color: var(--accent);
            margin-right: 0.75rem;
            font-size: 1.2rem;
        }}
        .tag {{
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        .tag-green {{ background: rgba(52, 211, 153, 0.15); color: var(--green); }}
        .tag-red {{ background: rgba(248, 113, 113, 0.15); color: var(--red); }}
        .tag-blue {{ background: rgba(96, 165, 250, 0.15); color: var(--accent); }}
        .tag-yellow {{ background: rgba(251, 191, 36, 0.15); color: var(--yellow); }}
        .footer {{
            text-align: center;
            padding: 2rem;
            color: var(--text-muted);
            font-size: 0.85rem;
            border-top: 1px solid var(--border);
            margin-top: 2rem;
        }}
        .mobile-toggle {{
            display: none;
            position: fixed;
            top: 1rem;
            left: 1rem;
            z-index: 200;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 0.5rem;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1.25rem;
        }}
        @media (max-width: 768px) {{
            .sidebar {{ transform: translateX(-100%); }}
            .sidebar.open {{ transform: translateX(0); }}
            .main {{ margin-left: 0; padding: 1rem; padding-top: 4rem; }}
            .charts-grid {{ grid-template-columns: 1fr; }}
            .stats-grid {{ grid-template-columns: repeat(2, 1fr); }}
            .mobile-toggle {{ display: block; }}
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .animate-in {{
            animation: fadeIn 0.5s ease forwards;
            opacity: 0;
        }}
    </style>
</head>
<body>
    <button class="mobile-toggle" onclick="document.querySelector('.sidebar').classList.toggle('open')">☰</button>

    <aside class="sidebar">
        <div class="sidebar-logo"><span>🚢</span> Titanic EDA</div>
        <nav class="sidebar-nav">
            <a href="#overview"><span class="icon">📊</span> Overview</a>
            <a href="#survival"><span class="icon">💚</span> Survival Analysis</a>
            <a href="#demographics"><span class="icon">👥</span> Demographics</a>
            <a href="#distributions"><span class="icon">📈</span> Distributions</a>
            <a href="#correlation"><span class="icon">🔗</span> Correlations</a>
            <a href="#missing"><span class="icon">⚠️</span> Missing Values</a>
            <a href="#insights"><span class="icon">💡</span> Key Insights</a>
        </nav>
    </aside>

    <main class="main">
        <div class="top-bar">
            <h1>Titanic Dashboard <span>Exploratory Data Analysis</span></h1>
            <button class="theme-toggle" onclick="toggleTheme()">🌙 Theme</button>
        </div>

        <section id="overview" class="animate-in">
            <div class="stats-grid">
                <div class="stat-card info">
                    <div class="value" data-count="{data['passenger_stats']['total']}">0</div>
                    <div class="label">Total Passengers</div>
                </div>
                <div class="stat-card survived">
                    <div class="value" data-count="{data['survival_rate']['survived_count']}">0</div>
                    <div class="label">Survived</div>
                </div>
                <div class="stat-card perished">
                    <div class="value" data-count="{data['survival_rate']['perished_count']}">0</div>
                    <div class="label">Perished</div>
                </div>
                <div class="stat-card info">
                    <div class="value" data-count="{data['age_stats']['mean']}" data-decimals="1">0</div>
                    <div class="label">Avg Age</div>
                </div>
                <div class="stat-card info">
                    <div class="value" data-count="{data['fare_stats']['mean']}" data-decimals="2" data-prefix="$">0</div>
                    <div class="label">Avg Fare</div>
                </div>
                <div class="stat-card warning">
                    <div class="value" data-count="{data['age_stats']['missing']}">0</div>
                    <div class="label">Missing Ages</div>
                </div>
            </div>
        </section>

        <section id="survival" class="animate-in">
            <div class="charts-grid">
                <div class="chart-container">
                    <h3>Overall Survival</h3>
                    <div class="chart-wrapper">
                        <canvas id="survivalChart"></canvas>
                    </div>
                </div>
                <div class="chart-container">
                    <h3>Survival by Sex</h3>
                    <div class="chart-wrapper">
                        <canvas id="sexChart"></canvas>
                    </div>
                </div>
                <div class="chart-container">
                    <h3>Survival by Class</h3>
                    <div class="chart-wrapper">
                        <canvas id="pclassChart"></canvas>
                    </div>
                </div>
                <div class="chart-container">
                    <h3>Survival by Embarked Port</h3>
                    <div class="chart-wrapper">
                        <canvas id="embarkedChart"></canvas>
                    </div>
                </div>
            </div>
        </section>

        <section id="demographics" class="animate-in">
            <div class="charts-grid">
                <div class="chart-container">
                    <h3>Survival by Age Group</h3>
                    <div class="chart-wrapper">
                        <canvas id="ageChart"></canvas>
                    </div>
                </div>
                <div class="chart-container">
                    <h3>Survival by Fare Range</h3>
                    <div class="chart-wrapper">
                        <canvas id="fareChart"></canvas>
                    </div>
                </div>
                <div class="chart-container">
                    <h3>Survival by Title</h3>
                    <div class="chart-wrapper small">
                        <canvas id="titleChart"></canvas>
                    </div>
                </div>
                <div class="chart-container">
                    <h3>Survival by Family Size</h3>
                    <div class="chart-wrapper small">
                        <canvas id="familyChart"></canvas>
                    </div>
                </div>
            </div>
        </section>

        <section id="distributions" class="animate-in">
            <div class="charts-grid">
                <div class="chart-container">
                    <h3>Age Distribution</h3>
                    <div class="chart-wrapper">
                        <canvas id="ageDistChart"></canvas>
                    </div>
                </div>
                <div class="chart-container">
                    <h3>Fare Distribution</h3>
                    <div class="chart-wrapper">
                        <canvas id="fareDistChart"></canvas>
                    </div>
                </div>
            </div>
        </section>

        <section id="correlation" class="animate-in">
            <div class="section">
                <h2>🔗 Correlation Heatmap</h2>
                <div class="heatmap" id="heatmap"></div>
            </div>
        </section>

        <section id="missing" class="animate-in">
            <div class="section">
                <h2>⚠️ Missing Values</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Column</th>
                            <th>Missing</th>
                            <th>Percentage</th>
                            <th>Severity</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(f"""<tr>
                            <td>{r['column']}</td>
                            <td>{r['missing']}</td>
                            <td>{r['percent']:.1f}%</td>
                            <td><span class="tag {'tag-red' if r['percent'] > 50 else 'tag-yellow' if r['percent'] > 10 else 'tag-green'}">{'{:.0f}'.format(r['percent'])}%</span></td>
                        </tr>
                        <tr><td colspan="4"><div class="progress-bar"><div class="progress-bar-fill" style="width:{r['percent']}%;background:{'#f87171' if r['percent'] > 50 else '#fbbf24' if r['percent'] > 10 else '#34d399'}"></div></div></td></tr>""" for r in data['missing'])}
                    </tbody>
                </table>
            </div>
        </section>

        <section id="insights" class="animate-in">
            <div class="insights">
                <h2>💡 Key Insights</h2>
                <ul>
                    <li>Only <strong>{data['survival_rate']['survived']:.1f}%</strong> of passengers survived the Titanic disaster.</li>
                    <li><strong>Gender was the strongest predictor</strong> — women survived at {data['survival_by_sex'][1]['survival_rate']:.1f}% vs men at {data['survival_by_sex'][0]['survival_rate']:.1f}%.</li>
                    <li><strong>Class hierarchy</strong> — 1st class: {data['survival_by_pclass'][0]['survival_rate']:.1f}%, 2nd: {data['survival_by_pclass'][1]['survival_rate']:.1f}%, 3rd: {data['survival_by_pclass'][2]['survival_rate']:.1f}%.</li>
                    <li><strong>Children (0-16)</strong> had a {data['survival_by_age'][0]['survival_rate']:.1f}% survival rate — "women and children first" held true.</li>
                    <li><strong>Higher fare = better survival</strong> — passengers paying $102-$205 survived at {data['survival_by_fare'][1]['survival_rate']:.1f}%.</li>
                    <li><strong>Cherbourg (C)</strong> passengers had the highest survival rate at {data['survival_by_embarked'][0]['survival_rate']:.1f}%.</li>
                    <li><strong>Cabin data is 77.1% missing</strong> — too sparse for analysis without imputation.</li>
                    <li><strong>Fare-survived correlation (0.257)</strong> is the strongest numerical predictor.</li>
                </ul>
            </div>
        </section>

        <div class="footer">
            Generated from Titanic EDA Project | Data: {data['passenger_stats']['total']} passengers, {data['shape'][1]} features | Built with Chart.js
        </div>
    </main>

    <script>
        const chartDefaults = {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{
                legend: {{ labels: {{ color: getComputedStyle(document.documentElement).getPropertyValue('--text-secondary').trim(), font: {{ size: 12 }} }} }}
            }},
            scales: {{
                x: {{ ticks: {{ color: getComputedStyle(document.documentElement).getPropertyValue('--text-secondary').trim() }}, grid: {{ color: getComputedStyle(document.documentElement).getPropertyValue('--border').trim() }} }},
                y: {{ ticks: {{ color: getComputedStyle(document.documentElement).getPropertyValue('--text-secondary').trim() }}, grid: {{ color: getComputedStyle(document.documentElement).getPropertyValue('--border').trim() }} }}
            }}
        }};

        function getThemeColors() {{
            const style = getComputedStyle(document.documentElement);
            return {{
                text: style.getPropertyValue('--text-secondary').trim(),
                border: style.getPropertyValue('--border').trim(),
                bg: style.getPropertyValue('--bg-primary').trim(),
            }};
        }}

        new Chart(document.getElementById('survivalChart'), {{
            type: 'doughnut',
            data: {{
                labels: ['Perished', 'Survived'],
                datasets: [{{
                    data: [{data['survival_rate']['perished_count']}, {data['survival_rate']['survived_count']}],
                    backgroundColor: ['#f87171', '#34d399'],
                    borderWidth: 2,
                    borderColor: getThemeColors().bg,
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                cutout: '65%',
                plugins: {{
                    legend: {{ position: 'bottom', labels: {{ color: getThemeColors().text }} }},
                    tooltip: {{
                        callbacks: {{
                            label: ctx => `{{ctx.label}}: {{ctx.raw}} ({{(ctx.raw / {data['passenger_stats']['total']} * 100).toFixed(1)}}%)`
                        }}
                    }},
                    datalabels: {{
                        color: '#fff',
                        font: {{ weight: 'bold', size: 14 }},
                        formatter: (val) => {{
                            const pct = (val / {data['passenger_stats']['total']} * 100).toFixed(1);
                            return `${{pct}}%\\n(${{val}})`;
                        }},
                    }}
                }}
            }}
        }});

        new Chart(document.getElementById('sexChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps([r['Sex'] for r in data['survival_by_sex']])},
                datasets: [{{
                    label: 'Survival Rate (%)',
                    data: {json.dumps([r['survival_rate'] for r in data['survival_by_sex']])},
                    backgroundColor: ['#60a5fa', '#a78bfa'],
                    borderRadius: 8,
                    barPercentage: 0.6,
                }}]
            }},
            options: {{
                ...chartDefaults,
                plugins: {{ legend: {{ display: false }},
                    datalabels: {{
                        anchor: 'end',
                        align: 'end',
                        offset: 4,
                        color: getThemeColors().text,
                        font: {{ weight: 'bold', size: 12 }},
                        formatter: val => `${{val}}%`,
                    }}
                }},
                scales: {{
                    ...chartDefaults.scales,
                    y: {{ ...chartDefaults.scales.y, max: 100, title: {{ display: true, text: 'Survival Rate (%)', color: getThemeColors().text }} }}
                }}
            }}
        }});

        new Chart(document.getElementById('pclassChart'), {{
            type: 'bar',
            data: {{
                labels: ['1st Class', '2nd Class', '3rd Class'],
                datasets: [{{
                    label: 'Survival Rate (%)',
                    data: {json.dumps([r['survival_rate'] for r in data['survival_by_pclass']])},
                    backgroundColor: ['#60a5fa', '#a78bfa', '#f59e0b'],
                    borderRadius: 8,
                    barPercentage: 0.6,
                }}]
            }},
            options: {{
                ...chartDefaults,
                plugins: {{ legend: {{ display: false }},
                    datalabels: {{
                        anchor: 'end',
                        align: 'end',
                        offset: 4,
                        color: getThemeColors().text,
                        font: {{ weight: 'bold', size: 12 }},
                        formatter: val => `${{val}}%`,
                    }}
                }},
                scales: {{
                    ...chartDefaults.scales,
                    y: {{ ...chartDefaults.scales.y, max: 100, title: {{ display: true, text: 'Survival Rate (%)', color: getThemeColors().text }} }}
                }}
            }}
        }});

        new Chart(document.getElementById('embarkedChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps([r['Embarked'] for r in data['survival_by_embarked']])},
                datasets: [{{
                    label: 'Survival Rate (%)',
                    data: {json.dumps([r['survival_rate'] for r in data['survival_by_embarked']])},
                    backgroundColor: ['#34d399', '#fbbf24', '#f87171'],
                    borderRadius: 8,
                    barPercentage: 0.6,
                }}]
            }},
            options: {{
                ...chartDefaults,
                plugins: {{ legend: {{ display: false }},
                    datalabels: {{
                        anchor: 'end',
                        align: 'end',
                        offset: 4,
                        color: getThemeColors().text,
                        font: {{ weight: 'bold', size: 12 }},
                        formatter: val => `${{val}}%`,
                    }}
                }},
                scales: {{
                    ...chartDefaults.scales,
                    y: {{ ...chartDefaults.scales.y, max: 100, title: {{ display: true, text: 'Survival Rate (%)', color: getThemeColors().text }} }}
                }}
            }}
        }});

        new Chart(document.getElementById('ageChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps([r['Age'] for r in data['survival_by_age']])},
                datasets: [{{
                    label: 'Survival Rate (%)',
                    data: {json.dumps([safe_num(r['survival_rate']) for r in data['survival_by_age']])},
                    backgroundColor: ['#34d399', '#60a5fa', '#a78bfa', '#f59e0b', '#f87171'],
                    borderRadius: 8,
                    barPercentage: 0.6,
                }}]
            }},
            options: {{
                ...chartDefaults,
                plugins: {{ legend: {{ display: false }},
                    datalabels: {{
                        anchor: 'end',
                        align: 'end',
                        offset: 4,
                        color: getThemeColors().text,
                        font: {{ weight: 'bold', size: 12 }},
                        formatter: val => val > 0 ? `${{val}}%` : '',
                    }}
                }},
                scales: {{
                    ...chartDefaults.scales,
                    y: {{ ...chartDefaults.scales.y, max: 100, title: {{ display: true, text: 'Survival Rate (%)', color: getThemeColors().text }} }}
                }}
            }}
        }});

        new Chart(document.getElementById('fareChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps([r['Fare'] for r in data['survival_by_fare']])},
                datasets: [{{
                    label: 'Survival Rate (%)',
                    data: {json.dumps([safe_num(r['survival_rate']) for r in data['survival_by_fare']])},
                    backgroundColor: ['#34d399', '#60a5fa', '#a78bfa', '#f59e0b', '#f87171'],
                    borderRadius: 8,
                    barPercentage: 0.6,
                }}]
            }},
            options: {{
                ...chartDefaults,
                plugins: {{ legend: {{ display: false }},
                    datalabels: {{
                        anchor: 'end',
                        align: 'end',
                        offset: 4,
                        color: getThemeColors().text,
                        font: {{ weight: 'bold', size: 12 }},
                        formatter: val => val > 0 ? `${{val}}%` : '',
                    }}
                }},
                scales: {{
                    ...chartDefaults.scales,
                    y: {{ ...chartDefaults.scales.y, max: 100, title: {{ display: true, text: 'Survival Rate (%)', color: getThemeColors().text }} }}
                }}
            }}
        }});

        new Chart(document.getElementById('titleChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps([r['Title'] for r in data['title_survival']])},
                datasets: [{{
                    label: 'Survival Rate (%)',
                    data: {json.dumps([safe_num(r['survival_rate']) for r in data['title_survival']])},
                    backgroundColor: ['#60a5fa', '#34d399', '#a78bfa', '#f59e0b', '#f87171'],
                    borderRadius: 8,
                    barPercentage: 0.6,
                }}]
            }},
            options: {{
                ...chartDefaults,
                plugins: {{ legend: {{ display: false }},
                    datalabels: {{
                        anchor: 'end',
                        align: 'end',
                        offset: 4,
                        color: getThemeColors().text,
                        font: {{ weight: 'bold', size: 11 }},
                        formatter: val => val > 0 ? `${{val}}%` : '',
                    }}
                }},
                scales: {{
                    ...chartDefaults.scales,
                    y: {{ ...chartDefaults.scales.y, max: 100, title: {{ display: true, text: 'Survival Rate (%)', color: getThemeColors().text }} }}
                }}
            }}
        }});

        new Chart(document.getElementById('familyChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps([str(r['FamilySize']) for r in data['family_survival']])},
                datasets: [{{
                    label: 'Survival Rate (%)',
                    data: {json.dumps([safe_num(r['survival_rate']) for r in data['family_survival']])},
                    backgroundColor: ['#34d399', '#60a5fa', '#a78bfa', '#f59e0b', '#f87171', '#ec4899', '#8b5cf6'],
                    borderRadius: 8,
                    barPercentage: 0.6,
                }}]
            }},
            options: {{
                ...chartDefaults,
                plugins: {{ legend: {{ display: false }},
                    datalabels: {{
                        anchor: 'end',
                        align: 'end',
                        offset: 4,
                        color: getThemeColors().text,
                        font: {{ weight: 'bold', size: 11 }},
                        formatter: val => val > 0 ? `${{val}}%` : '',
                    }}
                }},
                scales: {{
                    ...chartDefaults.scales,
                    y: {{ ...chartDefaults.scales.y, max: 100, title: {{ display: true, text: 'Survival Rate (%)', color: getThemeColors().text }} }},
                    x: {{ ...chartDefaults.scales.x, title: {{ display: true, text: 'Family Size', color: getThemeColors().text }} }}
                }}
            }}
        }});

        new Chart(document.getElementById('ageDistChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps([r['bin'] for r in data['age_dist']])},
                datasets: [{{
                    label: 'Passenger Count',
                    data: {json.dumps([r['count'] for r in data['age_dist']])},
                    backgroundColor: 'rgba(96, 165, 250, 0.6)',
                    borderColor: '#60a5fa',
                    borderWidth: 1,
                    borderRadius: 4,
                    barPercentage: 1.0,
                    categoryPercentage: 0.95,
                }}]
            }},
            options: {{
                ...chartDefaults,
                plugins: {{ legend: {{ display: false }},
                    datalabels: {{
                        anchor: 'end',
                        align: 'end',
                        offset: 4,
                        color: getThemeColors().text,
                        font: {{ weight: 'bold', size: 11 }},
                        formatter: val => val,
                    }}
                }},
                scales: {{
                    ...chartDefaults.scales,
                    y: {{ ...chartDefaults.scales.y, title: {{ display: true, text: 'Count', color: getThemeColors().text }} }}
                }}
            }}
        }});

        new Chart(document.getElementById('fareDistChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps([r['bin'] for r in data['fare_dist']])},
                datasets: [{{
                    label: 'Passenger Count',
                    data: {json.dumps([r['count'] for r in data['fare_dist']])},
                    backgroundColor: 'rgba(167, 139, 250, 0.6)',
                    borderColor: '#a78bfa',
                    borderWidth: 1,
                    borderRadius: 4,
                    barPercentage: 1.0,
                    categoryPercentage: 0.95,
                }}]
            }},
            options: {{
                ...chartDefaults,
                plugins: {{ legend: {{ display: false }},
                    datalabels: {{
                        anchor: 'end',
                        align: 'end',
                        offset: 4,
                        color: getThemeColors().text,
                        font: {{ weight: 'bold', size: 11 }},
                        formatter: val => val,
                    }}
                }},
                scales: {{
                    ...chartDefaults.scales,
                    y: {{ ...chartDefaults.scales.y, title: {{ display: true, text: 'Count', color: getThemeColors().text }} }}
                }}
            }}
        }});

        const corrFeatures = {json.dumps(data['corr_features'])};
        const corrMatrix = {json.dumps(data['corr_matrix'])};
        const heatmapEl = document.getElementById('heatmap');
        const headerRow = [''].concat(corrFeatures);
        headerRow.forEach(label => {{
            const div = document.createElement('div');
            div.className = 'heatmap-label';
            div.textContent = label;
            heatmapEl.appendChild(div);
        }});
        corrMatrix.forEach((row, i) => {{
            const labelDiv = document.createElement('div');
            labelDiv.className = 'heatmap-label';
            labelDiv.textContent = corrFeatures[i];
            heatmapEl.appendChild(labelDiv);
            row.forEach((val, j) => {{
                const cell = document.createElement('div');
                cell.className = 'heatmap-cell';
                const intensity = Math.abs(val);
                const r = val > 0 ? Math.round(96 + intensity * (248 - 96)) : Math.round(248 - intensity * (248 - 96));
                const g = val > 0 ? Math.round(165 - intensity * (165 - 113)) : Math.round(165 - intensity * (165 - 113));
                const b = val > 0 ? Math.round(250 - intensity * (250 - 113)) : Math.round(250 - intensity * (250 - 113));
                cell.style.background = `rgb(${{r}}, ${{g}}, ${{b}})`;
                cell.style.color = intensity > 0.5 ? '#fff' : getThemeColors().text;
                cell.textContent = val.toFixed(2);
                cell.title = `${{corrFeatures[i]}} × ${{corrFeatures[j]}}: ${{val}}`;
                heatmapEl.appendChild(cell);
            }});
        }});

        function animateCounters() {{
            document.querySelectorAll('[data-count]').forEach(el => {{
                const target = parseFloat(el.dataset.count);
                const decimals = parseInt(el.dataset.decimals) || 0;
                const prefix = el.dataset.prefix || '';
                const duration = 1500;
                const start = performance.now();
                function update(now) {{
                    const elapsed = now - start;
                    const progress = Math.min(elapsed / duration, 1);
                    const eased = 1 - Math.pow(1 - progress, 3);
                    const current = target * eased;
                    el.textContent = prefix + (decimals > 0 ? current.toFixed(decimals) : Math.round(current));
                    if (progress < 1) requestAnimationFrame(update);
                }}
                requestAnimationFrame(update);
            }});
        }}

        function toggleTheme() {{
            document.body.classList.toggle('light-theme');
            const btn = document.querySelector('.theme-toggle');
            btn.textContent = document.body.classList.contains('light-theme') ? '☀️ Theme' : '🌙 Theme';
        }}

        const observer = new IntersectionObserver((entries) => {{
            entries.forEach(entry => {{
                if (entry.isIntersecting) {{
                    entry.target.style.animationDelay = '0.1s';
                    entry.target.style.animationPlayState = 'running';
                }}
            }});
        }}, {{ threshold: 0.1 }});
        document.querySelectorAll('.animate-in').forEach(el => observer.observe(el));

        document.addEventListener('DOMContentLoaded', animateCounters);

        document.querySelectorAll('.sidebar-nav a').forEach(link => {{
            link.addEventListener('click', function() {{
                document.querySelectorAll('.sidebar-nav a').forEach(l => l.classList.remove('active'));
                this.classList.add('active');
                if (window.innerWidth <= 768) document.querySelector('.sidebar').classList.remove('open');
            }});
        }});
    </script>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    print(f"Dashboard generated at {output_path}")

if __name__ == "__main__":
    df = load_titanic()
    output = Path(__file__).parent / "index.html"
    generate_dashboard(df, output)
