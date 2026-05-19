#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import math
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
        "correlation": corr.to_dict(),
        "summary_stats": df.describe().round(2).to_dict(),
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
        "class_distribution": df["Pclass"].value_counts().sort_index().to_dict(),
        "embarked_distribution": df["Embarked"].value_counts().to_dict(),
    }
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Titanic EDA Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            line-height: 1.6;
        }}
        .header {{
            background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%);
            padding: 2rem;
            text-align: center;
            border-bottom: 1px solid #334155;
        }}
        .header h1 {{
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(90deg, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}
        .header p {{ color: #94a3b8; font-size: 1.1rem; }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 2rem; }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .stat-card {{
            background: #1e293b;
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid #334155;
            text-align: center;
        }}
        .stat-card .value {{
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }}
        .stat-card .label {{ color: #94a3b8; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.05em; }}
        .stat-card.survived .value {{ color: #34d399; }}
        .stat-card.perished .value {{ color: #f87171; }}
        .stat-card.info .value {{ color: #60a5fa; }}
        .stat-card.warning .value {{ color: #fbbf24; }}
        .section {{
            background: #1e293b;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            border: 1px solid #334155;
        }}
        .section h2 {{
            font-size: 1.5rem;
            margin-bottom: 1rem;
            color: #60a5fa;
            border-bottom: 1px solid #334155;
            padding-bottom: 0.5rem;
        }}
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 1.5rem;
        }}
        .chart-container {{
            background: #0f172a;
            border-radius: 8px;
            padding: 1rem;
            border: 1px solid #334155;
        }}
        .chart-container h3 {{
            text-align: center;
            margin-bottom: 1rem;
            color: #cbd5e1;
        }}
        .chart-wrapper {{ position: relative; height: 300px; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }}
        th, td {{
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid #334155;
        }}
        th {{
            background: #0f172a;
            color: #60a5fa;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85rem;
            letter-spacing: 0.05em;
        }}
        tr:hover {{ background: #2d3a4f; }}
        .insights {{
            background: #1e293b;
            border-radius: 12px;
            padding: 1.5rem;
            border-left: 4px solid #60a5fa;
            margin-bottom: 2rem;
        }}
        .insights h2 {{ color: #60a5fa; margin-bottom: 1rem; }}
        .insights ul {{ list-style: none; }}
        .insights li {{
            padding: 0.75rem 0;
            border-bottom: 1px solid #334155;
            display: flex;
            align-items: flex-start;
        }}
        .insights li:last-child {{ border-bottom: none; }}
        .insights li::before {{
            content: "▸";
            color: #60a5fa;
            margin-right: 0.75rem;
            font-size: 1.2rem;
        }}
        .footer {{
            text-align: center;
            padding: 2rem;
            color: #64748b;
            font-size: 0.9rem;
            border-top: 1px solid #334155;
        }}
        @media (max-width: 768px) {{
            .charts-grid {{ grid-template-columns: 1fr; }}
            .stats-grid {{ grid-template-columns: repeat(2, 1fr); }}
            .header h1 {{ font-size: 1.75rem; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Titanic EDA Dashboard</h1>
        <p>Exploratory Data Analysis of the Titanic Passenger Dataset</p>
    </div>

    <div class="container">
        <div class="stats-grid">
            <div class="stat-card info">
                <div class="value">{data['passenger_stats']['total']}</div>
                <div class="label">Total Passengers</div>
            </div>
            <div class="stat-card survived">
                <div class="value">{data['survival_rate']['survived']:.1f}%</div>
                <div class="label">Survived</div>
            </div>
            <div class="stat-card perished">
                <div class="value">{data['survival_rate']['perished']:.1f}%</div>
                <div class="label">Perished</div>
            </div>
            <div class="stat-card info">
                <div class="value">{data['age_stats']['mean']}</div>
                <div class="label">Avg Age</div>
            </div>
            <div class="stat-card info">
                <div class="value">${data['fare_stats']['mean']:.2f}</div>
                <div class="label">Avg Fare</div>
            </div>
            <div class="stat-card warning">
                <div class="value">{data['age_stats']['missing']}</div>
                <div class="label">Missing Ages</div>
            </div>
        </div>

        <div class="charts-grid">
            <div class="chart-container">
                <h3>Survival Rate</h3>
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
        </div>

        <div class="section">
            <h2>Correlation Matrix</h2>
            <table>
                <thead>
                    <tr>
                        <th>Feature</th>
                        <th>Age</th>
                        <th>Fare</th>
                        <th>SibSp</th>
                        <th>Parch</th>
                        <th>Survived</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(f"<tr><td>{feat}</td>{''.join(f'<td>{corr[feat][col]:.3f}</td>' for col in ['Age', 'Fare', 'SibSp', 'Parch', 'Survived'])}</tr>" for feat in ['Age', 'Fare', 'SibSp', 'Parch', 'Survived'])}
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>Missing Values Summary</h2>
            <table>
                <thead>
                    <tr>
                        <th>Column</th>
                        <th>Missing Count</th>
                        <th>Percentage</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(f"<tr><td>{r['column']}</td><td>{r['missing']}</td><td>{r['percent']:.1f}%</td></tr>" for r in data['missing'])}
                </tbody>
            </table>
        </div>

        <div class="insights">
            <h2>Key Insights</h2>
            <ul>
                <li>Only {data['survival_rate']['survived']:.1f}% of passengers survived the Titanic disaster.</li>
                <li>Women had a much higher survival rate ({data['survival_by_sex'][1]['survival_rate']:.1f}%) compared to men ({data['survival_by_sex'][0]['survival_rate']:.1f}%).</li>
                <li>First-class passengers survived at {data['survival_by_pclass'][0]['survival_rate']:.1f}% vs third-class at {data['survival_by_pclass'][2]['survival_rate']:.1f}%.</li>
                <li>Children (age 0-16) had a {data['survival_by_age'][0]['survival_rate']:.1f}% survival rate.</li>
                <li>Higher fare passengers had better survival odds — those paying $102-$205 survived at {data['survival_by_fare'][1]['survival_rate']:.1f}%.</li>
                <li>Passengers from Cherbourg (C) had the highest survival rate at {data['survival_by_embarked'][0]['survival_rate']:.1f}%.</li>
                <li>Cabin data is 77.1% missing — too sparse for meaningful analysis without imputation.</li>
            </ul>
        </div>
    </div>

    <div class="footer">
        Generated from Titanic EDA Project | Data: 891 passengers, 12 features
    </div>

    <script>
        const chartDefaults = {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{
                legend: {{ labels: {{ color: '#94a3b8', font: {{ size: 12 }} }} }}
            }},
            scales: {{
                x: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }} }},
                y: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }} }}
            }}
        }};

        new Chart(document.getElementById('survivalChart'), {{
            type: 'doughnut',
            data: {{
                labels: ['Perished', 'Survived'],
                datasets: [{{
                    data: [{data['survival_rate']['perished_count']}, {data['survival_rate']['survived_count']}],
                    backgroundColor: ['#f87171', '#34d399'],
                    borderWidth: 0
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ position: 'bottom', labels: {{ color: '#94a3b8' }} }},
                    tooltip: {{
                        callbacks: {{
                            label: ctx => `{{ctx.label}}: {{ctx.raw}} ({{(ctx.raw / {data['passenger_stats']['total']} * 100).toFixed(1)}}%)`
                        }}
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
                    backgroundColor: ['#34d399', '#60a5fa'],
                    borderRadius: 8
                }}]
            }},
            options: {{
                ...chartDefaults,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    ...chartDefaults.scales,
                    y: {{ ...chartDefaults.scales.y, max: 100, title: {{ display: true, text: 'Survival Rate (%)', color: '#94a3b8' }} }}
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
                    borderRadius: 8
                }}]
            }},
            options: {{
                ...chartDefaults,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    ...chartDefaults.scales,
                    y: {{ ...chartDefaults.scales.y, max: 100, title: {{ display: true, text: 'Survival Rate (%)', color: '#94a3b8' }} }}
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
                    borderRadius: 8
                }}]
            }},
            options: {{
                ...chartDefaults,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    ...chartDefaults.scales,
                    y: {{ ...chartDefaults.scales.y, max: 100, title: {{ display: true, text: 'Survival Rate (%)', color: '#94a3b8' }} }}
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
                    borderRadius: 8
                }}]
            }},
            options: {{
                ...chartDefaults,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    ...chartDefaults.scales,
                    y: {{ ...chartDefaults.scales.y, max: 100, title: {{ display: true, text: 'Survival Rate (%)', color: '#94a3b8' }} }}
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
                    borderRadius: 8
                }}]
            }},
            options: {{
                ...chartDefaults,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    ...chartDefaults.scales,
                    y: {{ ...chartDefaults.scales.y, max: 100, title: {{ display: true, text: 'Survival Rate (%)', color: '#94a3b8' }} }}
                }}
            }}
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
