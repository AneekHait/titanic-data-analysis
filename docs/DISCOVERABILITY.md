# Discoverability checklist

A short list of one-time actions to take on github.com that can't be done from a commit. Do these once and the repo gets meaningfully more findable.

## 1. Repo description (top of the GitHub page)

Set to:

> Titanic survival analysis (1,309 passengers): interactive Chart.js dashboard, statistical inference (95% Wilson CIs, odds ratios, chi-square, Cramer's V, Cohen's d), and a fully-tested reusable Python package.

## 2. Website (top of the GitHub page)

Set to:

> https://aneekhait.github.io/titanic-data-analysis/

## 3. Topics (the chips under the description)

Add all of these. They are the single biggest lever for GitHub search:

```
titanic
titanic-dataset
titanic-survival
exploratory-data-analysis
eda
data-analysis
data-science
statistics
statistical-inference
odds-ratio
chi-square
confidence-interval
effect-size
dashboard
chartjs
kaggle
python
pandas
numpy
scipy
seaborn
matplotlib
reproducible-research
portfolio-project
machine-learning
```

GitHub caps at 20 topics — trim from the bottom if needed.

## 4. Enable GitHub Pages

Settings → Pages → Source: **GitHub Actions**. The `.github/workflows/pages.yml` workflow will then publish to `https://aneekhait.github.io/titanic-data-analysis/` on every push that touches the dashboard or its inputs.

## 5. Social preview image

Settings → General → Social preview → Upload a 1280×640 PNG. A screenshot of the dashboard (dark theme) makes for a strong card on Twitter, LinkedIn, and Slack.

Capture it: open the live dashboard, set zoom to 100%, take a screenshot of the top of the page including the title, the overall survival overview card, and the class × sex chart. Save as `docs/social-preview.png` for safekeeping.

## 6. Pin the repo on your profile

github.com/AneekHait → "Customize your pins" → check this repo. Pinned repos rank higher in profile views and are what recruiters see first.

## 7. Cross-link from your homepage

Add a link to `https://aneekhait.github.io/titanic-data-analysis/` on `aneekhait.github.io`. Google ranks pages partly on inbound links from the same author's known sites.

## 8. (Optional) Index with Google Search Console

Once Pages is live, verify the property at search.google.com/search-console and submit `https://aneekhait.github.io/titanic-data-analysis/sitemap.xml` (or the page URL directly). Indexing usually takes 1–14 days.

## 9. (Optional) Zenodo DOI

zenodo.org → "GitHub" → enable for this repo → cut a GitHub release. Zenodo mints a DOI and the repo becomes citable from Google Scholar. The `CITATION.cff` is already in place.

## 10. (Optional) Submit to "awesome" lists

- `awesome-public-datasets` — under "Sports / History / Disasters"
- `awesome-data-science-projects`
- `awesome-jupyter` (if/when a notebook is the headline)

Each accepted listing is a permanent inbound link.
