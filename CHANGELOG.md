# Changelog

All notable changes to this project are documented here. Follows [Keep a Changelog](https://keepachangelog.com/) loosely; not strict semver.

## [0.2.0] — 2026-05-25

A major iteration: the project gained a polished dashboard, an author-written analyst report, and a proper documentation structure.

### Added
- **Author-written analyst report** ([reports/](reports/)) — narrative DOCX + PDF (16 pages, with executive summary, glossary, and "What this means in plain English" callouts after every statistical finding).
- **Inference helpers** in `src/analysis/inference.py`: Wilson confidence intervals for proportions, odds ratios with 95% CIs from the log-odds standard error, joint stratified tables.
- **Effect-size ranking** chart across all features on a single 0–1 scale.
- **Class × Sex joint heatmap** with proper RdYlGn diverging palette and luminance-aware text colour.
- **Lifeboat reality-check section** — surfaces 98.6% vs 2.6% survival for boat-recorded vs not.
- **Wilson 95% CIs as error bars** on all survival-rate charts in both dashboard and PDF.
- **Cross-stage validation script** ([reports/validate_dataset.py](reports/validate_dataset.py)) — reconciles row counts at every pipeline stage.
- **Documentation** — `docs/METHODOLOGY.md`, `docs/DATA.md`, `docs/DEVELOPMENT.md`; README polish; per-folder READMEs in `reports/` and `dashboard/`.
- **License, contributing guide, changelog, roadmap** at root.

### Changed
- **Light theme is now genuinely readable.** Root cause: `themeColors()` was reading CSS variables from `<html>`, but the `.light-theme` class lives on `<body>`. Switched the reader to `document.body` and added dedicated light-mode palette overrides. All chart text, datalabels, tags, and KPI accents are now properly contrasted on both backgrounds.
- **Dashboard hover tooltips** now include sample size and 95% CI on every rate chart; the doughnut tooltip surfaces both count and percentage.
- **Severity column** in the Missing Values table — was duplicating the percent column; now shows "High / Medium / Low" categorical tags.
- **Empty bins (e.g. the (307, 410] fare bin with zero passengers)** are rendered as "N/A" gaps instead of misleading 0% bars.
- **PDF inference boxes** redesigned — was dark fill with light-gray text (illegible in print); now a 6% pale tint with dark navy body text + a colored left bar.
- **PDF cover page** — replaced the staircase metric-box layout with properly anchored, horizontally-centred boxes; added prominent author byline + dataset link.
- **Classical PDF report** — sparse pages packed with substantive content; one orphan page eliminated by relocating the age × sex chart to the Demographics section.

### Fixed
- **`_parse_boat_body`** was treating whitespace-only `BoatBody` strings as valid lifeboat numbers, inflating the apparent boat-occupant count. Now correctly returns `(None, None)` for blanks. The lifeboat analysis now reports the real numbers: 486 on a boat (98.6% survived) vs 823 without a record (2.6%).
- **Table typos** — "2st" / "3st" → "1st / 2nd / 3rd"; float counts (`324.0`) → integers.
- **Joint heatmap colour gradient** — was producing muddy brown/olive tones for mid-range survival cells; now uses the 11-stop RdYlGn palette familiar from matplotlib/seaborn.

### Validated
- 49/49 pytest tests pass.
- All counts reconcile to 1,309 rows across raw → cleaned → engineered stages.
- Every static number and hover tooltip in the dashboard verified against the underlying DataFrame via a Playwright-driven probe.

---

## [0.1.0] — 2026-04

Initial release.

### Added
- `src/` Python package: data loading, cleaning, feature engineering, EDA, statistics, visualisation.
- CLI tools: `make eda`, `make download`.
- Classical EDA PDF report ([dashboard/titanic_eda_report.pdf](dashboard/titanic_eda_report.pdf)).
- First-pass Chart.js dashboard.
- Test suite (49 tests across loader / processing / EDA / statistics).
- CI workflow.
- Dockerfile and Makefile.
- Pre-commit hooks (ruff + pytest).
