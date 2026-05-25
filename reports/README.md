# Reports

The **author-written** analyst report. This is the polished, narrative-driven deliverable — written as a data analyst would present findings to a stakeholder, not assembled programmatically from chart templates.

For the chart-heavy programmatic alternative, see [dashboard/titanic_eda_report.pdf](../dashboard/titanic_eda_report.pdf).

---

## Outputs

| File | Audience | Notes |
|---|---|---|
| [Titanic_Survival_Analyst_Report.docx](Titanic_Survival_Analyst_Report.docx) | Stakeholders, peer review | Editable Word document |
| [Titanic_Survival_Analyst_Report.pdf](Titanic_Survival_Analyst_Report.pdf) | Distribution / archive | Fixed-layout, same content |

Both are 16 pages and contain the same narrative.

---

## What's inside

1. **Cover** — title, byline, dataset attribution
2. **Executive Summary** — 5 headline bullets + "How to read this report" glossary
3. **§1 Background & Question** — what happened in 1912 and the four concrete questions this analysis answers
4. **§2 Data & Method** — dataset link, column dictionary, statistical machinery
5. **§3 The Big Picture** — why the 38.2% overall rate is misleading
6. **§4 The Three Biggest Drivers** — Sex, Class, and their interaction
7. **§5 Secondary Factors** — Age, family size, fare (a class proxy), port (confounded), title
8. **§6 The Mechanism: Lifeboats** — the proximate cause everything else reduces to
9. **§7 Statistical Robustness** — significance tests, odds ratios with 95% CIs
10. **§8 Conclusions & Recommendations** — including modelling guidance and limitations
11. **Appendix A** — full survival/CI tables and correlation matrix
12. **Appendix B** — glossary of every statistical term used

---

## Rebuilding

```bash
# Re-render both DOCX and PDF
make report

# Or directly:
python reports/build_analyst_report.py
```

The PDF is produced by converting the DOCX via Word/LibreOffice (`docx2pdf`). On a fresh machine:

```bash
pip install -e ".[reports]"
pip install docx2pdf
```

---

## Validation

To confirm every number in the report ties back to the underlying data:

```bash
python reports/validate_dataset.py
```

This reconciles row counts at every stage of the pipeline (raw → cleaned → engineered) and dumps survival breakdowns for sex, class, port, joint Class × Sex, family size, age groups, lifeboat split, and titles. All subtotals should sum to **1,309**.

---

## Source files

| File | Purpose |
|---|---|
| [build_analyst_report.py](build_analyst_report.py) | The one-off DOCX builder. Authored content lives here as Python prose. |
| [validate_dataset.py](validate_dataset.py) | Cross-stage count reconciliation |

---

Prepared by **[Aneek Hait](https://aneekhait.github.io)**.
