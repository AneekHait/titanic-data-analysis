# Roadmap

What's done lives in [CHANGELOG.md](CHANGELOG.md). What's below is the realistic next step — and a few stretch ideas I might not get to.

The focus of this repo is classical inference + EDA done well. Anything below that materially expands that scope (predictive modelling, serving) belongs in a sibling repo so this one stays focused.

---

## Near-term — high-value, low scope creep

- [ ] **Logistic regression with Class × Sex interaction.** Marginal odds ratios already tell a clear story, but a joint model would let us separate independent effects of sex and class from their interaction, and adjust for fare / age / family size simultaneously. Output: an "adjusted odds ratio" table next to the marginal one already in §7 of the analyst report.
- [ ] **Wilson CI for joint Class × Sex cells.** They exist in the appendix table already; surface them in the dashboard heatmap tooltip and the PDF heatmap.
- [ ] **Cabin-letter extraction from `Occupation`.** Even with 47% missing, the recovered deck information would let us test whether upper-deck cabins had a measurable independent survival advantage on top of class.
- [ ] **Survival lift curve.** A single chart showing how survival probability shifts as you add features one at a time (Sex → +Class → +Age → +Fare). Useful for the "which features actually matter" narrative.

## Medium-term — substantial but bounded

- [ ] **Sphinx documentation** built from docstrings. Currently the API is documented inline; a generated site at `docs/api/` would make it browsable.
- [ ] **Bayesian survival rate posteriors** for the small joint cells (e.g. 1st-class women, n=144). Beta(α, β) priors give a more honest uncertainty picture than Wilson when n is small.

## Stretch — interesting but explicit scope expansion

- [ ] **Predictive modelling sibling repo.** Logistic regression, Random Forest, XGBoost with SHAP attributions. Belongs in its own repo to keep this one focused on inference rather than prediction.
- [ ] **Streamlit interactive explorer.** The current dashboard is single-page; an app would let users filter / re-stratify on the fly.
- [ ] **A/B comparison against the Kaggle subset.** Quantify how much the smaller, more-missing Kaggle training set distorts effect-size estimates.

---

If you have a strong opinion about prioritisation, see [CONTRIBUTING.md](CONTRIBUTING.md).
