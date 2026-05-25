# Methodology

How the statistical analysis is structured, why each test was chosen, and how to read the numbers in the outputs.

---

## Analytical flow

```
raw CSV  →  clean_data()  →  engineer_features()  →  analysis  →  outputs
```

Each stage is deterministic. Counts reconcile to **1,309 rows** at every stage — run `python reports/validate_dataset.py` to verify.

1. **Load** — `src/data/loader.py` downloads / reads `data/raw/titanic5.csv`.
2. **Clean** — `src/data/processing.clean_data` fills missing `Embarked` with the mode (`S`) and missing `Fare` with the median fare within `Pclass`. Age is left for stratum-aware imputation downstream.
3. **Engineer** — `src/data/processing.engineer_features` imputes Age by median within Sex × Pclass strata; derives `Title`, `FamilySize`, `IsAlone`, `AgeGroup`, `FareGroup`, `FarePerPerson`, `HasCabin`, `Lifeboat`, `BodyRecovered`.
4. **Analyse** — `src/analysis/` modules compute survival rates, effect sizes, odds ratios, and significance tests on the engineered DataFrame.

---

## What each metric answers

| Question | Metric | Module |
|---|---|---|
| "What share of group X survived?" | **Survival rate** + 95% Wilson CI | `analysis.inference.survival_rates_with_ci` |
| "How strong is feature X's signal overall?" | **Effect size** (Cramer's V or `|r|`) | `analysis.statistics.effect_sizes` |
| "How much do group X's survival odds differ from group Y's?" | **Odds ratio** + 95% CI from log-odds SE | `analysis.inference.odds_ratio` |
| "Could the X→Survived association be chance?" | **Chi-square** test of independence | `analysis.statistics.chi_square_test` |
| "Do numeric values differ between survivors and victims?" | **Welch's t-test** + Cohen's d | `analysis.statistics.t_test_survival` |
| "Do survival rates differ across multiple groups?" | **One-way ANOVA** | `analysis.statistics.anova_survival` |
| "How are the two factors interacting?" | **Joint stratified table** with cell CIs | `analysis.inference.joint_survival` |

---

## Why these tests and not others

- **Wilson score interval** (not the normal approximation) for proportion CIs. The Wilson interval has better coverage when proportions are near 0 or 1, or when sample sizes are small — relevant for cells like 1st-class women (n = 144, rate = 96.5%).
- **Welch's t-test** (not Student's) because survivor and non-survivor groups have unequal variances and very different sample sizes.
- **Cramer's V** as the categorical effect size because it's bounded [0, 1] and comparable across features with different cardinalities (binary Sex vs. four-port Embarked).
- **Fisher's exact test** for 2×2 odds-ratio p-values rather than chi-square — Fisher is exact and correct even when expected cell counts are small.
- **Log-odds standard error** for odds-ratio CIs because the sampling distribution of `log(OR)` is approximately normal, even when the OR distribution is heavily skewed.

---

## Effect-size thresholds

Both Cramer's V and `|r|` (point-biserial) sit on a 0–1 scale so we can rank features on a single chart. Conventional cutoffs (Cohen 1988):

| Range | Label |
|---|---|
| < 0.1 | Negligible |
| 0.1 – 0.3 | Small |
| 0.3 – 0.5 | Medium |
| ≥ 0.5 | Large |

For Cohen's d (mean differences):

| Range | Label |
|---|---|
| < 0.2 | Negligible |
| 0.2 – 0.5 | Small |
| 0.5 – 0.8 | Medium |
| ≥ 0.8 | Large |

---

## Confounding & target leakage

Two patterns to watch for:

- **Fare and Class are essentially the same feature** at different resolutions. Including both in a predictive model creates collinearity. Treat fare as class-with-finer-grain.
- **Embarkation port appears to predict survival** but mostly because port composition correlates with class. Most of the port effect vanishes after conditioning on class.
- **Lifeboat and BodyRecovered are downstream of the target.** Using them as features in a survival model is target leakage. They are treated as outcomes / sanity checks, never as predictors.

---

## A note on the dataset

`titanic5` is materially more complete than the well-known Kaggle training subset (891 rows). Specifically:

| | titanic5 | Kaggle train |
|---|---|---|
| Rows | **1,309** | 891 |
| Missing Age | **3.9%** | 19.9% |
| Embarked completeness | 99.8% | ≥99% |
| Cabin / Occupation completeness | 52.6% | 22.7% |

This means age-stratified analyses are reliably powered here in a way they aren't on the Kaggle subset.

---

## References

- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences.*
- Cramér, H. (1946). *Mathematical Methods of Statistics.*
- Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *Journal of the American Statistical Association.*
- Agresti, A. (2002). *Categorical Data Analysis* — for Fisher exact and chi-square in 2×N tables.
