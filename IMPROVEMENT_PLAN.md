# Project Improvement Plan

## Short-term (next session)

- [ ] **Feature engineering**: extract titles from `Name`, create `FamilySize` from `SibSp + Parch`, extract deck letter from `Cabin`, bin `Age` and `Fare` into categorical groups
- [ ] **Missing value handling**: impute `Age` (median by sex/class), drop `Cabin` (77% missing), fill `Embarked` with mode
- [ ] **Data pipeline**: add `src/data/processing.py` with reusable `clean()` and `engineer_features()` functions that transform raw data into a model-ready DataFrame
- [ ] **CLI improvement**: add `--output` flag to save results as CSV/JSON; add `--impute` flag to toggle missing value handling
- [ ] **More tests**: unit tests for `eda.py` functions (missing_summary, survival_rate, survival_by_categorical) using `pytest`

## Medium-term

- [ ] **Machine learning**: add `src/models/train.py` with Logistic Regression, Random Forest, and XGBoost classifiers; add `src/models/evaluate.py` for accuracy, precision, recall, F1, confusion matrix, ROC-AUC
- [ ] **Feature importance**: add `src/analysis/importance.py` with permutation importance and SHAP explanations; add `src/visualization/importance.py` for SHAP summary/dependence plots
- [ ] **Cross-validation**: integrate `sklearn.model_selection.StratifiedKFold` into model evaluation
- [ ] **Hyperparameter tuning**: add `src/models/tune.py` with `GridSearchCV` or `Optuna`
- [ ] **Pipeline orchestration**: unify data loading → cleaning → feature engineering → modeling into a single `pipeline.py` script
- [ ] **Streamlit dashboard**: add `app/` directory with interactive Streamlit app for exploring data and model predictions without code

## Long-term

- [ ] **CI/CD**: add `.github/workflows/ci.yml` running lint (ruff), type check (mypy), and tests on every push
- [ ] **Docker**: add `Dockerfile` and `docker-compose.yml` for reproducible environment
- [ ] **Config management**: migrate hardcoded paths and params to YAML/TOML config files with `src/config.py` loading them
- [ ] **Experiment tracking**: integrate MLflow or W&B for logging model runs, parameters, and metrics
- [ ] **API deployment**: add `api/` with FastAPI serving predictions; add `deploy/` with Kubernetes manifests or Cloud Run config
- [ ] **Pre-commit hooks**: add `.pre-commit-config.yaml` with ruff, mypy, and pytest checks
- [ ] **Documentation**: add Sphinx docs with `docs/` directory; expand docstrings across all modules

## Stretch goals

- [ ] Hyperparameter optimization with Optuna
- [ ] Neural network baseline (PyTorch/TabNet)
- [ ] Model calibration analysis
- [ ] Automated EDA report generation (ydata-profiling)
- [ ] A/B test framework for model comparison
