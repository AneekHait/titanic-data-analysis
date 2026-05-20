"""Tests for statistical analysis module."""

import pandas as pd
import pytest
from src.analysis.statistics import (
    chi_square_test,
    t_test_survival,
    anova_survival,
    effect_sizes,
)


class TestChiSquareTest:
    def test_returns_dict(self, raw_df: pd.DataFrame):
        result = chi_square_test(raw_df, "Sex")
        assert isinstance(result, dict)

    def test_has_required_keys(self, raw_df: pd.DataFrame):
        result = chi_square_test(raw_df, "Sex")
        for key in ["chi2", "p_value", "dof", "significant", "strength"]:
            assert key in result

    def test_sex_significant(self, raw_df: pd.DataFrame):
        result = chi_square_test(raw_df, "Sex")
        assert result["significant"] == True
        assert result["p_value"] < 0.001

    def test_pclass_significant(self, raw_df: pd.DataFrame):
        result = chi_square_test(raw_df, "Pclass")
        assert result["significant"] == True


class TestTTestSurvival:
    def test_returns_dict(self, raw_df: pd.DataFrame):
        result = t_test_survival(raw_df, "Age")
        assert isinstance(result, dict)

    def test_has_required_keys(self, raw_df: pd.DataFrame):
        result = t_test_survival(raw_df, "Age")
        for key in ["t_statistic", "p_value", "mean_survived", "mean_perished",
                     "cohens_d", "effect_size", "significant"]:
            assert key in result

    def test_fare_significant(self, raw_df: pd.DataFrame):
        result = t_test_survival(raw_df, "Fare")
        assert result["significant"] == True
        assert result["p_value"] < 0.001

    def test_means_different(self, raw_df: pd.DataFrame):
        result = t_test_survival(raw_df, "Fare")
        assert result["mean_survived"] != result["mean_perished"]


class TestAnovaSurvival:
    def test_returns_dict(self, raw_df: pd.DataFrame):
        result = anova_survival(raw_df, "Pclass")
        assert isinstance(result, dict)

    def test_has_required_keys(self, raw_df: pd.DataFrame):
        result = anova_survival(raw_df, "Pclass")
        for key in ["f_statistic", "p_value", "significant", "group_means"]:
            assert key in result

    def test_pclass_significant(self, raw_df: pd.DataFrame):
        result = anova_survival(raw_df, "Pclass")
        assert result["significant"] == True


class TestEffectSizes:
    def test_returns_dataframe(self, raw_df: pd.DataFrame):
        result = effect_sizes(raw_df)
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self, raw_df: pd.DataFrame):
        result = effect_sizes(raw_df)
        for col in ["feature", "type", "effect_size", "metric", "strength"]:
            assert col in result.columns

    def test_sex_has_largest_effect(self, raw_df: pd.DataFrame):
        result = effect_sizes(raw_df)
        assert result.iloc[0]["feature"] == "Sex"

    def test_all_features_present(self, raw_df: pd.DataFrame):
        result = effect_sizes(raw_df)
        expected = {"Sex", "Pclass", "Embarked", "Age", "Fare", "SibSp", "Parch"}
        assert set(result["feature"]) == expected
