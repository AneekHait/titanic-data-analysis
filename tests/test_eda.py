"""Tests for EDA analysis module."""

import pandas as pd
import pytest
from src.analysis.eda import (
    data_overview,
    missing_summary,
    survival_rate,
    survival_by_categorical,
    survival_by_numerical,
    correlation_analysis,
)


class TestDataOverview:
    def test_returns_dict(self, raw_df: pd.DataFrame):
        result = data_overview(raw_df)
        assert isinstance(result, dict)

    def test_has_shape(self, raw_df: pd.DataFrame):
        result = data_overview(raw_df)
        assert "shape" in result
        assert result["shape"] == (1309, 14)

    def test_has_dtypes(self, raw_df: pd.DataFrame):
        result = data_overview(raw_df)
        assert "dtypes" in result
        assert len(result["dtypes"]) == 14


class TestMissingSummary:
    def test_returns_dataframe(self, raw_df: pd.DataFrame):
        result = missing_summary(raw_df)
        assert isinstance(result, pd.DataFrame)

    def test_only_columns_with_missing(self, raw_df: pd.DataFrame):
        result = missing_summary(raw_df)
        assert (result["missing"] > 0).all()

    def test_sorted_by_percent(self, raw_df: pd.DataFrame):
        result = missing_summary(raw_df)
        assert result["percent"].is_monotonic_decreasing


class TestSurvivalRate:
    def test_returns_series(self, raw_df: pd.DataFrame):
        result = survival_rate(raw_df)
        assert isinstance(result, pd.Series)

    def test_values_sum_to_100(self, raw_df: pd.DataFrame):
        result = survival_rate(raw_df)
        assert abs(result.sum() - 100) < 0.01

    def test_has_0_and_1(self, raw_df: pd.DataFrame):
        result = survival_rate(raw_df)
        assert 0 in result.index
        assert 1 in result.index


class TestSurvivalByCategorical:
    def test_returns_dataframe(self, raw_df: pd.DataFrame):
        result = survival_by_categorical(raw_df, "Sex")
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self, raw_df: pd.DataFrame):
        result = survival_by_categorical(raw_df, "Sex")
        assert "passengers" in result.columns
        assert "survival_rate" in result.columns

    def test_survival_rate_in_range(self, raw_df: pd.DataFrame):
        result = survival_by_categorical(raw_df, "Pclass")
        assert (result["survival_rate"] >= 0).all()
        assert (result["survival_rate"] <= 100).all()


class TestSurvivalByNumerical:
    def test_returns_dataframe(self, raw_df: pd.DataFrame):
        result = survival_by_numerical(raw_df, "Age")
        assert isinstance(result, pd.DataFrame)

    def test_correct_number_of_bins(self, raw_df: pd.DataFrame):
        result = survival_by_numerical(raw_df, "Fare", bins=5)
        assert len(result) == 5


class TestCorrelationAnalysis:
    def test_returns_dataframe(self, raw_df: pd.DataFrame):
        result = correlation_analysis(raw_df)
        assert isinstance(result, pd.DataFrame)

    def test_diagonal_is_one(self, raw_df: pd.DataFrame):
        result = correlation_analysis(raw_df)
        assert all(result.iloc[i, i] == 1.0 for i in range(len(result)))

    def test_symmetric(self, raw_df: pd.DataFrame):
        result = correlation_analysis(raw_df)
        assert (result - result.T).abs().sum().sum() < 1e-10
