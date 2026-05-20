"""Tests for data processing module."""

import pandas as pd
import pytest
from src.data.processing import clean_data, engineer_features


class TestCleanData:
    def test_clean_data_preserves_rows(self, raw_df: pd.DataFrame):
        cleaned = clean_data(raw_df)
        assert len(cleaned) == len(raw_df)

    def test_clean_data_fills_embarked(self, raw_df: pd.DataFrame):
        cleaned = clean_data(raw_df)
        assert cleaned["Embarked"].isnull().sum() == 0

    def test_clean_data_fills_fare(self, raw_df: pd.DataFrame):
        cleaned = clean_data(raw_df)
        assert cleaned["Fare"].isnull().sum() == 0

    def test_clean_data_preserves_age_missing(self, raw_df: pd.DataFrame):
        cleaned = clean_data(raw_df)
        assert cleaned["Age"].isnull().sum() == 51


class TestEngineerFeatures:
    def test_engineer_features_adds_columns(self, clean_df: pd.DataFrame):
        eng = engineer_features(clean_df)
        new_cols = {"Title", "FamilySize", "IsAlone", "AgeGroup",
                    "FareGroup", "FarePerPerson", "HasCabin", "Lifeboat", "BodyRecovered"}
        assert new_cols.issubset(set(eng.columns))

    def test_family_size_correct(self, clean_df: pd.DataFrame):
        eng = engineer_features(clean_df)
        expected = clean_df["SibSp"] + clean_df["Parch"] + 1
        assert (eng["FamilySize"] == expected).all()

    def test_is_alone_binary(self, clean_df: pd.DataFrame):
        eng = engineer_features(clean_df)
        assert set(eng["IsAlone"].unique()).issubset({0, 1})

    def test_age_imputed(self, clean_df: pd.DataFrame):
        eng = engineer_features(clean_df, impute_age=True)
        assert eng["Age"].isnull().sum() == 0

    def test_age_not_imputed_when_disabled(self, clean_df: pd.DataFrame):
        eng = engineer_features(clean_df, impute_age=False)
        assert eng["Age"].isnull().sum() == 51

    def test_title_extraction(self, clean_df: pd.DataFrame):
        eng = engineer_features(clean_df)
        valid_titles = {"Mr", "Mrs", "Miss", "Master", "Officer", "Royalty", "Other", "Unknown"}
        assert set(eng["Title"].unique()).issubset(valid_titles)

    def test_fare_per_person_positive(self, clean_df: pd.DataFrame):
        eng = engineer_features(clean_df)
        assert (eng["FarePerPerson"] >= 0).all()

    def test_has_cabin_binary(self, clean_df: pd.DataFrame):
        eng = engineer_features(clean_df)
        assert set(eng["HasCabin"].unique()).issubset({0, 1})

    def test_age_group_categories(self, clean_df: pd.DataFrame):
        eng = engineer_features(clean_df)
        valid_groups = {"Child", "Teen", "Adult", "Senior"}
        assert set(eng["AgeGroup"].dropna().unique()).issubset(valid_groups)
