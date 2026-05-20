"""Shared pytest fixtures for Titanic EDA tests."""

import pytest
import pandas as pd

from src.data.loader import load_titanic
from src.data.processing import clean_data, engineer_features


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    """Raw titanic5 DataFrame."""
    return load_titanic()


@pytest.fixture(scope="session")
def clean_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Cleaned DataFrame with missing values handled."""
    return clean_data(raw_df)


@pytest.fixture(scope="session")
def engineered_df(clean_df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame with engineered features."""
    return engineer_features(clean_df)
