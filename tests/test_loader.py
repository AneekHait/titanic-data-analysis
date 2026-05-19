import pandas as pd
import pytest
from src.data.loader import load_titanic


def test_load_titanic_returns_dataframe(titanic_df: pd.DataFrame):
    assert isinstance(titanic_df, pd.DataFrame)


def test_load_titanic_has_expected_columns(titanic_df: pd.DataFrame):
    expected = {"PassengerId", "Survived", "Pclass", "Name", "Sex",
                "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"}
    assert expected.issubset(set(titanic_df.columns))


def test_load_titanic_non_empty(titanic_df: pd.DataFrame):
    assert len(titanic_df) > 0


def test_load_titanic_survived_binary(titanic_df: pd.DataFrame):
    assert titanic_df["Survived"].isin({0, 1}).all()


@pytest.fixture
def titanic_df() -> pd.DataFrame:
    return load_titanic()
