"""Data cleaning and feature engineering for the Titanic dataset."""

import re

import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw Titanic dataset.

    - Fills missing Embarked with mode (Southampton)
    - Fills missing Fare with median by Pclass
    - Leaves Age missing for imputation later (only 51 missing)

    Args:
        df: Raw titanic5 DataFrame.

    Returns:
        Cleaned DataFrame with missing values handled.
    """
    df = df.copy()

    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Fare"] = df["Fare"].fillna(df.groupby("Pclass")["Fare"].transform("median"))

    return df


def _impute_age(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing Age using median by Sex and Pclass."""
    df = df.copy()
    medians = df.groupby(["Sex", "Pclass"])["Age"].median()
    mask = df["Age"].isnull()
    df.loc[mask, "Age"] = df.loc[mask].apply(
        lambda row: medians[(row["Sex"], row["Pclass"])], axis=1
    )
    return df


def _extract_title(name: str) -> str:
    """Extract title from passenger name."""
    match = re.search(r", ([A-Za-z]+) ", name)
    if not match:
        return "Unknown"
    raw = match.group(1)
    title_map = {
        "Mr": "Mr", "Mrs": "Mrs", "Miss": "Miss", "Master": "Master",
        "Dr": "Officer", "Rev": "Officer", "Revd": "Officer",
        "Major": "Officer", "Col": "Officer", "Colonel": "Officer",
        "Capt": "Officer", "Captain": "Officer",
        "Mlle": "Miss", "Mme": "Mrs", "Ms": "Miss",
        "Countess": "Royalty", "Lady": "Royalty",
        "Sir": "Royalty", "Don": "Royalty",
        "Jonkheer": "Royalty", "Dona": "Royalty",
        "Sra": "Mrs", "Sr": "Mr", "Fr": "Officer",
    }
    return title_map.get(raw, "Other")


def _parse_boat_body(val) -> tuple[str | None, str | None]:
    """Parse BoatBody column into lifeboat and body recovery.

    Raw values include real boat numbers/letters ("1", "A"), bracketed body
    recovery codes ("[190]"), and blank whitespace strings for passengers with
    no recorded disposition.
    """
    if pd.isna(val):
        return None, None
    s = str(val).strip()
    if not s:
        return None, None
    if s.startswith("[") and s.endswith("]"):
        return None, s.strip("[]")
    return s, None


def engineer_features(df: pd.DataFrame, impute_age: bool = True) -> pd.DataFrame:
    """Create derived features from the cleaned Titanic dataset.

    Features created:
        - Title: extracted from Name (Mr, Mrs, Miss, Master, Officer, Royalty, Other)
        - FamilySize: SibSp + Parch + 1
        - IsAlone: 1 if FamilySize == 1, else 0
        - AgeGroup: Child (0-16), Teen (17-25), Adult (26-50), Senior (51+)
        - FareGroup: Low, Medium, High, Very High (quartile-based)
        - FarePerPerson: Fare / FamilySize
        - HasCabin: 1 if Occupation is not null (proxy for cabin info)
        - Lifeboat: extracted lifeboat number from BoatBody
        - BodyRecovered: 1 if body was recovered, 0 otherwise
        - Age (imputed): if impute_age=True, fills missing ages

    Args:
        df: Cleaned Titanic DataFrame.
        impute_age: Whether to impute missing ages using median by Sex+Pclass.

    Returns:
        DataFrame with engineered features.
    """
    df = df.copy()

    if impute_age:
        df = _impute_age(df)

    df["Title"] = df["Name"].apply(_extract_title)
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["FarePerPerson"] = df["Fare"] / df["FamilySize"]

    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 16, 25, 50, 100],
        labels=["Child", "Teen", "Adult", "Senior"],
    )

    df["FareGroup"] = pd.qcut(
        df["Fare"], q=4, labels=["Low", "Medium", "High", "Very High"], duplicates="drop"
    )

    df["HasCabin"] = df["Occupation"].notna().astype(int)

    boat_body = df["BoatBody"].apply(_parse_boat_body)
    df["Lifeboat"] = boat_body.apply(lambda x: x[0])
    df["BodyRecovered"] = boat_body.apply(lambda x: 1 if x[1] else 0)

    return df
