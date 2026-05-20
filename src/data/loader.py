import re
import pandas as pd
from pathlib import Path
from src.config import DATA_RAW


def _parse_price(s: str) -> float | None:
    if pd.isna(s):
        return None
    s = str(s).replace("£", "").strip()
    parts = s.split()
    total = 0.0
    for p in parts:
        if "s" in p:
            total += int(p.replace("s", "")) / 20
        elif "d" in p:
            total += int(p.replace("d", "")) / 240
        elif p.isdigit():
            total += int(p)
    return round(total, 2)


def _map_embarked(port: str) -> str:
    mapping = {"Southampton": "S", "Cherbourg": "C", "Queenstown": "Q", "Belfast": "B"}
    return mapping.get(port, port)


def download_titanic(dest: Path | None = None) -> Path:
    dest = dest or DATA_RAW
    dest.mkdir(parents=True, exist_ok=True)

    url = "https://hbiostat.org/data/repo/titanic5.csv"
    out_path = dest / "titanic5.csv"

    raw = pd.read_csv(url)

    df = pd.DataFrame()
    df["PassengerId"] = raw["Name_ID"]
    df["Survived"] = raw["Survived"]
    df["Pclass"] = raw["Class"].astype(int)
    df["Name"] = raw["Name"].str.strip()
    df["Sex"] = raw["Sex"]
    df["Age"] = raw["Age"]
    df["SibSp"] = raw["sibsp"]
    df["Parch"] = raw["parch"]
    df["Ticket"] = raw["Ticket"]
    df["Fare"] = raw["Price"].apply(_parse_price)
    df["Embarked"] = raw["Joined"].apply(_map_embarked)
    df["Occupation"] = raw["Occupation"]
    df["BoatBody"] = raw["Boat [Body]"]
    df["NameId"] = raw["Name_ID"]

    df.to_csv(out_path, index=False)
    print(f"Downloaded titanic5 dataset to {out_path} ({len(df)} rows, {len(df.columns)} cols)")
    return out_path


def load_titanic(path: Path | None = None) -> pd.DataFrame:
    path = path or DATA_RAW / "titanic5.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `python scripts/download_data.py` first."
        )
    return pd.read_csv(path)
