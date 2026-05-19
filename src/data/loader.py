import pandas as pd
from pathlib import Path
from src.config import DATA_RAW


def download_titanic(dest: Path | None = None) -> Path:
    dest = dest or DATA_RAW
    dest.mkdir(parents=True, exist_ok=True)

    train_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    out_path = dest / "titanic.csv"

    df = pd.read_csv(train_url)
    df.to_csv(out_path, index=False)
    print(f"Downloaded Titanic dataset to {out_path} ({len(df)} rows)")
    return out_path


def load_titanic(path: Path | None = None) -> pd.DataFrame:
    path = path or DATA_RAW / "titanic.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `python scripts/download_data.py` first."
        )
    return pd.read_csv(path)
