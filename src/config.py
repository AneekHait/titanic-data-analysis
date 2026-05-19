from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS_FIGURES = ROOT / "outputs" / "figures"

TARGET = "Survived"

NUM_COLS = ["Age", "Fare", "SibSp", "Parch"]
CAT_COLS = ["Sex", "Embarked", "Pclass"]
ID_COL = "PassengerId"

ALL_FEATURES = [TARGET] + NUM_COLS + CAT_COLS + ["Name", "Ticket", "Cabin"]
