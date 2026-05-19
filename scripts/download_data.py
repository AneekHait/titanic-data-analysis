#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import download_titanic

if __name__ == "__main__":
    download_titanic()
