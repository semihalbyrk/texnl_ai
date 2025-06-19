import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def load_csv(relative_path: str) -> pd.DataFrame:
    """data/... klasöründen CSV okur"""
    return pd.read_csv(PROJECT_ROOT / "data" / relative_path)
