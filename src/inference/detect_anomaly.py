# src/inference/detect_anomaly.py

import torch
import joblib
import pandas as pd
from pathlib import Path
from src.models.autoencoder import BetaVAE

ROOT = Path(__file__).resolve().parents[2]

# load the freshly retrained scaler and AE
SCL = joblib.load(ROOT / "models" / "scaler.gz")

# 8-dimensional input (must match build_features)
AE = BetaVAE(in_dim=8)
AE.load_state_dict(torch.load(ROOT / "models" / "ae.pt"))
AE.eval()

def label_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    feats = [
        "total_kg",
        "total_capacity_kg",
        "tasks_per_week",
        "avg_kg",
        "avg_bags",
        "weekly_fill_pct",
        "avg_fill_pct",
        "container_count",
    ]
    X = df[feats].fillna(0)
    Xn = torch.tensor(SCL.transform(X), dtype=torch.float32)

    recon, _, _ = AE(Xn)
    err = torch.mean((Xn - recon).pow(2), dim=1).detach().numpy()

    # threshold = mean + 2*std
    thresh = err.mean() + 2 * err.std()
    df["recon_error"] = err
    df["is_anomaly"]  = err > thresh

    return df
