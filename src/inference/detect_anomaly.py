# src/inference/detect_anomaly.py

import torch
import joblib
import pandas as pd
from pathlib import Path
from src.models.autoencoder import BetaVAE

ROOT = Path(__file__).resolve().parents[2]

# ─── Load the freshly retrained scaler and AE ────────────────────────────────
SCL = joblib.load(ROOT / "models" / "scaler.gz")

# ─── Must match the 10‐dim input used in train_ae.py ─────────────────────────
AE = BetaVAE(in_dim=10)
AE.load_state_dict(torch.load(ROOT / "models" / "ae.pt"))
AE.eval()

def label_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    # exactly the same 10 columns you scaled & trained on:
    feats = [
        "total_kg",
        "total_capacity_kg",
        "tasks_per_week",
        "avg_kg",
        "avg_bags",
        "container_count",
        "capacity_per_container",
        "waste_per_container",
        "tasks_per_container",
        "waste_per_task_per_container",
    ]

    # extract, fill na, scale, then feed into AE
    X     = df[feats].fillna(0)
    Xn    = torch.tensor(SCL.transform(X), dtype=torch.float32)
    recon, mu, logvar = AE(Xn)

    # MSE per row
    err   = torch.mean((Xn - recon).pow(2), dim=1).detach().numpy()
    thresh = err.mean() + 2 * err.std()

    df["recon_error"] = err
    df["is_anomaly"]  = err > thresh
    return df
