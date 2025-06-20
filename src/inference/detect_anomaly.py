# src/inference/detect_anomaly.py

import torch
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from src.models.autoencoder import BetaVAE

ROOT = Path(__file__).resolve().parents[2]

# ——— load VAE & scaler ———
SCL = joblib.load(ROOT / "models" / "scaler.gz")
FEATURES = [
    "tasks_per_week",
    "waste_per_task",
    "waste_per_task_per_ctr",
    "capacity_per_ctr",
    "fill_pct_per_task",
    "fill_pct_weekly",
]
AE = BetaVAE(in_dim=len(FEATURES))
AE.load_state_dict(torch.load(ROOT / "models" / "ae.pt"))
AE.eval()

# ——— iş kuralları eşikleri ———
LOWER_FILL = 30.0   # %30 altı → under‐util
UPPER_FILL = 90.0   # %90 üstü → over‐util

def label_anomalies(df: pd.DataFrame,
                    percentile: float = 90.0,
                    lower_fill: float = LOWER_FILL,
                    upper_fill: float = UPPER_FILL) -> pd.DataFrame:
    """
    1) anomaly_score = VAE reconstruction error
    2) model_threshold = percentile’inci skor
       model_anom = score > model_threshold
    3) under_util = fill_pct_per_task < lower_fill
    4) over_util  = fill_pct_per_task > upper_fill
    5) anomaly_type = one of ["normal","model","underutil","overutil","mixed"]
    6) is_anomaly = anomaly_type != "normal"
    """

    # — compute anomaly_score —
    X = df[FEATURES].fillna(0)
    Xn = torch.tensor(SCL.transform(X), dtype=torch.float32)
    recon, _, _ = AE(Xn)
    err = ((Xn - recon).pow(2).mean(dim=1)
               .detach().cpu().numpy())
    df["anomaly_score"] = err

    # — model‐based anomalies via percentile —
    thresh = np.percentile(err, percentile)
    model_anom = err > thresh

    # — rule‐based under/over util —
    under_util = df["fill_pct_per_task"] < lower_fill
    over_util  = df["fill_pct_per_task"] > upper_fill

    # — assign anomaly_type —
    types = []
    for m, u, o in zip(model_anom, under_util, over_util):
        flags = []
        if m: flags.append("model")
        if u: flags.append("underutil")
        if o: flags.append("overutil")
        if not flags:
            types.append("normal")
        elif len(flags) == 1:
            types.append(flags[0])
        else:
            types.append("mixed")
    df["anomaly_type"] = types
    df["is_anomaly"]   = df["anomaly_type"] != "normal"

    return df
