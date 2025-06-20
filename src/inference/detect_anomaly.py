# src/inference/detect_anomaly.py

import torch
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from src.models.autoencoder import BetaVAE

ROOT = Path(__file__).resolve().parents[2]

# load scaler & AE
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

# thresholds
LOWER_FILL = 30.0
UPPER_FILL = 90.0

def label_anomalies(df: pd.DataFrame,
                    percentile: float = 95.0,
                    lower_fill: float = LOWER_FILL,
                    upper_fill: float = UPPER_FILL) -> pd.DataFrame:
    # 1) model‐based scores
    X  = df[FEATURES].fillna(0)
    Xn = torch.tensor(SCL.transform(X), dtype=torch.float32)
    recon, _, _ = AE(Xn)
    err = ((Xn - recon).pow(2).mean(dim=1)
               .detach().cpu().numpy())
    df["anomaly_score"] = err

    # 2) model‐threshold
    thr = np.percentile(err, percentile)
    is_model_anom = err > thr

    # 3) rule‐based types for those flagged
    types = []
    for flag, fill, ctr in zip(is_model_anom,
                               df["fill_pct_per_task"],
                               df["container_count"]):
        if ctr == 0:
            types.append("normal")
        elif not flag:
            types.append("normal")
        elif fill < lower_fill:
            types.append("underutil")
        elif fill > upper_fill:
            types.append("overutil")
        else:
            types.append("model")

    df["anomaly_type"] = types
    df["is_anomaly"]  = df["anomaly_type"] != "normal"

    return df
