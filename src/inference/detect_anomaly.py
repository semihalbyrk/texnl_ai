import torch
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from src.models.autoencoder import BetaVAE

ROOT = Path(__file__).resolve().parents[2]

# 1) Load scaler & AE
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

# 2) Business‐rule thresholds
LOWER_FILL = 30.0   # %30 altı → under‐util
UPPER_FILL = 90.0   # %90 üstü → over‐util

def label_anomalies(df: pd.DataFrame,
                    percentile: float = 95.0,
                    lower_fill: float = LOWER_FILL,
                    upper_fill: float = UPPER_FILL) -> pd.DataFrame:
    """
    - anomaly_score: VAE reconstruction error
    - model_threshold: percentile’inci skor
    - is_anomaly: anomaly_score > model_threshold
    - anomaly_type:
       * "underutil"  if is_anomaly & fill_pct_per_task < lower_fill
       * "overutil"   if is_anomaly & fill_pct_per_task > upper_fill
       * "model"      if is_anomaly & fill_pct_per_task in [lower_fill,upper_fill]
       * "normal"     otherwise
    """

    # — compute anomaly_score —
    X  = df[FEATURES].fillna(0)
    Xn = torch.tensor(SCL.transform(X), dtype=torch.float32)
    recon, _, _ = AE(Xn)
    err = ((Xn - recon).pow(2).mean(dim=1)
               .detach().cpu().numpy())
    df["anomaly_score"] = err

    # — model‐based anomalies via percentile threshold —
    thr = np.percentile(err, percentile)
    is_model_anom = err > thr
    df["is_anomaly"] = is_model_anom

    # — assign anomaly_type —
    types = []
    for flag, fill in zip(is_model_anom, df["fill_pct_per_task"]):
        if not flag:
            types.append("normal")
        elif fill < lower_fill:
            types.append("underutil")
        elif fill > upper_fill:
            types.append("overutil")
        else:
            types.append("model")
    df["anomaly_type"] = types

    return df
