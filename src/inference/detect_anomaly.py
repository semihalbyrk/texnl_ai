# src/inference/detect_anomaly.py

import torch
import joblib
import pandas as pd
from pathlib import Path
from src.models.autoencoder import BetaVAE

ROOT = Path(__file__).resolve().parents[2]

# 1) Load freshly‐trained scaler & AE
SCL = joblib.load(ROOT / "models" / "scaler.gz")

# 2) Tam olarak 6 özellik kullanıyoruz:
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

# 3) İş kurallarınız için varsayılan eşikler (yüzde cinsinden)
LOWER_FILL = 30.0   # %30’un altı → düşük doluluk → under‐utilized
UPPER_FILL = 90.0   # %90’ın üstü → yüksek doluluk → over‐utilized

def label_anomalies(df: pd.DataFrame,
                    lower_fill_pct: float = LOWER_FILL,
                    upper_fill_pct: float = UPPER_FILL) -> pd.DataFrame:
    """
    - model_anom: VAE reconstruct error > μ + 2σ
    - under_util: fill_pct_per_task < lower_fill_pct  AND container_count > 1
    - over_util:  fill_pct_per_task > upper_fill_pct  AND container_count <= 1
    """

    # --- 1) Model‐temelli anomaly_score & eşik
    X = df[FEATURES].fillna(0)
    Xn = torch.tensor(SCL.transform(X), dtype=torch.float32)
    recon, _, _ = AE(Xn)
    err = ((Xn - recon).pow(2).mean(dim=1)
               .detach().cpu().numpy())

    thresh = err.mean() + 2 * err.std()
    model_anom = err > thresh

    df["anomaly_score"] = err

    # --- 2) Kural‐temelli anomalies
    under_util = (
        (df["fill_pct_per_task"] < lower_fill_pct) &
        (df["container_count"]    > 1)
    )
    over_util  = (
        (df["fill_pct_per_task"] > upper_fill_pct) &
        (df["container_count"]    <= 1)
    )

    # --- 3) Sonuç: hem VAE hem business‐rule
    df["is_anomaly"] = model_anom | under_util | over_util

    return df
