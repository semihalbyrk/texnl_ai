import torch, joblib, pandas as pd
from pathlib import Path
from src.models.autoencoder import BetaVAE

ROOT = Path(__file__).resolve().parents[2]

# load freshly‐trained scaler & AE
SCL = joblib.load(ROOT / "models" / "scaler.gz")

# we now have exactly 6 features going into the AE:
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

def label_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].fillna(0)
    Xn = torch.tensor(SCL.transform(X), dtype=torch.float32)
    recon, _, _ = AE(Xn)
    err = ((Xn - recon).pow(2).mean(dim=1)).detach().cpu().numpy()

    # dynamic threshold: μ + 2σ
    thresh = err.mean() + 2 * err.std()
    df["anomaly_score"] = err
    df["is_anomaly"]     = err > thresh
    return df
