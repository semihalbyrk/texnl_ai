import torch
import joblib
import pandas as pd
from pathlib import Path
from src.models.autoencoder import BetaVAE

ROOT = Path(__file__).resolve().parents[2]
# load freshly re-trained scaler and AE
SCL = joblib.load(ROOT / "models" / "scaler.gz")
AE  = BetaVAE(input_dim=7)                            # <-- now 7 dims
AE.load_state_dict(torch.load(ROOT / "models" / "ae.pt"))
AE.eval()

def label_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    # pick the same 7 features that we scaled at train time
    feats = [
        "total_kg",
        "tasks_per_week",
        "weekly_fill_pct",
        "avg_fill_pct",
        "avg_kg",
        "avg_bags",
        "container_count",
    ]
    # ensure missing columns are zeroed
    X = df.reindex(columns=feats, fill_value=0).values
    Xn = torch.tensor(SCL.transform(X), dtype=torch.float32)

    with torch.no_grad():
        recon, _, _ = AE(Xn)
        err = torch.mean((Xn - recon).pow(2), dim=1).cpu().numpy()

    # naive threshold: μ + 2σ
    thresh = err.mean() + 2 * err.std()

    df["recon_error"] = err
    df["is_anomaly"]  = err > thresh
    return df
