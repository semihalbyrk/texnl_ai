import torch, joblib, pandas as pd
from pathlib import Path
from src.models.autoencoder import BetaVAE

ROOT = Path(__file__).resolve().parents[2]
SCL  = joblib.load(ROOT / 'models' / 'scaler.gz')
AE   = BetaVAE(6)
AE.load_state_dict(torch.load(ROOT / 'models' / 'ae.pt'))
AE.eval()

def label_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    # Use per-container features for anomaly detection
    features = [
        'total_waste_per_container',
        'weekly_waste_per_container',
        'avg_waste_per_container',
        'tasks_per_week',
        'total_bags',
        'avg_bags'
    ]
    X = df[features].fillna(0)
    Xn = torch.tensor(SCL.transform(X), dtype=torch.float32)

    recon, _, _ = AE(Xn)
    err = torch.mean((Xn - recon).pow(2), dim=1).detach().numpy()
    thresh = err.mean() + 2 * err.std()

    df['recon_error'] = err
    df['is_anomaly']  = err > thresh
    return df
