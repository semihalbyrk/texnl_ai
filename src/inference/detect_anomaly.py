import torch, joblib, pandas as pd
from pathlib import Path
from src.models.autoencoder import BetaVAE

ROOT = Path(__file__).resolve().parents[2]
SCL  = joblib.load(ROOT/'models'/'scaler.gz')
AE   = BetaVAE(6); AE.load_state_dict(torch.load(ROOT/'models'/'ae.pt'))
AE.eval()

def label_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    Xn = torch.tensor(SCL.transform(df[['total_kg','total_capacity_kg',
                                        'tasks_per_week','util_ratio',
                                        'avg_kg','avg_bags']].fillna(0)),
                      dtype=torch.float32)
    recon, _, _ = AE(Xn)
    err = torch.mean((Xn - recon).pow(2), dim=1).detach().numpy()
    threshold = err.mean() + 2*err.std()
    df['recon_error'] = err
    df['is_anomaly']  = err > threshold
    return df
