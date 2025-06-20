# src/training/train_ae.py
import torch
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from src.models.autoencoder import BetaVAE

# ─── Paths ─────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MODELS = ROOT / "models"
FT = pd.read_csv(DATA / "sp_feature_table.csv")

# ─── Select the exact 8 features we feed into the AE ───────────
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

X = FT[feats].fillna(0).values

# ─── Standard‐scale them ───────────────────────────────────────
scaler = StandardScaler().fit(X)
joblib.dump(scaler, MODELS / "scaler.gz")
Xn = torch.tensor(scaler.transform(X), dtype=torch.float32)

# ─── Build & train the VAE ────────────────────────────────────
in_dim = Xn.shape[1]                # should be 8
model = BetaVAE(in_dim=in_dim)     # note `in_dim=...`
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 100
for epoch in range(epochs):
    recon, mu, lv = model(Xn)
    loss = model.loss(Xn, recon, mu, lv)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d}/{epochs:03d}  loss={loss.item():.4f}")

# ─── Save final weights ────────────────────────────────────────
torch.save(model.state_dict(), MODELS / "ae.pt")
print(f"✅ Trained AE saved → {MODELS/'ae.pt'}")
print(f"✅ Scaler saved    → {MODELS/'scaler.gz'}")
