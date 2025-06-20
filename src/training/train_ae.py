# src/training/train_ae.py
import torch
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from src.models.autoencoder import BetaVAE

# ─── Paths ─────────────────────────────────────────────────────
ROOT   = Path(__file__).resolve().parents[2]
DATA   = ROOT / "data"
MODELS = ROOT / "models"

# ─── 1) Load your feature table ────────────────────────────────
FT = pd.read_csv(DATA / "sp_feature_table.csv")

# ─── 2) Compute container-normalized metrics ────────────────────
# avoid division by zero by replacing 0→1 on denoms
FT["container_count_safe"] = FT["container_count"].replace(0, 1)

FT["capacity_per_container"]         = FT["total_capacity_kg"]  / FT["container_count_safe"]
FT["waste_per_container"]            = FT["total_kg"]           / FT["container_count_safe"]
FT["tasks_per_container"]            = FT["tasks_per_week"]     / FT["container_count_safe"]
FT["waste_per_task_per_container"]   = FT["waste_per_container"] / FT["tasks_per_week"].replace(0, 1)

# ─── 3) Select exactly the 10 dims we’ll feed into the AE ───────
feats = [
    "total_kg",
    "total_capacity_kg",
    "tasks_per_week",
    "avg_kg",
    "avg_bags",
    "container_count",            # raw count
    "capacity_per_container",
    "waste_per_container",
    "tasks_per_container",
    "waste_per_task_per_container"
]
X = FT[feats].fillna(0).values

# ─── 4) Standard‐scale ───────────────────────────────────────────
scaler = StandardScaler().fit(X)
joblib.dump(scaler, MODELS / "scaler.gz")
Xn = torch.tensor(scaler.transform(X), dtype=torch.float32)

# ─── 5) Build & train the VAE ─────────────────────────────────
in_dim = Xn.shape[1]                # should be 10 now
model  = BetaVAE(in_dim=in_dim)
opt    = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 100
for epoch in range(epochs):
    recon, mu, logvar = model(Xn)
    loss = model.loss(Xn, recon, mu, logvar)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d}/{epochs:03d}  loss={loss.item():.4f}")

# ─── 6) Save final artifacts ────────────────────────────────────
torch.save(model.state_dict(), MODELS / "ae.pt")
print(f"✅ Trained AE saved → {MODELS/'ae.pt'}")
print(f"✅ Scaler saved    → {MODELS/'scaler.gz'}")
