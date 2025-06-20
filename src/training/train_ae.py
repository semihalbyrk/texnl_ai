import torch, joblib, pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from src.models.autoencoder import BetaVAE

ROOT   = Path(__file__).resolve().parents[2]
DATA   = ROOT / "data"
MODELS = ROOT / "models"

# load features
FT = pd.read_csv(DATA / "sp_feature_table.csv")

# exactly the same 6 container‐aware fields
FEATS = [
    "tasks_per_week",
    "waste_per_task",
    "waste_per_task_per_ctr",
    "capacity_per_ctr",
    "fill_pct_per_task",
    "fill_pct_weekly",
]

X = FT[FEATS].fillna(0).values

# fit a fresh scaler
scaler = StandardScaler().fit(X)
joblib.dump(scaler, MODELS / "scaler.gz")

# prepare tensor
Xn = torch.tensor(scaler.transform(X), dtype=torch.float32)

# build & train
model = BetaVAE(in_dim=Xn.shape[1])
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(100):
    recon, mu, lv = model(Xn)
    loss = model.loss(Xn, recon, mu, lv)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d}/100  loss={loss.item():.4f}")

# save
torch.save(model.state_dict(), MODELS / "ae.pt")
print("✅ AE weights saved →", MODELS / "ae.pt")
print("✅ Scaler saved    →", MODELS / "scaler.gz")
