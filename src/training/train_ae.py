import torch, pandas as pd, joblib
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from src.models.autoencoder import BetaVAE

ROOT = Path(__file__).resolve().parents[2]
ft   = pd.read_csv(ROOT/'data/sp_feature_table.csv')
X    = ft[['total_kg','total_capacity_kg','tasks_per_week',
           'util_ratio','avg_kg','avg_bags']].fillna(0).values

scaler = StandardScaler().fit(X)
joblib.dump(scaler, ROOT/'models'/'scaler.gz')
Xn = torch.tensor(scaler.transform(X), dtype=torch.float32)

model = BetaVAE(Xn.shape[1])
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(100):
    recon, mu, lv = model(Xn)
    loss = model.loss(Xn, recon, mu, lv)
    opt.zero_grad(); loss.backward(); opt.step()
    if epoch % 10 == 0:
        print(epoch, loss.item())

torch.save(model.state_dict(), ROOT/'models'/'ae.pt')
