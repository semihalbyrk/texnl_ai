import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT  = DATA / "sp_feature_table.csv"

def build():
    # 1) Kaynak CSV'ler
    sp  = pd.read_csv(DATA / "service_points.csv")
    ass = pd.read_csv(DATA / "assets.csv")

    task = pd.read_csv(DATA / "tasks.csv")          # normal oku
    task['Date'] = pd.to_datetime(task['Date'],     # güvenli dönüştür
                                  errors='coerce')
    task = task.dropna(subset=['Date'])             # geçersiz tarihleri at

    # 2) Toplam kapasite (kg) — asset bazlı
    cap = (ass.groupby('Location Details')['Weight Capacity']
              .sum()
              .rename('total_capacity_kg')
              .reset_index())

    # 3) Task ağırlık / adet
    kg  = task[task['Item UOM'] == 'kg']
    pcs = task[task['Item UOM'] == 'pcs']

    agg = (kg.groupby('Service Point')['Actual Amount (Item)']
             .agg(total_kg='sum', avg_kg='mean')
             .reset_index()
             .merge(
              pcs.groupby('Service Point')['Actual Amount (Item)']
                  .agg(total_bags='sum', avg_bags='mean')
                  .reset_index(),
              on='Service Point', how='outer')
             .fillna(0))

    # 4) Haftalık görev frekansı
    total_days = (task['Date'].max() - task['Date'].min()).days + 1
    weeks      = max(total_days / 7, 1)

    freq = (task.groupby('Service Point')
                  .size()
                  .div(weeks)
                  .rename('tasks_per_week')
                  .reset_index())

    # 5) Birleştir
    df = (sp.merge(cap, left_on='Service Point Name',
                   right_on='Location Details', how='left')
            .merge(agg, left_on='Service Point Name',
                   right_on='Service Point', how='left')
            .merge(freq, left_on='Service Point Name',
                   right_on='Service Point', how='left')
            .fillna(0))

    df['util_ratio'] = df['total_kg'] / df['total_capacity_kg'].clip(lower=1)

    df.to_csv(OUT, index=False)
    print(f"✅ Feature table saved → {OUT}")

if __name__ == "__main__":
    build()
