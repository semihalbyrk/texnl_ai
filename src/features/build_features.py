import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT  = DATA / "sp_feature_table.csv"

def build():
    # 1) Load source CSVs
    sp  = pd.read_csv(DATA / "service_points.csv")
    ass = pd.read_csv(DATA / "assets.csv")

    # 2) Read & clean Task records
    task = pd.read_csv(DATA / "tasks.csv")
    task['Date'] = pd.to_datetime(task['Date'], errors='coerce')
    task = task.dropna(subset=['Date'])

    # 3) Total capacity per SP
    cap = (
        ass.groupby('Location Details')['Weight Capacity']
           .sum()
           .rename('total_capacity_kg')
           .reset_index()
    )

    # 4) Waste aggregation by UOM
    kg  = task[task['Item UOM'] == 'kg']
    pcs = task[task['Item UOM'] == 'pcs']
    agg = (
        kg.groupby('Service Point')['Actual Amount (Item)']
          .agg(total_kg='sum', avg_kg='mean')
          .reset_index()
          .merge(
              pcs.groupby('Service Point')['Actual Amount (Item)']
                 .agg(total_bags='sum', avg_bags='mean')
                 .reset_index(),
              on='Service Point', how='outer'
          )
          .fillna(0)
    )

    # 5) Weekly task frequency
    total_days = (task['Date'].max() - task['Date'].min()).days + 1
    weeks      = max(total_days / 7, 1)
    freq = (
        task.groupby('Service Point')
            .size()
            .div(weeks)
            .rename('tasks_per_week')
            .reset_index()
    )

    # 6) Container count per SP
    if 'Location Details' in ass.columns:
        cnt = (
            ass['Location Details']
               .value_counts()
               .rename('container_count')
               .reset_index()
        )
        cnt.columns = ['Location Details','container_count']
    elif 'Service Point Name' in ass.columns:
        cnt = (
            ass['Service Point Name']
               .value_counts()
               .rename('container_count')
               .reset_index()
        )
        cnt.columns = ['Service Point Name','container_count']
    else:
        cnt = pd.DataFrame(columns=['Service Point Name','container_count'])

    # 7) Merge all features
    df = (
        sp
          .merge(cap, on='Location Details', how='left')
          .merge(agg, left_on='Service Point Name', right_on='Service Point', how='left')
          .merge(freq, left_on='Service Point Name', right_on='Service Point', how='left')
          .merge(cnt, left_on='Service Point Name', right_on=cnt.columns[0], how='left')
    )
    df['container_count'] = df['container_count'].fillna(0).astype(int)

    # 8) Per-container metrics
    df['total_waste_per_container']   = df['total_kg']   / df['container_count'].replace(0,1)
    df['weekly_waste_per_container']  = (df['total_kg']/weeks) / df['container_count'].replace(0,1)
    df['avg_waste_per_container']     = df['avg_kg']     / df['container_count'].replace(0,1)

    # 9) Save
    df.to_csv(OUT, index=False)
    print(f"✅ Feature table saved → {OUT}")

if __name__ == '__main__':
    build()
