import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT  = DATA / "sp_feature_table.csv"

def build():
    # 1) Load raw CSVs
    sp   = pd.read_csv(DATA / "service_points.csv")
    ass  = pd.read_csv(DATA / "assets.csv")
    task = pd.read_csv(DATA / "tasks.csv", parse_dates=["Date"], dayfirst=True, 
                       infer_datetime_format=True, errors="coerce")
    task = task.dropna(subset=["Date"])

    # 2) Total capacity per SP (kg)
    cap = (
        ass
        .groupby("Location Details")["Weight Capacity"]
        .sum()
        .rename("total_capacity_kg")
        .reset_index()
    )

    # 3) Total & avg waste by UOM
    kg  = task[task["Item UOM"] == "kg"]
    pcs = task[task["Item UOM"] == "pcs"]

    agg = (
        kg.groupby("Service Point")["Actual Amount (Item)"]
          .agg(total_kg="sum", avg_kg="mean")
          .reset_index()
          .merge(
            pcs.groupby("Service Point")["Actual Amount (Item)"]
               .agg(total_bags="sum", avg_bags="mean")
               .reset_index(),
            on="Service Point", how="outer"
          )
          .fillna(0)
    )

    # 4) Weekly task frequency
    total_days = (task["Date"].max() - task["Date"].min()).days + 1
    weeks      = max(total_days / 7, 1)
    freq = (
        task.groupby("Service Point")
            .size()
            .div(weeks)
            .rename("tasks_per_week")
            .reset_index()
    )

    # 5) Container count per SP
    #    We assume ass["Location Details"] matches sp["Service Point Name"]
    cnt = (
        ass["Location Details"]
        .value_counts()
        .rename("container_count")
        .reset_index()
        .rename(columns={"index": "Location Details"})
    )

    # 6) Merge everything
    df = (
        sp
        .merge(cap,  left_on="Service Point Name", right_on="Location Details", how="left")
        .merge(agg,  left_on="Service Point Name", right_on="Service Point",    how="left")
        .merge(freq, left_on="Service Point Name", right_on="Service Point",    how="left")
        .merge(cnt,  left_on="Service Point Name", right_on="Location Details", how="left")
        .fillna({
            "total_capacity_kg": 0,
            "total_kg":          0,
            "avg_kg":            0,
            "total_bags":        0,
            "avg_bags":          0,
            "tasks_per_week":    0,
            "container_count":   0,
        })
    )

    # 7) New fill metrics
    df["weekly_fill_pct"] = (
        df["total_kg"]
          .div((df["total_capacity_kg"] * df["tasks_per_week"]).replace(0, pd.NA))
          .fillna(0)
          .clip(0, 1)
          * 100
    ).round(1)

    df["avg_fill_pct"] = (
        df["avg_kg"]
          .div(df["total_capacity_kg"].replace(0, pd.NA))
          .fillna(0)
          .clip(0, 1)
          * 100
    ).round(1)

    # 8) Write out
    df.to_csv(OUT, index=False)
    print(f"✅ Feature table saved → {OUT}")

if __name__ == "__main__":
    build()
