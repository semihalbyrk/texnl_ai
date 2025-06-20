# src/features/build_features.py

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT  = DATA / "sp_feature_table.csv"

def build():
    # 1) Read raw tables
    sp   = pd.read_csv(DATA / "service_points.csv")
    ass  = pd.read_csv(DATA / "assets.csv")
    task = pd.read_csv(DATA / "tasks.csv")
    task["Date"] = pd.to_datetime(task["Date"], errors="coerce")
    task = task.dropna(subset=["Date"])

    # 2) Total capacity per SP
    cap = (
        ass
        .groupby("Location Details")["Weight Capacity"]
        .sum()
        .rename("total_capacity_kg")
        .reset_index()
    )

    # 3) Waste aggregates (kg vs pcs)
    kg  = task[task["Item UOM"] == "kg"]
    pcs = task[task["Item UOM"] == "pcs"]

    agg = (
        kg
        .groupby("Service Point")["Actual Amount (Item)"]
        .agg(total_kg="sum", avg_kg="mean")
        .reset_index()
        .merge(
            pcs
            .groupby("Service Point")["Actual Amount (Item)"]
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
        task
        .groupby("Service Point")
        .size()
        .div(weeks)
        .rename("tasks_per_week")
        .reset_index()
    )

    # 5) Merge everything
    df = (
        sp
        .merge(cap,  left_on="Service Point Name", right_on="Location Details", how="left")
        .merge(agg,  left_on="Service Point Name", right_on="Service Point",      how="left")
        .merge(freq, left_on="Service Point Name", right_on="Service Point",      how="left")
        .fillna(0)
    )

    # 6) New “fill” metrics (0–1)
    df["weekly_fill_pct"] = (
        df["total_kg"]
        .div((df["total_capacity_kg"] * df["tasks_per_week"]).replace(0, 1))
        .fillna(0)
        .clip(0, 1)
    )
    df["avg_fill_pct"] = (
        df["avg_kg"]
        .div(df["total_capacity_kg"].replace(0, 1))
        .fillna(0)
        .clip(0, 1)
    )

    # 7) Container count
    cnt = (
        ass["Location Details"]
        .value_counts()
        .rename("container_count")
        .reset_index()
        .rename(columns={"index": "Location Details"})
    )
    df = df.merge(cnt, on="Location Details", how="left")
    df["container_count"] = df["container_count"].fillna(0).astype(int)

    # 8) Persist
    df.to_csv(OUT, index=False)
    print(f"✅ Feature table saved → {OUT}")


if __name__ == "__main__":
    build()
