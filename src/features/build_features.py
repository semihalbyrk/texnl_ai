import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT  = DATA / "sp_feature_table.csv"

def build():
    # 1) read raw
    sp   = pd.read_csv(DATA / "service_points.csv")
    ass  = pd.read_csv(DATA / "assets.csv")
    task = pd.read_csv(DATA / "tasks.csv")
    task["Date"] = pd.to_datetime(task["Date"], errors="coerce")
    task = task.dropna(subset=["Date"])

    # 2) compute total_days & weeks (you said: use 01.01.2024–18.06.2025 = 532 days)
    total_days = 532
    weeks = total_days / 7

    # 3) capacity per SP
    cap = (
        ass
        .groupby("Location Details")["Weight Capacity"]
        .sum()
        .rename("total_capacity_kg")
        .reset_index()
    )

    # 4) waste aggregates
    kg  = task[task["Item UOM"] == "kg"]
    pcs = task[task["Item UOM"] == "pcs"]

    agg_kg = (
        kg
        .groupby("Service Point")["Actual Amount (Item)"]
        .sum()
        .rename("total_kg")
    )
    agg_pcs = (
        pcs
        .groupby("Service Point")["Actual Amount (Item)"]
        .sum()
        .rename("total_bags")
    )

    # 5) weekly totals
    weekly = (
        agg_kg.div(weeks)
        .to_frame("weekly_total_kg")
        .join(
            agg_pcs.div(weeks).to_frame("weekly_total_bags"),
            how="outer"
        )
        .fillna(0)
        .reset_index().rename(columns={"Service Point":"Service Point Name"})
    )

    # 6) tasks per week
    freq = (
        task
        .groupby("Service Point")
        .size()
        .div(weeks)
        .rename("tasks_per_week")
        .reset_index().rename(columns={"Service Point":"Service Point Name"})
    )

    # 7) merge
    df = (
        sp.rename(columns={"Service Point Name":"Service Point Name"})
        .merge(cap,    left_on="Service Point Name", right_on="Location Details", how="left")
        .merge(weekly, on="Service Point Name",                       how="left")
        .merge(freq,   on="Service Point Name",                       how="left")
        .fillna(0)
    )

    # 8) container count
    cnt = (
        ass["Location Details"]
        .value_counts()
        .rename("container_count")
        .reset_index()
        .rename(columns={"index":"Location Details"})
    )
    df = df.merge(cnt, on="Location Details", how="left")
    df["container_count"] = df["container_count"].fillna(0).astype(int)

    # 9) normalized per-task/container
    df["waste_per_task"]         = (df["weekly_total_kg"] / df["tasks_per_week"]).fillna(0)
    df["waste_per_task_per_ctr"] = (
        df["weekly_total_kg"]
        .div(df["tasks_per_week"] * df["container_count"])
        .fillna(0)
    )
    df["capacity_per_ctr"]       = (
        df["total_capacity_kg"]
        .div(df["container_count"].replace(0,1))
    )
    df["fill_pct_per_task"]      = (
        df["waste_per_task"]
        .div(df["capacity_per_ctr"])
        .fillna(0)
        .clip(0,1) * 100
    )
    df["fill_pct_weekly"]        = (
        df["weekly_total_kg"]
        .div(df["capacity_per_ctr"] * df["tasks_per_week"])
        .fillna(0)
        .clip(0,1) * 100
    )

    # 10) save
    df.to_csv(OUT, index=False)
    print(f"✅ Feature table saved to {OUT}")
