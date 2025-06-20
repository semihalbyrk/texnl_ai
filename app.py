import pathlib
import pandas as pd
import numpy as np
import streamlit as st

from src.features.build_features import build as build_features
from src.inference.detect_anomaly import label_anomalies
from src.inference.recommend_assets import recommend

# Paths
ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data"

st.set_page_config(page_title="TexNL Efficiency AI", layout="wide")

# 1) Ensure feature table exists
csv_path = DATA / "sp_feature_table.csv"
if not csv_path.exists():
    st.info("CSV not found — generating from raw Excel…")
    xls = pd.ExcelFile(DATA / "TexNL_Data.xlsx")
    xls.parse("Service Points").to_csv(DATA /"service_points.csv", index=False)
    xls.parse("Assets").to_csv(DATA /"assets.csv", index=False)
    xls.parse("Task Record").to_csv(DATA /"tasks.csv", index=False)
    build_features()

df = pd.read_csv(csv_path)

# 2) Tag anomalies via AE
df = label_anomalies(df)

# 3) Sidebar filters
st.sidebar.header("🔍 Filters")
fa = st.sidebar.checkbox("🚨 Only anomalies", False)
fc = st.sidebar.checkbox("📦 Capacity > 0", True)
mc = st.sidebar.slider("Min. Containers/SP", 0, int(df.container_count.max()), 0)
s  = st.sidebar.text_input("Search SP")

df_view = df.copy()
if fa:
    df_view = df_view[df_view.is_anomaly]
if fc:
    df_view = df_view[df_view.total_capacity_kg > 0]
if mc:
    df_view = df_view[df_view.container_count >= mc]
if s:
    df_view = df_view[df_view["Service Point Name"].str.contains(s, case=False, na=False)]

# 4) KPI cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("SP Count", len(df_view))
c2.metric("Anomalous SP", int(df_view.is_anomaly.sum()))
c3.metric("Avg Fill %/Container", f"{df_view.fill_pct_per_container.mean():.1f}")
c4.metric("Avg Waste/Container", f"{df_view.waste_per_container.mean():.1f} kg")

st.divider()

# 5) Prepare table
cols = [
    "Service Point Name",
    "container_count",
    "total_capacity_kg",
    "capacity_per_container",
    "total_kg",
    "waste_per_container",
    "tasks_per_week",
    "tasks_per_container",
    "waste_per_task_per_container",
    "fill_pct_per_container",
    "recon_error",
    "is_anomaly",
]
df_tab = df_view[[c for c in cols if c in df_view.columns]].copy()

pretty = df_tab.rename(columns={
    "Service Point Name": "Service Point",
    "container_count":     "Containers",
    "total_capacity_kg":   "Capacity (kg)",
    "capacity_per_container": "Cap/Container (kg)",
    "total_kg":            "Total Waste (kg)",
    "waste_per_container": "Waste/Container (kg)",
    "tasks_per_week":      "Weekly Tasks",
    "tasks_per_container": "Tasks/Container",
    "waste_per_task_per_container": "Waste/Task/Cont (kg)",
    "fill_pct_per_container": "Fill %/Container",
    "recon_error":         "Anomaly Score",
    "is_anomaly":          "Anomalous?",
})

# 6) Highlighting
def highlight(r):
    if r["Anomalous?"]:
        c = "rgba(255,0,0,0.25)"
    elif r["Fill %/Container"] < 30:
        c = "rgba(220,220,220,0.25)"
    elif r["Fill %/Container"] > 90:
        c = "rgba(0,255,0,0.15)"
    else:
        c = ""
    return [f"background-color: {c}"] * len(r)

styled = (
    pretty
    .style
    .apply(highlight, axis=1)
    .format({
        "Containers":              "{:d}",
        "Capacity (kg)":           "{:,.0f}",
        "Cap/Container (kg)":      "{:,.1f}",
        "Total Waste (kg)":        "{:,.1f}",
        "Waste/Container (kg)":    "{:,.1f}",
        "Weekly Tasks":            "{:,.0f}",
        "Tasks/Container":         "{:,.1f}",
        "Waste/Task/Cont (kg)":    "{:,.2f}",
        "Fill %/Container":        "{:,.1f}",
        "Anomaly Score":           "{:.3f}",
    })
)

st.dataframe(styled, use_container_width=True, height=700, hide_index=True)
st.caption("🔴 Anomalous • ⚪ Low fill (<30%) • 🟢 High fill (>90%)")

# 7) DRL recommendations
st.header("📦 Asset Distribution Recommendations (DRL)")
if st.button("Compute Recommendations"):
    for move in recommend(df_view):
        st.write("➡️", move)
