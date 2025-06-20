# app.py  — TexNL AI dashboard (Streamlit)
import pathlib
import pandas as pd
import numpy as np
import streamlit as st

from src.features.build_features import build as build_features
from src.inference.detect_anomaly import label_anomalies
from src.inference.recommend_assets import recommend

# ────────────────────────────────────────────────────────────────
# Paths
# ────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data"

# ────────────────────────────────────────────────────────────────
# Streamlit page config
# ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="TexNL Efficiency AI", layout="wide")

# ────────────────────────────────────────────────────────────────
# 1) Ensure feature table exists (else build it)
# ────────────────────────────────────────────────────────────────
csv_path = DATA / "sp_feature_table.csv"
if not csv_path.exists():
    st.info("CSV bulunamadı ➜ Excel’den CSV’ler oluşturuluyor…")
    xls = pd.ExcelFile(DATA / "TexNL_Data.xlsx")
    xls.parse("Service Points").to_csv(DATA / "service_points.csv", index=False)
    xls.parse("Assets").to_csv(DATA / "assets.csv", index=False)
    xls.parse("Task Record").to_csv(DATA / "tasks.csv", index=False)
    build_features()

df = pd.read_csv(csv_path)

# ────────────────────────────────────────────────────────────────
# 2) Anomaly tagging
# ────────────────────────────────────────────────────────────────
df = label_anomalies(df)  # adds: is_anomaly, recon_error

# ────────────────────────────────────────────────────────────────
# 3) Compute weekly‐waste & per‐container waste
# ────────────────────────────────────────────────────────────────
# weekly waste ≈ avg_kg * tasks_per_week
df["weekly_waste_kg"] = df["avg_kg"].fillna(0) * df["tasks_per_week"].fillna(0)
# per-container weekly waste
df["waste_per_container"] = (
    df["weekly_waste_kg"]
      .div(df["container_count"].replace(0, np.nan))
      .fillna(df["weekly_waste_kg"])  # if 0 container, show total
      .round(1)
)

# ────────────────────────────────────────────────────────────────
# 4) Sidebar filters
# ────────────────────────────────────────────────────────────────
st.sidebar.header("🔍 Filtreler")
filter_anom  = st.sidebar.checkbox("🚨 Sadece Anomalous?", False)
filter_cap   = st.sidebar.checkbox("📦 Kapasite > 0",       True)
min_weekly   = st.sidebar.slider("Min. Weekly Fill (%)",   0, 100, 0)
min_tasks    = st.sidebar.slider("Min. Weekly Tasks",       0, int(df.tasks_per_week.max()), 0)
min_cont     = st.sidebar.slider("Min. Container Count",    0, int(df.container_count.max()), 0)
search       = st.sidebar.text_input("Service Point ara")

df_view = df.copy()
if filter_anom:
    df_view = df_view[df_view.is_anomaly]
if filter_cap:
    df_view = df_view[df_view.total_capacity_kg > 0]
df_view = df_view[df_view.weekly_fill_pct * 100 >= min_weekly]
df_view = df_view[df_view.tasks_per_week    >= min_tasks]
df_view = df_view[df_view.container_count   >= min_cont]
if search:
    df_view = df_view[df_view["Service Point Name"].str.contains(search, case=False, na=False)]

# ────────────────────────────────────────────────────────────────
# 5) KPI cards
# ────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Service Point Sayısı",      len(df_view))
c2.metric("Anomalous SP",              int(df_view.is_anomaly.sum()))
c3.metric("Ortalama Weekly Fill (%)",  f"{df_view.weekly_fill_pct.mean():.1f}")
c4.metric("Ortalama Containers/SP",    f"{df_view.container_count.mean():.1f}")

st.divider()

# ────────────────────────────────────────────────────────────────
# 6) Row‐highlighting (model anomaly + fill thresholds)
# ────────────────────────────────────────────────────────────────
def highlight_row(r):
    if r["is_anomaly"]:
        color = "rgba(255,0,0,0.25)"
    elif r["weekly_fill_pct"] < 0.30:
        color = "rgba(220,220,220,0.25)"
    elif r["weekly_fill_pct"] > 0.90:
        color = "rgba(0,255,0,0.15)"
    else:
        color = ""
    return [f"background-color: {color}"] * len(r)

# ────────────────────────────────────────────────────────────────
# 7) Table build & display
# ────────────────────────────────────────────────────────────────
cols = [
    "Service Point Name",
    "container_count",
    "total_capacity_kg",
    "total_kg",
    "avg_kg",
    "tasks_per_week",
    "weekly_fill_pct",
    "avg_fill_pct",
    "weekly_waste_kg",
    "waste_per_container",
    "recon_error",
    "is_anomaly",
]

df_tab = df_view[[c for c in cols if c in df_view.columns]].copy()

pretty = df_tab.rename(columns={
    "Service Point Name": "Service Point",
    "container_count":     "Containers",
    "total_capacity_kg":   "Capacity (kg)",
    "total_kg":            "Total Waste (kg)",
    "avg_kg":              "Avg Waste/Task (kg)",
    "tasks_per_week":      "Weekly Tasks",
    "weekly_fill_pct":     "Weekly Fill (%)",
    "avg_fill_pct":        "Avg Fill (%)",
    "weekly_waste_kg":     "Weekly Waste (kg)",
    "waste_per_container": "Waste/Container (kg)",
    "recon_error":         "Anomaly Score",
    "is_anomaly":          "Anomalous?",
})

styled = (
    pretty
      .style
      .apply(highlight_row, axis=1)
      .format({
         "Containers":             "{:d}",
         "Capacity (kg)":           "{:,.0f}",
         "Total Waste (kg)":        "{:,.1f}",
         "Avg Waste/Task (kg)":     "{:,.1f}",
         "Weekly Tasks":            "{:,.1f}",
         "Weekly Fill (%)":         "{:.1%}",
         "Avg Fill (%)":            "{:.1%}",
         "Weekly Waste (kg)":       "{:,.1f}",
         "Waste/Container (kg)":    "{:,.1f}",
         "Anomaly Score":           "{:.3f}",
      })
)

st.dataframe(styled, use_container_width=True, height=700, hide_index=True)
st.caption("🔴 Model Anomaly • ⚪ Low fill (<30%) • 🟢 High fill (>90%)")

# ────────────────────────────────────────────────────────────────
# 8) DRL recommendations
# ────────────────────────────────────────────────────────────────
st.header("📦 Asset Dağılım Önerileri (DRL)")
if st.button("Önerileri Hesapla"):
    for move in recommend(df_view):
        st.write("➡️", move)
