import pathlib
import pandas as pd
import numpy as np
import streamlit as st

from src.features.build_features import build as build_features
from src.inference.detect_anomaly import label_anomalies
from src.inference.recommend_assets import recommend

# ────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data"

st.set_page_config(page_title="TexNL Efficiency AI", layout="wide")

# 1) ensure features
csv_path = DATA / "sp_feature_table.csv"
if not csv_path.exists():
    st.info("Building feature‐table…")
    xls = pd.ExcelFile(DATA / "TexNL_Data.xlsx")
    xls.parse("Service Points").to_csv(DATA / "service_points.csv", index=False)
    xls.parse("Assets").to_csv(DATA / "assets.csv", index=False)
    xls.parse("Task Record").to_csv(DATA / "tasks.csv", index=False)
    build_features()

df = pd.read_csv(csv_path)

# 2) anomaly tag
df = label_anomalies(df)

# 3) sidebar
st.sidebar.header("🔍 Filtreler")
f_anom = st.sidebar.checkbox("🚨 Sadece Anomalous?", False)
f_cap  = st.sidebar.checkbox("📦 Kapasite > 0",     True)
min_fill   = st.sidebar.slider("Min. Fill%/Task", 0, 100, 0)
min_tasks  = st.sidebar.slider("Min. Weekly Tasks", 0, int(df.tasks_per_week.max()), 0)
min_ctrs   = st.sidebar.slider("Min. Containers/SP", 0, int(df.container_count.max()), 0)
search_sp  = st.sidebar.text_input("Service Point ara")

dfv = df.copy()
if f_anom:   dfv = dfv[dfv.is_anomaly]
if f_cap:    dfv = dfv[dfv.total_capacity_kg > 0]
dfv = dfv[dfv.fill_pct_per_task >= min_fill]
dfv = dfv[dfv.tasks_per_week     >= min_tasks]
dfv = dfv[dfv.container_count    >= min_ctrs]
if search_sp:
    dfv = dfv[dfv["Service Point Name"].str.contains(search_sp, case=False)]

# 4) KPIs
c1,c2,c3,c4 = st.columns(4)
c1.metric("Service Point Sayısı",   len(dfv))
c2.metric("Anomalous SP",           int(dfv.is_anomaly.sum()))
c3.metric("Ort. Fill%/Task",        f"{dfv.fill_pct_per_task.mean():.1f}")
c4.metric("Ort. Containers/SP",     f"{dfv.container_count.mean():.1f}")

st.divider()

# 5) table
cols = [
    "Service Point Name",
    "container_count",
    "total_capacity_kg",
    "weekly_total_kg",
    "tasks_per_week",
    "waste_per_task",
    "waste_per_task_per_ctr",
    "capacity_per_ctr",
    "fill_pct_per_task",
    "fill_pct_weekly",
    "anomaly_score",
    "is_anomaly",
]
df_tab = dfv[[c for c in cols if c in dfv.columns]].copy()

pretty = df_tab.rename(columns={
    "Service Point Name":    "Service Point",
    "container_count":        "Containers",
    "total_capacity_kg":      "Capacity (kg)",
    "weekly_total_kg":        "Weekly Total Waste (kg)",
    "tasks_per_week":         "Weekly Tasks",
    "waste_per_task":         "Waste/Task (kg)",
    "waste_per_task_per_ctr": "Waste/Task/Ctr (kg)",
    "capacity_per_ctr":       "Capacity/Ctr (kg)",
    "fill_pct_per_task":      "Fill%/Task",
    "fill_pct_weekly":        "Fill%/Weekly",
    "anomaly_score":          "Anomaly Score",
    "is_anomaly":             "Anomalous?",
})

def highlight(r):
    if r["Anomalous?"]:
        col = "rgba(255,0,0,0.25)"
    elif r["Fill%/Task"] < 30:
        col = "rgba(220,220,220,0.25)"
    elif r["Fill%/Task"] > 90:
        col = "rgba(0,255,0,0.15)"
    else:
        col = ""
    return [f"background-color: {col}"] * len(r)

styled = (
    pretty
      .style
      .apply(highlight, axis=1)
      .format({
          "Containers":            "{:d}",
          "Capacity (kg)":         "{:,.0f}",
          "Weekly Total Waste (kg)": "{:,.1f}",
          "Weekly Tasks":          "{:.1f}",
          "Waste/Task (kg)":       "{:.1f}",
          "Waste/Task/Ctr (kg)":   "{:.1f}",
          "Capacity/Ctr (kg)":     "{:,.0f}",
          "Fill%/Task":            "{:.1f}",
          "Fill%/Weekly":          "{:.1f}",
          "Anomaly Score":         "{:.3f}",
      })
)

st.dataframe(styled, use_container_width=True, height=700, hide_index=True)
st.caption("🔴 Anomalous • ⚪ Low fill (<30%) • 🟢 High fill (>90%)")

# 6) recommendations
st.header("📦 Asset Dağılım Önerileri (DRL)")
if st.button("Önerileri Hesapla"):
    for mv in recommend(dfv):
        st.write("➡️", mv)
