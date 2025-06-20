# app.py  — TexNL AI dashboard (Streamlit)
import pathlib
import pandas as pd
import numpy as np
import streamlit as st

from src.features.build_features import build as build_features
from src.inference.detect_anomaly import label_anomalies
from src.inference.recommend_assets import recommend

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data"
st.set_page_config(page_title="TexNL Efficiency AI", layout="wide")

# 1) feature‐table
csv_path = DATA / "sp_feature_table.csv"
if not csv_path.exists():
    st.info("CSV bulunamadı, oluşturuluyor…")
    xls = pd.ExcelFile(DATA / "TexNL_Data.xlsx")
    xls.parse("Service Points").to_csv(DATA/"service_points.csv", index=False)
    xls.parse("Assets").to_csv(DATA/"assets.csv",         index=False)
    xls.parse("Task Record").to_csv(DATA/"tasks.csv",      index=False)
    build_features()
df = pd.read_csv(csv_path)

# 2) anomaly tag
percentile = st.sidebar.slider("Model Percentile Eşiği", 80, 99, 95)
df = label_anomalies(df, percentile=percentile)

# 3) sidebar filtreler
st.sidebar.header("🔍 Filtreler")
f_ctr0    = st.sidebar.checkbox("📦 Konteynır > 0", True)
f_anom    = st.sidebar.checkbox("🚨 Sadece Anomalous?", False)
min_score = st.sidebar.slider("Min. Anomaly Score", 0.0, float(df.anomaly_score.max()), 0.0)
min_fill  = st.sidebar.slider("Min. Fill%/Task",  0, 100, 0)
min_tasks = st.sidebar.slider("Min. Weekly Tasks", 0, int(df.tasks_per_week.max()), 0)
search_sp = st.sidebar.text_input("Service Point ara")

dfv = df.copy()
if f_ctr0:     dfv = dfv[dfv.container_count > 0]
if f_anom:     dfv = dfv[dfv.is_anomaly]
dfv = dfv[dfv.anomaly_score >= min_score]
dfv = dfv[dfv.fill_pct_per_task >= min_fill]
dfv = dfv[dfv.tasks_per_week >= min_tasks]
if search_sp:
    dfv = dfv[dfv["Service Point Name"].str.contains(search_sp, case=False)]

# 4) KPI’lar
c1,c2,c3,c4 = st.columns(4)
c1.metric("SP Sayısı",          len(dfv))
c2.metric("Anomalous SP",       int(dfv.is_anomaly.sum()))
c3.metric("Ort. Anomaly Score", f"{dfv.anomaly_score.mean():.2f}")
c4.metric("Ort. Fill%/Task",    f"{dfv.fill_pct_per_task.mean():.1f}")

st.divider()

# 5) Tablo
cols = [
  "Service Point Name","container_count","total_capacity_kg","weekly_total_kg",
  "tasks_per_week","waste_per_task","waste_per_task_per_ctr","capacity_per_ctr",
  "fill_pct_per_task","anomaly_score","anomaly_type","is_anomaly"
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
  "anomaly_score":          "Anomaly Score",
  "anomaly_type":           "Anomaly Type",
  "is_anomaly":             "Anomalous?",
})

def highlight(r):
    return ["background-color: rgba(255,0,0,0.2)"]*len(r) if r["Anomalous?"] else [""]*len(r)

styled = (
    pretty
      .style
      .apply(highlight, axis=1)
      .format({
        "Containers":              "{:d}",
        "Capacity (kg)":           "{:,.0f}",
        "Weekly Total Waste (kg)": "{:,.1f}",
        "Weekly Tasks":            "{:.1f}",
        "Waste/Task (kg)":         "{:.1f}",
        "Waste/Task/Ctr (kg)":     "{:.1f}",
        "Capacity/Ctr (kg)":       "{:,.0f}",
        "Fill%/Task":              "{:.1f}",
        "Anomaly Score":           "{:.3f}",
      })
)

st.dataframe(styled, use_container_width=True, height=700, hide_index=True)
st.caption(f"🔴 Anomalous = VAE-model (> {percentile}p)")

# 6) DRL önerileri
st.header("📦 Asset Dağılım Önerileri (DRL)")
if st.button("Önerileri Hesapla"):
    for mv in recommend(dfv):
        st.write("➡️", mv)


# ──────────────────────────────────────────────────────────────────────
# How It Works (Detailed)
# ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## How It Works")
st.markdown("""
This dashboard leverages a **Beta-Variational Autoencoder (BetaVAE)** to detect anomalous Service Points.  
Below is the step-by-step workflow:

1. **Feature Extraction**  
   We compute six normalized metrics per Service Point:  
   - **Weekly Tasks**: Average number of weekly visits.  
   - **Waste/Task (kg)**: Average kilograms of waste collected per visit.  
   - **Waste/Task/Ctr (kg)**: Waste/Task normalized by container count.  
   - **Capacity/Ctr (kg)**: Average capacity per container.  
   - **Fill%/Task (%)**: (Waste/Task ÷ Capacity/Ctr) × 100.   

2. **Deep-Learning-Based Scoring**  
   A **BetaVAE** is trained on these six features to learn normal patterns.  
   - **Reconstruction Error** is computed for each point:  
     $$\text{anomaly\_score} = \frac{1}{6}\sum_{i=1}^6 (x_i - \hat x_i)^2$$  
   - A **dynamic threshold** (percentile slider) decides which errors are high enough to flag.  

3. **Business-Rule Overrides**  
   In addition to the model, we apply simple fill-level rules:  
   - **underutil** if Fill%/Task < 30% **and** container_count > 1  
   - **overutil**  if Fill%/Task > 90% **and** container_count ≤ 1  

4. **Final Anomaly Type**  
   Each Service Point is labeled:  
   - `_model_`      → only VAE-based anomaly  
   - `_underutil_`  → under-utilized by business rule  
   - `_overutil_`   → over-utilized by business rule  

**Business Value:** You can immediately spot which sites are collecting too little or too much waste relative to their container infrastructure—enabling dynamic re-allocation of assets and routes.
""")

