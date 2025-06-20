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
# 1) Veriyi hazırla & feature-table üret
# ────────────────────────────────────────────────────────────────
csv_path = DATA / "sp_feature_table.csv"
if not csv_path.exists():
    st.info("CSV bulunamadı ➜ Excel’den CSV’ler oluşturuluyor…")
    xls = pd.ExcelFile(DATA / "TexNL_Data.xlsx")
    xls.parse("Service Points").to_csv(DATA /"service_points.csv", index=False)
    xls.parse("Assets").to_csv(DATA /"assets.csv", index=False)
    xls.parse("Task Record").to_csv(DATA /"tasks.csv", index=False)
    build_features()

df = pd.read_csv(csv_path)

# ────────────────────────────────────────────────────────────────
# 2) Güvenli utilization hesaplamaları
# ────────────────────────────────────────────────────────────────
# 2a) Haftalık doluluk (%) = toplam atık / (kapasite * görev sayısı)
df["weekly_fill_pct"] = (
    df["total_kg"]
      .div((df["total_capacity_kg"].replace(0, np.nan) * df.get("tasks_per_week", 0)).replace(0, np.nan))
      .fillna(0)
      .clip(0, 1)
      * 100
).round(1)
# 2b) Ortalama doluluk her ziyaret (%) = ort. atık kg / kapasite
df["avg_fill_pct"] = (
    df.get("avg_kg", 0)
      .div(df["total_capacity_kg"].replace(0, np.nan))
      .fillna(0)
      .clip(0, 1)
      * 100
).round(1)

# ────────────────────────────────────────────────────────────────
# 3) Anomali tespiti (autoencoder)
# ────────────────────────────────────────────────────────────────
df = label_anomalies(df)  # ekler: is_anomaly (bool), recon_error (float)

# ────────────────────────────────────────────────────────────────
# 4) Konteyner sayısını ekle (Assets.csv → Location Details veya SP)
# ────────────────────────────────────────────────────────────────
assets_csv = DATA / "assets.csv"
if assets_csv.exists():
    a = pd.read_csv(assets_csv)
    if "Location Details" in a.columns:
        counts = a["Location Details"].value_counts()
        df["container_count"] = df["Service Point Name"].map(counts).fillna(0).astype(int)
    elif "Service Point Name" in a.columns:
        counts = a["Service Point Name"].value_counts()
        df["container_count"] = df["Service Point Name"].map(counts).fillna(0).astype(int)
    else:
        df["container_count"] = 0
else:
    df["container_count"] = 0

# ────────────────────────────────────────────────────────────────
# 5) Sidebar filtreleri
# ────────────────────────────────────────────────────────────────
st.sidebar.header("🔍 Filtreler")
filter_anom = st.sidebar.checkbox("🚨 Sadece anomaliler", False)
filter_cap = st.sidebar.checkbox("📦 Kapasite > 0", True)
min_weekly = st.sidebar.slider("Min. Haftalık Doluluk (%)", 0, 100, 0)
min_task   = st.sidebar.slider("Min. Haftalık Task", 0, int(df.get("tasks_per_week", 0).max()), 0)
search     = st.sidebar.text_input("Service Point ara")

# filtre uygula
df_view = df.copy()
if filter_anom:
    df_view = df_view[df_view.is_anomaly]
if filter_cap:
    df_view = df_view[df_view.total_capacity_kg > 0]
df_view = df_view[df_view.weekly_fill_pct >= min_weekly]
df_view = df_view[df_view.get("tasks_per_week", 0) >= min_task]
if search:
    df_view = df_view[df_view["Service Point Name"].str.contains(search, case=False, na=False)]

# ────────────────────────────────────────────────────────────────
# 6) KPI Kartları
# ────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Toplam SP",           len(df_view))
c2.metric("Anomali sayısı",      int(df_view.is_anomaly.sum()))
c3.metric("Ort. Haftalık Doluluk (%)", f"{df_view.weekly_fill_pct.mean():.1f}")
c4.metric("Ort. Ortalama Doluluk (%)",  f"{df_view.avg_fill_pct.mean():.1f}")

st.divider()

# ────────────────────────────────────────────────────────────────
# 7) Renklendirme fonksiyonu (weekly_fill esas alınarak)
# ────────────────────────────────────────────────────────────────
def highlight_row(r):
    if r["is_anomaly"]:
        color = "rgba(255,0,0,0.25)"
    elif r["weekly_fill_pct"] < 30:
        color = "rgba(220,220,220,0.25)"
    elif r["weekly_fill_pct"] > 90:
        color = "rgba(0,255,0,0.15)"
    else:
        color = ""
    return [f"background-color: {color}"] * len(r)

# ────────────────────────────────────────────────────────────────
# 8) Tablo hazırlığı & gösterimi
# ────────────────────────────────────────────────────────────────
cols = [
    "Service Point Name",
    "container_count",
    "total_capacity_kg",
    "total_kg",
    "avg_kg",
    "total_bags",
    "avg_bags",
    "weekly_fill_pct",
    "avg_fill_pct",
    "tasks_per_week",
    "recon_error",
    "is_anomaly",
]

# mevcut kolonlardan seç
df_tab = df_view[[c for c in cols if c in df_view.columns]].copy()

# Türkçe başlıklandır
pretty = df_tab.rename(columns={
    "Service Point Name": "Service Point",
    "container_count":     "Container Count",
    "total_capacity_kg":   "Capacity (kg)",
    "total_kg":            "Total Waste (kg)",
    "avg_kg":              "Avg Waste (kg)",
    "total_bags":          "Total Bags",
    "avg_bags":            "Avg Bags",
    "weekly_fill_pct":     "Weekly Fill (%)",
    "avg_fill_pct":        "Avg Fill (%)",
    "tasks_per_week":      "Weekly Tasks",
    "recon_error":         "Anomaly Score",
    "is_anomaly":          "Anomalous?",
})

styled = (
    pretty
      .style
      .apply(highlight_row, axis=1)
      .format({
         "Container Count":   "{:d}",
         "Capacity (kg)":      "{:,.0f}",
         "Total Waste (kg)":   "{:,.1f}",
         "Avg Waste (kg)":     "{:,.1f}",
         "Total Bags":         "{:,}",
         "Avg Bags":           "{:,.1f}",
         "Weekly Fill (%)":    "{:,.1f}",
         "Avg Fill (%)":       "{:,.1f}",
         "Weekly Tasks":       "{:,.1f}",
         "Anomaly Score":      "{:,.3f}",
      })
)

st.dataframe(styled, use_container_width=True, height=700, hide_index=True)
st.caption("🔴 Anomaly • ⚪ Low weekly fill (<30%) • 🟢 High weekly fill (>90%)")

# ────────────────────────────────────────────────────────────────
# 9) Öneri Motoru (DRL)
# ────────────────────────────────────────────────────────────────
st.header("📦 Asset Dağılım Önerileri (DRL)")
if st.button("Önerileri Hesapla"):
    for move in recommend(df_view):
        st.write("➡️", move)
