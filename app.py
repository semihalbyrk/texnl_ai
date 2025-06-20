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
# 1) Feature table hazırla
# ────────────────────────────────────────────────────────────────
csv_path = DATA / "sp_feature_table.csv"
if not csv_path.exists():
    st.info("CSV bulunamadı ➜ Excel’den dönüştürülüyor…")
    xls = pd.ExcelFile(DATA / "TexNL_Data.xlsx")
    xls.parse("Service Points").to_csv(DATA / "service_points.csv", index=False)
    xls.parse("Assets").to_csv(DATA / "assets.csv", index=False)
    xls.parse("Task Record").to_csv(DATA / "tasks.csv", index=False)
    build_features()

df = pd.read_csv(csv_path)

# ────────────────────────────────────────────────────────────────
# 2) Yeni doluluk metrikleri (container başına)
# ────────────────────────────────────────────────────────────────
# Haftalık Doluluk (%) = Toplam atık kg / (container_count * tasks_per_week * avg_capacity)
# Önce container sayısını ekleyelim
assets_csv = DATA / "assets.csv"
if assets_csv.exists():
    a = pd.read_csv(assets_csv)
    key = "Location Details" if "Location Details" in a.columns else (
          "Service Point Name" if "Service Point Name" in a.columns else None)
    if key:
        cnts = a[key].value_counts()
        df["container_count"] = df["Service Point Name"].map(cnts).fillna(0).astype(int)
    else:
        df["container_count"] = 0
else:
    df["container_count"] = 0

# Ortalama kapasite bir container başına
# (total_capacity_kg / container_count), sıfırlardan kaçınmak için clip
df["avg_container_capacity"] = (
    df["total_capacity_kg"]
      .div(df["container_count"].replace(0, np.nan))
      .fillna(0)
)

# Haftalık atık miktarı = total_kg / weeks (features tablosunda haftalık görev zaten var)
# Haftalık doluluk yüzdesi = weekly waste per container / avg_container_capacity
weekly_waste = df["total_kg"]
weekly_waste_per_sp = weekly_waste  # build_features görev sıklığı zaten haftalık normalize

# Haftalık Doluluk (%)
df["weekly_fill_pct"] = (
    weekly_waste_per_sp
      .div(df["container_count"].replace(0, np.nan) * df.get("tasks_per_week", 0))
      .fillna(0)
      .clip(0, 1)
      * 100
).round(1)

# Ortalama doluluk (her ziyaret başına) (%) = avg_kg / avg_container_capacity
df["avg_fill_pct"] = (
    df.get("avg_kg", 0)
      .div(df["avg_container_capacity"].replace(0, np.nan))
      .fillna(0)
      .clip(0, 1)
      * 100
).round(1)

# ────────────────────────────────────────────────────────────────
# 3) Anomali tespiti (autoencoder)
# ────────────────────────────────────────────────────────────────
df = label_anomalies(df)

# ────────────────────────────────────────────────────────────────
# 4) Sidebar filtreleri
# ────────────────────────────────────────────────────────────────
st.sidebar.header("🔍 Filtreler")
filter_anom = st.sidebar.checkbox("🚨 Sadece anomaliler", False)
filter_cap = st.sidebar.checkbox("📦 Konteyner > 0", True)
min_weekly = st.sidebar.slider("Min. Haftalık Doluluk (%)", 0, 100, 0)
min_visits = st.sidebar.slider("Min. Haftalık Task", 0, int(df.get("tasks_per_week",0).max()), 0)
search = st.sidebar.text_input("Service Point ara")

df_view = df.copy()
if filter_anom:
    df_view = df_view[df_view.is_anomaly]
if filter_cap:
    df_view = df_view[df_view.container_count > 0]
if min_weekly:
    df_view = df_view[df_view.weekly_fill_pct >= min_weekly]
if min_visits:
    df_view = df_view[df_view.tasks_per_week >= min_visits]
if search:
    df_view = df_view[df_view["Service Point Name"].str.contains(search, case=False, na=False)]

# ────────────────────────────────────────────────────────────────
# 5) KPI Kartları
# ────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Service Points", len(df_view))
c2.metric("Anomali Sayısı", int(df_view.is_anomaly.sum()))
c3.metric("Ort. Haftalık Doluluk (%)", f"{df_view.weekly_fill_pct.mean():.1f}")
c4.metric("Ort. Ortalama Doluluk (%)", f"{df_view.avg_fill_pct.mean():.1f}")

st.divider()

# ────────────────────────────────────────────────────────────────
# 6) Tablo hazırlığı & renklendirme
# ────────────────────────────────────────────────────────────────
cols = [
    "Service Point Name", "container_count", "total_capacity_kg", "total_kg",
    "avg_kg", "total_bags", "avg_bags",
    "weekly_fill_pct", "avg_fill_pct", "tasks_per_week",
    "recon_error", "is_anomaly"
]

df_tab = df_view[[c for c in cols if c in df_view.columns]].copy()

# Türkçe başlıklandır
pretty = df_tab.rename(columns={
    "Service Point Name": "Service Point",
    "container_count":     "Containers",
    "total_capacity_kg":   "Total Capacity (kg)",
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

# St decor function for highlighting
 def highlight(r):
    if r["Anomalous?"]:
        return ["background-color: rgba(255,0,0,0.25)"]*len(r)
    if r["Weekly Fill (%)"] < 30:
        return ["background-color: rgba(220,220,220,0.25)"]*len(r)
    if r["Weekly Fill (%)"] > 90:
        return ["background-color: rgba(0,255,0,0.15)"]*len(r)
    return [""]*len(r)

styled = (
    pretty
      .style
      .apply(highlight, axis=1)
      .format({
         "Containers":       "{:d}",
         "Total Capacity (kg)":   "{:,.0f}",
         "Total Waste (kg)":      "{:,.1f}",
         "Avg Waste (kg)":        "{:,.1f}",
         "Total Bags":            "{:,.0f}",
         "Avg Bags":              "{:,.1f}",
         "Weekly Fill (%)":       "{:,.1f}",
         "Avg Fill (%)":          "{:,.1f}",
         "Weekly Tasks":          "{:,.1f}",
         "Anomaly Score":         "{:.3f}",
      })
)

st.dataframe(styled, use_container_width=True, height=700, hide_index=True)
st.caption("🔴 Anomali • ⚪ Düşük Doluluk (<30%) • 🟢 Yüksek Doluluk (>90%)")

# ────────────────────────────────────────────────────────────────
# 7) Öneri Motoru (DRL)
# ────────────────────────────────────────────────────────────────
st.header("📦 Asset Dağılım Önerileri (DRL)")
if st.button("Önerileri Hesapla"):
    for mv in recommend(df_view):
        st.write("➡️", mv)
