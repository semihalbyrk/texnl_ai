# app.py — TexNL AI dashboard (Streamlit)
import pathlib
import pandas as pd
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
    xls.parse("Service Points").to_csv(DATA / "service_points.csv", index=False)
    xls.parse("Assets").to_csv(DATA / "assets.csv", index=False)
    xls.parse("Task Record").to_csv(DATA / "tasks.csv", index=False)
    build_features()

df = pd.read_csv(csv_path)

# ────────────────────────────────────────────────────────────────
# 2) Utilization’ı (0–1) güvenlice hesapla
# ────────────────────────────────────────────────────────────────
valid = df["total_capacity_kg"] > 0
df["util_ratio"] = 0.0
df.loc[valid, "util_ratio"] = df.loc[valid, "total_kg"] / df.loc[valid, "total_capacity_kg"]
df["util_ratio"] = df["util_ratio"].clip(0, 1)

# ────────────────────────────────────────────────────────────────
# 3) Anomali tespiti (autoencoder)
# ────────────────────────────────────────────────────────────────
df = label_anomalies(df)  # ekler: is_anomaly, recon_error

# ────────────────────────────────────────────────────────────────
# 4) Konteyner sayısını ekle (assets.csv → Location Details)
# ────────────────────────────────────────────────────────────────
assets_csv = DATA / "assets.csv"
if assets_csv.exists():
    a = pd.read_csv(assets_csv)
    if "Location Details" in a.columns:
        cnt = a["Location Details"].value_counts().rename("container_count")
        df["container_count"] = df["Service Point Name"].map(cnt).fillna(0).astype(int)
    else:
        df["container_count"] = 0
else:
    df["container_count"] = 0

# ────────────────────────────────────────────────────────────────
# 5) Sidebar filtreleri
# ────────────────────────────────────────────────────────────────
st.sidebar.header("🔍 Filtreler")
if st.sidebar.checkbox("🚨 Sadece anomaliler"):
    df = df[df.is_anomaly]
if st.sidebar.checkbox("📦 Kapasite > 0", True):
    df = df[df.total_capacity_kg > 0]

min_util = st.sidebar.slider("Minimum Kullanım (%)", 0, 100, 0)
df = df[df.util_ratio * 100 >= min_util]

min_task = st.sidebar.slider("Min. Haftalık Task", 0, int(df.tasks_per_week.max()), 0)
df = df[df.tasks_per_week >= min_task]

search = st.sidebar.text_input("Service Point ara")
if search:
    df = df[df["Service Point Name"].str.contains(search, case=False, na=False)]

# ────────────────────────────────────────────────────────────────
# 6) KPI Kartları
# ────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Toplam SP",           len(df))
c2.metric("Anomali sayısı",      int(df.is_anomaly.sum()))
c3.metric("Ort. Kullanım (%)",   f"{df.util_ratio.mean() * 100:.1f}")
c4.metric("Ort. Task Yoğunluğu", f"{df.tasks_per_week.mean():.2f}")

st.divider()

# ────────────────────────────────────────────────────────────────
# 7) Renklendirme fonksiyonu (yeniden isimlendirilmiş kolonlara göre)
# ────────────────────────────────────────────────────────────────
def highlight_row(r):
    # Burada artık "Anomali?" ve "Kullanım (%)" isimleri kullanılıyor
    if r["Anomali?"]:
        color = "rgba(255,0,0,0.25)"
    elif r["Kullanım (%)"] < 30:
        color = "rgba(220,220,220,0.25)"
    elif r["Kullanım (%)"] > 90:
        color = "rgba(0,255,0,0.15)"
    else:
        color = ""
    return [f"background-color: {color}"] * len(r)

# ────────────────────────────────────────────────────────────────
# 8) Tablo hazırlığı & gösterimi
# ────────────────────────────────────────────────────────────────
pretty = df[[
    "Service Point Name",
    "container_count",
    "total_kg",
    "total_capacity_kg",
    "util_ratio",
    "tasks_per_week",
    "is_anomaly",
    "recon_error",
]].copy()

# ➊ yüzdeyi 0–100 aralığına getir
pretty["util_ratio"] = (pretty["util_ratio"] * 100).round(1)

# ➋ Türkçe kolon adları
pretty = pretty.rename(columns={
    "Service Point Name": "Service Point",
    "container_count":     "Konteyner Sayısı",
    "total_kg":            "Atık (kg)",
    "total_capacity_kg":   "Kapasite (kg)",
    "util_ratio":          "Kullanım (%)",
    "tasks_per_week":      "Haftalık Task",
    "is_anomaly":          "Anomali?",
    "recon_error":         "Skor",
})

# ➌ Stil ve format
styled = (
    pretty
      .style
      .apply(highlight_row, axis=1)
      .format({
         "Konteyner Sayısı": "{:d}",
         "Atık (kg)":        "{:.1f}",
         "Kapasite (kg)":     "{:.0f}",
         "Kullanım (%)":      "{:.1f}",
         "Haftalık Task":     "{:.1f}",
         "Skor":              "{:.3f}",
      })
)

st.dataframe(styled, use_container_width=True, height=650, hide_index=True)
st.caption("🔴 Anomali • ⚪ Düşük kullanım (<30%) • 🟢 Yüksek kullanım (>90%)")

# ────────────────────────────────────────────────────────────────
# 9) Öneri Motoru (DRL)
# ────────────────────────────────────────────────────────────────
st.header("📦 Asset Dağılım Önerileri (DRL)")
if st.button("Önerileri Hesapla"):
    for move in recommend(df):
        st.write("➡️", move)
