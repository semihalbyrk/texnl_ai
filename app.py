# app.py — TexNL AI dashboard
import pathlib, pandas as pd, streamlit as st

from src.features.build_features import build as build_features
from src.inference.detect_anomaly import label_anomalies
from src.inference.recommend_assets import recommend

ROOT  = pathlib.Path(__file__).resolve().parent
DATA  = ROOT / "data"
st.set_page_config(page_title="TexNL Efficiency AI", layout="wide")

# ─── 1) Veriyi hazırla ────────────────────────────────────────────────────────
csv_path = DATA / "sp_feature_table.csv"
if not csv_path.exists():
    st.info("CSV yok ➜ Excel okunuyor & feature-table oluşturuluyor…")
    xls = pd.ExcelFile(DATA / "TexNL_Data.xlsx")
    xls.parse("Service Points").to_csv(DATA / "service_points.csv", index=False)
    xls.parse("Assets").to_csv(DATA / "assets.csv", index=False)
    xls.parse("Task Record").to_csv(DATA / "tasks.csv", index=False)
    build_features()

df = pd.read_csv(csv_path)

# Konteyner sayısı (asset adedi) ekle
if "container_count" not in df.columns:
    cnt = pd.read_csv(DATA / "assets.csv").groupby("Service Point Name")["Asset ID"].count()
    df = df.merge(cnt.rename("container_count"), on="Service Point Name", how="left").fillna({"container_count":0})

# Utilizasyonu % olarak (≤100) hesapla
df["util_ratio"] = (df["total_kg"] / df["total_capacity_kg"].replace(0, pd.NA)).clip(0, 1).fillna(0)

# ─── 2) Anomali etiketle ─────────────────────────────────────────────────────
df = label_anomalies(df)

# ─── 3) Sidebar filtreleri ───────────────────────────────────────────────────
st.sidebar.header("🔍 Filtreler")
if st.sidebar.checkbox("🚨 Yalnız anomaliler"):
    df = df[df.is_anomaly]
if st.sidebar.checkbox("📦 Kapasite > 0", True):
    df = df[df.total_capacity_kg > 0]

min_util = st.sidebar.slider("Min. kullanım (%)", 0, 100, 0)
df = df[df.util_ratio*100 >= min_util]

min_task = st.sidebar.slider("Min. haftalık task", 0, int(df.tasks_per_week.max()), 0)
df = df[df.tasks_per_week >= min_task]

q = st.sidebar.text_input("Service Point ara")
if q:
    df = df[df["Service Point Name"].str.contains(q, case=False, na=False)]

# ─── 4) KPI kartları ──────────────────────────────────────────────────────────
c1,c2,c3,c4 = st.columns(4)
c1.metric("Toplam SP", len(df))
c2.metric("Anomali sayısı", int(df.is_anomaly.sum()))
c3.metric("Ort. Kullanım (%)", f"{(df.util_ratio.mean()*100):.1f}")
c4.metric("Ort. Task Yoğunluğu", f"{df.tasks_per_week.mean():.2f}")

st.divider()

# ─── 5) Renklendirilmiş tablo ────────────────────────────────────────────────
def highlight(row):
    if row["is_anomaly"]:          color = "rgba(255,0,0,0.25)"
    elif row["util_ratio"] < .30:  color = "rgba(220,220,220,0.25)"
    elif row["util_ratio"] > .90:  color = "rgba(0,255,0,0.15)"
    else:                          color = ""
    return [f"background-color:{color}"]*len(row)

pretty = df[[
    "Service Point Name","container_count","total_kg","total_capacity_kg",
    "util_ratio","tasks_per_week","is_anomaly","recon_error"
]].copy()

pretty["util_ratio"] = (pretty["util_ratio"]*100).round(1)

pretty = pretty.rename(columns={
    "Service Point Name":"Service Point",
    "container_count":"Konteyner",
    "total_kg":"Atık (kg)",
    "total_capacity_kg":"Kapasite (kg)",
    "util_ratio":"Kullanım (%)",
    "tasks_per_week":"Haftalık Task",
    "is_anomaly":"Anomali?",
    "recon_error":"Skor"
})

styled = (pretty
          .style
          .apply(highlight, axis=1)
          .format({
              "Atık (kg)":"{:.1f}",
              "Kapasite (kg)":"{:.0f}",
              "Kullanım (%)":"{:.1f}",
              "Haftalık Task":"{:.1f}",
              "Skor":"{:.3f}",
              "Konteyner":"{:d}",
          }))

st.dataframe(styled, use_container_width=True, height=650, hide_index=True)
st.caption("Kırmızı = anomali • Gri = düşük kullanım • Yeşil = yüksek kullanım")

# ─── 6) DRL önerileri ────────────────────────────────────────────────────────
st.header("📦 Asset Dağılım Önerileri (DRL)")
if st.button("Önerileri Hesapla"):
    for move in recommend(df):
        st.write("➡️ ", move)
