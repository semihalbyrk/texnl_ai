# app.py  — TexNL AI dashboard (Streamlit)
import pathlib, pandas as pd, streamlit as st

from src.features.build_features import build as build_features
from src.inference.detect_anomaly import label_anomalies
from src.inference.recommend_assets import recommend

ROOT   = pathlib.Path(__file__).resolve().parent
DATA   = ROOT / "data"
MODELS = ROOT / "src" / "models"

st.set_page_config(page_title="TexNL Efficiency AI", layout="wide")

# ---------- 1) Veriyi hazırla & feature table üret ----------
csv_path = DATA / "sp_feature_table.csv"
if not csv_path.exists():
    st.info("CSV bulunamadı ➜ Excel dönüştürülüyor & feature table oluşturuluyor…")
    xls = pd.ExcelFile(DATA / "TexNL_Data.xlsx")
    xls.parse("Service Points").to_csv(DATA / "service_points.csv", index=False)
    xls.parse("Assets").to_csv(DATA / "assets.csv", index=False)
    xls.parse("Task Record").to_csv(DATA / "tasks.csv", index=False)
    build_features()

df = pd.read_csv(csv_path)

# ---------- 2) Anomali tespiti ----------
df = label_anomalies(df)   # adds is_anomaly, recon_error

# ---------- 2.5) Container sayısını ekle ----------
if "container_count" not in df.columns:
    assets_csv = DATA / "assets.csv"
    if assets_csv.exists():
        a = pd.read_csv(assets_csv)
        # SP kolonu adı değişken olabilir:
        for cand in ["Service Point Name", "Service Point"]:
            if cand in a.columns:
                sp_col = cand
                break
        else:
            sp_col = None

        if sp_col:
            cnt = a.groupby(sp_col).size().rename("container_count")
            df = df.merge(cnt, left_on="Service Point Name", right_index=True, how="left")
    # doldur
    df["container_count"] = df.get("container_count", 0).fillna(0).astype(int)

# ---------- 3) Sidebar filtreler ----------
st.sidebar.header("🔍 Filtreler")
f_anom   = st.sidebar.checkbox("🚨 Sadece anomaliler", False)
f_cap    = st.sidebar.checkbox("📦 Kapasitesi > 0", True)
min_util = st.sidebar.slider("Minimum kullanım (%)", 0, 100, 0)
min_task = st.sidebar.slider("Min. haftalık task", 0, int(df.tasks_per_week.max()), 0)
search   = st.sidebar.text_input("Service Point ara")

df_view = df.copy()
if f_anom:   df_view = df_view[df_view.is_anomaly]
if f_cap:    df_view = df_view[df_view.total_capacity_kg > 0]
df_view = df_view[df_view.util_ratio*100 >= min_util]
df_view = df_view[df_view.tasks_per_week >= min_task]
if search:
    df_view = df_view[
        df_view["Service Point Name"]
             .str.contains(search, case=False, na=False)
    ]

# ---------- 4) KPI Kartları ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Toplam SP",           len(df))
c2.metric("Anomali sayısı",      int(df.is_anomaly.sum()))
c3.metric("Ort. Kullanım (%)",   f"{df.util_ratio.mean()*100:.1f}")
c4.metric("Ort. Task Yoğunluğu", f"{df.tasks_per_week.mean():.2f}")

st.divider()

# ---------- 5) Renklendirme fonksiyonu ----------
def highlight_row(r):
    an = r.get("is_anomaly", False)
    u  = r.get("util_ratio", 0.0)
    if an:
        return ["background-color: rgba(255,0,0,0.25)"]*len(r)
    if u < 0.3:
        return ["background-color: rgba(220,220,220,0.25)"]*len(r)
    if u > 0.9:
        return ["background-color: rgba(0,255,0,0.15)"]*len(r)
    return [""]*len(r)

# ---------- 6) Tablo hazırlığı ----------
pretty = df_view[[
    "Service Point Name",
    "total_kg",
    "total_capacity_kg",
    "util_ratio",
    "tasks_per_week",
    "container_count",
    "is_anomaly",
    "recon_error",
]].copy()

# yüzdeyi sınırlı tut
pretty["util_ratio"] = (pretty.util_ratio*100).clip(0,100).round(1)

# önce sütun isimlerini güzelleştir
pretty = pretty.rename(columns={
    "Service Point Name": "Service Point",
    "total_kg":            "Atık (kg)",
    "total_capacity_kg":   "Kapasite (kg)",
    "util_ratio":          "Kullanım (%)",
    "tasks_per_week":      "Haftalık Task",
    "container_count":     "Konteyner Sayısı",
    "is_anomaly":          "Anomali?",
    "recon_error":         "Skor",
})

# sonra stil + format
styled = (
    pretty
      .style
      .apply(highlight_row, axis=1)
      .format({
         "Atık (kg)":         "{:.1f}",
         "Kapasite (kg)":      "{:.0f}",
         "Kullanım (%)":       "{:.1f}",
         "Haftalık Task":      "{:.1f}",
         "Konteyner Sayısı":   "{}",
         "Skor":               "{:.3f}",
      })
)

st.dataframe(styled, use_container_width=True, height=600, hide_index=True)
st.caption("🔴 Anomali • ⚪ Düşük kullanım (<30%) • 🟢 Yüksek kullanım (>90%)")

# ---------- 7) Öneri motoru ----------
st.header("📦 Asset Dağılım Önerileri (DRL)")
if st.button("Önerileri Hesapla"):
    for m in recommend(df):
        st.write("➡️", m)
