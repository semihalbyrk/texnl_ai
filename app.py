# app.py  — TexNL AI dashboard (Streamlit)
import pathlib, pandas as pd, streamlit as st
from src.features.build_features import build as build_features
from src.inference.detect_anomaly import label_anomalies
from src.inference.recommend_assets import recommend

ROOT  = pathlib.Path(__file__).resolve().parent
DATA  = ROOT / "data"
MODELS = ROOT / "src" / "models"

st.set_page_config(page_title="TexNL Efficiency AI", layout="wide")

# ---------- 1) Veriyi hazırla & özellik tablosu üret ----------
csv_path = DATA / "sp_feature_table.csv"
if not csv_path.exists():
    st.info("CSV bulunamadı ➜ Excel dönüştürülüyor & feature table oluşturuluyor…")
    # Excel → CSV (bir kerelik)
    xls = pd.ExcelFile(DATA / "TexNL_Data.xlsx")
    xls.parse("Service Points").to_csv(DATA/"service_points.csv", index=False)
    xls.parse("Assets").to_csv(DATA/"assets.csv", index=False)
    xls.parse("Task Record").to_csv(DATA/"tasks.csv", index=False)
    build_features()         #  src/features/build_features.py

df = pd.read_csv(csv_path)

# ---------- 2) Anomali tespiti ----------
df = label_anomalies(df)

st.sidebar.header("Filters")
show_anom = st.sidebar.checkbox("Sadece anomalileri göster", False)
if show_anom:
    df_view = df[df.is_anomaly].copy()
else:
    df_view = df.copy()

# ---------- 3) Ana metrikler ----------
c1, c2, c3 = st.columns(3)
c1.metric("Toplam Service Point", len(df))
c2.metric("Tespit edilen anomali", int(df.is_anomaly.sum()))
c3.metric("Ortalama Utilization", f"{df.util_ratio.mean():.2f}")

st.dataframe(df_view[
    ["Service Point Name","total_kg","total_capacity_kg",
     "util_ratio","is_anomaly","recon_error"]
])

# ---------- 4) Öneri motoru ----------
st.write("### 📦 Asset Dağılım Önerileri (DRL)")
st.write(" ")

for rec in recommend_asset_reallocation():
    st.markdown(f"➡️ {rec}")
