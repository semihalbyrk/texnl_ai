import streamlit as st
import sys
from pathlib import Path

# sys.path düzelt → src modülü erişilebilir olsun
sys.path.append(str(Path(__file__).resolve().parent))

from src.inference.detect_anomaly import run_anomaly_detection
from src.inference.recommend_assets import recommend_asset_reallocation

st.set_page_config(page_title="TexNL AI", layout="wide")

st.title("🧠 TexNL AI Analiz Paneli")
st.markdown("Servis noktalarının verimlilik analizini ve kaynak önerilerini görüntüleyin.")

st.divider()
st.header("📊 Anomali Tespiti")

if st.button("Anomalileri Tespit Et"):
    anomalies = run_anomaly_detection()
    if anomalies:
        for item in anomalies:
            st.warning(item)
    else:
        st.success("Anomali tespit edilmedi.")

st.divider()
st.header("📦 Asset Dağılım Önerileri (DRL)")

if st.button("Önerileri Hesapla"):
    recs = recommend_asset_reallocation()
    if recs:
        for rec in recs:
            st.markdown(f"➡️ {rec}")
    else:
        st.info("Herhangi bir öneri üretilmedi.")
