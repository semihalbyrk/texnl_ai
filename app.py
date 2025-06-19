import sys
from pathlib import Path

# src klasörünü Python path'ine ekle
sys.path.append(str(Path(__file__).resolve().parent / "src"))

import streamlit as st
from inference.detect_anomaly import run_anomaly_detection
from inference.recommend_assets import recommend_asset_reallocation

st.set_page_config(page_title="TexNL AI Analiz Paneli", layout="wide")

st.title("🧠 TexNL AI Analiz Paneli")
st.markdown("Servis noktalarının verimliliğini analiz edin, sistem önerilerini görüntüleyin.")

st.markdown("### ⚠️ Anomali Tespiti")
if st.button("Anomalileri Tespit Et"):
    anomalies = run_anomaly_detection()
    if anomalies:
        for anomaly in anomalies:
            st.warning(anomaly)
    else:
        st.success("Herhangi bir anomali tespit edilmedi.")

st.divider()
st.markdown("### 📦 Asset Dağılım Önerileri (DRL)")
if st.button("Önerileri Hesapla"):
    recs = recommend_asset_reallocation()
    if recs:
        for rec in recs:
            st.markdown(f"➡️ {rec}")
    else:
        st.info("Öneri üretilemedi veya model yüklenemedi.")
