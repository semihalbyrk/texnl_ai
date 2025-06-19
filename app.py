import streamlit as st
from src.inference.detect_anomaly import run_anomaly_detection
from src.inference.recommend_assets import recommend_asset_reallocation

st.set_page_config(page_title="TexNL AI", layout="wide")

st.title("📊 Service Point Verimlilik Analizi")
st.write(" ")

if st.button("Anomalileri Tespit Et"):
    anomalies = run_anomaly_detection()
    for item in anomalies:
        st.markdown(f"⚠️ {item}")

st.markdown("---")

st.title("📦 Asset Dağılım Önerileri (DRL)")
st.write(" ")

if st.button("Önerileri Hesapla"):
    recs = recommend_asset_reallocation()
    for rec in recs:
        st.markdown(f"➡️ {rec}")
