import sys
from pathlib import Path

# Pastikan src bisa diimport saat run dari root project
sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.models.train import load_config, train_pipeline, save_artifact
from src.models.infer import forecast_iterative_xgb, load_artifact
from src.pipeline.data_prep import load_and_validate
from src.models.baselines import naive_forecast, seasonal_naive_forecast


st.set_page_config(page_title="Prediksi Permohonan KIA", layout="wide")
st.title("Sistem Prediksi Permohonan KIA - Disdukcapil Kota Bogor")

cfg = load_config()

with st.sidebar:
    st.header("Pengaturan")
    horizon = st.number_input(
        "Horizon Prediksi (bulan)",
        min_value=1, max_value=24, value=int(cfg["forecast"]["horizon"])
    )
    use_existing_model = st.checkbox("Gunakan model tersimpan (jika ada)", value=False)
    train_button = st.button("Latih Model")
    predict_button = st.button("Prediksi ke Depan")

st.subheader("1) Unggah Data Historis")
uploaded = st.file_uploader("Unggah CSV (kolom: periode, permohonan_kia)", type=["csv"])

if uploaded is not None:
    df_raw = pd.read_csv(uploaded)
    try:
        df = load_and_validate(df_raw, cfg)
    except Exception as e:
        st.error(f"Validasi data gagal: {e}")
        st.stop()

    st.success(f"Data OK. Jumlah periode: {df.shape[0]}")
    fig = px.line(
        df,
        x=cfg["data"]["date_column"],
        y=cfg["data"]["target_column"],
        title="Historis Permohonan KIA"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("2) Pelatihan dan Evaluasi")
    if use_existing_model:
        try:
            artifact = load_artifact("models/kia_forecast")
            st.info(f"Model tersimpan: {artifact['model_name']}")
            st.json(artifact["scores"])
        except Exception as e:
            st.warning(f"Gagal memuat model tersimpan: {e}")
    elif train_button:
        with st.spinner("Melatih model..."):
            artifact = train_pipeline(df, cfg)
            model_path, meta_path = save_artifact(artifact, out_dir="models", filename_prefix="kia_forecast")
        st.success(f"Model '{artifact['model_name']}' tersimpan.")
        st.json(artifact["scores"])

    st.subheader("3) Prediksi ke Depan")
    if predict_button:
        try:
            artifact = load_artifact("models/kia_forecast")
        except Exception as e:
            st.warning(f"Tidak menemukan model tersimpan, mencoba melatih cepat: {e}")
            with st.spinner("Melatih model..."):
                artifact = train_pipeline(df, cfg)
                _ = save_artifact(artifact, out_dir="models", filename_prefix="kia_forecast")

        try:
            if artifact["model_name"] == "xgboost":
                fc_df = forecast_iterative_xgb(df[["periode", "permohonan_kia"]], artifact, horizon=horizon)
                fc_df = fc_df.rename(columns={"y_pred": "prediksi"})
            else:
                # Baseline
                y_hist = df[cfg["data"]["target_column"]].values
                dates = []
                last = df[cfg["data"]["date_column"]].iloc[-1]
                for _ in range(horizon):
                    last = (last + pd.offsets.MonthBegin(1))
                    dates.append(last)
                if artifact["model_name"] == "seasonal_naive":
                    yhat = seasonal_naive_forecast(y_hist, horizon=horizon, season_length=12)
                else:
                    yhat = naive_forecast(y_hist, horizon=horizon)
                fc_df = pd.DataFrame({"periode": dates, "prediksi": yhat})

            st.write("Hasil Prediksi:")
            st.dataframe(fc_df)

            # Gabungkan actual + prediksi untuk plot
            hist = df[[cfg["data"]["date_column"], cfg["data"]["target_column"]]].rename(columns={
                cfg["data"]["date_column"]: "periode",
                cfg["data"]["target_column"]: "aktual"
            })
            pred_plot = fc_df.copy()
            pred_plot["aktual"] = np.nan
            plot_df = pd.concat([hist, pred_plot], ignore_index=True, sort=False)

            fig2 = px.line(plot_df, x="periode", y=["aktual", "prediksi"], title="Aktual vs Prediksi")
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.error(f"Gagal melakukan prediksi: {e}")

else:
    st.info("Unggah data untuk memulai.")
