"""
Laptop Price Prediction - Streamlit App
=========================================
Aplikasi ini melakukan deployment model Machine Learning (Gradient Boosting
Regressor, hasil hyperparameter tuning) yang sudah dilatih pada notebook
untuk memprediksi harga laptop (dalam Euro)
berdasarkan spesifikasi yang diinput oleh pengguna.

Pipeline preprocessing HARUS identik dengan yang dipakai saat training:
1. Susun fitur mentah -> Company, TypeName, Ram, CpuTier, SSD, Not_SSD,
   Screen_Width, Inches, Weight, GpuBrand
2. One-Hot Encoding untuk kolom kategorikal (pakai encoder yang sudah di-fit)
3. Gabungkan fitur numerik + hasil OHE, urutan kolom HARUS sama seperti
   scaler.feature_names_in_
4. Standard Scaling (pakai scaler yang sudah di-fit)
5. Predict pakai model
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Prediksi Harga Laptop",
    page_icon="💻",
    layout="centered",
)

# ---------------------------------------------------------------------------
# 1. LOAD ARTIFACTS (model, encoder, scaler) - di-cache supaya tidak reload
#    setiap kali user berinteraksi dengan widget
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_laptop_price_model.pkl")
    encoder = joblib.load("onehot_encoder.pkl")
    scaler = joblib.load("standard_scaler.pkl")
    return model, encoder, scaler


model, encoder, scaler = load_artifacts()

# Kolom kategorikal & numerik sesuai urutan saat training (lihat notebook)
CAT_COLS = ["Company", "TypeName", "CpuTier", "GpuBrand"]
NUM_COLS = ["Ram", "SSD", "Not_SSD", "Screen_Width", "Inches", "Weight"]

# Ambil daftar kategori asli dari encoder yang sudah di-fit
company_options = list(encoder.categories_[0])
typename_options = list(encoder.categories_[1])
cputier_options = list(encoder.categories_[2])
# GpuBrand: kategori 'ARM' sengaja dibuang saat training (data ARM di-drop
# karena jumlahnya terlalu sedikit dan mengganggu model), jadi tidak
# ditampilkan sebagai pilihan di form.
gpubrand_options = [g for g in encoder.categories_[3] if g != "ARM"]

RAM_OPTIONS = [2, 4, 6, 8, 12, 16, 24, 32, 64]


# ---------------------------------------------------------------------------
# 2. FUNGSI PREPROCESSING + PREDIKSI
# ---------------------------------------------------------------------------
def predict_price(raw_input: dict) -> float:
    """Terima 1 baris input mentah (dict), kembalikan prediksi harga (EUR)."""
    df = pd.DataFrame([raw_input])

    # One-Hot Encoding kolom kategorikal
    encoded = encoder.transform(df[CAT_COLS])
    encoded_df = pd.DataFrame(
        encoded, columns=encoder.get_feature_names_out(CAT_COLS), index=df.index
    )

    # Gabungkan numerik + hasil encoding
    X = pd.concat([df[NUM_COLS], encoded_df], axis=1)

    # Buang kolom GpuBrand_ARM jika muncul (tidak dipakai saat training)
    if "GpuBrand_ARM" in X.columns:
        X = X.drop(columns=["GpuBrand_ARM"])

    # Pastikan urutan & kelengkapan kolom identik dengan saat scaler di-fit
    X = X.reindex(columns=scaler.feature_names_in_, fill_value=0)

    # Scaling lalu prediksi
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    return float(pred)


# ---------------------------------------------------------------------------
# 3. UI - FORM INPUT
# ---------------------------------------------------------------------------
st.title("💻 Prediksi Harga Laptop")
st.write(
    "Masukkan spesifikasi laptop di bawah ini, lalu klik **Prediksi Harga** "
    "untuk mengestimasi harga laptop menggunakan model Machine Learning "
    "(Gradient Boosting Regressor, hasil hyperparameter tuning) yang telah "
    "dilatih sebelumnya."
)

with st.form("laptop_form"):
    col1, col2 = st.columns(2)

    with col1:
        company = st.selectbox("Brand / Company", company_options)
        typename = st.selectbox("Tipe Laptop", typename_options)
        cputier = st.selectbox("CPU Tier", cputier_options)
        gpubrand = st.selectbox("GPU Brand", gpubrand_options)
        ram = st.selectbox("RAM (GB)", RAM_OPTIONS, index=RAM_OPTIONS.index(8))

    with col2:
        ssd = st.number_input("Kapasitas SSD (GB)", min_value=0, max_value=4000, value=256, step=32)
        not_ssd = st.number_input(
            "Kapasitas storage selain SSD (HDD/Hybrid/Flash, GB)",
            min_value=0, max_value=4000, value=0, step=32,
            help="Isi 0 jika laptop hanya menggunakan SSD.",
        )
        screen_width = st.number_input("Screen Width (piksel)", min_value=800, max_value=4000, value=1920, step=10)
        inches = st.number_input("Ukuran layar (Inches)", min_value=10.0, max_value=20.0, value=15.6, step=0.1)
        weight = st.number_input("Berat (kg)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)

    submitted = st.form_submit_button("🔮 Prediksi Harga")

if submitted:
    raw_input = {
        "Company": company,
        "TypeName": typename,
        "Ram": ram,
        "CpuTier": cputier,
        "SSD": ssd,
        "Not_SSD": not_ssd,
        "Screen_Width": screen_width,
        "Inches": inches,
        "Weight": weight,
        "GpuBrand": gpubrand,
    }

    try:
        price_eur = predict_price(raw_input)
        price_eur = max(price_eur, 0)  # jaga-jaga kalau ada prediksi negatif

        st.success("Prediksi berhasil!")
        st.metric("Estimasi Harga Laptop", f"€ {price_eur:,.2f}")

        with st.expander("Lihat detail input yang digunakan"):
            st.json(raw_input)

    except Exception as e:
        st.error(f"Terjadi error saat melakukan prediksi: {e}")

st.divider()
st.caption(
    "⚠️ Catatan: Model ini dilatih pada dataset laptop_price.csv (dataset Kaggle "
    "'muhammetvarl/laptop-price') dan hanya untuk keperluan demo/bootcamp. "
    "Hasil prediksi adalah estimasi statistik, bukan harga pasti."
)
