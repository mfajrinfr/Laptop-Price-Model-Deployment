"""
Laptop Price Prediction - Streamlit App (single-file, manual navigation)
===========================================================================
Aplikasi ini punya 3 "halaman" (Home, Panduan Pemakaian, Laptop Price
Prediction) yang di-navigasi lewat dropdown "Menu" di sidebar (bukan
multipage otomatis bawaan Streamlit), supaya tampilannya sesuai kebutuhan.
"""

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Laptop Price Prediction",
    page_icon="💻",
    layout="centered",
)

# ---------------------------------------------------------------------------
# NAVIGASI - dropdown "Menu" di sidebar (bukan multipage bawaan Streamlit)
# ---------------------------------------------------------------------------
PAGES = ["Home", "Panduan Pemakaian", "Laptop Price Prediction"]

if "menu_page" not in st.session_state:
    st.session_state.menu_page = "Home"

page = st.sidebar.selectbox("Menu", PAGES, key="menu_page")


def goto(page_name: str):
    """Helper untuk pindah halaman secara programatik (dipakai tombol/link)."""
    st.session_state.menu_page = page_name
    st.rerun()


# ---------------------------------------------------------------------------
# KONSTANTA & LOADER MODEL (dipakai khusus di halaman Prediction)
# ---------------------------------------------------------------------------
CAT_COLS = ["Company", "TypeName", "CpuTier", "GpuBrand"]
NUM_COLS = ["Ram", "SSD", "Not_SSD", "Screen_Width", "Inches", "Weight"]
RAM_OPTIONS = [2, 4, 6, 8, 12, 16, 24, 32, 64]
STORAGE_TYPES = ["SSD", "HDD", "Hybrid", "Flash Storage"]
LAYER2_TYPES = ["Tidak ada"] + STORAGE_TYPES


@st.cache_resource
def load_artifacts():
    model = joblib.load("best_laptop_price_model.pkl")
    encoder = joblib.load("onehot_encoder.pkl")
    scaler = joblib.load("standard_scaler.pkl")
    return model, encoder, scaler


def predict_price(raw_input: dict, model, encoder, scaler) -> float:
    """Terima 1 baris input mentah (dict), kembalikan prediksi harga (EUR)."""
    df = pd.DataFrame([raw_input])

    encoded = encoder.transform(df[CAT_COLS])
    encoded_df = pd.DataFrame(
        encoded, columns=encoder.get_feature_names_out(CAT_COLS), index=df.index
    )

    X = pd.concat([df[NUM_COLS], encoded_df], axis=1)
    if "GpuBrand_ARM" in X.columns:
        X = X.drop(columns=["GpuBrand_ARM"])

    X = X.reindex(columns=scaler.feature_names_in_, fill_value=0)
    X_scaled = scaler.transform(X)
    return float(model.predict(X_scaled)[0])


# ---------------------------------------------------------------------------
# HALAMAN 1: HOME
# ---------------------------------------------------------------------------
def show_home():
    st.title("💻 Laptop Price Prediction App")

    st.write(
        "Aplikasi ini dibuat untuk memprediksi estimasi harga laptop (dalam "
        "**Euro**) berdasarkan spesifikasinya, menggunakan model Machine "
        "Learning yang sudah dilatih sebelumnya. "
    )

    st.subheader("📊 Dataset")
    st.write(
        "Dataset yang digunakan adalah **Laptop Price Dataset** dari Kaggle, "
        "berisi spesifikasi dan harga (dalam Euro) dari berbagai laptop — "
        "seperti brand, tipe, RAM, storage, ukuran layar, berat, CPU, dan GPU."
    )
    st.link_button(
        "🔗 Buka Dataset di Kaggle",
        "https://www.kaggle.com/datasets/muhammetvarl/laptop-price/code",
    )

    st.subheader("🧪 Data Preprocessing & Training")
    st.write(
        "Seluruh proses data preprocessing, feature engineering, pemilihan "
        "model, hingga hyperparameter tuning dilakukan pada notebook berikut "
        "ini. Model terbaik yang dihasilkan (Gradient Boosting Regressor, "
        "hasil tuning) adalah model yang dipakai di aplikasi ini."
    )
    st.link_button(
        "🔗 Buka Notebook Preprocessing (Google Colab)",
        "https://colab.research.google.com/drive/11Ptl2KECMnaPo8-I49E__MTa84xteGZm?usp=sharing",
    )

    st.subheader("📁 Isi Aplikasi")
    st.write("Aplikasi ini terdiri dari 3 halaman, bisa dipilih lewat dropdown **Menu** di sidebar kiri:")
    st.markdown(
        """
- **🏠 Home** *(halaman ini)* — penjelasan umum aplikasi, dataset, dan sumber preprocessing
- **📖 Panduan Pemakaian** — panduan cara mengisi form dan contoh penggunaan
- **💻 Laptop Price Prediction** — form untuk memprediksi harga laptop
"""
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📖 Buka Panduan Pemakaian", use_container_width=True):
            goto("Panduan Pemakaian")
    with col2:
        if st.button("💻 Coba Prediksi Harga", use_container_width=True):
            goto("Laptop Price Prediction")

    st.divider()
    st.caption(
        "⚠️ Catatan: Model ini dilatih pada dataset laptop_price.csv (dataset "
        "Kaggle 'muhammetvarl/laptop-price') dan hanya untuk keperluan demonstrasi."
        " Hasil prediksi adalah estimasi statistik, bukan harga pasti."
    )


# ---------------------------------------------------------------------------
# HALAMAN 2: PANDUAN PEMAKAIAN
# ---------------------------------------------------------------------------
def show_panduan():
    st.title("📖 Panduan Pemakaian")
    st.write(
        "Halaman ini menjelaskan cara mengisi form di halaman "
        "**💻 Laptop Price Prediction**, langkah demi langkah."
    )

    st.subheader("1️⃣ Spesifikasi Umum")
    st.markdown(
        """
- **Brand / Company** — merek laptop (Dell, Apple, Asus, dst)
- **Tipe Laptop** — kategori laptop (Notebook, Ultrabook, Gaming, dst)
- **CPU Tier** — kelas prosesor (i3, i5, i7, Ryzen, atau Other)
- **GPU Brand** — merek kartu grafis (Intel, Nvidia, AMD)
- **RAM (GB)** — geser slider sesuai kapasitas RAM laptop
"""
    )

    st.subheader("2️⃣ Storage")
    st.markdown(
        """
Isi storage berdasarkan kondisi fisik laptop:

- **Storage Layer 1** *(wajib)* — pilih tipe storage-nya (SSD / HDD / Hybrid /
  Flash Storage), lalu isi kapasitas dan satuannya (GB atau TB)
- **Storage Layer 2** *(opsional)* — isi kalau laptop punya storage kedua
  (misalnya kombinasi SSD + HDD, atau 2 SSD sekaligus). Kalau laptop cuma
  punya 1 storage, pilih **"Tidak ada"** di Layer 2 — field kapasitas dan
  satuannya akan otomatis nonaktif

Aplikasi akan **otomatis menjumlahkan** total kapasitas SSD dan total
kapasitas non-SSD (HDD/Hybrid/Flash) dari Layer 1 & Layer 2 — jadi kamu
tidak perlu menjumlahkan atau mengonversi satuan secara manual.
"""
    )

    st.markdown("**Contoh pengisian Storage:**")
    st.table(
        {
            "Kondisi Laptop": [
                "256GB SSD saja",
                "1TB HDD saja",
                "256GB SSD + 2TB HDD",
                "512GB SSD + 512GB SSD",
                "64GB Flash Storage + 1TB HDD",
            ],
            "Layer 1": [
                "SSD, 256, GB",
                "HDD, 1, TB",
                "SSD, 256, GB",
                "SSD, 512, GB",
                "Flash Storage, 64, GB",
            ],
            "Layer 2": [
                "Tidak ada",
                "Tidak ada",
                "HDD, 2, TB",
                "SSD, 512, GB",
                "HDD, 1, TB",
            ],
        }
    )

    st.subheader("3️⃣ Dimensi Fisik")
    st.markdown(
        """
- **Screen Width (piksel)** — lebar resolusi layar (contoh: 1920 untuk Full HD)
- **Ukuran Layar (Inches)** — geser slider sesuai ukuran layar laptop
- **Berat (kg)** — geser slider sesuai berat laptop
"""
    )

    st.subheader("4️⃣ Prediksi")
    st.write(
        "Setelah semua field terisi, klik tombol **🔮 Prediksi Harga**. "
        "Aplikasi akan menampilkan estimasi harga laptop dalam **Euro**, "
        "beserta detail input yang digunakan (bisa dilihat di bagian "
        "*'Lihat detail input yang digunakan'*)."
    )

    st.divider()
    st.subheader("✅ Contoh Penggunaan Lengkap")
    st.write("Misalnya kamu ingin mengestimasi harga laptop dengan spesifikasi berikut:")
    st.markdown(
        """
| Field | Nilai |
|---|---|
| Brand / Company | Dell |
| Tipe Laptop | Notebook |
| CPU Tier | i5 |
| GPU Brand | Intel |
| RAM | 8 GB |
| Storage Layer 1 | SSD, 256 GB |
| Storage Layer 2 | HDD, 2 TB |
| Screen Width | 1920 px |
| Ukuran Layar | 15.6 inch |
| Berat | 2.1 kg |
"""
    )
    st.info("Dengan spesifikasi di atas, model memberikan estimasi harga sekitar **€ 927**.")

    if st.button("💻 Coba sendiri di halaman Laptop Price Prediction →"):
        goto("Laptop Price Prediction")


# ---------------------------------------------------------------------------
# HALAMAN 3: LAPTOP PRICE PREDICTION
# ---------------------------------------------------------------------------
def show_prediction():
    model, encoder, scaler = load_artifacts()

    company_options = list(encoder.categories_[0])
    typename_options = list(encoder.categories_[1])
    cputier_options = list(encoder.categories_[2])
    # 'ARM' sengaja dibuang: saat training baris dengan GPU brand ARM di-drop
    gpubrand_options = [g for g in encoder.categories_[3] if g != "ARM"]

    st.title("💻 Prediksi Harga Laptop")
    st.write(
        "Masukkan spesifikasi laptop di bawah ini, lalu klik **Prediksi "
        "Harga** untuk mengestimasi harga laptop menggunakan model Machine "
        "Learning (Gradient Boosting Regressor, hasil hyperparameter "
        "tuning) yang telah dilatih sebelumnya. Belum tahu cara isinya? "
        "Lihat halaman **📖 Panduan Pemakaian** di sidebar."
    )

    col1, col2 = st.columns(2)
    with col1:
        company = st.selectbox("Brand / Company", company_options)
        cputier = st.selectbox("CPU Tier", cputier_options)
        ram = st.select_slider("RAM (GB)", options=RAM_OPTIONS, value=8)
    with col2:
        typename = st.selectbox("Tipe Laptop", typename_options)
        gpubrand = st.selectbox("GPU Brand", gpubrand_options)

    st.markdown("---")
    st.markdown(
        "**💾 Storage**  \n"
    )

    st.caption("Storage Layer 1 (wajib)")
    l1_type_col, l1_val_col, l1_unit_col = st.columns([2, 2, 1])
    with l1_type_col:
        layer1_type = st.selectbox("Tipe storage", STORAGE_TYPES, key="l1_type")
    with l1_val_col:
        layer1_value = st.number_input("Kapasitas", min_value=0.0, value=256.0, step=32.0, key="l1_val")
    with l1_unit_col:
        layer1_unit = st.selectbox("Satuan", ["GB", "TB"], key="l1_unit")

    st.caption("Storage Layer 2 (opsional — pilih 'Tidak ada' jika laptop cuma punya 1 storage)")
    l2_type_col, l2_val_col, l2_unit_col = st.columns([2, 2, 1])
    with l2_type_col:
        layer2_type = st.selectbox("Tipe storage", LAYER2_TYPES, key="l2_type")

    # Field kapasitas & satuan Layer 2 otomatis nonaktif kalau tipe = "Tidak ada".
    # Ini butuh widget di luar st.form supaya perubahan dropdown langsung
    # memicu rerun dan status disabled ter-update secara real-time.
    layer2_disabled = (layer2_type == "Tidak ada")
    with l2_val_col:
        layer2_value = st.number_input(
            "Kapasitas", min_value=0.0, value=0.0, step=32.0,
            key="l2_val", disabled=layer2_disabled,
        )
    with l2_unit_col:
        layer2_unit = st.selectbox(
            "Satuan", ["GB", "TB"], key="l2_unit", disabled=layer2_disabled,
        )

    st.markdown("---")
    dim_col1, dim_col2, dim_col3 = st.columns(3)
    with dim_col1:
        screen_width = st.number_input("Screen Width (piksel)", min_value=800, max_value=4000, value=1920, step=10)
    with dim_col2:
        inches = st.slider("Ukuran layar (Inches)", min_value=10.0, max_value=20.0, value=15.6, step=0.1)
    with dim_col3:
        weight = st.slider("Berat (kg)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)

    submitted = st.button("🔮 Prediksi Harga", type="primary", use_container_width=True)

    if submitted:
        layer1_gb = layer1_value * 1000 if layer1_unit == "TB" else layer1_value
        layer2_gb = 0.0
        if not layer2_disabled:
            layer2_gb = layer2_value * 1000 if layer2_unit == "TB" else layer2_value

        ssd = (layer1_gb if layer1_type == "SSD" else 0) + (
            layer2_gb if (not layer2_disabled and layer2_type == "SSD") else 0
        )
        not_ssd = (
            (layer1_gb if layer1_type in ("HDD", "Hybrid", "Flash Storage") else 0)
            + (layer2_gb if (not layer2_disabled and layer2_type in ("HDD", "Hybrid", "Flash Storage")) else 0)
        )

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
            price_eur = predict_price(raw_input, model, encoder, scaler)
            price_eur = max(price_eur, 0)

            st.success("Prediksi berhasil!")
            st.metric("Estimasi Harga Laptop", f"€ {price_eur:,.2f}")

            with st.expander("Lihat detail input yang digunakan"):
                st.json(raw_input)

        except Exception as e:
            st.error(f"Terjadi error saat melakukan prediksi: {e}")

    st.divider()
    st.caption(
        "⚠️ Catatan: Model ini dilatih pada dataset laptop_price.csv (dataset "
        "Kaggle 'muhammetvarl/laptop-price') dan hanya untuk keperluan demonstrasi."
        " Hasil prediksi adalah estimasi statistik, bukan harga pasti."
    )


# ---------------------------------------------------------------------------
# ROUTING
# ---------------------------------------------------------------------------
if page == "Home":
    show_home()
elif page == "Panduan Pemakaian":
    show_panduan()
elif page == "Laptop Price Prediction":
    show_prediction()
