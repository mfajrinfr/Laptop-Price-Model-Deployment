# Panduan Deployment - Laptop Price Prediction

Folder ini berisi semua yang kamu butuhkan untuk deploy model prediksi harga
laptop (Gradient Boosting Regressor, hasil hyperparameter tuning) ke sebuah
web app sederhana menggunakan **Streamlit**.

## 📁 Isi Folder

```
laptop_price_deployment/
├── app.py                        # Kode aplikasi Streamlit
├── requirements.txt               # Daftar library yang dibutuhkan
├── best_laptop_price_model.pkl    # Model hasil training (Gradient Boosting, tuned)
├── onehot_encoder.pkl              # OneHotEncoder yang sudah di-fit
├── standard_scaler.pkl             # StandardScaler yang sudah di-fit
└── README.md                       # File ini
```

**Penting:** keempat file (`app.py`, `requirements.txt`, dan ketiga file `.pkl`)
harus berada dalam **folder yang sama**, karena `app.py` me-load ketiganya
menggunakan path relatif (`joblib.load("best_laptop_price_model.pkl")`, dst).

## 📌 Riwayat Model

| Versi | Model | Test R² | MAPE |
|---|---|---|---|
| v1 | Extra Trees Regressor | 0.81 | 17.41% |
| v2 (current) | Gradient Boosting Regressor (hasil hyperparameter tuning) | 0.8434 | 19.00%* |

\* Test R² lebih baik, meski MAPE sedikit lebih tinggi dari v1 — model v2 dipilih karena
proses seleksi model & tuning yang lebih menyeluruh di notebook, dan Adjusted R²-nya
juga lebih tinggi (0.8157 vs sebelumnya).

## 🔧 Bagaimana Pipeline Prediksi Bekerja

Model kalian dilatih dengan alur berikut (lihat notebook), jadi urutannya
harus persis sama saat inference:

1. **Fitur mentah** yang dipakai (8 kolom sebelum encoding):
   `Company`, `TypeName`, `Ram`, `CpuTier`, `SSD`, `Not_SSD`, `Screen_Width`,
   `Inches`, `Weight`, `GpuBrand`
2. **One-Hot Encoding** untuk 4 kolom kategorikal: `Company`, `TypeName`,
   `CpuTier`, `GpuBrand` → menggunakan `onehot_encoder.pkl`
3. Kolom `GpuBrand_ARM` dibuang (saat training, baris dengan GPU brand ARM
   di-drop karena datanya terlalu sedikit)
4. Semua fitur numerik + hasil one-hot digabung, lalu di-**scale** dengan
   `standard_scaler.pkl` (39 fitur total)
5. Hasil scaling masuk ke `best_laptop_price_model.pkl` untuk prediksi harga
   (dalam **Euro**)

Semua logika ini sudah dituliskan ulang di fungsi `predict_price()` pada
`app.py`, jadi kalian tidak perlu mengulang training atau menulis ulang
preprocessing — cukup jalankan app-nya.

## ▶️ 1. Menjalankan di Lokal (Testing)

```bash
# 1. (Opsional tapi disarankan) buat virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Jalankan aplikasi
streamlit run app.py
```

Browser akan otomatis terbuka di `http://localhost:8501`. Coba isi form
spesifikasi laptop lalu klik **Prediksi Harga** untuk memastikan semuanya
berjalan lancar sebelum deploy.

## ☁️ 2. Deploy ke Streamlit Community Cloud (Gratis)

Ini cara paling cepat kalau kelompok kalian butuh link publik untuk demo
bootcamp.

1. **Buat repository GitHub baru**, lalu upload seluruh isi folder
   `laptop_price_deployment/` (termasuk file `.pkl`-nya) ke repo tersebut.
   - Pastikan file `.pkl` ter-upload dengan benar (total ukuran model ± 25 MB,
     masih dalam batas wajar untuk GitHub biasa; kalau lebih dari 100 MB
     kalian perlu Git LFS).
2. Buka **[share.streamlit.io](https://share.streamlit.io)** dan login
   dengan akun GitHub.
3. Klik **"New app"**, pilih repository, branch, dan file utama `app.py`.
4. Klik **Deploy**. Streamlit Cloud akan otomatis install dependencies dari
   `requirements.txt` dan menjalankan aplikasinya.
5. Setelah selesai build (biasanya 1-3 menit), kalian akan mendapatkan URL
   publik seperti `https://namaapp.streamlit.app` yang bisa dibagikan ke
   dosen/mentor/tim.

## 🐳 3. Alternatif: Deploy dengan Docker (kalau butuh platform lain)

Kalau kelompok kalian perlu deploy ke platform selain Streamlit Cloud
(misalnya Railway, Render, atau server sendiri), tinggal tambahkan
`Dockerfile` berikut di folder yang sama:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build & run:
```bash
docker build -t laptop-price-app .
docker run -p 8501:8501 laptop-price-app
```

## ⚠️ Catatan Penting

- **Versi scikit-learn**: model, encoder, dan scaler disimpan menggunakan
  scikit-learn `1.6.1`. Versi ini sudah dipin di `requirements.txt`. Kalau
  kalian pakai versi scikit-learn yang jauh berbeda, ada risiko pickle gagal
  dimuat atau hasil prediksi berubah — jadi jangan asal update versi tanpa
  testing ulang.
- **Kategori yang tidak dikenal**: `OneHotEncoder` dibuat dengan
  `handle_unknown='ignore'`, jadi kalau ada input kategori baru yang belum
  pernah dilihat model saat training, encoder tidak akan error — tapi baris
  itu di-encode jadi semua nol (efeknya prediksi bisa kurang akurat untuk
  kombinasi yang benar-benar baru).
- **Satuan harga**: model memprediksi harga dalam **Euro**, sesuai kolom
  `Price_euros` di dataset asli. Tidak ada konversi ke mata uang lain di
  aplikasi ini.
- Kalau nanti mau retrain/update model, cukup ganti ketiga file `.pkl`
  dengan yang baru (asalkan nama file & struktur fitur akhir tetap sama),
  tidak perlu mengubah `app.py`.

## 🧪 Testing Cepat (Tanpa UI)

Kalau mau cek pipeline prediksi tanpa buka browser, kalian bisa jalankan
snippet ini dari Python:

```python
from app import predict_price

contoh_input = {
    "Company": "Dell",
    "TypeName": "Notebook",
    "Ram": 8,
    "CpuTier": "i5",
    "SSD": 256,
    "Not_SSD": 0,
    "Screen_Width": 1920,
    "Inches": 15.6,
    "Weight": 2.1,
    "GpuBrand": "Intel",
}

print(predict_price(contoh_input))
```

Selamat mengerjakan bagian deployment-nya! 🚀
