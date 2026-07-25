# Panduan Deployment - Laptop Price Prediction

Folder ini berisi semua yang kamu butuhkan untuk deploy model prediksi harga
laptop (Gradient Boosting Regressor, hasil hyperparameter tuning) ke sebuah
web app menggunakan **Streamlit**.

## 📁 Isi Folder

```
laptop_price_deployment/
├── app.py                            # Seluruh kode aplikasi (1 file, 3 "halaman")
├── requirements.txt                  # Daftar library yang dibutuhkan
├── runtime.txt                       # Pin versi Python untuk Streamlit Cloud
├── best_laptop_price_model.pkl       # Model hasil training (Gradient Boosting, tuned)
├── company_segment_mapping.pkl       # Mapping brand -> segmen harga (1/2/3)
├── cpu_tier_mapping.pkl              # Mapping CPU tier -> skor ordinal (0-3)
├── gpu_brand_mapping.pkl             # Mapping GPU brand -> skor ordinal (0-3)
├── typename_onehot_encoder.pkl       # OneHotEncoder khusus kolom TypeName
├── standard_scaler.pkl               # StandardScaler yang sudah di-fit
└── README.md                         # File ini
```

Aplikasi ini **1 file (`app.py`)** dengan 3 "halaman" yang dipilih lewat
dropdown **Menu** di sidebar kiri (bukan multipage otomatis bawaan
Streamlit):

1. **🏠 Home** — penjelasan aplikasi, link dataset Kaggle, link notebook
   preprocessing/training
2. **📖 Panduan Pemakaian** — cara mengisi form beserta contoh penggunaan
3. **💻 Laptop Price Prediction** — form utama untuk memprediksi harga laptop

**Penting:** semua file `.pkl` harus tetap berada di folder **root** yang
sama dengan `app.py`, karena dimuat dengan path relatif
(`joblib.load("best_laptop_price_model.pkl")`, dst) yang dihitung dari
direktori tempat `streamlit run app.py` dijalankan.

## 📌 Riwayat Model & Pipeline

| Versi | Encoding Company/CpuTier/GpuBrand | Encoding TypeName | Model |
|---|---|---|---|
| v1 | One-Hot Encoding (1 encoder gabungan) | One-Hot Encoding | Extra Trees Regressor |
| v2 | One-Hot Encoding (1 encoder gabungan) | One-Hot Encoding | Gradient Boosting (tuned) |
| **v3 (current)** | **Ordinal mapping** (segmentasi/agregasi harga) | One-Hot Encoding (encoder terpisah) | Gradient Boosting (tuned ulang) |

Revisi v3 mengganti cara encoding `Company`, `CpuTier`, dan `GpuBrand` dari
One-Hot Encoding menjadi **mapping manual berbasis segmentasi harga** (bukan
lagi dummy variable), berdasarkan hasil analisis mutual information &
p-value importance di notebook. `TypeName` tetap pakai One-Hot Encoding,
tapi sekarang dengan encoder-nya sendiri (`typename_onehot_encoder.pkl`).

## 🔧 Bagaimana Pipeline Prediksi Bekerja

Model dilatih dengan alur berikut (lihat notebook), jadi urutannya harus
persis sama saat inference:

1. **Fitur mentah** yang dipakai (10 kolom sebelum encoding):
   `Company`, `TypeName`, `Ram`, `CpuTier`, `SSD`, `Not_SSD`, `Screen_Width`,
   `Inches`, `Weight`, `GpuBrand`
2. **Mapping ordinal** (bukan One-Hot):
   - `Company` → segmen harga brand (1 = entry-level, 2 = mid-level,
     3 = premium) via `company_segment_mapping.pkl`
   - `CpuTier` → skor 0-3 (`Other`=0, `i3`=1, `i5`/`Ryzen`=2, `i7`=3) via
     `cpu_tier_mapping.pkl`
   - `GpuBrand` → skor 0-3 (`ARM`=0, `AMD`=1, `Intel`=2, `Nvidia`=3) via
     `gpu_brand_mapping.pkl`
3. **One-Hot Encoding** untuk `TypeName` saja, via `typename_onehot_encoder.pkl`
4. Semua fitur (numerik + hasil mapping + hasil one-hot TypeName) digabung,
   lalu di-**scale** dengan `standard_scaler.pkl` (15 fitur total)
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

1. **Buat repository GitHub baru**, lalu upload seluruh isi folder
   `laptop_price_deployment/` (termasuk semua file `.pkl` dan `runtime.txt`)
   ke repo tersebut.
2. Buka **[share.streamlit.io](https://share.streamlit.io)** dan login
   dengan akun GitHub.
3. Klik **"New app"**, pilih repository, branch, dan file utama `app.py`.
4. **Sebelum klik Deploy**, buka **"Advanced settings"** dan pilih versi
   Python **3.11 atau 3.12** secara manual (lihat catatan bug di bawah).
5. Klik **Deploy**. Setelah build selesai (biasanya 1-3 menit), kalian akan
   mendapatkan URL publik seperti `https://namaapp.streamlit.app`.

> ⚠️ **Bug platform yang sedang terjadi (per pertengahan 2026):** Streamlit
> Community Cloud kadang memaksa pakai Python 3.14 meski sudah diset lewat
> `runtime.txt` maupun Advanced Settings, yang menyebabkan build gagal/lambat
> karena banyak library (scikit-learn, pandas, dst) belum punya wheel untuk
> Python 3.14. Ini bug resmi yang sudah dilaporkan ke tim Streamlit
> ([issue #15326](https://github.com/streamlit/streamlit/issues/15326)) dan
> di luar kendali kita. Kalau kena, coba delete & redeploy app beberapa kali,
> atau pakai opsi Docker di bawah sebagai alternatif sementara.

## 🐳 3. Alternatif: Deploy dengan Docker (kalau butuh platform lain)

Kalau kelompok kalian perlu deploy ke platform selain Streamlit Cloud
(misalnya Hugging Face Spaces, Railway, Render, atau server sendiri),
tinggal tambahkan `Dockerfile` berikut di folder yang sama:

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

- **Versi scikit-learn**: model, mapping, encoder, dan scaler disimpan
  menggunakan scikit-learn `1.6.1`. Versi ini dipin persis di
  `requirements.txt`. Kalau kalian pakai versi scikit-learn yang jauh
  berbeda, ada risiko pickle gagal dimuat atau hasil prediksi berubah.
- **Kategori yang tidak dikenal**: `typename_onehot_encoder.pkl` dibuat
  dengan `handle_unknown='ignore'`, jadi input `TypeName` baru yang belum
  pernah dilihat model tidak akan error, tapi di-encode jadi semua nol.
  Untuk `Company`, `CpuTier`, dan `GpuBrand` yang pakai mapping manual,
  pastikan pilihan di form (dropdown) selalu mengikuti key yang ada di
  masing-masing file mapping `.pkl` — form di `app.py` sudah otomatis
  mengambil opsi dari situ, jadi aman selama file mapping-nya tidak diubah.
- **Satuan harga**: model memprediksi harga dalam **Euro**, sesuai kolom
  `Price_euros` di dataset asli. Tidak ada konversi ke mata uang lain di
  aplikasi ini.
- Kalau nanti mau retrain/update model, cukup ganti keenam file `.pkl`
  dengan yang baru (asalkan nama file & struktur fitur akhir tetap sama),
  tidak perlu mengubah kode di `app.py` — kecuali kalau ada kolom fitur yang
  ditambah/dihapus, baru perlu update fungsi `predict_price()`.

## 🧪 Testing Cepat (Tanpa UI)

```python
import importlib.util

spec = importlib.util.spec_from_file_location("laptop_app", "app.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)  # beberapa warning Streamlit di sini normal, aman diabaikan

model, company_map, cputier_map, gpubrand_map, typename_encoder, scaler = module.load_artifacts()

contoh_input = {
    "Company": "Dell",
    "TypeName": "Notebook",
    "Ram": 8,
    "CpuTier": "i5",
    "SSD": 256,
    "Not_SSD": 2000,
    "Screen_Width": 1920,
    "Inches": 15.6,
    "Weight": 2.1,
    "GpuBrand": "Intel",
}

print(module.predict_price(
    contoh_input, model, company_map, cputier_map, gpubrand_map, typename_encoder, scaler
))
```

Cara paling praktis tetap lewat browser: jalankan `streamlit run app.py`,
lalu pilih **💻 Laptop Price Prediction** di dropdown Menu sidebar.

Selamat mengerjakan bagian deployment-nya! 🚀
