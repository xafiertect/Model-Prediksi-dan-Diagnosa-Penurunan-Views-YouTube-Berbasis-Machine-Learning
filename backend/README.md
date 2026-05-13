# YouTube Prediction Backend API 🚀

Ini adalah layanan Backend berbasis **FastAPI** untuk memprediksi performa video YouTube (Viral, Normal, atau Declining) menggunakan model *Machine Learning* (Random Forest/XGBoost).

## 📁 Struktur Direktori
```text
backend/
├── main.py                # Titik masuk utama aplikasi FastAPI
├── routers/               # Endpoint modular API
│   ├── predict.py         # Endpoint prediksi ML
│   ├── history.py         # Endpoint historis (skeleton)
│   └── stats.py           # Endpoint statistik (skeleton)
├── schemas/               # Model validasi data (Pydantic)
│   └── prediction.py      
├── models/                # Penyimpanan model ML terlatih (.pkl)
│   ├── save_model.py      # Script pembuat dummy model lokal
│   └── ... (.pkl files)
├── requirements.txt       # Daftar dependensi Python
└── .env                   # File konfigurasi environment (secrets)
```

---

## 🛠️ Persiapan Lingkungan (Setup)

Pastikan Anda berada di direktori *root* proyek (folder yang memuat `venv/` dan `backend/`).

### 1. Aktifkan Virtual Environment
Buka terminal dan aktifkan *virtual environment* Python Anda:
**Linux / macOS:**
```bash
source venv/bin/activate
```
**Windows:**
```bash
venv\Scripts\activate
```

### 2. Pindah ke Folder Backend
Semua perintah backend harus dijalankan dari dalam folder `backend`.
```bash
cd backend
```

### 3. Install Dependensi
Jika Anda belum menginstal dependensi khusus backend, jalankan:
```bash
pip install -r requirements.txt
```

---

## ⚙️ Menyiapkan Model Machine Learning (Penting!)
Agar endpoint `/predict` bisa bekerja dengan baik, API membutuhkan file `.pkl`. Jika Anda belum menjalankan proses *training* penuh di Jupyter Notebook, Anda dapat men-generate *dummy models* terlebih dahulu:
```bash
python models/save_model.py
```
> Script di atas akan menghasilkan file `rf_classifier.pkl`, `scaler.pkl`, dll. di dalam folder `models/`.

---

## 🚀 Menjalankan Server API

Untuk menyalakan *server* lokal dengan fitur *live-reload* (server otomatis *restart* jika ada perubahan kode):
```bash
uvicorn main:app --reload
```

Jika berhasil, server akan berjalan di:
- **URL Dasar:** `http://localhost:8000`
- **Dokumentasi API Interaktif (Swagger UI):** `http://localhost:8000/docs`
- **Redoc UI:** `http://localhost:8000/redoc`

---

## 🧪 Menguji Endpoint API (Testing)

Anda bisa menguji endpoint API melalui Swagger UI (`http://localhost:8000/docs`) atau menggunakan alat seperti `cURL`/Postman.

### Contoh Request (cURL)
```bash
curl -X 'POST' \
  'http://localhost:8000/predict/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "views": 15000,
  "ctr": 7.5,
  "impressions": 200000,
  "avg_view_duration": 180,
  "engagement_rate": 5.2
}'
```

### Contoh Response Sukses (200 OK)
```json
{
  "status": "Viral",
  "confidence": 0.85,
  "predicted_views": 16650,
  "recommendation": "Excellent performance! Engage with comments and consider a follow-up video on the same topic."
}
```

### Contoh Response Gagal (503 Service Unavailable)
Muncul jika file model `.pkl` tidak ditemukan.
```json
{
  "error": "HTTP Exception",
  "detail": "Model or Scaler file not found. Please train and save models first.",
  "timestamp": "2026-05-11T00:35:00.000Z"
}
```

---
## 🔒 Environment Variables
Gunakan file `.env.example` sebagai referensi untuk membuat file `.env` lokal Anda. File `.env` mengatur *port*, pengaturan *CORS*, dan kunci *API rahasia* (jika kelak ditambahkan fungsi otentikasi).
