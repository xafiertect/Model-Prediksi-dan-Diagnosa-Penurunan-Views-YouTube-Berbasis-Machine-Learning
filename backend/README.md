# Hippo Academy — Backend API

Backend ini dibangun menggunakan **FastAPI**, **Python**, dan memuat model **Machine Learning** untuk melakukan prediksi performa video YouTube.

## Prasyarat
- Python 3.10 atau versi di atasnya
- Lingkungan virtual (opsional namun sangat direkomendasikan, misalnya `venv` atau `conda`)

## Instalasi

1. Buka terminal dan masuk ke folder `backend`:
   ```bash
   cd backend
   ```

2. Buat lingkungan virtual (virtual environment) dan aktifkan:
   - **Linux/Mac**:
     ```bash
     python -m venv venv
     source venv/bin/activate
     ```
   - **Windows**:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```

3. Instal dependensi:
   ```bash
   pip install -r requirements.txt
   ```

## Konfigurasi Lingkungan (Environment Variables)

1. Buat file `.env` dengan menyalin konfigurasi dari `.env.example`:
   ```bash
   cp .env.example .env
   ```
2. Sesuaikan konfigurasi di dalam file `.env` dengan kredensial API dan direktori yang sesuai (terutama `MODEL_PATH` untuk mengarah ke model `.pkl` dan API key LLM/YouTube jika digunakan).

## Menjalankan Aplikasi Lokal

Untuk menjalankan *server development* lokal:

```bash
uvicorn main:app --reload --port 8000
```

## Dokumentasi API (Swagger)

Setelah server berjalan, Anda dapat mengakses dokumentasi interaktif bawaan FastAPI di browser:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc UI**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
