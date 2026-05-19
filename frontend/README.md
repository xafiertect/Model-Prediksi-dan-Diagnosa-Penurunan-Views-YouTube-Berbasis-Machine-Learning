# Hippo Academy — Frontend

Aplikasi frontend ini dibangun menggunakan **React**, **Vite**, dan **Tailwind CSS**.

## Prasyarat
- Node.js (direkomendasikan versi 18+)
- npm (atau yarn/pnpm)

## Instalasi

1. Buka terminal dan masuk ke folder `frontend`:
   ```bash
   cd frontend
   ```

2. Instal semua dependensi:
   ```bash
   npm install
   ```

## Menjalankan Aplikasi Lokal

1. Salin file environment (jika ada `.env.example`):
   ```bash
   cp .env.example .env
   ```
   *(Pastikan `VITE_API_URL` mengarah ke URL backend Anda, biasanya `http://localhost:8000/api/v1`)*

2. Jalankan server pengembangan (Vite):
   ```bash
   npm run dev
   ```

3. Buka browser dan akses aplikasi pada URL yang ditampilkan di terminal (secara default biasanya `http://localhost:5173`).

## Perintah Tambahan
- `npm run build`: Membangun versi production aplikasi.
- `npm run preview`: Melihat *preview* aplikasi setelah di-*build* secara lokal.
