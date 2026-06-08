# Build context = project root (lihat docker-compose.yml: context: .)
# Ini supaya path relatif yang dipakai kode (model_loader._backend_dir,
# stats._PROJECT_ROOT, youtube_oauth.TOKEN_FILE) tetap resolve dengan benar:
# image jadi /app/backend (= "backend dir") + /app/data (= "project root"/data).
FROM python:3.12-slim

WORKDIR /app/backend

# user non-root buat keamanan
RUN useradd -m -u 1000 appuser

# requirements.txt sekarang dikonsolidasikan di root (lihat CLAUDE.md/.gitignore
# history) — backend/requirements.txt sudah tidak ada di branch ini.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

# Folder yang di-mount sebagai volume (lihat docker-compose.yml) — siapkan dulu
# biar permission appuser benar saat volume kosong di-mount di atasnya.
RUN mkdir -p models scalers data /app/data/processed /app/data/cleaned \
    && touch token_store.json \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
