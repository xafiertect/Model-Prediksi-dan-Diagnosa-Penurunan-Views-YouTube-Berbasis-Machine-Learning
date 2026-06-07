"""
Router: /auth
=============
Endpoint OAuth 2.0 untuk koneksi akun YouTube.
Mengelola login, callback, status autentikasi, dan pengambilan data real-time channel.
"""

import os
import secrets

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse

from utils.youtube_oauth import (
    build_flow, save_token, is_authenticated,
    is_configured, delete_token
)
from utils.youtube_api import (
    fetch_channel_info, fetch_recent_videos, fetch_video_analytics
)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_LIVE_CSV = os.path.join(_PROJECT_ROOT, "data", "processed", "youtube_live.csv")

router = APIRouter(prefix="/auth", tags=["YouTube OAuth"])

# State CSRF & PKCE — disimpan di memory (cukup untuk single-user local app)
_oauth_state: str = ""
_oauth_verifier: str = ""


@router.get("/youtube/login")
async def youtube_login():
    """Memulai flow OAuth 2.0. Redirect ke halaman consent Google."""
    global _oauth_state, _oauth_verifier

    if not is_configured():
        raise HTTPException(
            status_code=503,
            detail="Integrasi YouTube belum dikonfigurasi. Tambahkan YOUTUBE_CLIENT_ID & YOUTUBE_CLIENT_SECRET di .env"
        )

    flow = build_flow()
    _oauth_state = secrets.token_urlsafe(16)

    auth_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        state=_oauth_state,
        prompt="consent"
    )
    # Simpan PKCE verifier yang di-generate otomatis
    _oauth_verifier = getattr(flow, "code_verifier", "")
    
    return RedirectResponse(url=auth_url)


@router.get("/youtube/callback")
async def youtube_callback(code: str = "", state: str = "", error: str = ""):
    """Callback dari Google setelah user menyetujui OAuth."""
    global _oauth_state, _oauth_verifier

    frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173")

    # Handle user rejection
    if error:
        return RedirectResponse(url=f"{frontend_url}/?yt_error=access_denied")

    # Validasi CSRF state
    if not state or state != _oauth_state:
        raise HTTPException(status_code=400, detail="Request OAuth tidak valid. Coba login ulang.")

    if not code:
        raise HTTPException(status_code=400, detail="Authorization code tidak ditemukan.")

    try:
        flow = build_flow()
        # Restore PKCE verifier
        if _oauth_verifier:
            flow.code_verifier = _oauth_verifier

        flow.fetch_token(code=code)
        save_token(flow.credentials)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal menukar token: {str(e)}")

    return RedirectResponse(url=f"{frontend_url}/?yt_connected=true")


@router.get("/youtube/status")
async def youtube_status():
    """Cek apakah user sudah terhubung dengan akun YouTube."""
    authenticated = is_authenticated()
    configured = is_configured()
    return {
        "is_authenticated": authenticated,
        "is_configured": configured,
    }


@router.get("/youtube/logout")
async def youtube_logout():
    """Hapus token OAuth — user harus login ulang untuk reconnect."""
    delete_token()
    return {"message": "Berhasil logout dari akun YouTube."}


@router.post("/youtube/sync")
async def sync_youtube_videos(max_videos: int = 20, include_analytics: bool = False):
    """
    Sinkronisasi video terbaru dari YouTube ke youtube_live.csv.
    Data disimpan di data/processed/youtube_live.csv dan otomatis digabung
    ke /stats/videos tanpa mengubah abis_cleaning.csv.
    """
    if not is_authenticated():
        raise HTTPException(
            status_code=401,
            detail="Sesi YouTube telah habis. Silakan login ulang."
        )

    try:
        channel = fetch_channel_info()
        channel_id = channel.get("channel_id", "")
        videos = fetch_recent_videos(max_results=max_videos)

        if not videos:
            return {"status": "ok", "synced": 0, "message": "Tidak ada video ditemukan di channel."}

        rows = []
        for v in videos:
            analytics: dict = {}
            if include_analytics:
                try:
                    analytics = fetch_video_analytics(v["video_id"], channel_id)
                except Exception:
                    pass

            views = v["views"]
            row = {
                "video_id":          v["video_id"],
                "title":             v["title"],
                "views":             views,
                "ctr":               analytics.get("ctr", 0.0),
                "date":              v["published_at"],
                "impressions":       analytics.get("impressions", 0),
                "avg_view_duration": analytics.get("avg_view_duration", "00:03:00"),
                "video_duration":    v["video_duration"],
                "likes":             v["likes"],
                "comments":          v["comments"],
                "retention_rate":    analytics.get("retention_rate", 0.0),
                "subscriber_gained": analytics.get("subscriber_gained", 0),
                "lag_views_7d":      analytics.get("lag_views_7d", 0),
                "rolling_mean_14d":  analytics.get("rolling_mean_views_14d", 0),
                "video_age_days":    v["video_age_days"],
                "anomaly_label_model": 0,
                "anomaly_score":       0.0,
                "source":            "youtube_live",
            }
            rows.append(row)

        new_df = pd.DataFrame(rows)

        if os.path.exists(_LIVE_CSV):
            existing = pd.read_csv(_LIVE_CSV)
            existing = existing[~existing["video_id"].isin(new_df["video_id"])]
            merged = pd.concat([existing, new_df], ignore_index=True)
        else:
            os.makedirs(os.path.dirname(_LIVE_CSV), exist_ok=True)
            merged = new_df

        merged.to_csv(_LIVE_CSV, index=False)

        return {
            "status":         "ok",
            "synced":         len(rows),
            "total_in_cache": len(merged),
            "channel":        channel.get("title", ""),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal sinkronisasi video: {str(e)}")


@router.get("/youtube/sync/status")
async def sync_status():
    """Cek jumlah video yang tersimpan di youtube_live.csv."""
    if not os.path.exists(_LIVE_CSV):
        return {"cached": 0, "last_sync": None}
    try:
        df = pd.read_csv(_LIVE_CSV)
        last_sync = None
        if "date" in df.columns and not df.empty:
            last_sync = str(df["date"].max())
        return {"cached": len(df), "last_sync": last_sync}
    except Exception:
        return {"cached": 0, "last_sync": None}


@router.get("/youtube/channel")
async def get_channel_data(max_videos: int = 20):
    """
    Mengambil data channel dan daftar video terbaru secara real-time.
    Memerlukan autentikasi OAuth aktif.
    """
    if not is_authenticated():
        raise HTTPException(
            status_code=401,
            detail="Sesi YouTube telah habis. Silakan login ulang."
        )

    try:
        channel = fetch_channel_info()
        videos = fetch_recent_videos(max_results=max_videos)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal mengambil data channel: {str(e)}")

    return {
        "status": "success",
        "channel": channel,
        "videos": videos,
    }


@router.get("/youtube/video/{video_id}/metrics")
async def get_video_metrics(video_id: str):
    """
    Mengambil semua metrik real-time sebuah video dari YouTube Analytics API.
    Data dikembalikan dalam format siap pakai untuk form prediksi ML.
    """
    if not is_authenticated():
        raise HTTPException(
            status_code=401,
            detail="Sesi YouTube telah habis. Silakan login ulang."
        )

    try:
        channel = fetch_channel_info()
        channel_id = channel.get("channel_id", "")

        # Ambil info dasar video (durasi, age, likes, comments)
        videos = fetch_recent_videos(max_results=50)
        video_info = next((v for v in videos if v["video_id"] == video_id), None)

        if not video_info:
            raise HTTPException(status_code=404, detail="Video tidak ditemukan di channel Anda.")

        # Ambil data analytics (CTR, retensi, impressions, dll.)
        analytics = fetch_video_analytics(video_id, channel_id)

        # Gabungkan data video + analytics menjadi metrik lengkap
        metrics = {
            "views":                    video_info["views"],
            "likes":                    video_info["likes"],
            "comments":                 video_info["comments"],
            "video_duration":           video_info["video_duration"],
            "video_age_days":           video_info["video_age_days"],
            # Data dari Analytics API (jika tersedia)
            "ctr":                      analytics.get("ctr", 5.0),
            "impressions":              analytics.get("impressions", 100000),
            "avg_view_duration":        analytics.get("avg_view_duration", "00:03:00"),
            "retention_rate":           analytics.get("retention_rate", 35.0),
            "subscriber_gained":        analytics.get("subscriber_gained", 0),
            "lag_views_7d":             analytics.get("lag_views_7d", int(video_info["views"] * 0.8)),
            "rolling_mean_views_14d":   analytics.get("rolling_mean_views_14d", int(video_info["views"] * 0.9)),
        }

        return {
            "status": "success",
            "video_title": video_info["title"],
            "thumbnail": video_info["thumbnail"],
            "published_at": video_info["published_at"],
            "analytics_available": "_analytics_error" not in analytics,
            "metrics": metrics,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal mengambil metrik video: {str(e)}")
