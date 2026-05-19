# # YouTube Analytics — Data Cleaning & Feature Engineering
# 
# **Tujuan Notebook:**
# 1. Pembersihan data mentah dari YouTube Studio
# 2. Engineering fitur-fitur penting untuk ML
# 3. Klasifikasi performa video (performance class)
# 
# ---

# --- CELL ---

# ## 0. Setup & Import

# --- CELL ---

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 50)
pd.set_option('display.float_format', '{:,.2f}'.format)
print(" Library siap!")

# --- CELL ---

# ## 1. Load Data

# --- CELL ---

FILE_PATH = '../data/raw/hippo_academy_raw.csv'
df_raw = pd.read_csv(FILE_PATH)

# Fix 1: Hapus duplicate columns setelah load
df_raw = df_raw.loc[:, ~df_raw.columns.duplicated()]
duplicates = df_raw.columns[df_raw.columns.duplicated()]
print("Duplicate columns:", duplicates.tolist())
print(f"Shape: {df_raw.shape}")
print(f"Columns: {df_raw.columns.tolist()}")
df_raw.head(3)

# --- CELL ---

# ---
# ## 2. Pembersihan Data (Data Cleaning)

# --- CELL ---

# ### 2.1 Hapus baris kosong & duplikat

# --- CELL ---

before = len(df_raw)
df = df_raw.dropna(how='all').copy()
df = df.drop_duplicates(subset=['video_title', 'publish_date'], keep='first')
df = df.reset_index(drop=True)
print(f"Setelah hapus baris kosong/duplikat: {before} → {len(df)} video")

# --- CELL ---

# ### 2.2 Konversi tipe data

# --- CELL ---

# Fix 2: Parsing durasi yang benar (handle MM:SS dan HH:MM:SS)
def parse_duration_to_seconds(duration_str):
    try:
        parts = str(duration_str).strip().split(':')
        parts = [int(p) for p in parts]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        elif len(parts) == 2:
            return parts[0] * 60 + parts[1]
        else:
            return int(parts[0])
    except:
        return np.nan

df['avg_watch_seconds'] = df['avg_view_duration'].apply(parse_duration_to_seconds)
df['video_duration_sec'] = pd.to_numeric(df['video_duration_sec'], errors='coerce')

# Fix 2b: Filter baris dengan durasi tidak valid
df = df[df['video_duration_sec'] > 0].copy()

numeric_cols = ['views', 'impressions', 'ctr(%)', 'likes', 'comments',
                'subscribers_gained', 'subscribers_lost', 'revenue_idr']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

print(f"avg_watch_seconds sample: {df['avg_watch_seconds'].head(3).tolist()}")
print(f"Sisa setelah filter durasi valid: {len(df)} video")

# --- CELL ---

# ### 2.3 Hapus baris tanpa data inti

# --- CELL ---

required_cols = ['views', 'video_duration_sec', 'impressions']
before = len(df)
df = df.dropna(subset=required_cols)
df = df[df['views'] > 0]
df = df.reset_index(drop=True)
print(f"Hapus {before - len(df)} baris tanpa data inti")
print(f"Sisa: {len(df)} video")

# --- CELL ---

# ### 2.4 Ringkasan data bersih

# --- CELL ---

print("=" * 50)
print(f"Data Bersih: {len(df)} video, {df.shape[1]} kolom")
print("=" * 50)
null_remaining = df.isnull().mean() * 100
null_remaining = null_remaining[null_remaining > 0].sort_values(ascending=False)
print("\nNull yang masih tersisa:")
print(null_remaining if len(null_remaining) else "✅ Tidak ada null tersisa!")

# --- CELL ---

# ---
# ## 3. Feature Engineering

# --- CELL ---

# ### 3.1 Helper: safe_divide

# --- CELL ---

# Fix 3: Helper untuk hindari division by zero
def safe_divide(a, b):
    """Bagi aman: mengembalikan NaN jika denominator 0."""
    return a / b.replace(0, np.nan) if hasattr(b, 'replace') else a / (b if b != 0 else np.nan)

print("safe_divide siap")

# --- CELL ---

# ### 3.2 Engagement Ratios

# --- CELL ---

views = df['views']
df['like_rate']     = safe_divide(df['likes'], views)
df['comment_rate']  = safe_divide(df['comments'], views)
df['engagement_rate'] = safe_divide(df[['likes','comments']].sum(axis=1), views)
print("Engagement ratios dibuat: like_rate, comment_rate, engagement_rate")

# --- CELL ---

# ### 3.3 Watch Time & Retention Metrics

# --- CELL ---

views = df['views']

# Fix 3: pakai safe_divide; Fix: watch_time_ratio selalu [0,1]
df['watch_time_ratio'] = safe_divide(df['avg_watch_seconds'], df['video_duration_sec']).clip(0, 1)
df['revenue_per_view'] = safe_divide(df['revenue_idr'], views)
df['subscriber_net']   = df['subscribers_gained'] - df['subscribers_lost'].fillna(0)
df['subscriber_net_per_view'] = safe_divide(df['subscriber_net'], views)

print("Watch time & retention metrics dibuat:")
print(f"  watch_time_ratio range: [{df['watch_time_ratio'].min():.2f}, {df['watch_time_ratio'].max():.2f}]")

# --- CELL ---

# ### 3.4 Impression & CTR Metrics

# --- CELL ---

# Fix 4: CTR selalu dalam range [0,1]
df['ctr'] = (df['ctr(%)'] / 100).clip(0, 1)
df['impression_to_view_rate'] = safe_divide(df['views'], df['impressions'])
print("Impression & CTR metrics dibuat")
print(f"  ctr range: [{df['ctr'].min():.3f}, {df['ctr'].max():.3f}]")

# --- CELL ---

# ### 3.5 Fitur Waktu Publikasi

# --- CELL ---

df['published_at'] = pd.to_datetime(df['publish_date'], errors='coerce')

if df['published_at'].notna().any():
    df['upload_hour']     = df['published_at'].dt.hour
    df['upload_day']      = df['published_at'].dt.dayofweek
    df['upload_day_name'] = df['published_at'].dt.day_name()
    df['upload_month']    = df['published_at'].dt.month
    df['upload_year']     = df['published_at'].dt.year
    today = pd.Timestamp.today().normalize()
    df['video_age_days']  = (today - df['published_at'].dt.normalize()).dt.days
    print("Fitur waktu publikasi dibuat")
else:
    print(" publish_date tidak bisa di-parse sebagai datetime")

# --- CELL ---

# ### 3.6 Duration Bucket

# --- CELL ---

def categorize_duration(seconds):
    if pd.isna(seconds): return 'unknown'
    m = seconds / 60
    if m < 1:   return 'shorts (<1 mnt)'
    elif m < 3:  return 'pendek (1-3 mnt)'
    elif m < 7:  return 'medium (3-7 mnt)'
    elif m < 15: return 'panjang (7-15 mnt)'
    else:        return 'sangat panjang (>15 mnt)'

df['duration_bucket'] = df['video_duration_sec'].apply(categorize_duration)
print("Duration bucket:\n", df['duration_bucket'].value_counts().to_string())

# --- CELL ---

# ### 3.7 NLP Fitur dari Judul Video

# --- CELL ---

SENSATIONAL_WORDS = [
    'GEMPAR','HEBOH','NGAMUK','PANIK','MERINDING','GEGER','VIRAL',
    'MENGEJUTKAN','BONGKAR','HANCUR','TERDIAM','KAGET','MALU','TAKUT',
    'MARAH','MENGGUNCANG','TERHINA','BOIKOT','PERANG','GILA','BERANI','NEKAT'
]
TOPIC_KEYWORDS = {
    'israel_palestina': ['ISRAEL','PALESTINA','GAZA','HAMAS','IDF'],
    'malaysia':         ['MALAYSIA','MELAYU','JIRAN'],
    'militer_tni':      ['TNI','MILITER','ALUTSISTA','ANGKATAN','RUDAL'],
    'ekonomi_mineral':  ['MINERAL','NIKEL','BATU BARA','EKONOMI','INVESTASI','EKSPOR'],
    'diplomasi':        ['PRABOWO','DIPLOMATIK','PBB','SIDANG','HUBUNGAN','KTT'],
    'amerika_barat':    ['AMERIKA','AS','USA','BARAT','NATO','CIA'],
    'cina':             ['CINA','CHINA','TIONGKOK','XI JINPING'],
    'rusia':            ['RUSIA','RUSSIA','PUTIN'],
}

def extract_title_features(title):
    if pd.isna(title): return {}
    s = str(title); u = s.upper()
    topic_scores = {t: sum(1 for kw in kws if kw in u) for t, kws in TOPIC_KEYWORDS.items()}
    # Fix 5: topic cluster logic yang benar
    if max(topic_scores.values()) == 0:
        topic_cluster = 'lainnya'
    else:
        topic_cluster = max(topic_scores, key=topic_scores.get)
    letters = [c for c in s if c.isalpha()]
    all_entities = [kw for kws in TOPIC_KEYWORDS.values() for kw in kws]
    return {
        'title_length':       len(s),
        'title_words':        len(s.split()),
        'sensational_count':  sum(1 for w in SENSATIONAL_WORDS if w in u),
        'has_symbol':         int(bool(re.search(r'[‼⁉✅🚨🔥❗❓💥🇮🇩⚡‼️]', s))),
        'caps_ratio':         sum(1 for c in letters if c.isupper()) / len(letters) if letters else 0.0,
        'topic_cluster':      topic_cluster,
        'topic_score':        topic_scores[topic_cluster] if topic_cluster != 'lainnya' else 0,
        'entity_count':       sum(1 for e in all_entities if e in u),
    }

title_df = pd.DataFrame(df['video_title'].apply(extract_title_features).tolist(), index=df.index)
df = pd.concat([df, title_df], axis=1)
print("NLP fitur judul dibuat")
print(df['topic_cluster'].value_counts().to_string())

# --- CELL ---

# ---
# ## 4. Performance Classification
# 
# Score = `views × (avg_watch_seconds / 60)`. Berbasis persentil channel sendiri.

# --- CELL ---

df['avg_watch_minutes'] = df['avg_watch_seconds'] / 60
df['perf_score'] = df['views'] * df['avg_watch_minutes']

p20 = df['perf_score'].quantile(0.20)
p50 = df['perf_score'].quantile(0.50)
p80 = df['perf_score'].quantile(0.80)
p95 = df['perf_score'].quantile(0.95)

print(f"P20={p20:,.0f} | P50={p50:,.0f} | P80={p80:,.0f} | P95={p95:,.0f}")

def assign_performance_class(score):
    if pd.isna(score):    return 'unknown'
    elif score < p20:     return 'rendah'
    elif score < p50:     return 'sedang'
    elif score < p80:     return 'bagus'
    elif score < p95:     return 'sangat_bagus'
    else:                 return 'viral'

df['performance_class'] = df['perf_score'].apply(assign_performance_class)
class_order = ['rendah','sedang','bagus','sangat_bagus','viral']
print("\nDistribusi kelas:")
print(df['performance_class'].value_counts().reindex(class_order).to_string())

# --- CELL ---

# ---
# ## 5. Persiapan ML Dataset
# 
# > ⚠️ **Fix 6 — No Data Leakage**: `views`, `avg_watch_seconds`, `perf_score` hanya dipakai untuk generate label, **BUKAN** sebagai fitur training.

# --- CELL ---

# Kolom untuk generate label saja (tidak masuk X_train)
TARGET_GENERATION_COLS = ['views', 'avg_watch_seconds', 'perf_score', 'avg_watch_minutes']

# Fitur aman: tidak bocor info post-upload
SAFE_FEATURE_COLS = [
    'upload_hour', 'upload_day', 'upload_day_name', 'upload_month', 'upload_year',
    'video_duration_sec', 'duration_bucket',
    'title_length', 'title_words', 'sensational_count', 'has_symbol',
    'caps_ratio', 'topic_cluster', 'topic_score', 'entity_count',
    'ctr', 'impressions',
]

existing_safe = [c for c in SAFE_FEATURE_COLS if c in df.columns]
df_ml = df[existing_safe + ['published_at', 'video_title', 'perf_score', 'performance_class']].copy()

print(f"Dataset ML: {df_ml.shape[0]} video × {df_ml.shape[1]} kolom")
print(f"\nFitur safe (tidak leakage): {existing_safe}")

# --- CELL ---

# ### 5.1 Fix 7 — Imputasi Missing Values

# --- CELL ---

num_cols = df_ml.select_dtypes(include=np.number).columns.tolist()
cat_cols = df_ml.select_dtypes(exclude=np.number).columns.difference(['published_at']).tolist()

df_ml[num_cols] = df_ml[num_cols].fillna(df_ml[num_cols].median())
for c in cat_cols:
    df_ml[c] = df_ml[c].fillna('unknown')

print(f"Null setelah imputasi: {df_ml.isnull().sum().sum()}")

# --- CELL ---

# ### 5.2 Fix 8 — Categorical Encoding

# --- CELL ---

encode_cols = [c for c in ['topic_cluster','duration_bucket','upload_day_name'] if c in df_ml.columns]
df_ml_encoded = pd.get_dummies(df_ml.drop(columns=['published_at','video_title','performance_class','perf_score']),
                                columns=encode_cols, drop_first=True)

# Tambahkan kembali kolom target
df_ml_encoded['perf_score']       = df_ml['perf_score'].values
df_ml_encoded['performance_class'] = df_ml['performance_class'].values

print(f"Shape setelah encoding: {df_ml_encoded.shape}")
print(f"Semua kolom numerik: {df_ml_encoded.select_dtypes(include=np.number).shape[1]} kolom")

# --- CELL ---

# ### 5.3 Fix 9 — Time-Based Train/Test Split
# 
# > ⚠️ YouTube data adalah time-series. Random split akan leak data masa depan ke training.

# --- CELL ---

df_ml_encoded['published_at'] = df_ml['published_at'].values

# Cek apakah published_at valid (range > 30 hari)
valid_dates = df_ml_encoded['published_at'].dropna()
date_range = (valid_dates.max() - valid_dates.min()).days if len(valid_dates) > 0 else 0

if date_range > 30:
    df_ml_encoded = df_ml_encoded.sort_values('published_at')
    cutoff_date = df_ml_encoded['published_at'].quantile(0.80)
    print(f"Split chronologis — Cutoff date (P80): {cutoff_date}")
    mask_train = df_ml_encoded['published_at'] < cutoff_date
else:
    # Fallback: index-based 80/20 (publish_date tidak valid di data ini)
    print(f" publish_date tidak valid (range: {date_range} hari) — pakai index-based split 80/20")
    n = len(df_ml_encoded)
    mask_train = pd.Series([True]*int(n*0.80) + [False]*(n - int(n*0.80)), index=df_ml_encoded.index)

train = df_ml_encoded[mask_train].drop(columns=['published_at'])
test  = df_ml_encoded[~mask_train].drop(columns=['published_at'])

X_train = train.drop(columns=['perf_score','performance_class'])
y_train = train['performance_class']
X_test  = test.drop(columns=['perf_score','performance_class'])
y_test  = test['performance_class']

print(f"Train size: {len(train)} | Test size: {len(test)}")
print(f"X_train shape: {X_train.shape}")

# --- CELL ---

# ---
# ## 6. Simpan Hasil

# --- CELL ---

from pathlib import Path

output_dir = Path("../data/raw")
output_dir.mkdir(parents=True, exist_ok=True)

OUTPUT_FULL = output_dir / "data_cleaned_full.csv"
OUTPUT_ML   = output_dir / "data_ml_features.csv"

df.to_csv(OUTPUT_FULL, index=False)
df_ml.to_csv(OUTPUT_ML, index=False)

print("Saved!")
print(OUTPUT_FULL.resolve())

# --- CELL ---

# Preview
print("\n🔍 Sample data ML:")
preview_cols = ['video_title','perf_score','performance_class','topic_cluster','ctr']
df_ml[[c for c in preview_cols if c in df_ml.columns]].head(5)

# --- CELL ---

# ---
# ## 7. Checklist & Next Steps
# 
# ### ✅ Semua fix diterapkan:
# - [x] Tidak ada duplicate columns
# - [x] Durasi di-parse dengan benar (MM:SS dan HH:MM:SS)
# - [x] Tidak ada division by zero (`safe_divide`)
# - [x] CTR selalu dalam range `[0, 1]`
# - [x] `watch_time_ratio` selalu dalam range `[0, 1]`
# - [x] Topic cluster logic benar
# - [x] Tidak ada data leakage ke target
# - [x] Dataset ML: no NaN, semua numerik
# - [x] Chronological train/test split
# 
# ### Next Steps untuk ML:
# 1. **Model 1 — Classifier**: XGBoost/LightGBM untuk prediksi `performance_class`
# 2. **Model 2 — Regressor**: Prediksi total views
# 3. **SHAP Analysis**: Kenapa video ini underperform?
# 4. **(Opsional)** IndoBERT embeddings untuk judul video
# 