# # 🎬 Data Preparation & Analisis Ekstrem (Outlier Handling)
# **Penanggung Jawab:** Wildan
# 
# Notebook ini didedikasikan untuk melakukan pembersihan data (*Data Cleaning*) secara mendalam dan menangani nilai ekstrem (*Outliers*). Berbeda dengan penanganan data konvensional, di domain YouTube, **Outlier bukanlah error, melainkan 'Video Viral'**. Oleh karena itu, notebook ini mengimplementasikan logika bisnis khusus untuk menyelamatkan data viral menggunakan **IQR Flagging** dan **Log Transformation**.

# --- CELL ---

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import matplotlib as plt
import seaborn as sns

# --- CELL ---

!pip3 install seaborn

# --- CELL ---

# ### 1. Memuat Dataset

# --- CELL ---

df = pd.read_csv('../../data/Data_Merged_Fix.csv')
print(f"Ukuran data awal: {df.shape}")
df.head(3)

# --- CELL ---

# ### 2. Standardisasi Nama Kolom
# Mengubah semua kolom menjadi *snake_case* (huruf kecil semua, spasi diganti underscore, membuang karakter khusus seperti % dan kurung).

# --- CELL ---

import re

def clean_col_name(col):
    col = col.lower()
    col = col.replace(' (%)', '_pct')
    col = col.replace(' (idr)', '_idr')
    col = col.replace(' (hours)', '_hours')
    col = re.sub(r'[^a-z0-9]+', '_', col)
    return col.strip('_')

df.columns = [clean_col_name(c) for c in df.columns]
print("Kolom setelah distandarisasi:")
print(df.columns.tolist()[:10])

# --- CELL ---

# ### 3. Penyesuaian Tipe Data
# Mengubah format `average_view_duration` (String "JJ:MM:DD") menjadi numerik (Total Detik). Serta memastikan `publish_date_wib` menjadi tipe datetime.

# --- CELL ---

def time_to_seconds(time_str):
    if pd.isna(time_str):
        return 0
    try:
        parts = str(time_str).split(':')
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        return int(time_str)
    except:
        return 0

if 'average_view_duration' in df.columns:
    df['average_view_duration_sec'] = df['average_view_duration'].apply(time_to_seconds)
    df = df.drop(columns=['average_view_duration'])

if 'publish_date_wib' in df.columns:
    df['publish_date_wib'] = pd.to_datetime(df['publish_date_wib'], errors='coerce')

print(df[['average_view_duration_sec', 'publish_date_wib']].head())

# --- CELL ---

# ### 4. Penanganan Missing Values dan Redundansi
# 1. Menghapus indikator yang 100% kosong (seperti click teasers, dll).
# 2. Imputasi fitur Time Series (TS1_Views, dst) dengan nilai 0 bila ada yang kosong.
# 3. Menghapus kolom `content` yang redundan dengan `video_id`.

# --- CELL ---

# ### 🧹 Penanganan Missing Values (Logika Bisnis)
# Langkah ini tidak menggunakan imputasi *Median* secara buta. Alih-alih:
# 1. **Drop Kolom:** Kolom yang lebih dari 95% isinya kosong (NaN) akan dihapus karena tidak memberikan bobot informasi bagi Machine Learning.
# 2. **Imputasi Angka 0 pada Time Series:** Kolom *views* harian (, , dll) yang kosong diisi dengan angka . Logika bisnisnya: Jika data tidak terekam di hari pertama, kemungkinan besar video tersebut memang belum mendapatkan *views* (berlaku untuk *channel* kecil/menengah).

# --- CELL ---

# Drop redundansi
if 'content' in df.columns:
    df = df.drop(columns=['content'])

# Manual drop columns
cols_to_drop_manual = ['stayed_to_watch_pct', 'average_views_per_viewer', 'unique_viewers', 'new_viewers', 'returning_viewers', 'casual_viewers', 'regular_viewers', 'transaction_revenue_idr', 'transactions', 'revenue_per_transaction_idr', 'estimated_doubleclick_revenue_idr', 'playlist_watch_time_hours', 'views_from_playlist', 'views_per_playlist_start', 'hours_streamed', 'reminders_set', 'chat_messages', 'reactions', 'post_subscribers', 'community_clip_views', 'watch_time_from_community_clips_hours', 'card_clicks', 'cards_shown', 'card_teaser_clicks', 'card_teasers_shown','rubies']
df = df.drop(columns=[c for c in cols_to_drop_manual if c in df.columns])


# Drop kolom dengan missing > 95%
threshold = 0.95 * len(df)
df = df.dropna(axis=1, thresh=len(df) - threshold)

# Imputasi fitur Time Series
ts_cols = ['ts1_views', 'ts2_views', 'ts3_views', 'ts4_views']
for col in ts_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0)

print(f"Ukuran data setelah dibersihkan: {df.shape}")

# --- CELL ---

df.info()

# --- CELL ---

# ### 5. Export Data Bersih
# Menyimpan dataset yang sudah rapi agar siap digunakan pada tahap _Feature Engineering_ atau _Modelling_ selanjutnya.

# --- CELL ---

df.describe()

# --- CELL ---

# df.isnull().sum()

# --- CELL ---

df.isnull().sum()

# --- CELL ---

# * Menangani missing value
# 

# --- CELL ---

# ### 🧹 Penanganan Missing Values (Logika Bisnis)
# Langkah ini tidak menggunakan imputasi *Median* secara buta. Alih-alih:
# 1. **Drop Kolom:** Kolom yang lebih dari 95% isinya kosong (NaN) akan dihapus karena tidak memberikan bobot informasi bagi Machine Learning.
# 2. **Imputasi Angka 0 pada Time Series:** Kolom *views* harian (, , dll) yang kosong diisi dengan angka . Logika bisnisnya: Jika data tidak terekam di hari pertama, kemungkinan besar video tersebut memang belum mendapatkan *views* (berlaku untuk *channel* kecil/menengah).

# --- CELL ---

threshold = 0.95 * len(df)
df = df.dropna(axis=1, thresh=len(df) - threshold)
df.isnull().sum()


# --- CELL ---

# 

# --- CELL ---

df.describe(include="all")

# --- CELL ---

df.info()

# --- CELL ---

# Memeriksa jumlah nilai yang hilang di setiap kolom
missing_values = df.isnull().sum()

# Memisahkan kolom berdasarkan batas toleransi missing value (pada kasus ini disesuaikan menjadi 800)
less = missing_values[missing_values < 800].index
over = missing_values[missing_values >= 800].index

# Mengisi nilai yang hilang dengan median untuk kolom numerik
numeric_features = df[less].select_dtypes(include=['number']).columns
df[numeric_features] = df[numeric_features].fillna(df[numeric_features].median())

# Menghapus kolom dengan terlalu banyak nilai yang hilang
# Perubahan nama DataFrame dilakukan supaya data asli tidak berubah
df_cleaned = df.drop(columns=over)

# Terakhir, lakukan pemeriksaan terhadap data yang sudah melewati tahapan verifikasi missing value
missing_values_after = df_cleaned.isnull().sum()
print(missing_values_after[missing_values_after > 0])

# --- CELL ---

df.isnull().sum()

# --- CELL ---

df.head()

# --- CELL ---

# * Konversi date time

# --- CELL ---

import pandas as pd
# Mengonversi kolom 'publish_time_wib' menjadi tipe data datetime
df['publish_time_wib'] = pd.to_datetime(df['publish_time_wib'])

# Memeriksa kembali tipe data spesifik pada kolom tersebut untuk memastikan perubahannya
print(df[['publish_time_wib']].info())

# --- CELL ---

# * Penanganan Duplikat Data

# --- CELL ---

df.duplicated().sum()

# --- CELL ---

# * Analisis data Outlier berupa
# - views, watch_time_hours, engaged_views

# --- CELL ---

# ### 📊 Analisis Outlier (Identifikasi Video Viral)
# Boxplot di bawah ini digunakan untuk melihat sebaran data. Kita akan melihat banyak titik di luar batas atas (titik-titik *Outlier*). 
# 
# **Penting:** Dalam proyek ini, kita **TIDAK AKAN MENGHAPUS** titik-titik tersebut. Menghapus *outlier* di data YouTube sama dengan menghapus video-video tersukses kita dari memori model Machine Learning.

# --- CELL ---

import seaborn as sns
import matplotlib.pyplot as plt

# List fitur utama yang akan dianalisis
features_to_check = ['views', 'watch_time_hours', 'engaged_views']

for feature in features_to_check:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[feature])
    plt.title(f'Deteksi Outlier pada {feature}')
    plt.show()

# --- CELL ---

# * Kesimpulan:
# Data  tidak berdistribusi normal. Titik-titik kemungkinan besar adalah video-video "viral" yang performanya jauh melampaui rata-rata konten lainnya.

# --- CELL ---

# * Metrik pendapatan (revenue)

# --- CELL ---

# ### 📊 Analisis Outlier (Identifikasi Video Viral)
# Boxplot di bawah ini digunakan untuk melihat sebaran data. Kita akan melihat banyak titik di luar batas atas (titik-titik *Outlier*). 
# 
# **Penting:** Dalam proyek ini, kita **TIDAK AKAN MENGHAPUS** titik-titik tersebut. Menghapus *outlier* di data YouTube sama dengan menghapus video-video tersukses kita dari memori model Machine Learning.

# --- CELL ---


# List fitur utama yang akan dianalisis
features_to_check = ['estimated_revenue_idr', 'estimated_adsense_revenue_idr']

for feature in features_to_check:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[feature])
    plt.title(f'Deteksi Outlier pada {feature}')
    plt.show()

# --- CELL ---

# 

# --- CELL ---

# ### 📊 Analisis Outlier (Identifikasi Video Viral)
# Boxplot di bawah ini digunakan untuk melihat sebaran data. Kita akan melihat banyak titik di luar batas atas (titik-titik *Outlier*). 
# 
# **Penting:** Dalam proyek ini, kita **TIDAK AKAN MENGHAPUS** titik-titik tersebut. Menghapus *outlier* di data YouTube sama dengan menghapus video-video tersukses kita dari memori model Machine Learning.

# --- CELL ---



features_to_check = ['cpm_idr', 'rpm_idr']

for feature in features_to_check:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[feature])
    plt.title(f'Deteksi Outlier pada {feature}')
    plt.show()

# --- CELL ---

# * Retensi danEfisiensi

# --- CELL ---

# ### 📊 Analisis Outlier (Identifikasi Video Viral)
# Boxplot di bawah ini digunakan untuk melihat sebaran data. Kita akan melihat banyak titik di luar batas atas (titik-titik *Outlier*). 
# 
# **Penting:** Dalam proyek ini, kita **TIDAK AKAN MENGHAPUS** titik-titik tersebut. Menghapus *outlier* di data YouTube sama dengan menghapus video-video tersukses kita dari memori model Machine Learning.

# --- CELL ---

features_to_check = ['average_percentage_viewed_pct', 'impressions_click_through_rate_pct']

for feature in features_to_check:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[feature])
    plt.title(f'Deteksi Outlier pada {feature}')
    plt.show()

# --- CELL ---



# --- CELL ---

# 
# Rangkuman dari hasil analisis grafik outlier pada notebook tersebut:
# 
# 1. Outlier pada Metrik Performa (Views, Watch Time Hours, Engaged Views)
# 
# Grafik: Menunjukkan adanya banyak titik data yang berada jauh di atas batas atas (upper whisker) dari boxplot.
# Kesimpulan: Distribusi data tidak normal dan memiliki rentang yang sangat lebar ke arah atas (right-skewed). Titik-titik pencilan (outlier) ini kemungkinan besar merepresentasikan video-video "viral" yang performa dan jumlah penontonnya jauh melampaui rata-rata konten video lainnya.
# 
# 2. Outlier pada Metrik Pendapatan (Estimated Revenue IDR, Estimated AdSense Revenue IDR)
# 
# Grafik: Mirip dengan metrik performa, terdapat titik-titik ekstrem yang nilainya jauh lebih besar dari mayoritas data.
# Kesimpulan: Ada beberapa video spesifik yang menghasilkan pendapatan yang secara signifikan jauh lebih tinggi dibandingkan dengan video lainnya.
# 
# 3. Outlier pada Metrik Retensi dan Efisiensi (Average Percentage Viewed, Impressions Click-Through Rate / CTR)
# 
# Grafik: Walaupun boxplot menunjukkan data lebih banyak terpusat (terkonsentrasi) pada rentang nilai tertentu (karena berupa persentase), tetap ditemukan adanya nilai ekstrem.
# Maka Nilai-nilai ekstrem ini mengindikasikan adanya pola retensi penonton yang spesifik atau rasio klik-tayang (CTR) yang sangat tinggi/sangat rendah pada beberapa video tertentu dibandingkan rata-rata umum.
# 

# --- CELL ---

# * Dari hasil yang telah saya diskusikan dengan GEMINI AI, maka kesimpulan yang saya ambil saya akan menggunakan IQR sebagai flagging, artinya data yang video viral whic is diluar garis akan saya tandai sebagai is viral dan saya akan padukan dengan algoritma robust agar membuat prediksi semakin akurat.

# --- CELL ---

# * IQR Flagging

# --- CELL ---

# ### 🚀 Feature Engineering Lanjutan: IQR Flagging ()
# Karena kita tidak menghapus video viral, kita harus memberi 'tanda' agar algoritma Machine Learning tahu bahwa video ini spesial. 
# Kita menggunakan batas statistik matematis **Interquartile Range (IQR)**. Jika *views* sebuah video melewati batas , maka video tersebut diberi bendera .

# --- CELL ---

# 1. Tentukan kolom target yang ingin dianalisis (contoh: 'views' atau 'engaged_views')
kolom_target = 'views'
# 2. Hitung Q1 (Kuartil 1) dan Q3 (Kuartil 3)
Q1 = df[kolom_target].quantile(0.25)
Q3 = df[kolom_target].quantile(0.75)
# 3. Hitung rentang IQR
IQR = Q3 - Q1
# 4. Tentukan Batas Atas (Upper Bound)
# Formula standar untuk batas atas pencilan (outlier) adalah Q3 + (1.5 * IQR)
batas_atas = Q3 + 1.5 * IQR
print(f"Batas atas (Viral Threshold) untuk {kolom_target}: {batas_atas}")
# 5. Lakukan Flagging dengan membuat kolom baru (1 jika viral, 0 jika normal)
df['is_viral'] = np.where(df[kolom_target] > batas_atas, 1, 0)
# Cek sebaran video viral vs normal
print(df['is_viral'].value_counts())

# --- CELL ---

df.info()

# --- CELL ---

# ### 📉 Normalisasi Data: Log Transformation (`np.log1p`)
# Video viral memiliki rentang angka yang sangat jauh (jutaan) dibandingkan video normal (ribuan). Hal ini membuat algoritma (seperti Regresi) menjadi bias. 
# Untuk menjinakkannya, kita menerapkan **Logaritma Natural (Log1p)**. Logika ini akan merapatkan skala angka ekstrem menjadi skala belasan, sehingga model bisa mempelajarinya dengan stabil tanpa kehilangan informasi esensial.

# --- CELL ---

# 1. Daftar kolom yang terdeteksi memiliki outlier ekstrem dari box plot sebelumnya
outlier_features = ['watch_time_hours', 'engaged_views']

# 2. Lakukan Transformasi Logaritmik (log1p)
for feature in outlier_features:
    df[f'{feature}_log'] = np.log1p(df[feature])

# 3. Visualisasi Perbandingan Sebelum vs Sesudah
for feature in outlier_features:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot Sebelum (Data Asli - Skewed)
    sns.histplot(df[feature], bins=30, ax=axes[0], kde=True, color='skyblue')
    axes[0].set_title(f'Distribusi {feature} Asli')
    
    # Plot Sesudah (Data Log - Lebih Normal/Terpusat)
    sns.histplot(df[f'{feature}_log'], bins=30, ax=axes[1], kde=True, color='salmon')
    axes[1].set_title(f'Distribusi {feature} Setelah Log Transform')
    
    plt.tight_layout()
    plt.show()

# 4. Verifikasi perubahan distribusi secara numerik (Opsional)
# Ini membantu memastikan distribusi data menjadi lebih merata dan terpusat [cite: 155]
print(df[[f'{f}_log' for f in outlier_features]].describe())

# --- CELL ---



# --- CELL ---

# ## CATATAN
# * Nanti di tahap modeling...
# from sklearn.ensemble import RandomForestRegressor
# *  Fitur yang digunakan (X) termasuk 'is_viral' untuk memberi konteks
# X = df[['duration', 'average_percentage_viewed_pct', 'is_viral', ...]] 
# * Target yang diprediksi (y) BISA menggunakan data yang sudah di-log
# y = df['views_log'] 
# model = RandomForestRegressor(random_state=42)
# model.fit(X, y)

# --- CELL ---

# 

# --- CELL ---

# ### 📊 Analisis Outlier (Identifikasi Video Viral)
# Boxplot di bawah ini digunakan untuk melihat sebaran data. Kita akan melihat banyak titik di luar batas atas (titik-titik *Outlier*). 
# 
# **Penting:** Dalam proyek ini, kita **TIDAK AKAN MENGHAPUS** titik-titik tersebut. Menghapus *outlier* di data YouTube sama dengan menghapus video-video tersukses kita dari memori model Machine Learning.

# --- CELL ---


fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# Boxplot sebelum di-log (Outlier terlihat sangat jauh)
sns.boxplot(x=df['views'], ax=axes[0])
axes[0].set_title('Boxplot Views Asli (Outlier Ekstrem)')
# Boxplot sesudah di-log (Outlier sudah "dijinakkan")
sns.boxplot(x=df['views_log'], ax=axes[1], color='orange')
axes[1].set_title('Boxplot Views Log (Outlier Jinak)')
plt.tight_layout()
plt.show()

# --- CELL ---

df.info()

# --- CELL ---

df.head()
df.isnull().sum()

# --- CELL ---

# ### 💾 Export Master Data
# Data telah berhasil dibersihkan, di-*flag*, dan di-*log*. Dataset ini sekarang adalah **'Master Data'** yang secara matematis sangat aman dan siap digunakan oleh tim untuk tahap *Feature Engineering* selanjutnya.

# --- CELL ---

# Mundur dua folder dari notebooks/versi_wildan/ ke folder utama, lalu masuk ke folder data
path_simpan = '../../data/Data_Cleaned_Wildan.csv'
df.to_csv(path_simpan, index=False)
print(f"Data berhasil diekspor dan disimpan di: {path_simpan}")
print(f"Total baris dan kolom final: {df.shape}")


# --- CELL ---

# ### 🚀 Feature Engineering Lanjutan: IQR Flagging ()
# Karena kita tidak menghapus video viral, kita harus memberi 'tanda' agar algoritma Machine Learning tahu bahwa video ini spesial. 
# Kita menggunakan batas statistik matematis **Interquartile Range (IQR)**. Jika *views* sebuah video melewati batas , maka video tersebut diberi bendera .

# --- CELL ---

# ### 📉 Normalisasi Data: Log Transformation (`np.log1p`)
# Video viral memiliki rentang angka yang sangat jauh (jutaan) dibandingkan video normal (ribuan). Hal ini membuat algoritma (seperti Regresi) menjadi bias. 
# Untuk menjinakkannya, kita menerapkan **Logaritma Natural (Log1p)**. Logika ini akan merapatkan skala angka ekstrem menjadi skala belasan, sehingga model bisa mempelajarinya dengan stabil tanpa kehilangan informasi esensial.

# --- CELL ---

# Berikut adalah rangkuman dari semua hal krusial yang telah Anda selesaikan di data_prep_wildan.ipynb. Anda bisa menyalin teks ini ke dalam sel Markdown di akhir notebook sebagai dokumentasi proyek Anda:
# 
# Tahap 1: Pengumpulan & Standardisasi Data
# Loading Data: Membaca dataset mentah hasil gabungan (Data_Merged_Fix.csv).
# Pembersihan Nama Kolom: Menyeragamkan semua nama kolom menjadi format snake_case (huruf kecil semua dan spasi diganti underscore) agar mudah dipanggil dalam kode Python.
# Tahap 2: Pembersihan Data (Data Cleaning)
# Menghapus Kolom Tidak Penting: Membuang kolom-kolom yang redundan atau tidak relevan dengan tujuan model (seperti content dan beberapa indikator revenue yang tidak terpakai).
# Menangani Missing Values:
# Menghapus kolom yang memiliki lebih dari 95% data kosong (NaN).
# Mengisi (impute) nilai kosong pada kolom Time Series (seperti ts1_views, ts2_views, dll) dengan angka 0.
# Tahap 3: Transformasi Tipe Data
# Konversi Waktu: Mengubah format average_view_duration yang awalnya berformat "JJ:MM:DD" menjadi total detik numerik (average_view_duration_sec).
# Konversi Tanggal: Memastikan kolom publish_date_wib terbaca dengan benar sebagai format datetime.
# Tahap 4: Analisis Outlier (Exploratory Data Analysis)
# Melakukan analisis visual menggunakan Boxplot pada metrik-metrik kunci (views, watch_time_hours, revenue, CTR, dll).
# Temuan Utama: Ditemukan bahwa data memiliki distribusi right-skewed (miring ke kanan) dengan banyaknya outlier ekstrem di sisi atas. Outlier ini tidak dihapus karena merepresentasikan real-case dari video-video "viral".
# Tahap 5: Penanganan Outlier (Feature Engineering & Robust Transformation)
# IQR Flagging: Menerapkan metode Interquartile Range (IQR) bukan untuk menghapus data, melainkan untuk membuat fitur baru bernama is_viral (1 = viral, 0 = normal). Ini memberikan "konteks" khusus bagi Machine Learning.
# Log Transformation: Menerapkan np.log1p pada kolom metrik yang ekstrem (seperti views_log, watch_time_hours_log, engaged_views_log) untuk menormalkan distribusi data secara matematis agar algoritma model tidak bias atau error.
# Status Saat Ini: Dataset telah bersih, memiliki fitur tambahan yang sangat representatif untuk kondisi dunia nyata (real-case), dan sudah berhasil diekspor. Data kini 100% siap untuk digunakan pada tahap Feature Engineering (Lanjutan) atau langsung masuk ke Model Training!

# --- CELL ---

