import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Data Preparation & Analisis - Versi Wildan\n",
    "\n",
    "Notebook ini dibuat untuk membersihkan dan menyiapkan data `Data_Merged_Fix.csv`. Berdasarkan analisis sebelumnya, ada beberapa hal yang difokuskan pada tahap ini:\n",
    "1. **Standardisasi Nama Kolom** (mengubah ke `snake_case`).\n",
    "2. **Penyesuaian Tipe Data** (mengubah durasi menjadi detik, dan mengatur tipe *datetime*).\n",
    "3. **Penanganan Missing Values** (menghapus kolom kosong, imputasi data penting).\n",
    "4. **Menghapus Redundansi Data** (menghapus duplikasi ID Video)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 1. Memuat Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.pd.read_csv('../../data/Data_Merged_Fix.csv')\n",
    "print(f\"Ukuran data awal: {df.shape}\")\n",
    "df.head(3)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2. Standardisasi Nama Kolom\n",
    "Mengubah semua kolom menjadi *snake_case* (huruf kecil semua, spasi diganti underscore, membuang karakter khusus seperti % dan kurung)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import re\n",
    "\n",
    "def clean_col_name(col):\n",
    "    col = col.lower()\n",
    "    col = col.replace(' (%)', '_pct')\n",
    "    col = col.replace(' (idr)', '_idr')\n",
    "    col = col.replace(' (hours)', '_hours')\n",
    "    col = re.sub(r'[^a-z0-9]+', '_', col)\n",
    "    return col.strip('_')\n",
    "\n",
    "df.columns = [clean_col_name(c) for c in df.columns]\n",
    "print(\"Kolom setelah distandarisasi:\")\n",
    "print(df.columns.tolist()[:10])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3. Penyesuaian Tipe Data\n",
    "Mengubah format `average_view_duration` (String \"JJ:MM:DD\") menjadi numerik (Total Detik). Serta memastikan `publish_date_wib` menjadi tipe datetime."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def time_to_seconds(time_str):\n",
    "    if pd.isna(time_str):\n",
    "        return 0\n",
    "    try:\n",
    "        parts = str(time_str).split(':')\n",
    "        if len(parts) == 3:\n",
    "            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])\n",
    "        elif len(parts) == 2:\n",
    "            return int(parts[0]) * 60 + int(parts[1])\n",
    "        return int(time_str)\n",
    "    except:\n",
    "        return 0\n",
    "\n",
    "if 'average_view_duration' in df.columns:\n",
    "    df['average_view_duration_sec'] = df['average_view_duration'].apply(time_to_seconds)\n",
    "    df = df.drop(columns=['average_view_duration'])\n",
    "\n",
    "if 'publish_date_wib' in df.columns:\n",
    "    df['publish_date_wib'] = pd.to_datetime(df['publish_date_wib'], errors='coerce')\n",
    "\n",
    "print(df[['average_view_duration_sec', 'publish_date_wib']].head())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4. Penanganan Missing Values dan Redundansi\n",
    "1. Menghapus indikator yang 100% kosong (seperti click teasers, dll).\n",
    "2. Imputasi fitur Time Series (TS1_Views, dst) dengan nilai 0 bila ada yang kosong.\n",
    "3. Menghapus kolom `content` yang redundan dengan `video_id`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Drop redundansi\n",
    "if 'content' in df.columns:\n",
    "    df = df.drop(columns=['content'])\n",
    "\n",
    "# Drop kolom dengan missing > 95%\n",
    "threshold = 0.95 * len(df)\n",
    "df = df.dropna(axis=1, thresh=len(df) - threshold)\n",
    "\n",
    "# Imputasi fitur Time Series\n",
    "ts_cols = ['ts1_views', 'ts2_views', 'ts3_views', 'ts4_views']\n",
    "for col in ts_cols:\n",
    "    if col in df.columns:\n",
    "        df[col] = df[col].fillna(0)\n",
    "\n",
    "print(f\"Ukuran data setelah dibersihkan: {df.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 5. Export Data Bersih\n",
    "Menyimpan dataset yang sudah rapi agar siap digunakan pada tahap _Feature Engineering_ atau _Modelling_ selanjutnya."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.to_csv('../../data/Data_Cleaned_Wildan.csv', index=False)\n",
    "print(\"Data berhasil disimpan ke 'data/Data_Cleaned_Wildan.csv'\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open('notebooks/versi_wildan/data_prep_wildan.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)
print("Notebook created successfully.")
