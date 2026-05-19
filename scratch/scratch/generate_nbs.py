import json
import os

base_dir = "/run/media/rizqimaulidiyah/7542d4da-568c-4bbf-b867-1295fe534e4e/Capstone-project/Model-Prediksi-dan-Diagnosa-Penurunan-Views-YouTube-Berbasis-Machine-Learning/notebooks"

# Helper to create a basic notebook structure
def create_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.9"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

def markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [s + "\n" for s in source.split("\n")]
    }

def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [s + "\n" for s in source.split("\n")]
    }

# MODEL 1
m1_cells = [
    markdown_cell("# 🔮 Model 1: Time Series Forecasting (Prediksi Views ke Depan)\n**Tujuan:** Memproyeksikan estimasi *views* harian sebuah video untuk rentang waktu ke depan (H+1 hingga H+7) guna mengetahui potensi jangkauan konten."),
    markdown_cell("## 1. Import Library"),
    code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport xgboost as xgb\nfrom prophet import Prophet\nfrom sklearn.metrics import mean_squared_error, mean_absolute_error"),
    markdown_cell("## 2. Load Data\nData yang digunakan adalah fitur yang sudah digabungkan."),
    code_cell("# df = pd.read_csv('../data/processed/final_features.csv')\n# df['publish_date'] = pd.to_datetime(df['publish_date'])\n# df = df.sort_values('publish_date')"),
    markdown_cell("## 3. Data Splitting (Time-based)\n**Aturan:** Tidak boleh menggunakan random split. Wajib Time-based Splitting."),
    code_cell("# cutoff_date = '2025-01-01'\n# train = df[df['publish_date'] < cutoff_date]\n# test  = df[df['publish_date'] >= cutoff_date]"),
    markdown_cell("## 4. Modeling: XGBoost + Prophet\nFitur Input Wajib:\n- Lag Features: `TS1_Views`, `TS2_Views`, `TS3_Views`\n- Time Decay: `days_since_upload` dan `decay_weight`\n- Rolling Stats: `rolling_avg_views_7d`"),
    code_cell("# xgb_model = xgb.XGBRegressor(objective='reg:squarederror')\n# # + Prophet implementation"),
    markdown_cell("## 5. Evaluasi (RMSE & MAE)"),
    code_cell("# rmse = np.sqrt(mean_squared_error(y_test, y_pred))\n# mae = mean_absolute_error(y_test, y_pred)")
]

with open(os.path.join(base_dir, "08_model_1_time_series.ipynb"), "w") as f:
    json.dump(create_notebook(m1_cells), f, indent=2)

# MODEL 2
m2_cells = [
    markdown_cell("# 📉 Model 2: Decline Detection (Deteksi Penurunan & Akar Masalah)\n**Tujuan:** Mendeteksi kapan sebuah video mulai 'mati' atau anjlok secara tidak wajar, lalu secara otomatis membedah alasan penurunannya (Root Cause Analysis)."),
    markdown_cell("## 1. Import Library"),
    code_cell("import pandas as pd\nimport numpy as np\nfrom sklearn.ensemble import IsolationForest\nimport shap\nimport warnings\nwarnings.filterwarnings('ignore')"),
    markdown_cell("## 2. Load Prediksi dari Model 1\nModel ini dieksekusi JIKA `daily_growth_rate` negatif > threshold (misal drop > 20%)."),
    code_cell("# df_forecast = pd.read_csv('../data/processed/forecast_results.csv')\n# declining_videos = df_forecast[df_forecast['daily_growth_rate'] < -0.2]"),
    markdown_cell("## 3. Anomaly Detection (Isolation Forest)\nMenangkap kejanggalan harian yang sangat drastis."),
    code_cell("# iso_forest = IsolationForest(contamination=0.1, random_state=42)\n# anomalies = iso_forest.fit_predict(features)"),
    markdown_cell("## 4. Root Cause Analysis dengan SHAP\nFitur Input Wajib: `growth_acceleration`, `ctr_normalized`, `engagement_score`"),
    code_cell("# explainer = shap.TreeExplainer(xgb_model_from_model1)\n# shap_values = explainer.shap_values(X_declining)"),
    markdown_cell("## 5. Ekstraksi Kesimpulan (Teks)\nHarus menghasilkan kesimpulan teks diagnostik otomatis."),
    code_cell("def generate_diagnosis_report(video_id, top_causes):\n    return f\"Alert Penurunan! Views diprediksi drop. Top penyebab: {', '.join(top_causes)}.\"\n# print(generate_diagnosis_report('VID123', ['CTR turun', 'Komentar berkurang']))")
]

with open(os.path.join(base_dir, "09_model_2_decline_detection.ipynb"), "w") as f:
    json.dump(create_notebook(m2_cells), f, indent=2)

# MODEL 3
m3_cells = [
    markdown_cell("# 📊 Model 3: Revenue Forecasting (Prediksi Pendapatan)\n**Tujuan:** Memproyeksikan pendapatan finansial (IDR) di masa depan berdasarkan tren performa views dan monetisasi."),
    markdown_cell("## 1. Import Library"),
    code_cell("import pandas as pd\nimport numpy as np\nfrom sklearn.ensemble import RandomForestRegressor\nfrom sklearn.linear_model import ElasticNet\nfrom sklearn.metrics import mean_absolute_error"),
    markdown_cell("## 2. Data Filtering\n**Aturan:** Hanya eksekusi pada `monetization_rate = 1` atau `estimated_revenue_idr > 0`."),
    code_cell("# df = pd.read_csv('../data/processed/final_features.csv')\n# df_monetized = df[(df['monetization_rate'] == 1) | (df['estimated_revenue_idr'] > 0)]"),
    markdown_cell("## 3. Modeling (Random Forest Regressor / ElasticNet)\nFitur Input Wajib:\n- Prediksi Views (Output Model 1)\n- `revenue_per_view`\n- `monetization_rate`\n- `ctr_impression_score`"),
    code_cell("# rf_model = RandomForestRegressor(random_state=42)\n# rf_model.fit(X_train, y_train)"),
    markdown_cell("## 4. Evaluasi (WAPE)\nWAPE (Weighted Absolute Percentage Error) agar model lebih fokus mengevaluasi akurasi pada video dengan *revenue* paling tinggi."),
    code_cell("def calculate_wape(y_true, y_pred):\n    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))\n# wape_score = calculate_wape(y_test, y_pred)")
]

with open(os.path.join(base_dir, "10_model_3_revenue_forecast.ipynb"), "w") as f:
    json.dump(create_notebook(m3_cells), f, indent=2)

print("Berhasil membuat 3 notebook ML!")
