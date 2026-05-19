import nbformat as nbf

nb = nbf.v4.new_notebook()

nb['cells'] = [
    nbf.v4.new_markdown_cell("# Model 2: Decline Detection (Deteksi Penurunan & Akar Masalah)\n**Tujuan:** Mendeteksi kapan sebuah video mulai 'mati' atau anjlok secara tidak wajar, lalu secara otomatis membedah alasan penurunannya (Root Cause Analysis)."),
    
    nbf.v4.new_markdown_cell("## 1. Import Library"),
    nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import shap
import warnings
import joblib
warnings.filterwarnings('ignore')"""),
    
    nbf.v4.new_markdown_cell("## 2. Load Prediksi dari Model 1\nModel ini dieksekusi JIKA `daily_growth_rate` negatif > threshold (misal drop > 20%)."),
    nbf.v4.new_code_cell("""df_forecast = pd.read_csv('../../data/processed/forecast_results.csv')
declining_videos = df_forecast[df_forecast['daily_growth_rate'] < -0.2].copy()

if declining_videos.empty:
    print("Tidak ada video yang mengalami penurunan > 20%.")
else:
    print(f"Ditemukan {len(declining_videos)} video yang mengalami penurunan signifikan.")
"""),
    
    nbf.v4.new_markdown_cell("## 3. Anomaly Detection (Isolation Forest)\nMenangkap kejanggalan harian yang sangat drastis."),
    nbf.v4.new_code_cell("""if not declining_videos.empty:
    # Kita menggunakan fitur diagnostik untuk Isolation Forest
    diag_features = ['growth_acceleration', 'ctr_normalized', 'engagement_score']
    
    # Fill NA jika ada
    declining_videos[diag_features] = declining_videos[diag_features].fillna(0)
    
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    declining_videos['anomaly'] = iso_forest.fit_predict(declining_videos[diag_features])
    
    print("Jumlah anomali terdeteksi (nilai -1):", (declining_videos['anomaly'] == -1).sum())
"""),
    
    nbf.v4.new_markdown_cell("## 4. Root Cause Analysis dengan SHAP\nFitur Input Wajib: `growth_acceleration`, `ctr_normalized`, `engagement_score`"),
    nbf.v4.new_code_cell("""if not declining_videos.empty:
    # Load Model 1
    xgb_model_from_model1 = joblib.load('../../data/processed/xgb_model.pkl')
    
    # Fitur yang dipakai oleh Model 1
    model1_features = ['ts1_views', 'ts2_views', 'ts3_views', 'days_since_upload', 'decay_weight', 'rolling_avg_views_7d']
    X_declining = declining_videos[model1_features].fillna(0)
    
    explainer = shap.TreeExplainer(xgb_model_from_model1)
    shap_values = explainer.shap_values(X_declining)
    
    # Kita bisa extract top causes dari fitur model 1 atau diag_features. 
    # Karena SHAP diekstrak dari model 1, maka fiturnya adalah model1_features.
    
    # Rata-rata absolute SHAP values untuk declining videos
    feature_importance = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({'feature': model1_features, 'importance': feature_importance})
    importance_df = importance_df.sort_values(by='importance', ascending=False)
    
    print("Top Fitur Penentu (Global SHAP for Declining Videos):")
    print(importance_df.head(3))
"""),
    
    nbf.v4.new_markdown_cell("## 5. Ekstraksi Kesimpulan (Teks)\nHarus menghasilkan kesimpulan teks diagnostik otomatis."),
    nbf.v4.new_code_cell("""def generate_diagnosis_report(video_row, shap_vals, feature_names):
    # Dapatkan top 3 fitur yang memiliki impact negatif paling besar (menurunkan prediksi)
    # SHAP value negatif berarti menurunkan prediksi views
    top_causes_idx = np.argsort(shap_vals)[:3]
    top_causes = [feature_names[i] for i in top_causes_idx if shap_vals[i] < 0]
    
    if not top_causes:
        top_causes = ["Faktor Eksternal/Lainnya"]
        
    # Map ke deskripsi yang lebih ramah
    mapping = {
        'decay_weight': 'Usia video melewati batas viral (Time Decay)',
        'ts1_views': 'Momentum views sebelumnya menurun',
        'rolling_avg_views_7d': 'Rata-rata performa mingguan turun',
        'days_since_upload': 'Usia video bertambah lama',
        'ts2_views': 'Momentum views H-2 menurun',
        'ts3_views': 'Momentum views H-3 menurun'
    }
    top_causes_mapped = [mapping.get(c, c) for c in top_causes]
    
    return f"Alert Penurunan! Views diprediksi drop {abs(video_row['daily_growth_rate'])*100:.1f}% hari ini. Top penyebab: {', '.join(top_causes_mapped)}."

if not declining_videos.empty:
    for i in range(min(3, len(declining_videos))):
        vid_row = declining_videos.iloc[i]
        vid_shap = shap_values[i]
        report = generate_diagnosis_report(vid_row, vid_shap, model1_features)
        print(f"Video {vid_row['video_id']}:")
        print(report)
        print("-" * 50)
""")
]

with open('notebooks/modelling/09_model_2_decline_detection.ipynb', 'w') as f:
    nbf.write(nb, f)
print("09_model_2_decline_detection.ipynb generated.")
