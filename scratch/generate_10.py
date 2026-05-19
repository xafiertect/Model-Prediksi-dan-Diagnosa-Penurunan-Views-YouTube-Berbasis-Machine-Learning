import nbformat as nbf

nb = nbf.v4.new_notebook()

nb['cells'] = [
    nbf.v4.new_markdown_cell("# Model 3: Revenue Forecasting (Prediksi Pendapatan)\n**Tujuan:** Memproyeksikan pendapatan finansial (IDR) di masa depan berdasarkan tren performa views dan monetisasi."),
    
    nbf.v4.new_markdown_cell("## 1. Import Library"),
    nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')"""),
    
    nbf.v4.new_markdown_cell("## 2. Data Filtering\n**Aturan:** Hanya eksekusi pada `monetization_rate = 1` atau `estimated_revenue_idr > 0`."),
    nbf.v4.new_code_cell("""# Gunakan output dari Model 1 (yang sudah di-merge dengan fitur lain)
df = pd.read_csv('../../data/processed/forecast_results.csv')

# Kolom revenue aktual (target). Diasumsikan estimated_doubleclick_revenue_idr adalah revenue utamanya.
# Kita tambahkan validasi untuk memastikan tipe data benar.
if 'estimated_doubleclick_revenue_idr' not in df.columns:
    df['estimated_doubleclick_revenue_idr'] = 0

df_monetized = df[(df['monetization_rate'] == 1) | (df['estimated_doubleclick_revenue_idr'] > 0)].copy()

# Kita asumsikan revenue_per_view tidak null, jika null diisi median
if 'revenue_per_view' in df_monetized.columns:
    df_monetized['revenue_per_view'] = df_monetized['revenue_per_view'].fillna(df_monetized['revenue_per_view'].median())
else:
    df_monetized['revenue_per_view'] = 0.0

if 'ctr_impression_score' not in df_monetized.columns:
    df_monetized['ctr_impression_score'] = 0.0

print(f"Jumlah video monetisasi: {len(df_monetized)}")
"""),
    
    nbf.v4.new_markdown_cell("## 3. Modeling (Random Forest Regressor)\nFitur Input Wajib:\n- Prediksi Views (Output Model 1)\n- `revenue_per_view`\n- `monetization_rate`\n- `ctr_impression_score`"),
    nbf.v4.new_code_cell("""if not df_monetized.empty:
    features = ['predicted_views', 'revenue_per_view', 'monetization_rate', 'ctr_impression_score']
    target = 'estimated_doubleclick_revenue_idr'
    
    # Isi NaN dengan 0 untuk fitur training
    df_monetized[features] = df_monetized[features].fillna(0)
    df_monetized[target] = df_monetized[target].fillna(0)
    
    # Split Data (Simple 80-20 untuk testing/training evaluasi)
    split_idx = int(len(df_monetized) * 0.8)
    train = df_monetized.iloc[:split_idx]
    test  = df_monetized.iloc[split_idx:]
    
    X_train = train[features]
    y_train = train[target]
    X_test = test[features]
    y_test = test[target]
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
"""),
    
    nbf.v4.new_markdown_cell("## 4. Evaluasi (WAPE)\nWAPE (Weighted Absolute Percentage Error) agar model lebih fokus mengevaluasi akurasi pada video dengan *revenue* paling tinggi."),
    nbf.v4.new_code_cell("""def calculate_wape(y_true, y_pred):
    sum_abs_error = np.sum(np.abs(y_true - y_pred))
    sum_abs_true = np.sum(np.abs(y_true))
    
    # Prevent division by zero
    if sum_abs_true == 0:
        return 0
    return sum_abs_error / sum_abs_true

if not df_monetized.empty:
    wape_score = calculate_wape(y_test, y_pred)
    print(f"WAPE Score: {wape_score:.4f} ({wape_score*100:.2f}%)")
    
    # Simpan prediksi revenue
    test['predicted_revenue'] = y_pred
    test[['video_id', 'predicted_views', 'predicted_revenue', 'estimated_doubleclick_revenue_idr']].head()
""")
]

with open('notebooks/modelling/10_model_3_revenue_forecast.ipynb', 'w') as f:
    nbf.write(nb, f)
print("10_model_3_revenue_forecast.ipynb generated.")
