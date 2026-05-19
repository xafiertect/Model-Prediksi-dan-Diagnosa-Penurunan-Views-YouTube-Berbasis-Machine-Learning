import json
import os
import subprocess

notebook_dir = "/run/media/rizqimaulidiyah/7542d4da-568c-4bbf-b867-1295fe534e4e/Capstone-project/Model-Prediksi-dan-Diagnosa-Penurunan-Views-YouTube-Berbasis-Machine-Learning/notebooks/feature_enginering"

replacements = {
    "04_feature_engineering_yusuf.ipynb": [
        ("./../data/Data_Cleaned_Wildan", "../../data/Data_Cleaned_Wildan.csv"),
        ("df['ctr'] = df['ctr(%)']", "df['ctr'] = df['impressions_click_through_rate_pct']"),
        ("if 'ctr' not in df.columns and 'ctr(%)' in df.columns:", "if 'ctr' not in df.columns:")
    ],
    "05_feature_engineering_akmal.ipynb": [
        ("../../data/processed/hippo_academy_clean.csv", "../../data/Data_Cleaned_Wildan.csv"),
        ("df['total_engagement'] = df['likes'] + df['comments']", "df['total_engagement'] = df['engaged_views']"),
        ("df['avg_view_duration']", "df['avg_view_duration_sec']"),
        ("df['ctr'] = df['ctr(%)']", "df['ctr'] = df['impressions_click_through_rate_pct']"),
        ("if 'ctr' not in df.columns and 'ctr(%)' in df.columns:", "if 'ctr' not in df.columns:")
    ],
    "06_feature_engineering_zahra.ipynb": [
        ("../../data/processed/hippo_academy_clean.csv", "../../data/Data_Cleaned_Wildan.csv"),
        ("df['subscribers_gained']", "df['subscribers']")
    ]
}

for nb_file, reps in replacements.items():
    nb_path = os.path.join(notebook_dir, nb_file)
    if os.path.exists(nb_path):
        with open(nb_path, 'r') as f:
            content = f.read()
        for old, new in reps:
            content = content.replace(old, new)
        with open(nb_path, 'w') as f:
            f.write(content)
        print(f"Fixed {nb_file}")

# Now execute all notebooks
notebooks = [
    "02_feature_engineering_wildan.ipynb",
    "03_feature_engineering_qiqi.ipynb",
    "04_feature_engineering_yusuf.ipynb",
    "05_feature_engineering_akmal.ipynb",
    "06_feature_engineering_zahra.ipynb"
]

jupyter_path = "/run/media/rizqimaulidiyah/7542d4da-568c-4bbf-b867-1295fe534e4e/Capstone-project/Model-Prediksi-dan-Diagnosa-Penurunan-Views-YouTube-Berbasis-Machine-Learning/capstonevnv/bin/jupyter"

for nb in notebooks:
    nb_path = os.path.join(notebook_dir, nb)
    if os.path.exists(nb_path):
        print(f"Executing {nb}...")
        try:
            subprocess.run([jupyter_path, "nbconvert", "--to", "notebook", "--execute", "--inplace", nb_path], check=True)
            print(f"Successfully executed {nb}")
        except subprocess.CalledProcessError as e:
            print(f"Error executing {nb}: {e}")
            
