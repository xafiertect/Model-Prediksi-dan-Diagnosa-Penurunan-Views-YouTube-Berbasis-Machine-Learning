import json

nb_path = '/run/media/rizqimaulidiyah/7542d4da-568c-4bbf-b867-1295fe534e4e/Capstone-project/Model-Prediksi-dan-Diagnosa-Penurunan-Views-YouTube-Berbasis-Machine-Learning/notebooks/feature_enginering/03_feature_engineering_qiqi.ipynb'

with open(nb_path, 'r') as f:
    nb = json.load(f)

# Update cell 3: Load Data
nb['cells'][3]['source'] = ["df = pd.read_csv('../../data/Data_Cleaned_Wildan.csv')\n"]

# Update cell 4: Markdown for 1. like_rate -> 1. engaged_view_rate
nb['cells'][4]['source'] = [
    "## 1. engaged_view_rate\n",
    "engaged_views / views\n"
]

# Update cell 5: Code for engaged_view_rate
nb['cells'][5]['source'] = [
    "df['engaged_view_rate'] = safe_divide(df['engaged_views'], df['views']).fillna(0)\n",
    "\n",
    "plt.figure(figsize=(8,4))\n",
    "sns.histplot(df['engaged_view_rate'], bins=50, kde=True)\n",
    "plt.title('Distribution of engaged_view_rate')\n",
    "plt.show()\n",
    "\n",
    "# IQR Outlier check\n",
    "Q1 = df['engaged_view_rate'].quantile(0.25)\n",
    "Q3 = df['engaged_view_rate'].quantile(0.75)\n",
    "IQR = Q3 - Q1\n",
    "outliers = df[(df['engaged_view_rate'] < (Q1 - 1.5 * IQR)) | (df['engaged_view_rate'] > (Q3 + 1.5 * IQR))]\n",
    "print(f'Outliers in engaged_view_rate: {len(outliers)}')\n"
]

# Update cell 6: Markdown for 2. comment_rate -> 2. ctr_normalized
nb['cells'][6]['source'] = [
    "## 2. ctr_normalized\n",
    "impressions_click_through_rate_pct / 100\n"
]

# Update cell 7: Code for ctr_normalized
nb['cells'][7]['source'] = [
    "df['ctr_normalized'] = (df['impressions_click_through_rate_pct'] / 100.0).fillna(0)\n",
    "\n",
    "plt.figure(figsize=(8,4))\n",
    "sns.histplot(df['ctr_normalized'], bins=50, kde=True)\n",
    "plt.title('Distribution of ctr_normalized')\n",
    "plt.show()\n",
    "\n",
    "# IQR Outlier check\n",
    "Q1 = df['ctr_normalized'].quantile(0.25)\n",
    "Q3 = df['ctr_normalized'].quantile(0.75)\n",
    "IQR = Q3 - Q1\n",
    "outliers = df[(df['ctr_normalized'] < (Q1 - 1.5 * IQR)) | (df['ctr_normalized'] > (Q3 + 1.5 * IQR))]\n",
    "print(f'Outliers in ctr_normalized: {len(outliers)}')\n"
]

# Update cell 8: Markdown for 3. retention_proxy
nb['cells'][8]['source'] = [
    "## 3. retention_proxy\n",
    "average_percentage_viewed_pct / 100\n"
]

# Update cell 9: Code for retention_proxy
nb['cells'][9]['source'] = [
    "df['retention_proxy'] = (df['average_percentage_viewed_pct'] / 100.0).fillna(0).clip(0, 1)\n",
    "\n",
    "plt.figure(figsize=(8,4))\n",
    "sns.histplot(df['retention_proxy'], bins=50, kde=True)\n",
    "plt.title('Distribution of retention_proxy')\n",
    "plt.show()\n",
    "\n",
    "# IQR Outlier check\n",
    "Q1 = df['retention_proxy'].quantile(0.25)\n",
    "Q3 = df['retention_proxy'].quantile(0.75)\n",
    "IQR = Q3 - Q1\n",
    "outliers = df[(df['retention_proxy'] < (Q1 - 1.5 * IQR)) | (df['retention_proxy'] > (Q3 + 1.5 * IQR))]\n",
    "print(f'Outliers in retention_proxy: {len(outliers)}')\n"
]

# Update cell 10: Markdown for 4. engagement_score
nb['cells'][10]['source'] = [
    "## 4. engagement_score\n",
    "(engaged_view_rate * 0.4) + (ctr_normalized * 0.3) + (retention_proxy * 0.3)\n"
]

# Update cell 11: Code for engagement_score
nb['cells'][11]['source'] = [
    "df['engagement_score'] = (df['engaged_view_rate'] * 0.4) + (df['ctr_normalized'] * 0.3) + (df['retention_proxy'] * 0.3)\n",
    "\n",
    "plt.figure(figsize=(8,4))\n",
    "sns.histplot(df['engagement_score'], bins=50, kde=True)\n",
    "plt.title('Distribution of engagement_score')\n",
    "plt.show()\n",
    "\n",
    "# IQR Outlier check\n",
    "Q1 = df['engagement_score'].quantile(0.25)\n",
    "Q3 = df['engagement_score'].quantile(0.75)\n",
    "IQR = Q3 - Q1\n",
    "outliers = df[(df['engagement_score'] < (Q1 - 1.5 * IQR)) | (df['engagement_score'] > (Q3 + 1.5 * IQR))]\n",
    "print(f'Outliers in engagement_score: {len(outliers)}')\n"
]

# Update cell 13: Code for Save Features
nb['cells'][13]['source'] = [
    "cols_to_save = ['video_id', 'engaged_view_rate', 'ctr_normalized', 'retention_proxy', 'engagement_score']\n",
    "if 'video_id' in df.columns:\n",
    "    df[cols_to_save].to_csv('../../data/processed/features_engagement.csv', index=False)\n",
    "    print('Saved to features_engagement.csv')\n",
    "else:\n",
    "    print('video_id not found in dataset')\n"
]

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=2)

print("Notebook updated successfully.")
