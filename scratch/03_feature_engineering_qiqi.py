# # Engagement Features
# 
# By Qiqi

# --- CELL ---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- CELL ---

df = pd.read_csv('../data/raw/data_cleaned_full.csv')

# --- CELL ---

# ## 1. Like Rate
# like_rate = likes / views

# --- CELL ---

df['like_rate'] = df['likes'] / (df['views'] + 1)
sns.histplot(df['like_rate'], bins=30, kde=True)
plt.title('Like Rate Distribution')
plt.show()

# --- CELL ---

# ## 2. Comment Rate
# comment_rate = comments / views

# --- CELL ---

df['comment_rate'] = df['comments'] / (df['views'] + 1)
sns.histplot(df['comment_rate'], bins=30, kde=True)
plt.title('Comment Rate Distribution')
plt.show()

# --- CELL ---

# ## 3. Retention Proxy
# retention_proxy = avg_view_duration / video_duration_sec
# Note: Need to convert duration string to seconds if necessary. Assuming avg_view_duration is already string HH:MM:SS or seconds.

# --- CELL ---

def time_to_sec(t):
    try:
        if pd.isna(t):
            return 0
        if isinstance(t, str):
            parts = t.split(':')
            if len(parts) == 3:
                return int(parts[0])*3600 + int(parts[1])*60 + int(parts[2])
            elif len(parts) == 2:
                return int(parts[0])*60 + int(parts[1])
            return int(t)
        return float(t)
    except:
        return 0

df['avg_view_sec'] = df['avg_view_duration'].apply(time_to_sec)
df['video_duration_sec'] = df['video_duration_sec'].apply(time_to_sec)

df['retention_proxy'] = df['avg_view_sec'] / (df['video_duration_sec'] + 1)
sns.histplot(df['retention_proxy'], bins=30, kde=True)
plt.title('Retention Proxy Distribution')
plt.show()

# --- CELL ---

# ## 4. Engagement Score
# engagement_score = (like_rate * 0.5) + (comment_rate * 0.3) + (retention_proxy * 0.2)

# --- CELL ---

df['engagement_score'] = (df['like_rate'] * 0.5) + (df['comment_rate'] * 0.3) + (df['retention_proxy'] * 0.2)
sns.boxplot(x=df['engagement_score'])
plt.title('Engagement Score Outliers')
plt.show()

# --- CELL ---

# ## Save Data

# --- CELL ---

df = df.replace([np.inf, -np.inf], np.nan)
df = df.apply(lambda x: x.fillna(0) if x.dtype.kind in "biufc" else x)
df.to_csv('../data/processed/features_engagement.csv', index=False)
print('Saved to ../data/processed/features_engagement.csv')

# --- CELL ---

