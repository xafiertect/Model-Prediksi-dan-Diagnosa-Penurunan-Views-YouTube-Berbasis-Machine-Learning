# # CTR Impact Features
# 
# By Yusuf

# --- CELL ---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- CELL ---

df = pd.read_csv('../data/raw/data_cleaned_full.csv')

# --- CELL ---

# ## 1. Impression to View Rate
# impression_to_view_rate = views / impressions

# --- CELL ---

df['impression_to_view_rate'] = df['views'] / (df['impressions'] + 1)
sns.histplot(df['impression_to_view_rate'], bins=30, kde=True)
plt.title('Impression to View Rate')
plt.show()

# --- CELL ---

# ## 2. CTR Normalized
# ctr_normalized = ctr / 100

# --- CELL ---

# ## 3. CTR Impression Score
# ctr_impression_score = ctr_normalized * impression_to_view_rate

# --- CELL ---

# ## 4. CTR Category
# ctr_category = bin ctr into Low(<3%), Mid(3-7%), High(>7%)

# --- CELL ---

# ### CTR Category Analysis

# --- CELL ---

# ## Save Data