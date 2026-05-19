# # Revenue Features
# 
# By Az-Zahrawani

# --- CELL ---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- CELL ---

df = pd.read_csv('../data/raw/data_cleaned_full.csv')

# --- CELL ---

# ## 1. Revenue per View

# --- CELL ---

df['revenue_per_view'] = df['revenue_idr'] / (df['views'] + 1)
sns.histplot(df['revenue_per_view'], bins=30, kde=True)
plt.title('Revenue per View')
plt.show()
print('Correlation with Views:', df['revenue_per_view'].corr(df['views']))

# --- CELL ---

# ## 2. Revenue per Subscriber

# --- CELL ---

df['revenue_per_subscriber'] = df['revenue_idr'] / (df['subscribers_gained'].abs() + 1)
sns.histplot(df['revenue_per_subscriber'], bins=30, kde=True)
plt.title('Revenue per Subscriber')
plt.show()
print('Correlation with Views:', df['revenue_per_subscriber'].corr(df['views']))

# --- CELL ---

# ## 3. Monetization Rate

# --- CELL ---

# ## 4. Avg Revenue Category

# --- CELL ---

# ## Save Data