# # EDA - Hippo Academy YouTube Dataset
# 
# By Qiqi

# --- CELL ---

# ## 1. Import Libraries

# --- CELL ---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# --- CELL ---

# ## 2. Load Data

# --- CELL ---

df = pd.read_csv('../data/raw/hippo_academy_raw.csv')
df.head()

# --- CELL ---

# ## 3. Data Overview (Shape, Dtypes, Missing Values)

# --- CELL ---

print('Shape:', df.shape)
print('\nData Types:')
print(df.dtypes)
print('\nMissing Values:')
print(df.isnull().sum())

# --- CELL ---

# ## 3.1 Missing Values Handler
# Fill numeric with median, categorical with mode.

# --- CELL ---

print('Missing before:', df.isnull().sum().sum())
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])
print('Missing after:', df.isnull().sum().sum())
assert df.isnull().sum().sum() == 0

# --- CELL ---

# ## 4. Descriptive Stats

# --- CELL ---

df.describe()

# --- CELL ---

# ## 5. Views Distribution

# --- CELL ---

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df['views'], bins=30, kde=True, ax=axes[0])
axes[0].set_title('Histogram of Views')
sns.boxplot(x=df['views'], ax=axes[1])
axes[1].set_title('Boxplot of Views')
plt.show()

# --- CELL ---

# ## 6. Correlation Heatmap

# --- CELL ---

plt.figure(figsize=(10, 8))
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# --- CELL ---

# ## 7. Monthly Views Trend

# --- CELL ---

df['publish_date'] = pd.to_datetime(df['publish_date'])
df['month'] = df['publish_date'].dt.to_period('M')
monthly_views = df.groupby('month')['views'].sum()
monthly_views.plot(kind='line', marker='o', figsize=(10, 5))
plt.title('Monthly Views Trend')
plt.xlabel('Month')
plt.ylabel('Total Views')
plt.grid(True)
plt.show()

# --- CELL ---

# ## 8. Top 10 Videos by Views

# --- CELL ---

df.nlargest(10, 'views')[['video_title', 'views', 'publish_date']]

# --- CELL ---

# ## 9. CTR vs Views Analysis

# --- CELL ---

df['engagement_rate'] = (df['likes'] + df['comments']) / (df['views'] + 1)
plt.figure(figsize=(8, 6))
sns.regplot(x='ctr(%)', y='views', data=df, scatter_kws={'alpha':0.5})
sns.scatterplot(x='ctr(%)', y='views', hue='engagement_rate', data=df, palette='viridis')
plt.title('CTR vs Views')
plt.show()
corr_val = df['ctr(%)'].corr(df['views'])
print('Pearson Correlation between CTR and Views:', corr_val)

# --- CELL ---

# ### Interpretation
# The scatter plot shows the relationship between CTR and Views. A higher CTR generally correlates with more views, as expected. The color represents the engagement rate.

# --- CELL ---

# ## 10. Conclusion
# The dataset provides insights into video performance over time. CTR and views show positive correlation, and engagement rate highlights high-performing content.