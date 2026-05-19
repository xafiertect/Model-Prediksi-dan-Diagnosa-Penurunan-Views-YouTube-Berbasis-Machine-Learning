# # Time Decay & Feature Aggregation
# 
# By Akmal

# --- CELL ---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- CELL ---

df = pd.read_csv('../data/raw/data_cleaned_full.csv')
df['publish_date'] = pd.to_datetime(df['publish_date'])

# --- CELL ---

# ## Part 1 — Time Decay
# We use lambda=0.05 to smoothly discount older videos. A 0.05 decay means the weight halves roughly every 14 days, which is reasonable for YouTube's algorithm lifecycle.

# --- CELL ---

# ### Decay Weight Comparison

# --- CELL ---

# ## Part 2 — Feature Aggregation

# --- CELL ---

# ### Top 10 Videos by Performance Score

# --- CELL ---

# ## Save Data