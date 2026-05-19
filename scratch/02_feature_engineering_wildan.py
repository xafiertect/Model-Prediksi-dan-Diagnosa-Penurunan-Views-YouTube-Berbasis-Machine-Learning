# # Growth Features
# 
# By Wildan

# --- CELL ---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- CELL ---

df = pd.read_csv('../data/raw/data_cleaned_full.csv')
df['publish_date'] = pd.to_datetime(df['publish_date'])

# --- CELL ---

# ## 1. Daily Growth Rate
# daily_growth_rate = (views_today - views_yesterday) / views_yesterday * 100

# --- CELL ---

# ## 2. Subscriber Net
# subscriber_net = subscribers_gained - subscribers_lost

# --- CELL ---

# ## 3. View Velocity
# view_velocity = views / days_since_upload

# --- CELL ---

# ## 4. Rolling Average Views

# --- CELL ---

# ## 5. Growth Acceleration
# growth_acceleration = diff of daily_growth_rate

# --- CELL ---

# ## Save Data