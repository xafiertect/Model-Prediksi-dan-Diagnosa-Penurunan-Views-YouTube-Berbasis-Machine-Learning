import pandas as pd
import os

base_path = 'data/processed'

# Load the datasets
df_cleaned = pd.read_csv(os.path.join(base_path, 'youtube_cleaned.csv'))
df_ctr = pd.read_csv(os.path.join(base_path, 'features_ctr.csv'))
df_engagement = pd.read_csv(os.path.join(base_path, 'features_engagement.csv'))
df_growth = pd.read_csv(os.path.join(base_path, 'features_growth.csv'))
df_revenue = pd.read_csv(os.path.join(base_path, 'features_revenue.csv'))
df_time_decay = pd.read_csv(os.path.join(base_path, 'features_time_decay.csv'))

# Merge datasets on 'video_id'
df_final = df_cleaned.merge(df_ctr, on='video_id', how='left')

# Helper function to merge and avoid duplicate _x / _y columns
def merge_clean(left, right, on_col):
    cols_to_use = right.columns.difference(left.columns).tolist() + [on_col]
    return left.merge(right[cols_to_use], on=on_col, how='left')

df_final = merge_clean(df_final, df_engagement, 'video_id')
df_final = merge_clean(df_final, df_growth, 'video_id')
df_final = merge_clean(df_final, df_revenue, 'video_id')
df_final = merge_clean(df_final, df_time_decay, 'video_id')

# Generate additional lag features if they don't exist
# We assume the dataframe can be sorted by publish_date to create lag features correctly
if 'publish_date' in df_final.columns:
    df_final['publish_date'] = pd.to_datetime(df_final['publish_date'])
    df_final = df_final.sort_values('publish_date').reset_index(drop=True)
    
    # We create ts1_views, ts2_views, ts3_views (lag 1, 2, 3 of views)
    # The requirement says "Lag Features: TS1_Views, TS2_Views, TS3_Views (untuk menangkap momentum masa lalu)."
    # These refer to previous video views.
    df_final['ts1_views'] = df_final['views'].shift(1).fillna(0)
    df_final['ts2_views'] = df_final['views'].shift(2).fillna(0)
    df_final['ts3_views'] = df_final['views'].shift(3).fillna(0)

# Save the final merged dataset
output_path = os.path.join(base_path, 'final_features.csv')
df_final.to_csv(output_path, index=False)
print(f'Successfully merged and saved final features to {output_path}')
