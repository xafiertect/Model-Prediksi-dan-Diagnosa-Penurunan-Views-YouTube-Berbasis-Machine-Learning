import json
import os

base_dir = "/run/media/rizqimaulidiyah/7542d4da-568c-4bbf-b867-1295fe534e4e/Capstone-project/Model-Prediksi-dan-Diagnosa-Penurunan-Views-YouTube-Berbasis-Machine-Learning/notebooks/feature_enginering"
os.makedirs(base_dir, exist_ok=True)

def create_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.9"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

def markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [s + "\n" for s in source.split("\n")]
    }

def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [s + "\n" for s in source.split("\n")]
    }

safe_divide_code = """def safe_divide(a, b):
    return a / b.replace(0, np.nan)"""

# -------------------------------------------------------------------
# 02_feature_engineering_wildan.ipynb
# -------------------------------------------------------------------
nb2_cells = [
    markdown_cell("# Feature Engineering: Growth Features\nBy Wildan"),
    code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport warnings\nwarnings.filterwarnings('ignore')"),
    markdown_cell("## Load Data"),
    code_cell("df = pd.read_csv('../../data/processed/hippo_academy_clean.csv')\ndf['publish_date'] = pd.to_datetime(df['publish_date'])\ndf = df.sort_values('publish_date')"),
    markdown_cell("## 1. daily_growth_rate\n(views_today - views_yesterday) / views_yesterday * 100"),
    code_cell("df['views_yesterday'] = df['views'].shift(1)\ndf['daily_growth_rate'] = ((df['views'] - df['views_yesterday']) / df['views_yesterday']) * 100\n\n# Handle NaN/Inf\ndf['daily_growth_rate'] = df['daily_growth_rate'].replace([np.inf, -np.inf], np.nan).fillna(0)\n\nplt.figure(figsize=(8,4))\nsns.histplot(df['daily_growth_rate'], bins=50, kde=True)\nplt.title('Distribution of daily_growth_rate')\nplt.show()"),
    markdown_cell("## 2. subscriber_net\nsubscribers_gained - subscribers_lost"),
    code_cell("df['subscriber_net'] = df['subscribers_gained'] - df['subscribers_lost']\n\nplt.figure(figsize=(8,4))\nsns.histplot(df['subscriber_net'], bins=50, kde=True)\nplt.title('Distribution of subscriber_net')\nplt.show()"),
    markdown_cell("## 3. view_velocity\nviews / days_since_upload"),
    code_cell("df['days_since_upload'] = (pd.to_datetime('today') - df['publish_date']).dt.days\ndf['days_since_upload'] = df['days_since_upload'].replace(0, 1) # Avoid division by zero\n\ndf['view_velocity'] = df['views'] / df['days_since_upload']\n\nplt.figure(figsize=(8,4))\nsns.histplot(df['view_velocity'], bins=50, kde=True)\nplt.title('Distribution of view_velocity')\nplt.show()"),
    markdown_cell("## 4. rolling_avg_views_7d\n7-day rolling mean of views"),
    code_cell("df['rolling_avg_views_7d'] = df['views'].rolling(window=7, min_periods=1).mean()\n\nplt.figure(figsize=(8,4))\nsns.histplot(df['rolling_avg_views_7d'], bins=50, kde=True)\nplt.title('Distribution of rolling_avg_views_7d')\nplt.show()"),
    markdown_cell("## 5. growth_acceleration\ndiff of daily_growth_rate"),
    code_cell("df['growth_acceleration'] = df['daily_growth_rate'].diff().fillna(0)\n\nplt.figure(figsize=(8,4))\nsns.histplot(df['growth_acceleration'], bins=50, kde=True)\nplt.title('Distribution of growth_acceleration')\nplt.show()"),
    markdown_cell("## Rolling Average Plot\n7-day and 30-day rolling avg views on one line chart."),
    code_cell("df['rolling_avg_views_30d'] = df['views'].rolling(window=30, min_periods=1).mean()\n\nplt.figure(figsize=(12,6))\nplt.plot(df['publish_date'], df['rolling_avg_views_7d'], label='7-Day Rolling Avg')\nplt.plot(df['publish_date'], df['rolling_avg_views_30d'], label='30-Day Rolling Avg')\nplt.title('7-Day vs 30-Day Rolling Average Views')\nplt.xlabel('Publish Date')\nplt.ylabel('Views')\nplt.legend()\nplt.show()"),
    markdown_cell("## Save Features"),
    code_cell("cols_to_save = ['video_id', 'daily_growth_rate', 'subscriber_net', 'view_velocity', 'rolling_avg_views_7d', 'growth_acceleration']\nif 'video_id' in df.columns:\n    df[cols_to_save].to_csv('../../data/processed/features_growth.csv', index=False)\n    print('Saved to features_growth.csv')\nelse:\n    print('video_id not found in dataset')")
]

with open(os.path.join(base_dir, "02_feature_engineering_wildan.ipynb"), "w") as f:
    json.dump(create_notebook(nb2_cells), f, indent=2)

# -------------------------------------------------------------------
# 03_feature_engineering_qiqi.ipynb
# -------------------------------------------------------------------
nb3_cells = [
    markdown_cell("# Feature Engineering: Engagement Features\nBy Rizqi"),
    code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport warnings\nwarnings.filterwarnings('ignore')\n\n" + safe_divide_code),
    markdown_cell("## Load Data"),
    code_cell("df = pd.read_csv('../../data/processed/hippo_academy_clean.csv')"),
    markdown_cell("## 1. like_rate\nlikes / views"),
    code_cell("df['like_rate'] = safe_divide(df['likes'], df['views']).fillna(0)\n\nplt.figure(figsize=(8,4))\nsns.histplot(df['like_rate'], bins=50, kde=True)\nplt.title('Distribution of like_rate')\nplt.show()\n\n# IQR Outlier check\nQ1 = df['like_rate'].quantile(0.25)\nQ3 = df['like_rate'].quantile(0.75)\nIQR = Q3 - Q1\noutliers = df[(df['like_rate'] < (Q1 - 1.5 * IQR)) | (df['like_rate'] > (Q3 + 1.5 * IQR))]\nprint(f'Outliers in like_rate: {len(outliers)}')"),
    markdown_cell("## 2. comment_rate\ncomments / views"),
    code_cell("df['comment_rate'] = safe_divide(df['comments'], df['views']).fillna(0)\n\nplt.figure(figsize=(8,4))\nsns.histplot(df['comment_rate'], bins=50, kde=True)\nplt.title('Distribution of comment_rate')\nplt.show()\n\n# IQR Outlier check\nQ1 = df['comment_rate'].quantile(0.25)\nQ3 = df['comment_rate'].quantile(0.75)\nIQR = Q3 - Q1\noutliers = df[(df['comment_rate'] < (Q1 - 1.5 * IQR)) | (df['comment_rate'] > (Q3 + 1.5 * IQR))]\nprint(f'Outliers in comment_rate: {len(outliers)}')"),
    markdown_cell("## 3. retention_proxy\navg_view_duration / video_duration_sec"),
    code_cell("df['retention_proxy'] = safe_divide(df['avg_view_duration'], df['video_duration_sec']).fillna(0).clip(0, 1)\n\nplt.figure(figsize=(8,4))\nsns.histplot(df['retention_proxy'], bins=50, kde=True)\nplt.title('Distribution of retention_proxy')\nplt.show()\n\n# IQR Outlier check\nQ1 = df['retention_proxy'].quantile(0.25)\nQ3 = df['retention_proxy'].quantile(0.75)\nIQR = Q3 - Q1\noutliers = df[(df['retention_proxy'] < (Q1 - 1.5 * IQR)) | (df['retention_proxy'] > (Q3 + 1.5 * IQR))]\nprint(f'Outliers in retention_proxy: {len(outliers)}')"),
    markdown_cell("## 4. engagement_score\n(like_rate * 0.5) + (comment_rate * 0.3) + (retention_proxy * 0.2)"),
    code_cell("df['engagement_score'] = (df['like_rate'] * 0.5) + (df['comment_rate'] * 0.3) + (df['retention_proxy'] * 0.2)\n\nplt.figure(figsize=(8,4))\nsns.histplot(df['engagement_score'], bins=50, kde=True)\nplt.title('Distribution of engagement_score')\nplt.show()\n\n# IQR Outlier check\nQ1 = df['engagement_score'].quantile(0.25)\nQ3 = df['engagement_score'].quantile(0.75)\nIQR = Q3 - Q1\noutliers = df[(df['engagement_score'] < (Q1 - 1.5 * IQR)) | (df['engagement_score'] > (Q3 + 1.5 * IQR))]\nprint(f'Outliers in engagement_score: {len(outliers)}')"),
    markdown_cell("## Save Features"),
    code_cell("cols_to_save = ['video_id', 'like_rate', 'comment_rate', 'retention_proxy', 'engagement_score']\nif 'video_id' in df.columns:\n    df[cols_to_save].to_csv('../../data/processed/features_engagement.csv', index=False)\n    print('Saved to features_engagement.csv')\nelse:\n    print('video_id not found in dataset')")
]

with open(os.path.join(base_dir, "03_feature_engineering_qiqi.ipynb"), "w") as f:
    json.dump(create_notebook(nb3_cells), f, indent=2)

# -------------------------------------------------------------------
# 04_feature_engineering_yusuf.ipynb
# -------------------------------------------------------------------
nb4_cells = [
    markdown_cell("# Feature Engineering: CTR Impact Features\nBy Yusuf"),
    code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport warnings\nwarnings.filterwarnings('ignore')\n\n" + safe_divide_code),
    markdown_cell("## Load Data"),
    code_cell("df = pd.read_csv('../../data/processed/hippo_academy_clean.csv')"),
    markdown_cell("## 1. impression_to_view_rate\nviews / impressions"),
    code_cell("df['impression_to_view_rate'] = safe_divide(df['views'], df['impressions']).fillna(0)\n\nplt.figure(figsize=(8,4))\nsns.histplot(df['impression_to_view_rate'], bins=50, kde=True)\nplt.title('Distribution of impression_to_view_rate')\nplt.show()"),
    markdown_cell("## 2. ctr_normalized\nctr / 100 (and clip to 0-1)"),
    code_cell("if 'ctr' not in df.columns and 'ctr(%)' in df.columns:\n    df['ctr'] = df['ctr(%)']\n\ndf['ctr_normalized'] = (df['ctr'] / 100).clip(0, 1)\n\nplt.figure(figsize=(8,4))\nsns.histplot(df['ctr_normalized'], bins=50, kde=True)\nplt.title('Distribution of ctr_normalized')\nplt.show()"),
    markdown_cell("## 3. ctr_impression_score\nctr_normalized * impression_to_view_rate"),
    code_cell("df['ctr_impression_score'] = df['ctr_normalized'] * df['impression_to_view_rate']\n\nplt.figure(figsize=(8,4))\nsns.histplot(df['ctr_impression_score'], bins=50, kde=True)\nplt.title('Distribution of ctr_impression_score')\nplt.show()"),
    markdown_cell("## 4. ctr_category & CTR Category Analysis\nBin CTR into Low(<3%), Mid(3-7%), High(>7%)"),
    code_cell("def categorize_ctr(x):\n    if x < 3:\n        return 'Low'\n    elif x <= 7:\n        return 'Mid'\n    else:\n        return 'High'\n\ndf['ctr_category'] = df['ctr'].apply(categorize_ctr)\nprint(df['ctr_category'].value_counts())\n\n# CTR Category Analysis\nctr_counts = df['ctr_category'].value_counts(normalize=True) * 100\n\nplt.figure(figsize=(8,5))\nax = sns.barplot(x=ctr_counts.index, y=ctr_counts.values)\nfor p in ax.patches:\n    ax.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom')\nplt.title('CTR Category Distribution')\nplt.ylabel('Percentage (%)')\nplt.show()"),
    markdown_cell("## Save Features"),
    code_cell("cols_to_save = ['video_id', 'impression_to_view_rate', 'ctr_normalized', 'ctr_impression_score', 'ctr_category']\nif 'video_id' in df.columns:\n    df[cols_to_save].to_csv('../../data/processed/features_ctr.csv', index=False)\n    print('Saved to features_ctr.csv')\nelse:\n    print('video_id not found in dataset')")
]

with open(os.path.join(base_dir, "04_feature_engineering_yusuf.ipynb"), "w") as f:
    json.dump(create_notebook(nb4_cells), f, indent=2)

# -------------------------------------------------------------------
# 05_feature_engineering_akmal.ipynb
# -------------------------------------------------------------------
nb5_cells = [
    markdown_cell("# Feature Engineering: Time Decay & Feature Aggregation\nBy Akmal"),
    code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport warnings\nwarnings.filterwarnings('ignore')\n\n" + safe_divide_code),
    markdown_cell("## Load Data"),
    code_cell("df = pd.read_csv('../../data/processed/hippo_academy_clean.csv')\ndf['publish_date'] = pd.to_datetime(df['publish_date'])"),
    markdown_cell("## Part 1 — Time Decay\n1. `days_since_upload`\n2. `decay_weight = exp(-0.05 * days_since_upload)`\nLambda 0.05 is chosen because YouTube videos typically lose rapid momentum after the first few weeks.\n3. `weighted_views = views * decay_weight`"),
    code_cell("df['days_since_upload'] = (pd.to_datetime('today') - df['publish_date']).dt.days\ndf['days_since_upload'] = df['days_since_upload'].clip(lower=0)\n\ndf['decay_weight'] = np.exp(-0.05 * df['days_since_upload'])\ndf['weighted_views'] = df['views'] * df['decay_weight']\n\nplt.figure(figsize=(10,5))\nplt.scatter(df['views'], df['weighted_views'], alpha=0.5)\nplt.xlabel('Original Views')\nplt.ylabel('Weighted Views')\nplt.title('Original Views vs Weighted Views Comparison')\nplt.show()"),
    markdown_cell("## Decay Weight Comparison\nPlot 3 decay curves using lambda values 0.01, 0.05, 0.1"),
    code_cell("days = np.arange(0, 366)\nlambda_1 = np.exp(-0.01 * days)\nlambda_2 = np.exp(-0.05 * days)\nlambda_3 = np.exp(-0.1 * days)\n\nplt.figure(figsize=(10,5))\nplt.plot(days, lambda_1, label='lambda = 0.01')\nplt.plot(days, lambda_2, label='lambda = 0.05')\nplt.plot(days, lambda_3, label='lambda = 0.1')\nplt.xlabel('Days (0-365)')\nplt.ylabel('Decay Weight')\nplt.title('Time Decay Curves Comparison')\nplt.legend()\nplt.grid(True)\nplt.show()"),
    markdown_cell("## Part 2 — Feature Aggregation"),
    code_cell("df['total_engagement'] = df['likes'] + df['comments']\ndf['engagement_to_view_ratio'] = safe_divide(df['total_engagement'], df['views']).fillna(0)\n\n# Normalize avg_view_duration (min-max scale)\nmax_dur = df['avg_view_duration'].max()\ndf['avg_view_duration_norm'] = df['avg_view_duration'] / max_dur if max_dur > 0 else 0\n\nif 'ctr' not in df.columns and 'ctr(%)' in df.columns:\n    df['ctr'] = df['ctr(%)']\n\ndf['performance_score'] = (0.4 * df['ctr']) + (0.3 * df['engagement_to_view_ratio']) + (0.3 * df['avg_view_duration_norm'])\n\ntop_10 = df.nlargest(10, 'performance_score')[['video_title', 'performance_score']]\ndisplay(top_10)"),
    markdown_cell("## Save Features"),
    code_cell("cols_to_save = ['video_id', 'days_since_upload', 'decay_weight', 'weighted_views', 'total_engagement', 'engagement_to_view_ratio', 'performance_score']\nif 'video_id' in df.columns:\n    df[cols_to_save].to_csv('../../data/processed/features_time_decay.csv', index=False)\n    print('Saved to features_time_decay.csv')\nelse:\n    print('video_id not found in dataset')")
]

with open(os.path.join(base_dir, "05_feature_engineering_akmal.ipynb"), "w") as f:
    json.dump(create_notebook(nb5_cells), f, indent=2)

# -------------------------------------------------------------------
# 06_feature_engineering_zahra.ipynb
# -------------------------------------------------------------------
nb6_cells = [
    markdown_cell("# Feature Engineering: Revenue Features\nBy Az-Zahrawani"),
    code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport warnings\nwarnings.filterwarnings('ignore')\n\n" + safe_divide_code),
    markdown_cell("## Load Data"),
    code_cell("df = pd.read_csv('../../data/processed/hippo_academy_clean.csv')"),
    markdown_cell("## 1. revenue_per_view\nrevenue_idr / views"),
    code_cell("if 'revenue_idr' not in df.columns and 'estimated_revenue_idr' in df.columns:\n    df['revenue_idr'] = df['estimated_revenue_idr']\n\ndf['revenue_per_view'] = safe_divide(df['revenue_idr'], df['views']).fillna(0)\n\nplt.figure(figsize=(8,4))\nsns.histplot(df['revenue_per_view'], bins=50, kde=True)\nplt.title('Distribution of revenue_per_view')\nplt.show()\n\nprint('Correlation with views:', df['revenue_per_view'].corr(df['views']))"),
    markdown_cell("## 2. revenue_per_subscriber\nrevenue_idr / subscribers_gained"),
    code_cell("df['revenue_per_subscriber'] = safe_divide(df['revenue_idr'], df['subscribers_gained']).fillna(0)\n\nplt.figure(figsize=(8,4))\nsns.histplot(df['revenue_per_subscriber'], bins=50, kde=True)\nplt.title('Distribution of revenue_per_subscriber')\nplt.show()\n\nprint('Correlation with views:', df['revenue_per_subscriber'].corr(df['views']))"),
    markdown_cell("## 3. monetization_rate\n(revenue_idr > 0).astype(int)"),
    code_cell("df['monetization_rate'] = (df['revenue_idr'] > 0).astype(int)\n\nplt.figure(figsize=(8,4))\nsns.countplot(x=df['monetization_rate'])\nplt.title('Distribution of monetization_rate')\nplt.show()\n\nprint('Correlation with views:', df['monetization_rate'].corr(df['views']))"),
    markdown_cell("## 4. avg_revenue_category\nbin revenue_per_view into Low/Mid/High"),
    code_cell("q33 = df['revenue_per_view'].quantile(0.33)\nq66 = df['revenue_per_view'].quantile(0.66)\n\ndef categorize_rev(x):\n    if x <= q33:\n        return 'Low'\n    elif x <= q66:\n        return 'Mid'\n    else:\n        return 'High'\n\ndf['avg_revenue_category'] = df['revenue_per_view'].apply(categorize_rev)\n\nplt.figure(figsize=(8,4))\nsns.countplot(x=df['avg_revenue_category'])\nplt.title('Distribution of avg_revenue_category')\nplt.show()"),
    markdown_cell("## Save Features"),
    code_cell("cols_to_save = ['video_id', 'revenue_per_view', 'revenue_per_subscriber', 'monetization_rate', 'avg_revenue_category']\nif 'video_id' in df.columns:\n    df[cols_to_save].to_csv('../../data/processed/features_revenue.csv', index=False)\n    print('Saved to features_revenue.csv')\nelse:\n    print('video_id not found in dataset')")
]

with open(os.path.join(base_dir, "06_feature_engineering_zahra.ipynb"), "w") as f:
    json.dump(create_notebook(nb6_cells), f, indent=2)

print("Berhasil membuat 5 notebook Feature Engineering!")
