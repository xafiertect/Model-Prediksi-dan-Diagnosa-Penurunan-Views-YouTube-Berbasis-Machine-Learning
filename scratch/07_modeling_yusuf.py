# # Predictive Modeling
# 
# By Yusuf

# --- CELL ---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import optuna
import pickle

# --- CELL ---

# ## Prepare Final Features
# We simulate `final_features.csv` by merging some features from our processed files.

# --- CELL ---

# ## 1. Load Data & Check Balance

# --- CELL ---

# ## 2. Train-Test Split

# --- CELL ---

# ## 3. Preprocessing (StandardScaler)

# --- CELL ---

# ## 4. Train 3 Models & 5. Evaluate

# --- CELL ---

# ## Optuna Hyperparameter Tuning

# --- CELL ---

# ## 7 & 8. Save Best Model and Scaler