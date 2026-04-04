"""
PHASE 3: RQ3 - Framing Effect (Controlled Regression)
-----------------------------------------------------
Goal:
Test whether framing variables (sentiment, readability, lexical tone, blurb length)
improve predictive performance beyond baseline controls.

Input:
    data/processed/kickstarter_cleaned.csv

Output:
    Printed AUC comparison for:
        - Control-only model
        - Control + Framing model
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import os

# ---------------------------------------------------------
# Load Data
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "kickstarter_cleaned.csv")

df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df):,} rows")

# Filter clean education projects
edu = df[(df['is_education'] == 1) & (df['biased_category'] == 0)].copy()
print(f"Education projects (clean): {len(edu):,}")
print(f"Success rate: {edu['success'].mean():.1%}")

# ---------------------------------------------------------
# Feature Definitions
# ---------------------------------------------------------
control_features = [
    'log_goal',
    'creator_total_projects',
    'year',
    'month',
    'category_avg_success'
]

framing_features = [
    'blurb_word_count',
    'sentiment_compound',
    'readability_fre',
    'lex_innovation',
    'lex_community',
    'lex_digital_online'
]

# ---------------------------------------------------------
# Prepare Data
# ---------------------------------------------------------
X_full = edu[control_features + framing_features]
y = edu['success']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, random_state=42
)

# Control-only subset
Xc_train = X_train[control_features]
Xc_test = X_test[control_features]

# ---------------------------------------------------------
# Model 1: Controls Only
# ---------------------------------------------------------
model_c = LogisticRegression(max_iter=2000)
model_c.fit(Xc_train, y_train)
auc_c = roc_auc_score(y_test, model_c.predict_proba(Xc_test)[:, 1])

# ---------------------------------------------------------
# Model 2: Controls + Framing
# ---------------------------------------------------------
model_f = LogisticRegression(max_iter=2000)
model_f.fit(X_train, y_train)
auc_f = roc_auc_score(y_test, model_f.predict_proba(X_test)[:, 1])

# ---------------------------------------------------------
# Results
# ---------------------------------------------------------
print("\n" + "=" * 50)
print("RQ3: Does Framing Affect Success?")
print("=" * 50)
print(f"Control only AUC: {auc_c:.3f}")
print(f"Control + Framing AUC: {auc_f:.3f}")
print(f"Improvement: +{auc_f - auc_c:.3f}")
print("=" * 50)

if auc_f > auc_c:
    print("YES — Framing improves predictive performance.")
else:
    print("NO — Framing does not add predictive value.")
