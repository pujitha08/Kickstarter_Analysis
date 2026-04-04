"""
PHASE 3: RQ2 & RQ3 - Predictive Modeling
-----------------------------------------
RQ2: Do NLP text features improve prediction vs basic features only?
RQ3: Do framing features improve prediction beyond control variables?

This script builds four logistic regression models:
- RQ2 Model 1: Basic features only
- RQ2 Model 2: Basic + ALL NLP features
- RQ3 Model 3: Control features only
- RQ3 Model 4: Control + Framing features

Outputs AUC scores for each comparison.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "kickstarter_cleaned.csv")

df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df):,} rows")

# Keep only clean education projects
edu = df[(df['is_education'] == 1) & (df['biased_category'] == 0)].copy()
print(f"Education projects (clean): {len(edu):,}")
print(f"Success rate: {edu['success'].mean():.1%}")

# ---------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------

basic_features = [
    'log_goal',
    'creator_total_projects',
    'year',
    'month',
    'category_avg_success'
]

nlp_features = [
    'blurb_word_count',
    'blurb_char_count',
    'has_exclamation',
    'has_question',
    'has_numbers',
    'n_exclamation',
    'sentiment_compound',
    'sentiment_pos',
    'sentiment_neg',
    'readability_fre',
    'readability_grade',
    'lex_professional',
    'lex_innovation',
    'lex_community',
    'lex_help_please',
    'lex_passion_dream',
    'lex_digital_online',
    'lex_interactive',
    'lex_children_kids'
]

control_features = basic_features.copy()

framing_features = [
    'blurb_word_count',
    'sentiment_compound',
    'readability_fre',
    'lex_innovation',
    'lex_community',
    'lex_digital_online'
]

y = edu['success']

# ---------------------------------------------------------
# Helper function to scale data
# ---------------------------------------------------------
def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# ---------------------------------------------------------
# RQ2: Basic vs Basic + NLP
# ---------------------------------------------------------

X_basic = edu[basic_features]
X_nlp = edu[basic_features + nlp_features]

# Split once for RQ2
X_train_rq2, X_test_rq2, y_train_rq2, y_test_rq2 = train_test_split(
    X_basic, y, test_size=0.2, random_state=42
)

train_idx = X_train_rq2.index
test_idx = X_test_rq2.index

X_nlp_train = X_nlp.loc[train_idx]
X_nlp_test = X_nlp.loc[test_idx]

# Scale features
X_basic_train_scaled, X_basic_test_scaled = scale_data(X_train_rq2, X_test_rq2)
X_nlp_train_scaled, X_nlp_test_scaled = scale_data(X_nlp_train, X_nlp_test)

# Model 1: Basic only
model_basic = LogisticRegression(max_iter=1000, random_state=42)
model_basic.fit(X_basic_train_scaled, y_train_rq2)
auc_basic = roc_auc_score(y_test_rq2, model_basic.predict_proba(X_basic_test_scaled)[:, 1])

# Model 2: Basic + NLP
model_nlp = LogisticRegression(max_iter=1000, random_state=42)
model_nlp.fit(X_nlp_train_scaled, y_train_rq2)
auc_nlp = roc_auc_score(y_test_rq2, model_nlp.predict_proba(X_nlp_test_scaled)[:, 1])

# ---------------------------------------------------------
# RQ3: Control vs Control + Framing
# ---------------------------------------------------------

X_control = edu[control_features]
X_framing = edu[control_features + framing_features]

# Split once for RQ3
X_train_rq3, X_test_rq3, y_train_rq3, y_test_rq3 = train_test_split(
    X_control, y, test_size=0.2, random_state=42
)

X_framing_train = X_framing.loc[X_train_rq3.index]
X_framing_test = X_framing.loc[X_test_rq3.index]

# Scale features
X_control_train_scaled, X_control_test_scaled = scale_data(X_train_rq3, X_test_rq3)
X_framing_train_scaled, X_framing_test_scaled = scale_data(X_framing_train, X_framing_test)

# Model 3: Control only
model_c = LogisticRegression(max_iter=1000, random_state=42)
model_c.fit(X_control_train_scaled, y_train_rq3)
auc_c = roc_auc_score(y_test_rq3, model_c.predict_proba(X_control_test_scaled)[:, 1])

# Model 4: Control + Framing
model_f = LogisticRegression(max_iter=1000, random_state=42)
model_f.fit(X_framing_train_scaled, y_train_rq3)
auc_f = roc_auc_score(y_test_rq3, model_f.predict_proba(X_framing_test_scaled)[:, 1])

# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

print("\n" + "=" * 50)
print("RQ2: Basic vs Basic + NLP")
print("=" * 50)
print(f"Basic only AUC: {auc_basic:.3f}")
print(f"Basic + NLP AUC: {auc_nlp:.3f}")
print(f"Improvement: +{auc_nlp - auc_basic:.3f}")

print("\n" + "=" * 50)
print("RQ3: Control vs Control + Framing")
print("=" * 50)
print(f"Control only AUC: {auc_c:.3f}")
print(f"Control + Framing AUC: {auc_f:.3f}")
print(f"Improvement: +{auc_f - auc_c:.3f}")

print("\n" + "=" * 50)
print("SUMMARY: RQ2 vs RQ3")
print("=" * 50)
print(f"RQ2 (adding NLP):     +{auc_nlp - auc_basic:.3f}")
print(f"RQ3 (adding framing): +{auc_f - auc_c:.3f}")
print("=" * 50)