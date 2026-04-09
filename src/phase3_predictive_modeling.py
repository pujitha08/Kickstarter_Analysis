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

Outputs mean ± std AUC across 5 stratified CV folds.

Fixes applied:
  [Fix 1] 5-fold stratified CV replaces single train/test split
  [Fix 2] class_weight='balanced' on all LR models
  [Fix 3] category_avg_success recomputed per fold (prevents leakage)
  [Fix 4] blurb_char_count dropped (collinear with blurb_word_count)
           sentiment_pos / sentiment_neg dropped (derived from compound)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold   # [Fix 1] replaces train_test_split
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
edu = edu.reset_index(drop=True)   # [Fix 3] clean integer index for fold slicing
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
    # [Fix 3] category_avg_success removed here — recomputed per fold
    #         in run_cv() to prevent leakage from test rows
]

nlp_features = [
    'blurb_word_count',
    # [Fix 4] blurb_char_count dropped: corr ~0.99 with blurb_word_count
    'has_exclamation',
    'has_question',
    'has_numbers',
    'n_exclamation',
    'sentiment_compound',
    # [Fix 4] sentiment_pos / sentiment_neg dropped: mathematically
    #         derived from compound — keeping all three is redundant
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

y = edu['success'].values

# ---------------------------------------------------------
# [Fix 3] Per-fold category_avg_success (leak-free)
# ---------------------------------------------------------
def get_cat_avg_train(train_idx):
    """Compute category success rate from training rows only."""
    train_df = edu.iloc[train_idx]
    cat_means = train_df.groupby('sub_category')['success'].mean()
    global_mean = train_df['success'].mean()
    return edu['sub_category'].map(cat_means).fillna(global_mean).values

# ---------------------------------------------------------
# Helper function to scale data
# ---------------------------------------------------------
def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# ---------------------------------------------------------
# [Fix 1] 5-fold stratified CV runner
# ---------------------------------------------------------
def run_cv(feature_list, n_splits=5):
    """
    Run stratified K-fold CV for a given feature list.
    category_avg_success is injected per fold [Fix 3].
    Returns array of per-fold AUC scores (mean ± std reported).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_aucs = []

    for tr_idx, te_idx in skf.split(edu, y):
        # [Fix 3] category average computed on training fold only
        cat_avg = get_cat_avg_train(tr_idx)

        X = edu[feature_list].copy()
        X['category_avg_success'] = cat_avg

        X_train = X.iloc[tr_idx].values
        X_test  = X.iloc[te_idx].values
        y_train = y[tr_idx]
        y_test  = y[te_idx]

        X_train_sc, X_test_sc = scale_data(X_train, X_test)

        # [Fix 2] class_weight='balanced' on all LR models
        model = LogisticRegression(max_iter=1000, random_state=42,
                                   class_weight='balanced')
        model.fit(X_train_sc, y_train)
        fold_aucs.append(roc_auc_score(y_test, model.predict_proba(X_test_sc)[:, 1]))

    return np.array(fold_aucs)

# ---------------------------------------------------------
# RQ2: Basic vs Basic + NLP
# ---------------------------------------------------------
aucs_basic = run_cv(basic_features)
aucs_nlp   = run_cv(basic_features + nlp_features)

# ---------------------------------------------------------
# RQ3: Control vs Control + Framing
# ---------------------------------------------------------
aucs_control = run_cv(basic_features)          # same as basic — control = basic
aucs_framing = run_cv(basic_features + framing_features)

# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

print("\n" + "=" * 50)
print("RQ2: Basic vs Basic + NLP")
print("=" * 50)
print(f"Basic only AUC:  {aucs_basic.mean():.3f} ± {aucs_basic.std():.3f}")
print(f"Basic + NLP AUC: {aucs_nlp.mean():.3f} ± {aucs_nlp.std():.3f}")
print(f"Improvement:     +{aucs_nlp.mean() - aucs_basic.mean():.3f}")

print("\n" + "=" * 50)
print("RQ3: Control vs Control + Framing")
print("=" * 50)
print(f"Control only AUC:      {aucs_control.mean():.3f} ± {aucs_control.std():.3f}")
print(f"Control + Framing AUC: {aucs_framing.mean():.3f} ± {aucs_framing.std():.3f}")
print(f"Improvement:           +{aucs_framing.mean() - aucs_control.mean():.3f}")

print("\n" + "=" * 50)
print("SUMMARY: RQ2 vs RQ3")
print("=" * 50)
print(f"RQ2 (adding NLP):     +{aucs_nlp.mean() - aucs_basic.mean():.3f}")
print(f"RQ3 (adding framing): +{aucs_framing.mean() - aucs_control.mean():.3f}")
print("=" * 50)