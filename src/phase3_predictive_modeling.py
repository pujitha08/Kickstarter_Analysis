"""
PHASE 3: RQ2 & RQ3 - Predictive Modeling
-----------------------------------------
RQ2: Do NLP text features improve prediction vs basic features only?
RQ3: Do framing features improve prediction beyond control variables?
RQ5: What structural/NLP features best predict continuous funding ratio?

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
  [Fix 5] matplotlib import added (was missing — ROC curve crashed)
  [Fix 6] duplicate cross_val_score import removed
  [Fix 7] RQ5 CV now runs on full dataset, not just the training split
  [Fix 8] RQ5 target log-transformed to handle right-skewed funding_ratio
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt                        # [Fix 5] was missing
from sklearn.model_selection import StratifiedKFold, cross_val_score   # [Fix 1]
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, roc_curve, r2_score, mean_squared_error
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

# Keep a fitted model + scaled data for the ROC curve later
_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
_tr, _te = next(_skf.split(edu, y))
_cat_avg  = get_cat_avg_train(_tr)
_X        = edu[basic_features + nlp_features].copy()
_X['category_avg_success'] = _cat_avg
X_nlp_train_scaled, X_nlp_test_scaled = scale_data(_X.iloc[_tr].values, _X.iloc[_te].values)
y_train_rq2, y_test_rq2 = y[_tr], y[_te]
model_nlp = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
model_nlp.fit(X_nlp_train_scaled, y_train_rq2)
auc_nlp = aucs_nlp.mean()   # canonical AUC is the CV mean

# ---------------------------------------------------------
# RQ3: Control vs Control + Framing
# ---------------------------------------------------------
aucs_control = run_cv(basic_features)          # control == basic
aucs_framing = run_cv(basic_features + framing_features)

# Keep a fitted model for RQ3 cross_val_score block below
_X_framing = edu[basic_features + framing_features].copy()
_X_framing['category_avg_success'] = _cat_avg
X_framing_train_scaled, X_framing_test_scaled = scale_data(
    _X_framing.iloc[_tr].values, _X_framing.iloc[_te].values
)
y_train_rq3 = y_train_rq2   # same fold

# Scalar aliases so the results block below stays readable
auc_basic = aucs_basic.mean()
auc_c     = aucs_control.mean()
auc_f     = aucs_framing.mean()

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

# ---------------------------------------------------------
# RQ3: Controlled Regression (Framing Effects)
# ---------------------------------------------------------

print("\n" + "=" * 50)
print("RQ3: Controlled Regression (Framing Effects)")
print("=" * 50)

import statsmodels.api as sm

# Build regression dataset
X_rq3_reg = edu[basic_features + framing_features].copy()

# Add category_avg_success (global version is fine for regression)
X_rq3_reg['category_avg_success'] = edu['sub_category'].map(
    edu.groupby('sub_category')['success'].mean()
).fillna(edu['success'].mean()).values

# Add constant for regression
X_rq3_reg = sm.add_constant(X_rq3_reg)

y_rq3_reg = edu['success']

# Fit logistic regression (interpretable coefficients)
model_rq3_reg = sm.Logit(y_rq3_reg, X_rq3_reg).fit(disp=0)

print(model_rq3_reg.summary())

# ---------------------------------------------------------
# RQ5: Funding Ratio Regression (Continuous Outcome)
# ---------------------------------------------------------

print("\n" + "=" * 50)
print("RQ5: Funding Ratio Regression")
print("=" * 50)

# Use same education dataset
X_rq5 = edu[basic_features + nlp_features].copy()
X_rq5['category_avg_success'] = edu['sub_category'].map(
    edu.groupby('sub_category')['success'].mean()
).fillna(edu['success'].mean()).values

y_rq5_raw = edu['funding_ratio']

# [Fix 8] Log-transform the target — raw funding_ratio is heavily right-skewed.
# log1p handles zero values safely. Cap at 99.5th percentile first.
cap = y_rq5_raw.quantile(0.995)
y_rq5 = np.log1p(y_rq5_raw.clip(upper=cap))

# Align X and y (drop any NaN in funding_ratio)
mask = y_rq5_raw.notna()
X_rq5 = X_rq5[mask].reset_index(drop=True)
y_rq5 = y_rq5[mask].reset_index(drop=True)

# Scale
scaler_rq5 = StandardScaler()
X_rq5_scaled = scaler_rq5.fit_transform(X_rq5)

# Model (fit on full set for coefficient inspection — CV gives honest R²)
model_rq5 = LinearRegression()
model_rq5.fit(X_rq5_scaled, y_rq5)

# [Fix 7] CV on full dataset, not just the training split
cv_rq5 = cross_val_score(LinearRegression(), X_rq5_scaled, y_rq5, cv=5, scoring='r2')
r2   = cv_rq5.mean()
rmse = np.sqrt(mean_squared_error(y_rq5, model_rq5.predict(X_rq5_scaled)))

print(f"R² (5-fold CV): {r2:.3f} ± {cv_rq5.std():.3f}")
print(f"RMSE (train):   {rmse:.3f}  (log scale)")

coeffs = pd.DataFrame({
    'feature': X_rq5.columns,
    'coef': model_rq5.coef_
}).sort_values('coef', ascending=False)

print("\nTop Positive Drivers:")
print(coeffs.head(10))

print("\nTop Negative Drivers:")
print(coeffs.tail(10))

print("\n" + "=" * 50)
print("MODEL VALIDATION: Cross-Validation")
print("=" * 50)

# [Fix 7] RQ2 NLP model CV — run on full dataset via run_cv() (already done above)
print(f"RQ2 (NLP model) CV AUC:     {aucs_nlp.mean():.3f} ± {aucs_nlp.std():.3f}")

# [Fix 7] RQ3 Framing model CV — same
print(f"RQ3 (Framing model) CV AUC: {aucs_framing.mean():.3f} ± {aucs_framing.std():.3f}")

# RQ5 cross-validation already computed above
print(f"RQ5 (Regression) CV R²:     {r2:.3f} ± {cv_rq5.std():.3f}")

print("\n" + "=" * 50)
print("FINAL MODEL COMPARISON")
print("=" * 50)

print(f"RQ2 Improvement (NLP):     +{aucs_nlp.mean() - aucs_basic.mean():.3f}")
print(f"RQ3 Improvement (Framing): +{aucs_framing.mean() - aucs_control.mean():.3f}")
print(f"RQ5 Model Strength (R²):   {r2:.3f}")

print("\nKey Insight:")
if (aucs_nlp.mean() - aucs_basic.mean()) > (aucs_framing.mean() - aucs_control.mean()):
    print("→ NLP features add more predictive power than framing features.")
else:
    print("→ Framing features add more predictive power than NLP features.")

# ---------------------------------------------------------
# ROC Curve  [Fix 5] matplotlib now imported at top
# ---------------------------------------------------------
fpr, tpr, _ = roc_curve(y_test_rq2, model_nlp.predict_proba(X_nlp_test_scaled)[:, 1])

plt.figure()
plt.plot(fpr, tpr, label=f"NLP Model (AUC = {auc_nlp:.3f})")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - NLP Model")
plt.legend()
plt.savefig(os.path.join(BASE_DIR, "outputs", "figures", "roc_curve_nlp.png"))
plt.close()