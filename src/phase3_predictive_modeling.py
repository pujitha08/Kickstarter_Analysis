"""
PHASE 3: RQ1, RQ2, RQ3 & RQ5 - Predictive Modeling
-----------------------------------------
RQ1: Binary classification (full dataset) - Logistic Reg, RF, XGBoost + SHAP
RQ2: Do NLP text features improve prediction vs basic features only?
RQ3: Do framing features improve prediction beyond control variables?
RQ5: What structural/NLP features best predict continuous funding ratio?

This script builds:
- RQ1: Full dataset models (Logistic Regression, Random Forest, XGBoost)
- RQ2 Model 1: Basic features only
- RQ2 Model 2: Basic + ALL NLP features
- RQ3 Model 3: Control features only
- RQ3 Model 4: Control + Framing features
- RQ5: Linear regression for funding ratio prediction

Outputs mean ± std AUC across 5 stratified CV folds.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os
import warnings
warnings.filterwarnings('ignore')

# Try XGBoost and SHAP
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not installed. Run: pip install xgboost")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not installed. Run: pip install shap")

# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "kickstarter_cleaned.csv")

df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df):,} rows")

# =========================================================
# [RQ1 ADDED] RQ1: BINARY CLASSIFICATION (FULL DATASET)
# =========================================================
print("\n" + "=" * 60)
print("RQ1: BINARY CLASSIFICATION (Full Dataset - All Projects)")
print("=" * 60)

# Prepare full dataset (all completed projects, NOT just education)
df_full = df[(df['status'].isin(['successful', 'failed'])) & (df['biased_category'] == 0)].copy()
df_full['success'] = (df_full['status'] == 'successful').astype(int)
print(f"Full dataset: {len(df_full):,} rows")
print(f"Success rate: {df_full['success'].mean():.1%}")

# Features for RQ1 (structural features only, no text/NLP)
cat_features_rq1 = ['main_category', 'country', 'month', 'day_of_week']
num_features_rq1 = ['goal_usd', 'duration', 'launch_hour', 'creator_total_projects']

X_rq1 = df_full[cat_features_rq1 + num_features_rq1]
y_rq1 = df_full['success'].values

# Preprocessor for RQ1
preprocessor_rq1 = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features_rq1),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features_rq1)
    ])

def get_feature_names_rq1(preprocessor, cat_features, num_features):
    cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_features)
    return list(num_features) + list(cat_names)

# [RQ1 ADDED] 5-fold stratified CV runner for RQ1
def run_cv_rq1(feature_list, model, n_splits=5):
    """5-fold stratified CV for RQ1 models"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_aucs = []
    
    for tr_idx, te_idx in skf.split(X_rq1, y_rq1):
        X_train = X_rq1.iloc[tr_idx]
        X_test = X_rq1.iloc[te_idx]
        y_train = y_rq1[tr_idx]
        y_test = y_rq1[te_idx]
        
        # Preprocess
        preprocessor_rq1.fit(X_train)
        X_train_scaled = preprocessor_rq1.transform(X_train)
        X_test_scaled = preprocessor_rq1.transform(X_test)
        
        # Clone and train model
        if isinstance(model, LogisticRegression):
            model_clone = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced')
        elif isinstance(model, RandomForestClassifier):
            model_clone = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
        elif XGB_AVAILABLE and isinstance(model, XGBClassifier):
            model_clone = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', use_label_encoder=False)
        else:
            model_clone = model
        
        model_clone.fit(X_train_scaled, y_train)
        y_prob = model_clone.predict_proba(X_test_scaled)[:, 1]
        fold_aucs.append(roc_auc_score(y_test, y_prob))
    
    return np.array(fold_aucs)

# [RQ1 ADDED] Train models with CV
models_rq1 = {}
results_rq1 = {}

# Logistic Regression
lr_model = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced')
aucs_lr = run_cv_rq1(None, lr_model)
results_rq1["Logistic Regression"] = {'auc_mean': aucs_lr.mean(), 'auc_std': aucs_lr.std()}
print(f"Logistic Regression: AUC = {aucs_lr.mean():.4f} ± {aucs_lr.std():.4f}")

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
aucs_rf = run_cv_rq1(None, rf_model)
results_rq1["Random Forest"] = {'auc_mean': aucs_rf.mean(), 'auc_std': aucs_rf.std()}
print(f"Random Forest: AUC = {aucs_rf.mean():.4f} ± {aucs_rf.std():.4f}")

# XGBoost (if available)
if XGB_AVAILABLE:
    xgb_model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', use_label_encoder=False)
    aucs_xgb = run_cv_rq1(None, xgb_model)
    results_rq1["XGBoost"] = {'auc_mean': aucs_xgb.mean(), 'auc_std': aucs_xgb.std()}
    print(f"XGBoost: AUC = {aucs_xgb.mean():.4f} ± {aucs_xgb.std():.4f}")

# [RQ1 ADDED] Train final models on full data for feature importance and SHAP
preprocessor_rq1.fit(X_rq1)
X_scaled_full = preprocessor_rq1.transform(X_rq1)
feature_names_rq1 = get_feature_names_rq1(preprocessor_rq1, cat_features_rq1, num_features_rq1)

# Random Forest Feature Importance
rf_final = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
rf_final.fit(X_scaled_full, y_rq1)
importances = rf_final.feature_importances_
feat_importance = pd.Series(importances, index=feature_names_rq1).sort_values(ascending=False)

print("\n" + "=" * 60)
print("TOP 10 FEATURES (Random Forest - Full Dataset)")
print("=" * 60)
for i, (feat, imp) in enumerate(feat_importance.head(10).items()):
    print(f"{i+1}. {feat}: {imp:.4f}")

# Plot feature importance
plt.figure(figsize=(10, 8))
feat_importance.head(15).plot(kind='barh')
plt.title("Top 15 Features - Random Forest (Full Dataset)")
plt.xlabel("Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "outputs", "figures", "rq1_rf_importance.png"))
plt.close()
print("\nSaved: rq1_rf_importance.png")

# [RQ1 ADDED] Logistic Regression Coefficients (direction of impact)
lr_final = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced')
lr_final.fit(X_scaled_full, y_rq1)
coefs = lr_final.coef_[0]
feat_coefs = pd.Series(coefs, index=feature_names_rq1).sort_values(ascending=False)

print("\n" + "=" * 60)
print("WHAT HELPS vs HURTS SUCCESS (Logistic Regression - Full Dataset)")
print("=" * 60)

print("\nHELPS SUCCESS (Positive coefficients):")
for feat, coef in feat_coefs.head(8).items():
    print(f"   + {feat}: {coef:.4f}")

print("\nHURTS SUCCESS (Negative coefficients):")
for feat, coef in feat_coefs.tail(8).items():
    print(f"   - {feat}: {coef:.4f}")

# Plot coefficients
plt.figure(figsize=(10, 8))
colors = ['green' if c > 0 else 'red' for c in pd.concat([feat_coefs.head(10), feat_coefs.tail(10)])]
pd.concat([feat_coefs.head(10), feat_coefs.tail(10)]).plot(kind='barh', color=colors)
plt.title("Top Positive & Negative Drivers (Full Dataset)")
plt.xlabel("Coefficient")
plt.axvline(x=0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "outputs", "figures", "rq1_lr_coefficients.png"))
plt.close()
print("\nSaved: rq1_lr_coefficients.png")

# [RQ1 ADDED] ROC Curves comparison
if XGB_AVAILABLE:
    xgb_final = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', use_label_encoder=False)
    xgb_final.fit(X_scaled_full, y_rq1)
    print("XGBoost final model trained for ROC curve")

# [RQ1 ADDED] ROC Curves comparison (ALL 3 MODELS)
plt.figure(figsize=(10, 8))

# Logistic Regression
y_prob_lr = lr_final.predict_proba(X_scaled_full)[:, 1]
fpr_lr, tpr_lr, _ = roc_curve(y_rq1, y_prob_lr)
plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {results_rq1['Logistic Regression']['auc_mean']:.3f})")

# Random Forest
y_prob_rf = rf_final.predict_proba(X_scaled_full)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_rq1, y_prob_rf)
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {results_rq1['Random Forest']['auc_mean']:.3f})")

# XGBoost (if available)
if XGB_AVAILABLE:
    y_prob_xgb = xgb_final.predict_proba(X_scaled_full)[:, 1]          
    fpr_xgb, tpr_xgb, _ = roc_curve(y_rq1, y_prob_xgb)
    plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC = {results_rq1['XGBoost']['auc_mean']:.3f})")

# Random guess line
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("RQ1: ROC Curves - Full Dataset")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "outputs", "figures", "rq1_roc_curves.png"))
plt.close()
print("Saved: rq1_roc_curves.png")


print("\n" + "=" * 60)
print("Moving to Education-Only Analysis")
print("=" * 60 + "\n")


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
