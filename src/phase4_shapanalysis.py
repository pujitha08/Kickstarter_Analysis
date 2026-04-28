"""
PHASE 4: SHAP Explainability Analysis
--------------------------------------
Fits an XGBoost classifier on the clean education subset and uses SHAP
(SHapley Additive exPlanations) to explain which features drive success
predictions for education campaigns.

Adapted to repo conventions:
  - Uses data/processed/kickstarter_cleaned.csv (Phase 1 output)
  - Filters to clean education projects (is_education==1, biased_category==0)
  - Replaces goal_usd with log_goal 
  - Replaces blurb_length with blurb_word_count 
  - Adds scale_pos_weight to handle class imbalance
  - Saves all figures to outputs/figures/

NOTE: The XGBoost model here is fit on the FULL clean education dataset
(no train/test split). This is intentional — SHAP is used for feature
interpretation, not evaluation. AUC scores come from Phase 3 CV results.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import xgboost as xgb

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "kickstarter_cleaned.csv")
FIG_DIR   = os.path.join(BASE_DIR, "outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Load & filter ──────────────────────────────────────────────────────────────
print("=" * 60)
print("  PHASE 4: SHAP EXPLAINABILITY ANALYSIS")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df):,} rows")

# Same filter as Phase 3 — clean education projects only
edu = df[(df['is_education'] == 1) & (df['biased_category'] == 0)].copy()
edu = edu.reset_index(drop=True)
print(f"Education projects (clean): {len(edu):,}")
print(f"Success rate: {edu['success'].mean():.1%}")

# ── Features ───────────────────────────────────────────────────────────────────
# log_goal replaces goal_usd (consistent with Phase 3, log scale handles skew)
# blurb_word_count replaces blurb_length (consistent with Phase 3)
features = [
    'log_goal',
    'duration',
    'blurb_word_count',
    'year',
    'month',
    'launch_hour',
    'creator_total_projects',
    'is_repeat_creator',
    'category_avg_success',
    'sentiment_compound',
    'readability_fre',
    'lex_professional',
    'lex_innovation',
    'lex_community',
    'lex_passion_dream',
    'lex_digital_online',
    'lex_interactive',
    'lex_children_kids',
    'lex_help_please',
]

X = edu[features]
y = edu['success']

print(f"\nFeatures: {len(features)}")
print(f"Class distribution: {y.value_counts().to_dict()}")

# ── XGBoost model ─────────────────────────────────────────────────────────────
# Fit on full education dataset — for interpretation only, not evaluation.
# scale_pos_weight handles class imbalance (failed > successful in education).
pos = int((y == 0).sum())   # failed
neg = int((y == 1).sum())   # successful
spw = pos / neg if neg > 0 else 1.0

print(f"\nFitting XGBoost (scale_pos_weight={spw:.2f})...")

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    scale_pos_weight=spw,
    random_state=42,
    verbosity=0,
    eval_metric='auc',
    use_label_encoder=False,
)
model.fit(X, y)
print("Model fit complete.")

# ── SHAP values ───────────────────────────────────────────────────────────────
print("\nComputing SHAP values...")
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# ── Plot 1: SHAP Summary (beeswarm) ───────────────────────────────────────────
print("\nGenerating SHAP summary plot...")
plt.figure(figsize=(10, 7))
shap.summary_plot(
    shap_values, X,
    feature_names=features,
    show=False,
)
plt.title("SHAP Feature Importance — Education Campaigns", fontsize=13, pad=15)
plt.tight_layout()
summary_path = os.path.join(FIG_DIR, "phase4_shap_summary.png")
plt.savefig(summary_path, bbox_inches='tight', facecolor='white', dpi=150)
plt.close()
print(f"  Saved → {summary_path}")

# ── Plot 2: Mean |SHAP| bar chart (top 15) ────────────────────────────────────
print("\nGenerating SHAP bar chart...")
mean_shap = pd.DataFrame({
    'feature':       features,
    'mean_abs_shap': np.abs(shap_values).mean(axis=0)
}).sort_values('mean_abs_shap', ascending=False)

top15 = mean_shap.head(15)

nlp_features = {
    'sentiment_compound', 'readability_fre', 'blurb_word_count',
    'lex_professional', 'lex_innovation', 'lex_community',
    'lex_passion_dream', 'lex_digital_online', 'lex_interactive',
    'lex_children_kids', 'lex_help_please',
}
colors = ['#27AE60' if f in nlp_features else '#2E75B6'
          for f in top15['feature']]

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(
    top15['feature'][::-1],
    top15['mean_abs_shap'][::-1],
    color=colors[::-1]
)
ax.set_xlabel('Mean |SHAP value|', fontsize=11)
ax.set_title('Top 15 Features by SHAP Importance — Education Campaigns',
             fontsize=13)

from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color='#27AE60', label='NLP / text feature'),
    Patch(color='#2E75B6', label='Structural feature'),
], loc='lower right', fontsize=10)

plt.tight_layout()
bar_path = os.path.join(FIG_DIR, "phase4_shap_top15_bar.png")
plt.savefig(bar_path, bbox_inches='tight', facecolor='white', dpi=150)
plt.close()
print(f"  Saved → {bar_path}")

# ── Top features summary ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TOP 10 FEATURES BY MEAN |SHAP|")
print("=" * 60)
print(mean_shap.head(10).to_string(index=False))

print("\n" + "=" * 60)
print("PHASE 4 COMPLETE")
print("=" * 60)
print(f"  → {summary_path}")
print(f"  → {bar_path}")