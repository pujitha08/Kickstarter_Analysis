"""
Phase 1: Data Preparation & Community Definition
=================================================
Funding Knowledge: How Education Projects Navigate Kickstarter's Entertainment-First Platform

This script performs:
  1. Data cleaning & deduplication
  2. Build education community (Layer 1 + Layer 2)
  3. Feature engineering (structural features)
  4. NLP feature extraction (sentiment, readability, lexicons)
  5. Data validation & quality checks
  6. Auto-generate phase1_report.md

Input:  data/raw/Kickstarter Campaigns DataSet.csv
Output: data/processed/kickstarter_cleaned.csv
        data/processed/kickstarter_education.csv
        outputs/phase1_report.md
"""

import pandas as pd
import numpy as np
import re
import os
import warnings
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings('ignore')

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "Kickstarter Campaigns DataSet.csv")
OUT_CLEAN = os.path.join(BASE_DIR, "data", "processed", "kickstarter_cleaned.csv")
OUT_EDU = os.path.join(BASE_DIR, "data", "processed", "kickstarter_education.csv")
OUT_REPORT = os.path.join(BASE_DIR, "outputs", "phase1_report.md")

# Stats dictionary populated by each step, consumed by report generator
stats = {}

LEXICONS = {
    'lex_professional':   r'\bprofession\w*\b|\bexpert\b|\bexperienc\w*\b|\baward\b|\byears?\b|\bdecade\b',
    'lex_innovation':     r'\binnovat\w*\b|\bnew\b|\bfirst\b|\brevolut\w*\b|\bunique\b|\bgroundbreak\w*\b',
    'lex_community':      r'\bcommunit\w*\b|\btogether\b|\bneighbor\w*\b|\bempower\w*\b|\bimpact\b|\bsocial\b',
    'lex_help_please':    r'\bhelp\b|\bsupport\b|\bneed\b|\bplease\b|\bfund\b|\bdonat\w*\b',
    'lex_passion_dream':  r'\bpassion\w*\b|\bdream\b|\blove\b|\bheart\b|\bsoul\b|\bdedicat\w*\b',
    'lex_digital_online': r'\bdigital\b|\bonline\b|\bapp\b|\bweb\b|\bplatform\b|\bvirtual\b|\bsoftware\b',
    'lex_interactive':    r'\binteract\w*\b|\bhands.?on\b|\bengage\w*\b|\bimmersive\b|\bplay\b',
    'lex_children_kids':  r'\bchild\w*\b|\bkid\w*\b|\byoung\b|\byouth\b|\bboy\b|\bgirl\b|\btoddler\b',
}


# ════════════════════════════════════════════════════════════
# STEP 1: DATA CLEANING & DEDUPLICATION
# ════════════════════════════════════════════════════════════
def load_and_clean(path):
    print("=" * 60)
    print("STEP 1: DATA CLEANING & DEDUPLICATION")
    print("=" * 60)

    df = pd.read_csv(path, low_memory=False)
    stats['raw_rows'], stats['raw_cols'] = df.shape[0], df.shape[1]
    print(f"Raw dataset: {stats['raw_rows']:,} rows x {stats['raw_cols']} columns")

    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)

    # Deduplicate
    stats['n_dupes'] = int(df.duplicated(subset='id').sum())
    df = df.drop_duplicates(subset='id', keep='first').copy()
    stats['after_dedup'] = df.shape[0]
    print(f"Duplicates: {stats['n_dupes']:,} removed -> {stats['after_dedup']:,} unique projects")

    # Parse dates
    df['launched_at'] = pd.to_datetime(df['launched_at'], errors='coerce')
    df['deadline'] = pd.to_datetime(df['deadline'], errors='coerce')

    # Missing values
    print("\nMissing value treatment:")
    nulls = df.isnull().sum()
    has_nulls = nulls[nulls > 0]
    stats['null_cols'] = len(has_nulls)
    stats['null_details'] = {col: int(n) for col, n in has_nulls.items()}

    if stats['null_cols'] == 0:
        print("  Null values: none across all columns")
    else:
        for col, n in has_nulls.items():
            print(f"  {col}: {n:,} nulls")
        for col in ['blurb', 'name', 'city']:
            if df[col].isnull().any():
                df[col] = df[col].fillna('')
        for col in ['goal_usd', 'duration', 'usd_pledged', 'backers_count']:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        if df['launched_at'].isnull().any():
            n_bad = df['launched_at'].isnull().sum()
            df = df.dropna(subset=['launched_at'])
            print(f"  Dropped {n_bad} rows with unparseable dates")

    # Empty strings
    stats['empty_strings'] = {}
    for col in ['name', 'blurb', 'city', 'country']:
        n_empty = (df[col].fillna('').str.strip() == '').sum()
        if n_empty > 0:
            stats['empty_strings'][col] = n_empty
    if stats['empty_strings']:
        for col, n in stats['empty_strings'].items():
            print(f"  {col}: {n:,} empty strings")
    else:
        print("  Empty strings: none")

    # Edge cases
    stats['goal_lt_1'] = int((df['goal_usd'] < 1).sum())
    stats['duration_gt_90'] = int((df['duration'] > 90).sum())
    stats['zero_backers'] = int((df['backers_count'] == 0).sum())
    print(f"\nEdge cases: goal<$1={stats['goal_lt_1']}, dur>90d={stats['duration_gt_90']}, 0 backers={stats['zero_backers']:,}")

    # Outlier capping
    goal_cap = df['goal_usd'].quantile(0.995)
    stats['goal_cap'] = float(goal_cap)
    stats['n_goal_capped'] = int((df['goal_usd'] > goal_cap).sum())
    df['goal_usd_raw'] = df['goal_usd']
    df['goal_usd'] = df['goal_usd'].clip(upper=goal_cap)

    stats['n_dur_capped'] = int((df['duration'] > 92).sum())
    df['duration'] = df['duration'].clip(upper=92)
    print(f"Outlier capping: {stats['n_goal_capped']:,} goals capped at ${goal_cap:,.0f}, {stats['n_dur_capped']} durations capped at 92d")

    # Filter to completed
    stats['status_dist'] = df['status'].value_counts().to_dict()
    df = df[df['status'].isin(['successful', 'failed'])].copy()
    df['success'] = (df['status'] == 'successful').astype(int)

    stats['n_completed'] = len(df)
    stats['n_success'] = int(df['success'].sum())
    stats['n_failed'] = int((df['success'] == 0).sum())
    stats['success_rate'] = df['success'].mean() * 100

    # Funding ratio
    df['funding_ratio'] = df['usd_pledged'] / df['goal_usd'].replace(0, np.nan)
    stats['funding_ratio_nan'] = int(df['funding_ratio'].isnull().sum())

    # Biased categories
    cat_agg = df.groupby('main_category').agg(n=('id', 'count'), n_fail=('success', lambda x: (x == 0).sum())).reset_index()
    biased_cats = cat_agg[(cat_agg['n'] > 50) & (cat_agg['n_fail'] == 0)]['main_category'].tolist()
    df['biased_category'] = df['main_category'].isin(biased_cats).astype(int)
    stats['n_biased_cats'] = len(biased_cats)
    stats['n_biased_projects'] = int(df['biased_category'].sum())
    stats['biased_pct'] = df['biased_category'].mean() * 100

    print(f"Completed: {stats['n_completed']:,} ({stats['success_rate']:.1f}% success)")
    print(f"Biased categories: {stats['n_biased_cats']} ({stats['n_biased_projects']:,} projects)")
    print("Step 1 complete\n")
    return df, biased_cats


# ════════════════════════════════════════════════════════════
# STEP 2: BUILD EDUCATION COMMUNITY
# ════════════════════════════════════════════════════════════
def build_education_community(df, biased_cats):
    print("=" * 60)
    print("STEP 2: BUILD EDUCATION COMMUNITY")
    print("=" * 60)

    edu_categories = ['Academic', "Children's Books", 'Workshops', 'Kids']
    layer1_mask = df['main_category'].isin(edu_categories)
    layer1_ids = set(df.loc[layer1_mask, 'id'])

    stats['edu_cats'] = {}
    for cat in edu_categories:
        n = int((df['main_category'] == cat).sum())
        parent = df[df['main_category'] == cat]['sub_category'].iloc[0] if n > 0 else ''
        stats['edu_cats'][cat] = {'n': n, 'biased': cat in biased_cats, 'parent': parent}
    stats['layer1_total'] = len(layer1_ids)

    edu_keywords = (r'\beducat\w*\b|\bschool\b|\bclassroom\b|\bstudent\b|'
                    r'\bteach\w*\b|\blearn(?:ing)?\b|\bcurriculum\b|'
                    r'\btutorial\b|\btextbook\b|\blesson\b|\bcourse\b|'
                    r'\btraining\b|\bliteracy\b|\bSTEM\b')
    layer2_mask = df['blurb'].fillna('').str.lower().str.contains(edu_keywords, regex=True)
    layer2_ids = set(df.loc[layer2_mask, 'id'])

    both_ids = layer1_ids & layer2_ids
    layer1_only = layer1_ids - layer2_ids
    layer2_only = layer2_ids - layer1_ids

    stats['layer2_total'] = len(layer2_ids)
    stats['layer2_new'] = len(layer2_only)
    stats['overlap'] = len(both_ids)

    all_edu_ids = layer1_ids | layer2_ids
    df['is_education'] = df['id'].isin(all_edu_ids).astype(int)
    df['edu_source'] = df['id'].apply(
        lambda x: 'Both' if x in both_ids
        else ('Category' if x in layer1_only
              else ('Keyword' if x in layer2_only else 'Non-Education')))

    stats['edu_total'] = int(df['is_education'].sum())
    stats['edu_cat_only'] = int((df['edu_source'] == 'Category').sum())
    stats['edu_kw_only'] = int((df['edu_source'] == 'Keyword').sum())
    stats['edu_both'] = int((df['edu_source'] == 'Both').sum())
    stats['edu_biased'] = int(df[(df['is_education'] == 1) & (df['biased_category'] == 1)].shape[0])
    stats['edu_clean'] = stats['edu_total'] - stats['edu_biased']

    edu_clean = df[(df['is_education'] == 1) & (df['biased_category'] == 0)]
    stats['edu_clean_success_rate'] = edu_clean['success'].mean() * 100
    stats['edu_clean_success'] = int(edu_clean['success'].sum())
    stats['edu_clean_failed'] = int((edu_clean['success'] == 0).sum())
    stats['edu_countries'] = int(edu_clean['country'].nunique())
    stats['edu_n_categories'] = int(edu_clean['sub_category'].nunique())

    stats['edu_host_cats'] = {}
    for cat in edu_clean['sub_category'].value_counts().index:
        sub = edu_clean[edu_clean['sub_category'] == cat]
        stats['edu_host_cats'][cat] = {'n': len(sub), 'success_rate': sub['success'].mean() * 100}

    print(f"Layer 1: {stats['layer1_total']:,} | Layer 2 new: {stats['layer2_new']:,} | Combined: {stats['edu_total']:,} (clean: {stats['edu_clean']:,})")
    print("Step 2 complete\n")
    return df


# ════════════════════════════════════════════════════════════
# STEP 3: FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════
def engineer_structural_features(df):
    print("=" * 60)
    print("STEP 3: FEATURE ENGINEERING (STRUCTURAL)")
    print("=" * 60)

    df['year'] = df['launched_at'].dt.year
    df['month'] = df['launched_at'].dt.month
    df['day_of_week'] = df['launched_at'].dt.day_name()
    df['launch_hour'] = df['launched_at'].dt.hour
    df['log_goal'] = np.log1p(df['goal_usd'])
    df['goal_bucket'] = pd.cut(df['goal_usd'],
        bins=[0, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000, float('inf')],
        labels=['<$500','$500-1K','$1K-2.5K','$2.5K-5K','$5K-10K','$10K-25K','$25K-50K','$50K-100K','$100K+'])

    blurbs = df['blurb'].fillna('')
    df['blurb_word_count'] = blurbs.apply(lambda x: len(x.split()))
    df['blurb_char_count'] = blurbs.str.len()
    df['has_exclamation'] = blurbs.str.contains('!').astype(int)
    df['has_question'] = blurbs.str.contains(r'\?').astype(int)
    df['has_numbers'] = blurbs.str.contains(r'\d').astype(int)
    df['n_exclamation'] = blurbs.str.count('!')

    cc = df.groupby('creator_id')['id'].count()
    df['creator_total_projects'] = df['creator_id'].map(cc)
    df['is_repeat_creator'] = (df['creator_total_projects'] >= 2).astype(int)

    top_countries = df['country'].value_counts().head(5).index.tolist()
    df['country_group'] = df['country'].apply(lambda x: x if x in top_countries else 'Other')
    df['category_avg_success'] = df['sub_category'].map(df.groupby('sub_category')['success'].mean())

    stats['year_range'] = f"{int(df['year'].min())}-{int(df['year'].max())}"
    stats['top_countries'] = top_countries
    stats['median_goal'] = float(df['goal_usd'].median())
    stats['n_struct_features'] = 16

    print(f"16 structural features created")
    print("Step 3 complete\n")
    return df


# ════════════════════════════════════════════════════════════
# STEP 4: NLP FEATURES
# ════════════════════════════════════════════════════════════
def _count_syllables(word):
    word = word.lower().strip()
    if not word: return 0
    if word.endswith('e') and len(word) > 2: word = word[:-1]
    count, prev = 0, False
    for c in word:
        v = c in 'aeiou'
        if v and not prev: count += 1
        prev = v
    return max(count, 1)

def _flesch_re(text):
    if not text or len(text.strip()) < 10: return 0.0
    s = max(len(re.split(r'[.!?]+', text.strip())), 1)
    w = text.split(); nw = max(len(w), 1)
    ns = sum(_count_syllables(x) for x in w)
    return round(206.835 - 1.015*(nw/s) - 84.6*(ns/nw), 2)

def _flesch_grade(text):
    if not text or len(text.strip()) < 10: return 0.0
    s = max(len(re.split(r'[.!?]+', text.strip())), 1)
    w = text.split(); nw = max(len(w), 1)
    ns = sum(_count_syllables(x) for x in w)
    return round(0.39*(nw/s) + 11.8*(ns/nw) - 15.59, 2)

def extract_nlp_features(df):
    print("=" * 60)
    print("STEP 4: NLP FEATURE EXTRACTION")
    print("=" * 60)

    blurbs = df['blurb'].fillna('')
    blurb_lower = blurbs.str.lower()

    print("  VADER sentiment...")
    analyzer = SentimentIntensityAnalyzer()
    sc = blurbs.apply(lambda x: analyzer.polarity_scores(x))
    df['sentiment_compound'] = sc.apply(lambda x: x['compound'])
    df['sentiment_pos'] = sc.apply(lambda x: x['pos'])
    df['sentiment_neg'] = sc.apply(lambda x: x['neg'])
    stats['sentiment_mean'] = float(df['sentiment_compound'].mean())
    stats['sentiment_success'] = float(df[df['success']==1]['sentiment_compound'].mean())
    stats['sentiment_failed'] = float(df[df['success']==0]['sentiment_compound'].mean())

    print("  Readability...")
    df['readability_fre'] = blurbs.apply(_flesch_re)
    df['readability_grade'] = blurbs.apply(_flesch_grade)
    stats['fre_mean'] = float(df['readability_fre'].mean())
    stats['fre_success'] = float(df[df['success']==1]['readability_fre'].mean())
    stats['fre_failed'] = float(df[df['success']==0]['readability_fre'].mean())
    stats['grade_mean'] = float(df['readability_grade'].mean())

    print("  Custom lexicons...")
    stats['lexicon_results'] = {}
    for name, pattern in LEXICONS.items():
        df[name] = blurb_lower.str.contains(pattern, regex=True).astype(int)
        ry = df[df[name]==1]['success'].mean()*100
        rn = df[df[name]==0]['success'].mean()*100
        nm = int(df[name].sum())
        stats['lexicon_results'][name] = {'n': nm, 'rate_yes': ry, 'rate_no': rn, 'delta': ry-rn}
        print(f"    {name:<25s} n={nm:>6,}  d={ry-rn:+.1f}pp")

    stats['n_nlp_features'] = 13
    print("13 NLP features created")
    print("Step 4 complete\n")
    return df


# ════════════════════════════════════════════════════════════
# STEP 5: VALIDATION
# ════════════════════════════════════════════════════════════
def validate(df):
    print("=" * 60)
    print("STEP 5: VALIDATION")
    print("=" * 60)

    stats['final_rows'] = len(df)
    stats['final_cols'] = df.shape[1]
    stats['final_nulls'] = int(df.isnull().sum().sum())

    edu_clean = df[(df['is_education']==1) & (df['biased_category']==0)]
    stats['final_edu_success'] = int(edu_clean['success'].sum())
    stats['final_edu_failed'] = int((edu_clean['success']==0).sum())
    stats['final_edu_balance'] = edu_clean['success'].mean() * 100

    ok = stats['final_nulls'] == 0
    print(f"  {stats['final_rows']:,} rows x {stats['final_cols']} cols | Nulls: {stats['final_nulls']}")
    print(f"  Education (clean): {stats['final_edu_success']:,} success / {stats['final_edu_failed']:,} failed ({stats['final_edu_balance']:.1f}%)")
    print(f"  {'ALL CHECKS PASSED' if ok else 'ISSUES FOUND'}\n")
    return ok


# ════════════════════════════════════════════════════════════
# STEP 6: GENERATE REPORT
# ════════════════════════════════════════════════════════════
def generate_report(output_path):
    print("=" * 60)
    print("STEP 6: GENERATING REPORT")
    print("=" * 60)

    s = stats
    today = datetime.now().strftime("%B %d, %Y")

    lex_rows = ""
    for name, r in sorted(s['lexicon_results'].items(), key=lambda x: -x[1]['delta']):
        label = name.replace('lex_', '').replace('_', ' ').title()
        lex_rows += f"| `{name}` | {label} | {r['n']:,} | {r['rate_yes']:.1f}% | {r['rate_no']:.1f}% | **{r['delta']:+.1f}pp** |\n"

    host_rows = ""
    for cat, info in sorted(s['edu_host_cats'].items(), key=lambda x: -x[1]['n']):
        pct = info['n'] / s['edu_clean'] * 100
        host_rows += f"| {cat} | {info['n']:,} | {pct:.1f}% | {info['success_rate']:.1f}% |\n"

    edu_cat_rows = ""
    for cat, info in s['edu_cats'].items():
        flag = "Biased" if info['biased'] else "Clean"
        edu_cat_rows += f"| {cat} | {info['parent']} | {info['n']:,} | {flag} |\n"

    report = f"""# Phase 1 Report: Data Preparation & Community Definition

**Project:** Funding Knowledge — How Education Projects Navigate Kickstarter  
**Generated:** {today} (auto-generated by `src/phase1_data_preparation.py`)

---

## 1. Data Cleaning & Deduplication

| Metric | Value |
|---|---|
| Raw dataset | {s['raw_rows']:,} rows x {s['raw_cols']} columns |
| Duplicates removed | {s['n_dupes']:,} ({s['n_dupes']/s['raw_rows']*100:.1f}%) |
| After deduplication | {s['after_dedup']:,} unique projects |
| Completed projects | {s['n_completed']:,} ({s['success_rate']:.1f}% successful) |

### Missing Value Treatment

| Check | Result | Action |
|---|---|---|
| Null values | {s['null_cols']} columns with nulls | {'No imputation needed' if s['null_cols']==0 else 'Text->empty, numeric->median'} |
| Empty strings | {len(s['empty_strings'])} columns affected | {'No treatment needed' if not s['empty_strings'] else 'Documented'} |

The script includes defensive fallback logic for nulls (text->empty string, numeric->median, bad dates->drop) that activates automatically if the dataset changes.

### Edge Cases & Outlier Treatment

| Item | Count | Action |
|---|---|---|
| `goal_usd` < $1 | {s['goal_lt_1']} | Kept (symbolic campaigns) |
| `goal_usd` outliers | {s['n_goal_capped']:,} | Capped at 99.5th percentile (${s['goal_cap']:,.0f}). Raw preserved in `goal_usd_raw` |
| `duration` > 90 days | {s['duration_gt_90']} | Capped at 92 days |
| `backers_count` = 0 | {s['zero_backers']:,} | Kept (all failed, valid data) |
| `funding_ratio` NaN | {s['funding_ratio_nan']} | From $0 goals (protected) |

### Biased Categories
{s['n_biased_cats']} `main_category` values contain only successful projects (data artifact), affecting {s['n_biased_projects']:,} projects ({s['biased_pct']:.1f}%). Flagged with `biased_category = 1`.

---

## 2. Education Community Definition

### Layer 1: Category-Based ({s['layer1_total']:,} projects)

| Category | Parent | Count | Status |
|---|---|---|---|
{edu_cat_rows}
### Layer 2: Keyword-Based ({s['layer2_new']:,} additional projects)

Keywords searched in blurbs: `educate, school, classroom, student, teach, learn, curriculum, tutorial, textbook, lesson, course, training, literacy, STEM`

Overlap with Layer 1: {s['overlap']:,} projects

### Combined Education Ecosystem

| Segment | Count | % |
|---|---|---|
| Category only | {s['edu_cat_only']:,} | {s['edu_cat_only']/s['edu_total']*100:.1f}% |
| Keyword only | {s['edu_kw_only']:,} | {s['edu_kw_only']/s['edu_total']*100:.1f}% |
| Both | {s['edu_both']:,} | {s['edu_both']/s['edu_total']*100:.1f}% |
| **Total** | **{s['edu_total']:,}** | |
| Biased (flagged) | {s['edu_biased']:,} | |
| **Clean** | **{s['edu_clean']:,}** | **{s['edu_clean_success_rate']:.1f}% success** |

### Where Education Lives on Kickstarter

| Host Category | Projects | Share | Success Rate |
|---|---|---|---|
{host_rows}
---

## 3. Feature Engineering

**{s['n_struct_features']} structural features:** year, month, day_of_week, launch_hour, log_goal, goal_bucket, blurb_word_count, blurb_char_count, has_exclamation, has_question, has_numbers, n_exclamation, creator_total_projects, is_repeat_creator, country_group, category_avg_success

**{s['n_nlp_features']} NLP features:** sentiment_compound, sentiment_pos, sentiment_neg, readability_fre, readability_grade, and 8 custom lexicon flags

---

## 4. Early NLP Findings

### Sentiment
| Group | Mean Compound |
|---|---|
| Successful | {s['sentiment_success']:.3f} |
| Failed | {s['sentiment_failed']:.3f} |
| *Overall* | *{s['sentiment_mean']:.3f}* |

Failed projects are slightly *more* positive in tone than successful ones.

### Readability
Mean Flesch Reading Ease: {s['fre_mean']:.1f} | Successful: {s['fre_success']:.1f} | Failed: {s['fre_failed']:.1f}. Minimal difference.

### Lexicon Analysis

| Lexicon | Theme | Matches | Success (yes) | Success (no) | Delta |
|---|---|---|---|---|---|
{lex_rows}
**Key insight:** Innovation language helps most, digital/online language hurts dramatically. Passion and community framing underperform.

---

## 5. Output Files

| File | Rows | Columns |
|---|---|---|
| `data/processed/kickstarter_cleaned.csv` | {s['final_rows']:,} | {s['final_cols']} |
| `data/processed/kickstarter_education.csv` | {s['edu_total']:,} | {s['final_cols']} |
| `outputs/phase1_report.md` | - | auto-generated |

---

## 6. Notes for Downstream Phases

1. Filter `biased_category == 0` when modeling at `main_category` level
2. Column naming: `sub_category` = broad (food, tech), `main_category` = detailed (Food Trucks, Academic)
3. Children's Books is both education AND biased — exclude from modeling
4. `goal_usd` capped at ${s['goal_cap']:,.0f}; original in `goal_usd_raw`
5. Class balance (education, clean): {s['final_edu_success']:,} success / {s['final_edu_failed']:,} failed ({s['final_edu_balance']:.1f}%)
"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)
    print(f"Report saved: {output_path}\n")


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
def main():
    print("\n" + "=" * 60)
    print("  PHASE 1: DATA PREPARATION & COMMUNITY DEFINITION")
    print("  Funding Knowledge: Education on Kickstarter")
    print("=" * 60 + "\n")

    df, biased_cats = load_and_clean(RAW_PATH)
    df = build_education_community(df, biased_cats)
    df = engineer_structural_features(df)
    df = extract_nlp_features(df)
    validate(df)

    os.makedirs(os.path.dirname(OUT_CLEAN), exist_ok=True)
    df.to_csv(OUT_CLEAN, index=False)
    df[df['is_education'] == 1].to_csv(OUT_EDU, index=False)
    print(f"Saved: {OUT_CLEAN} ({len(df):,} x {df.shape[1]})")
    print(f"Saved: {OUT_EDU} ({df['is_education'].sum():,} rows)")

    generate_report(OUT_REPORT)

    print("=" * 60)
    print("  PHASE 1 COMPLETE - MILESTONE: Clean dataset ready")
    print("=" * 60)


if __name__ == "__main__":
    main()
