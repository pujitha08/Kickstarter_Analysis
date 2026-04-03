"""
Phase 2: Exploratory Data Analysis (EDA)
=========================================
Funding Knowledge: How Education Projects Navigate Kickstarter's Entertainment-First Platform

This script performs platform-wide and education-focused EDA.
All visualizations are saved to outputs/figures/.

Prerequisite: Run phase1_data_preparation.py first.

Input:  data/processed/kickstarter_cleaned.csv
Output: outputs/figures/*.png (all visualizations)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from wordcloud import WordCloud, STOPWORDS

warnings.filterwarnings('ignore')

# ── Paths ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "kickstarter_cleaned.csv")
FIG_DIR = os.path.join(BASE_DIR, "outputs", "figures")

# ── Style ────────────────────────────────────────────────────
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 150

COLORS = {
    'primary': '#2C3E50',
    'success': '#27AE60',
    'fail': '#E74C3C',
    'accent': '#3498DB',
    'orange': '#E67E22',
    'purple': '#8E44AD',
}

fig_count = 0


def save_fig(name):
    """Save current figure to outputs/figures/ and close."""
    global fig_count
    fig_count += 1
    path = os.path.join(FIG_DIR, f"{name}.png")
    plt.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [{fig_count}] Saved: {name}.png")


# ════════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════════
def load_data():
    print("Loading cleaned dataset...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df['launched_at'] = pd.to_datetime(df['launched_at'])

    # Separate education and platform data
    edu = df[(df['is_education'] == 1) & (df['biased_category'] == 0)].copy()
    platform = df[df['biased_category'] == 0].copy()

    print(f"  Platform (clean): {len(platform):,}")
    print(f"  Education (clean): {len(edu):,}")
    return df, platform, edu


# ════════════════════════════════════════════════════════════
# SECTION 1: PLATFORM-WIDE EDA
# ════════════════════════════════════════════════════════════
def platform_eda(df):
    """Platform-wide exploratory charts."""
    print("\n" + "=" * 60)
    print("SECTION 1: PLATFORM-WIDE EDA")
    print("=" * 60)

    # 1. Success rate by broad category (sub_category)
    cat_success = df.groupby('sub_category')['success'].agg(['mean', 'count']).reset_index()
    cat_success.columns = ['category', 'success_rate', 'n']
    cat_success = cat_success.sort_values('success_rate', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(cat_success['category'], cat_success['success_rate'], color=COLORS['accent'])
    ax.set_xlabel('Success Rate')
    ax.set_title('Success Rate by Category (Platform-Wide)')
    ax.set_xlim(0, 1)
    for bar, rate, n in zip(bars, cat_success['success_rate'], cat_success['n']):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{rate:.0%} (n={n:,})', va='center', fontsize=9)
    save_fig("01_success_rate_by_category")

    # 2. Success rate by country (top 10)
    country_counts = df['country'].value_counts()
    valid_countries = country_counts[country_counts >= 50].index
    country_df = df[df['country'].isin(valid_countries)]
    country_success = country_df.groupby('country')['success'].mean().sort_values(ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(10, 5))
    country_success.plot(kind='bar', color=COLORS['accent'], ax=ax)
    ax.set_title('Top 10 Countries by Success Rate')
    ax.set_ylabel('Success Rate')
    ax.set_ylim(0, 1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    save_fig("02_success_rate_by_country")

    # 3. Projects per year
    fig, ax = plt.subplots(figsize=(10, 5))
    df['year'].value_counts().sort_index().plot(kind='bar', color=COLORS['primary'], ax=ax)
    ax.set_title('Number of Projects by Year')
    ax.set_ylabel('Projects')
    ax.set_xlabel('Year')
    save_fig("03_projects_by_year")

    # 4. Success rate per year
    fig, ax = plt.subplots(figsize=(10, 5))
    df.groupby('year')['success'].mean().plot(marker='o', color=COLORS['orange'], linewidth=2, ax=ax)
    ax.set_title('Success Rate by Year')
    ax.set_ylabel('Success Rate')
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    save_fig("04_success_rate_by_year")

    # 5. Projects by month
    fig, ax = plt.subplots(figsize=(10, 5))
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_counts = df['month'].value_counts().sort_index()
    ax.bar(month_names, month_counts.values, color=COLORS['purple'])
    ax.set_title('Projects by Month')
    ax.set_ylabel('Projects')
    save_fig("05_projects_by_month")

    # 6. Success rate by day of week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    fig, ax = plt.subplots(figsize=(10, 5))
    day_success = df.groupby('day_of_week')['success'].mean().reindex(day_order)
    day_success.plot(kind='bar', color=COLORS['success'], ax=ax)
    ax.set_title('Success Rate by Day of Week')
    ax.set_ylabel('Success Rate')
    ax.set_ylim(0, 1)
    ax.set_xticklabels(day_order, rotation=45, ha='right')
    save_fig("06_success_rate_by_day")

    # 7. Creator experience by success
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x='success', y='creator_total_projects', ax=ax,
                palette=[COLORS['fail'], COLORS['success']])
    ax.set_title('Creator Experience by Campaign Outcome')
    ax.set_xticklabels(['Failed', 'Successful'])
    ax.set_ylabel('Total Projects by Creator')
    save_fig("07_creator_experience_boxplot")

    # 8. Goal amount by success (log scale)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x='success', y='goal_usd', ax=ax,
                palette=[COLORS['fail'], COLORS['success']])
    ax.set_yscale('log')
    ax.set_title('Goal Amount by Campaign Outcome (Log Scale)')
    ax.set_xticklabels(['Failed', 'Successful'])
    ax.set_ylabel('Goal (USD, log scale)')
    save_fig("08_goal_amount_boxplot")

    # 9. Blurb word count by success
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(data=df, x='success', y='blurb_word_count', ax=axes[0],
                palette=[COLORS['fail'], COLORS['success']])
    axes[0].set_title('Blurb Word Count by Outcome')
    axes[0].set_xticklabels(['Failed', 'Successful'])

    sns.boxplot(data=df, x='success', y='sentiment_compound', ax=axes[1],
                palette=[COLORS['fail'], COLORS['success']])
    axes[1].set_title('Blurb Sentiment by Outcome')
    axes[1].set_xticklabels(['Failed', 'Successful'])
    plt.tight_layout()
    save_fig("09_blurb_features_boxplot")

    # 10. Funding ratio vs success rate over time
    yearly = df.groupby('year').agg({'funding_ratio': 'mean', 'success': 'mean'}).dropna()
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(yearly.index, yearly['funding_ratio'], marker='o', color=COLORS['accent'],
             linewidth=2, label='Avg Funding Ratio')
    ax1.set_ylabel('Average Funding Ratio', color=COLORS['accent'])
    ax1.axhline(y=1, color=COLORS['accent'], linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(yearly.index, yearly['success'], marker='s', color=COLORS['orange'],
             linewidth=2, label='Success Rate')
    ax2.set_ylabel('Success Rate', color=COLORS['orange'])
    ax1.set_title('Funding Ratio vs Success Rate Over Time')
    fig.tight_layout()
    save_fig("10_funding_ratio_vs_success_time")

    # 11. Goal vs funding ratio scatter
    fig, ax = plt.subplots(figsize=(8, 5))
    sample = df.sample(min(10000, len(df)), random_state=42)
    ax.scatter(sample['goal_usd'], sample['funding_ratio'], alpha=0.2, s=5, color=COLORS['primary'])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Goal Amount (USD, log)')
    ax.set_ylabel('Funding Ratio (log)')
    ax.set_title('Goal Amount vs Funding Ratio')
    ax.axhline(y=1, color=COLORS['fail'], linestyle='--', alpha=0.5, label='100% funded')
    ax.legend()
    save_fig("11_goal_vs_funding_ratio_scatter")

    # 12. Success rate by goal bucket
    goal_success = df.groupby('goal_bucket', observed=True)['success'].agg(['mean', 'count']).reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(goal_success['goal_bucket'].astype(str), goal_success['mean'], color=COLORS['accent'])
    ax.set_title('Success Rate by Goal Amount Range')
    ax.set_ylabel('Success Rate')
    ax.set_xlabel('Goal Range')
    ax.set_ylim(0, 1)
    for bar, rate, n in zip(bars, goal_success['mean'], goal_success['count']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{rate:.0%}\n(n={n:,})', ha='center', fontsize=8)
    plt.xticks(rotation=45, ha='right')
    save_fig("12_success_rate_by_goal_bucket")


# ════════════════════════════════════════════════════════════
# SECTION 2: EDUCATION-FOCUSED EDA
# ════════════════════════════════════════════════════════════
def education_eda(df, edu):
    """Education community specific charts."""
    print("\n" + "=" * 60)
    print("SECTION 2: EDUCATION-FOCUSED EDA")
    print("=" * 60)

    # 13. Education vs platform success rate
    overall_rate = df[df['biased_category'] == 0]['success'].mean()
    edu_rate = edu['success'].mean()

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(['All Projects\n(clean)', 'Education\n(clean)'],
                  [overall_rate, edu_rate],
                  color=[COLORS['accent'], COLORS['orange']])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Success Rate')
    ax.set_title('Success Rate: Education vs Platform Overall')
    for bar, rate in zip(bars, [overall_rate, edu_rate]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{rate:.1%}', ha='center', fontsize=14, fontweight='bold')
    save_fig("13_education_vs_platform_success")

    # 14. Education success rate over time vs platform
    overall_trend = df[df['biased_category'] == 0].groupby('year')['success'].mean()
    edu_trend = edu.groupby('year')['success'].mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(overall_trend.index, overall_trend.values, marker='o', label='All Projects',
            color=COLORS['accent'], linewidth=2)
    ax.plot(edu_trend.index, edu_trend.values, marker='s', label='Education Projects',
            color=COLORS['orange'], linewidth=2)
    ax.set_title('Success Rate Over Time: Education vs Platform')
    ax.set_ylabel('Success Rate')
    ax.set_xlabel('Year')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig("14_education_vs_platform_trend")

    # 15. Education project volume over time
    overall_counts = df[df['biased_category'] == 0].groupby('year').size()
    edu_counts = edu.groupby('year').size()

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(overall_counts.index, overall_counts.values, alpha=0.4, color=COLORS['accent'],
            label='All Projects')
    ax1.set_ylabel('All Projects', color=COLORS['accent'])

    ax2 = ax1.twinx()
    ax2.plot(edu_counts.index, edu_counts.values, marker='o', color=COLORS['orange'],
             linewidth=2, label='Education')
    ax2.set_ylabel('Education Projects', color=COLORS['orange'])
    ax1.set_title('Project Volume: Education vs Platform Over Time')
    ax1.set_xlabel('Year')
    fig.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95))
    save_fig("15_education_volume_over_time")

    # 16. Education success by host category (THE FRAMING EFFECT)
    host_stats = edu.groupby('sub_category')['success'].agg(['mean', 'count']).reset_index()
    host_stats.columns = ['category', 'edu_rate', 'n']
    host_stats = host_stats[host_stats['n'] >= 20].sort_values('edu_rate', ascending=True)

    # Add platform baseline for comparison
    platform_rates = df[df['biased_category'] == 0].groupby('sub_category')['success'].mean()
    host_stats['platform_rate'] = host_stats['category'].map(platform_rates)

    fig, ax = plt.subplots(figsize=(12, 7))
    y_pos = range(len(host_stats))
    ax.barh(y_pos, host_stats['edu_rate'], height=0.4, color=COLORS['orange'],
            label='Education Projects', alpha=0.9)
    ax.barh([y + 0.4 for y in y_pos], host_stats['platform_rate'], height=0.4,
            color=COLORS['accent'], label='Category Average', alpha=0.5)
    ax.set_yticks([y + 0.2 for y in y_pos])
    ax.set_yticklabels(host_stats['category'])
    ax.set_xlabel('Success Rate')
    ax.set_title('The Framing Effect: Education Success Rate by Host Category')
    ax.legend()
    ax.set_xlim(0, 1)

    for y, edu_r, n in zip(y_pos, host_stats['edu_rate'], host_stats['n']):
        ax.text(edu_r + 0.01, y, f'{edu_r:.0%} (n={n})', va='center', fontsize=8)
    save_fig("16_framing_effect_by_category")

    # 17. Education goal distribution: success vs failure
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=edu, x='success', y='goal_usd', ax=ax,
                palette=[COLORS['fail'], COLORS['success']])
    ax.set_yscale('log')
    ax.set_title('Education: Goal Amount by Outcome (Log Scale)')
    ax.set_xticklabels(['Failed', 'Successful'])
    ax.set_ylabel('Goal (USD, log scale)')
    save_fig("17_education_goal_boxplot")

    # 18. Education lexicon impact
    lexicon_cols = [c for c in edu.columns if c.startswith('lex_')]
    lex_impact = []
    for col in lexicon_cols:
        rate_yes = edu[edu[col] == 1]['success'].mean() * 100
        rate_no = edu[edu[col] == 0]['success'].mean() * 100
        n = edu[col].sum()
        lex_impact.append({'lexicon': col.replace('lex_', '').replace('_', ' ').title(),
                          'delta': rate_yes - rate_no, 'n': n})
    lex_df = pd.DataFrame(lex_impact).sort_values('delta')

    fig, ax = plt.subplots(figsize=(10, 6))
    colors_bar = [COLORS['success'] if d > 0 else COLORS['fail'] for d in lex_df['delta']]
    ax.barh(lex_df['lexicon'], lex_df['delta'], color=colors_bar)
    ax.set_xlabel('Δ Success Rate (percentage points)')
    ax.set_title('Education Projects: Impact of Language on Success')
    ax.axvline(x=0, color='black', linewidth=0.5)
    for i, (delta, n) in enumerate(zip(lex_df['delta'], lex_df['n'])):
        ax.text(delta + (0.3 if delta >= 0 else -0.3), i,
                f'{delta:+.1f}pp (n={n:,})', va='center', fontsize=9,
                ha='left' if delta >= 0 else 'right')
    save_fig("18_education_lexicon_impact")

    # 19. Word cloud: Successful education projects
    custom_stopwords = set(STOPWORDS)
    custom_stopwords.update(['project', 'projects', 'campaign', 'will', 'one',
                             'make', 'help', 'create', 'support', 'new', 'us',
                             'first', 'need', 'want', 'world', 'way', 'time',
                             'people', 'life', 'story', 'book', 'video', 'game'])

    success_text = " ".join(edu[edu['success'] == 1]['blurb'].dropna().astype(str))
    if success_text.strip():
        wc = WordCloud(width=1200, height=600, background_color='white',
                       stopwords=custom_stopwords, collocations=False,
                       colormap='Greens').generate(success_text)
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Word Cloud: Successful Education Projects', fontsize=16)
        save_fig("19_wordcloud_education_success")

    # 20. Word cloud: Failed education projects
    failed_text = " ".join(edu[edu['success'] == 0]['blurb'].dropna().astype(str))
    if failed_text.strip():
        wc = WordCloud(width=1200, height=600, background_color='white',
                       stopwords=custom_stopwords, collocations=False,
                       colormap='Reds').generate(failed_text)
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Word Cloud: Failed Education Projects', fontsize=16)
        save_fig("20_wordcloud_education_failed")

    # 21. Education: Sentiment distribution success vs fail
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(edu[edu['success'] == 1]['sentiment_compound'], bins=50, alpha=0.6,
            color=COLORS['success'], label='Successful', density=True)
    ax.hist(edu[edu['success'] == 0]['sentiment_compound'], bins=50, alpha=0.6,
            color=COLORS['fail'], label='Failed', density=True)
    ax.set_title('Education: Sentiment Distribution by Outcome')
    ax.set_xlabel('VADER Compound Sentiment')
    ax.set_ylabel('Density')
    ax.legend()
    save_fig("21_education_sentiment_distribution")

    # 22. Education: Success rate by country
    edu_country = edu.groupby('country')['success'].agg(['mean', 'count']).reset_index()
    edu_country.columns = ['country', 'rate', 'n']
    edu_country = edu_country[edu_country['n'] >= 20].sort_values('rate', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(edu_country['country'], edu_country['rate'], color=COLORS['accent'])
    ax.set_title('Education: Success Rate by Country (min 20 projects)')
    ax.set_ylabel('Success Rate')
    ax.set_ylim(0, 1)
    for bar, rate, n in zip(bars, edu_country['rate'], edu_country['n']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{rate:.0%}\n(n={n})', ha='center', fontsize=8)
    plt.xticks(rotation=45, ha='right')
    save_fig("22_education_success_by_country")

    # 23. Education: Repeat vs first-time creators
    edu_creator = edu.groupby('is_repeat_creator')['success'].mean()
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(['First-Time', 'Repeat Creator'], edu_creator.values,
                  color=[COLORS['accent'], COLORS['success']])
    ax.set_title('Education: Success Rate by Creator Experience')
    ax.set_ylabel('Success Rate')
    ax.set_ylim(0, 1)
    for bar, rate in zip(bars, edu_creator.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{rate:.1%}', ha='center', fontsize=14, fontweight='bold')
    save_fig("23_education_repeat_vs_firsttime")


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
def main():
    print("\n" + "=" * 60)
    print("  PHASE 2: EXPLORATORY DATA ANALYSIS")
    print("  Funding Knowledge: Education on Kickstarter")
    print("=" * 60)

    os.makedirs(FIG_DIR, exist_ok=True)

    df, platform, edu = load_data()
    platform_eda(platform)
    education_eda(platform, edu)

    print(f"\n{'='*60}")
    print(f"  PHASE 2 COMPLETE — {fig_count} figures saved to outputs/figures/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
