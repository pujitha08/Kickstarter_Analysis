# Funding Knowledge: Education on Kickstarter

Exploring why education-focused crowdfunding campaigns struggle on a platform built for entertainment — and what creators can do about it. We analyze project descriptions, funding patterns, and creator strategies to uncover what separates the successes from the silence.

## The Story

Kickstarter was built for creative projects — films, albums, games, art. But thousands of creators come to the platform each year trying to fund **knowledge**: textbooks, coding tools, language courses, STEM toys, science documentaries. They're trying to educate, not entertain — and the platform often punishes them for it.

Our signature finding is the **framing effect**: identical educational content succeeds or fails at dramatically different rates depending on which Kickstarter category it's filed under. Education projects thrive when framed as entertainment but struggle when labeled as academic.

## Research Questions

1. **RQ1:** What project characteristics best predict whether an education-focused Kickstarter campaign succeeds?
2. **RQ2:** How does the language of education campaign descriptions relate to campaign outcomes?
3. **RQ3:** Does the category a creator chooses to host their education project shape its likelihood of success?
4. **RQ4:** What temporal and geographic patterns exist in education campaign launches and outcomes?
5. **RQ5:** What distinguishes education projects that dramatically exceed their funding goals from those that fail in silence?

## Defining the Education Community

Education isn't a single Kickstarter category — it's a mission scattered across the entire platform. We define our community using two layers:

- **Layer 1 (Category-based):** Projects in explicitly educational categories — Academic, Children's Books, Workshops, Kids
- **Layer 2 (Keyword-based):** Projects from any category whose description signals educational intent (e.g., "learn," "teach," "student," "STEM," "curriculum")

This two-layer approach captures ~11,000 education-themed projects across 15 Kickstarter categories and 23 countries.

## Project Structure

```
├── data/
│   ├── raw/                          ← Original Kickstarter dataset (not tracked in git)
│   └── processed/                    ← Cleaned & feature-engineered datasets
├── src/
│   ├── phase1_data_preparation.py    ← Data cleaning, community definition, feature engineering
│   ├── phase2_eda.py                 ← Exploratory data analysis & visualizations
│   ├── phase3_predictive_modeling.py ← Classification, regression
│   │── phase4_shapanalysis.py.       -  SHAP explainability
├── notebooks/                        ← Jupyter notebooks for exploration (optional)
├── outputs/
│   ├── figures/                      ← Plots and charts
│   └── tables/                       ← Summary tables
├── requirements.txt
└── README.md
```

## Dataset

- **Source:** [Kickstarter Campaigns Dataset 2.0](https://www.kaggle.com/datasets/yashkantharia/kickstarter-campaigns-dataset-20) (Kaggle)
- **Raw size:** 217,245 rows × 19 columns
- **After deduplication:** 192,888 unique projects
- **Completed projects (success/fail):** 180,675
- **Education community:** ~11,000 projects

## Methods

- **EDA:** Geographic, temporal, and category-level analysis with emphasis on the framing effect
- **NLP:** Sentiment analysis (VADER), readability scoring, custom lexicon matching, topic modeling
- **Modeling:** Logistic Regression, Random Forest, XGBoost with SHAP explainability
- **Continuous outcome:** Funding ratio regression (pledged / goal)

## Setup

```bash
git clone https://github.com/pujitha08/Kickstarter_Analysis.git
cd Kickstarter_Analysis
pip install -r requirements.txt
```

Place the raw dataset (`Kickstarter Campaigns DataSet.csv`) in `data/raw/`, then run the phases in order:

```bash
python src/phase1_data_preparation.py    # Clean data, define community, engineer features
python src/phase2_eda.py                 # Generate EDA visualizations
python src/phase3_predictive_modeling.py  # Train models, SHAP analysis
python src/phase4_story_narrative.py      # Education community deep-dive
```

## Team

DSBA 6211 — Spring 2026
