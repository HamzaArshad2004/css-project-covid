# UAE COVID-19 Behavioral Analysis
### Capstone Project — Cross-Domain Mobility × Emotion Discovery via Formal Concept Analysis

This project investigates how **public mobility patterns** and **online emotional sentiment** co-evolved during the UAE COVID-19 pandemic (March 2020 – December 2021). Using Formal Concept Analysis (FCA) and LLM-assisted rule evaluation, it surfaces non-obvious, policy-relevant associations between physical behavior and public discourse.

---

## Research Question

> *Do changes in UAE population mobility during COVID-19 reliably co-occur with measurable shifts in public emotional sentiment — and if so, which cross-domain patterns are most policy-actionable?*

---

## Pipeline Overview

```
Raw Data                  Processing               Analysis              Output
─────────                 ──────────               ────────              ──────
Google Mobility  ──►  preprocess_and_features  ──►  fca_analysis    ──►  association_rules.csv
Reddit Posts     ──►       (binary matrix)     ──►  (FCA lattice)   ──►  formal_concepts.csv
                                                         │
                                               evaluate_rules_llm   ──►  top_cross_domain_rules.txt
                                                         │
                                               visualize_results    ──►  summary_report.txt
                                                                         charts/
```

### Step 1 — Data Collection
| Source | Script | Output |
|--------|--------|--------|
| Google Community Mobility Reports (UAE national) | `collect_mobility_data.py` → `process_global_mobility_uae.py` | `mobility_data_processed.csv` |
| Reddit (UAE + COVID subreddits, 2020–2021) | `collect_reddit_data.py` | `reddit_wildfire_posts_by_day.json` |

Reddit collection uses a multi-subreddit, multi-query, multi-sort strategy across UAE-local (`r/UAE`, `r/dubai`, `r/abudhabi`) and global subreddits (`r/Coronavirus`, `r/worldnews`) to maximise post coverage. Posts are deduplicated by ID.

### Step 2 — Feature Engineering (`preprocess_and_features.py`)
671 daily observations are converted into a **29-column binary matrix** for FCA:

**Mobility features** (from Google Mobility % change vs. baseline):
- `mobility_drop_retail` — retail & recreation < −25 %
- `mobility_drop_transit` — transit stations < −30 %
- `mobility_drop_workplace` — workplaces < −25 %
- `residential_increase` — residential > +8 %
- `grocery_spike` — grocery & pharmacy > +10 %
- `severe_lockdown_behavior` — retail drop ∧ workplace drop ∧ residential increase
- `partial_restrictions` — one or more drops, but not severe lockdown

**Social/Emotion features** (from Reddit posts via VADER sentiment):
- Sentiment: `high_negative_sentiment`, `high_positive_sentiment`, `sentiment_improved`, `sentiment_worsened`, `sentiment_shift_detected`, `dominant_emotion_fear`, `mixed_emotions`
- Keywords: `fear_keywords_present`, `anger_mentioned`, `anxiety_keywords_present`, `sadness_keywords_present`, `solidarity_messages`
- Topics: `covid_topic_detected`, `lockdown_mentioned`, `vaccine_mentioned`, `health_concern`, `compliance_discussed`, `policy_governance_discussion`
- Composite: `emotion_with_mobility_signal`, `emotion_mobility_mismatch`, `low_emotion_low_mobility_signal`

### Step 3 — Formal Concept Analysis (`fca_analysis.py`)
- Builds a **Galois lattice** (formal context) over the binary matrix
- To prevent combinatorial explosion, objects > 150 are **aggregated to weekly majority-vote** (97 weekly objects) for lattice construction; rule mining runs on the full 671-day daily data
- Extracts **association rules** with configurable thresholds:
  - `min_support = 0.05` (≥ 5 % of days)
  - `min_confidence = 0.80`
  - `min_lift = 1.05`
  - `max_conclusion_prevalence = 0.75` (blocks near-constant conclusions)
- A **tautology filter** removes definitionally true rules (e.g. concluding `severe_lockdown_behavior` when its three required components are all in the premise)
- Rules are tagged `cross_domain = True` when they span both mobility and emotion feature sets

### Step 4 — LLM Rule Evaluation (`evaluate_rules_llm.py`)
*(Optional — requires `OPENAI_API_KEY`)*

- Sends up to 100 stratified candidate rules to `gpt-4o-mini` in a single batch call
- Stratification: 50 social-conclusion rules + 50 mobility-conclusion rules (both cross-domain directions represented)
- LLM scores each rule on **novelty** (1–10) and **policy relevance** (1–10), returns a ranked list of the top 20 with:
  - "Why it matters" reasoning
  - Concrete policy recommendation for UAE health/government officials
- Without `--llm`, rules are ranked statistically (cross-domain first, then lift)

### Step 5 — Visualisation & Report (`visualize_results.py`)
Generates four charts and a `summary_report.txt`:
- `mobility_trends.png` — UAE mobility category time series
- `sentiment_timeline.png` — daily sentiment compound score over the pandemic
- `features_heatmap.png` — binary feature activation correlation matrix
- `combined_analysis.png` — overlaid mobility and sentiment signals
- `summary_report.txt` — full LLM-selected rule set with scores and policy recommendations (falls back to statistical ranking if LLM step was skipped)

---

## Key Findings

Top cross-domain associations discovered (LLM novelty ≥ 7/10):

| Rule | Finding | Support | Confidence | Lift |
|------|---------|---------|------------|------|
| `dominant_emotion_fear + low_mobility → grocery_spike` | Background fear drives retail surges even on otherwise-quiet days — pandemic pantry-loading behaviour | 33 % of days | 83 % | 1.96 |
| `sentiment_shift_detected + low_mobility → grocery_spike` | Any sentiment movement (not just fear) predicts stocking behaviour | 22 % of days | 83 % | 1.94 |
| `grocery_spike + high_positive_sentiment → low_crisis_baseline` | Positive emotional days with retail activity signal calm, non-lockdown periods | 15 % of days | 85 % | 2.16 |
| `workplace_drop + compliance_discussed → lockdown_mentioned` | Workplace mobility drops co-occur with compliance discourse as a leading indicator of lockdown announcements | 6 % of days | 93 % | 2.90 |
| `grocery_spike + compliance_discussed → vaccine_mentioned` | Compliance-engaged shoppers bridge to vaccination discourse — a signal for targeted rollout messaging | 13 % of days | 81 % | 1.80 |

---

## Setup

### Prerequisites
- Python 3.10+
- Reddit API credentials (for data collection only)
- OpenAI API key (for LLM rule evaluation only)

### Installation
```sh
git clone https://github.com/<your-username>/capstone-project-covid.git
cd capstone-project-covid
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration
```sh
cp .env.example .env
# Edit .env and fill in your credentials:
#   REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT
#   OPENAI_API_KEY  (optional — only needed for --llm evaluation)
```

---

## Usage

### Interactive pipeline runner
```sh
python scripts/run_full_pipeline.py
```
Options:
1. Full pipeline (collect → process → FCA → visualise → evaluate)
2. Skip data collection (use existing data)
3. Analysis only (FCA + visualise)
4. Custom steps

You will be prompted whether to enable LLM scoring.

### Make targets
```sh
make help        # list all targets
make mobility    # process raw mobility CSV
make features    # build binary feature matrix
make fca         # run FCA + rule mining
make visualize   # generate charts and summary report
make pipeline    # run steps 2–5 non-interactively
make analysis    # fca + visualize only
make clean       # remove results/
make clean-all   # remove results/ and data/processed/
```

### LLM evaluation only (after FCA has been run)
```sh
python scripts/evaluate_rules_llm.py          # statistical ranking
python scripts/evaluate_rules_llm.py --llm    # LLM batch selection
```

---

## Project Structure

```
capstone-project-covid/
├── data/
│   ├── raw/                         # Source data (large files git-ignored)
│   │   ├── Global_Mobility_Report.csv
│   │   └── reddit_wildfire_posts_by_day.json
│   └── processed/                   # Generated feature files
│       ├── fca_binary_matrix.csv
│       ├── mobility_data_processed.csv
│       └── reddit_sentiment_features.csv
├── results/
│   ├── fca/
│   │   ├── association_rules.csv
│   │   ├── association_rules_evaluated.csv
│   │   ├── formal_concepts.csv
│   │   ├── crisis_context.cxt          # Galicia-compatible FCA context
│   │   └── top_cross_domain_rules.txt  # LLM-selected top rules
│   ├── visualizations/
│   └── summary_report.txt
├── scripts/
│   ├── collect_mobility_data.py
│   ├── collect_reddit_data.py
│   ├── process_global_mobility_uae.py
│   ├── preprocess_and_features.py
│   ├── fca_analysis.py
│   ├── evaluate_rules_llm.py
│   ├── visualize_results.py
│   └── run_full_pipeline.py
├── .env.example
├── Makefile
└── requirements.txt
```

---

## Data Sources

- **Google Community Mobility Reports** — [google.com/covid19/mobility](https://www.google.com/covid19/mobility/) — UAE national-level daily % change from baseline across 6 location categories (2020-03-01 to 2021-12-31, 671 days)
- **Reddit** — posts collected via PRAW from UAE-local and global COVID subreddits, filtered to the same date range