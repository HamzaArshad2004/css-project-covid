# UAE COVID-19 Behavioral Analysis
### Capstone Project — Cross-Domain Mobility × Emotion Discovery via Formal Concept Analysis

This project investigates how **public mobility patterns** and **online emotional sentiment** co-evolved during the UAE COVID-19 pandemic (March 2020 – December 2021). Using Formal Concept Analysis (FCA) and LLM-assisted rule evaluation, it surfaces non-obvious, policy-relevant associations between physical behavior and public discourse.

---

## Research Question

> *Do changes in UAE population mobility during COVID-19 reliably co-occur with measurable shifts in public emotional sentiment — and if so, which cross-domain patterns are most policy-actionable?*

---

## Pipeline Overview

```
Raw Data                   Processing                Analysis                  Output
─────────                  ──────────                ────────                  ──────
Google Mobility  ──►  preprocess_and_features  ──►  fca_analysis         ──►  association_rules.csv
Reddit Posts     ──►       (binary matrix)     ──►  (FCA lattice)        ──►  formal_concepts.csv
                                │                        │
                     lagged_features.py          prune_rules.py          ──►  association_rules_predictive_pruned.csv
                     (lag 1/3/7, lead 2/7)       (6-stage filter)             (40 clean predictive rules)
                                                         │
                                               evaluate_rules_llm        ──►  association_rules_evaluated.csv
                                               (UAE-calibrated LLM)      ──►  association_rules_predictive_evaluated.csv
                                                         │
                                               temporal_backtest         ──►  association_rules_stable.csv
                                                         │
                                               policy_briefs             ──►  policy_briefs.txt
                                                         │
                                               visualize_results         ──►  summary_report.txt
                                                                              6 × charts/
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
- Composite: `emotion_with_mobility_signal`, `emotion_mobility_mismatch`, `calm_mobile_baseline`
  - `calm_mobile_baseline`: days where **neither** elevated negative emotion **nor** any mobility-disruption signal is present — a positive behavioural-state indicator representing routine/recovery days (distinct from a residual catch-all)

### Step 3 — Formal Concept Analysis (`fca_analysis.py`)
- Builds a **Galois lattice** (formal context) over the binary matrix
- To prevent combinatorial explosion, objects > 150 are **aggregated to weekly majority-vote** for lattice construction; rule mining runs on the full 671-day daily data
- Extracts **same-day association rules** with hard statistical thresholds:
  - `min_support ≥ 10 %`, `min_confidence ≥ 75 %`, `min_lift ≥ 1.8`
- A **tautology filter** removes definitionally true rules
- Rules are tagged `cross_domain = True` when they span both mobility and emotion feature sets
- **Result: 23 raw same-day rules → 9 after full pruning**

### Step 3b — Lagged Feature Matrix (`src/features/lagged_features.py`)
- Generates a **671 × 111** feature matrix with lags (1, 3, 7 days prior) and leads (2, 7 days ahead) for all 29 base features
- Rules are then mined on this matrix to discover **cross-time-window associations**

### Step 3c — Predictive Rule Pruning (`src/rules/prune_rules.py`)
A 6-stage filter pipeline removes artefact rules before any evaluation:
1. **Same-lag artefacts** — drops rules where all features share the same lag number
2. **Same-day contamination** — removes temporally-unsuffixed rules (already covered by same-day analysis)
3. **Leakage** — drops rules with `_lead` features in the antecedent
4. **Tautology** — drops rules where premise and conclusion belong to the same mobility or sentiment group
5. **Bidirectional deduplication** — keeps the higher-conviction direction of A↔B pairs
6. **Subsumption** — removes rules whose antecedent is a superset of a stronger rule
- **Result: 3483 raw predictive rules → 40 clean rules**

### Step 4 — LLM Rule Evaluation (`evaluate_rules_llm.py`)
*(Optional — requires `OPENAI_API_KEY`)*

Two separate LLM evaluation passes run in sequence:

**Same-day rules:**
- Sends up to 100 stratified candidate rules to `gpt-4o-mini` in a single batch call
- Scores each rule on **novelty** (1–10) and **policy relevance** (1–10)
- System prompt embeds the UAE COVID-19 policy timeline (4 phases, exact dates) and mandates UAE-specific agency recommendations (NCEMA, DHA, MoHAP, WAM)
- Without `--llm`, rules are ranked statistically (cross-domain first, then lift)

**Predictive rules:**
- Separate `PREDICTIVE_SYSTEM_PROMPT` explains lag/lead notation and rewards cross-domain temporal rules
- Novelty rubric anchors 9–10 to rules where an emotion/discourse signal today forecasts a mobility change in 2–7 days (the most operationally valuable direction)
- Policy rubric requires a `lead_time_days` estimate and a named UAE body + pre-emptive action
- **Result: top 10 predictive rules selected with LLM reasoning and policy recommendations**

### Step 5 — Temporal Backtest (`src/validation/temporal_backtest.py`)
- 70/30 chronological split (train 2020-03–2021-06, holdout 2021-07–2021-12)
- Drops rules where confidence drops > 10 percentage points on the holdout period
- **Result: 8/9 same-day rules stable (88%)**; `grocery_spike, weekend → calm_mobile_baseline` dropped (12.2 pp drop)

### Step 6 — Policy Briefs (`src/reporting/policy_briefs.py`)
- Generates structured operational briefs per rule: signal window, lead time, action playbook, false-alarm risk
- Outputs `results/policy_briefs.txt`; predictive rule briefs in `results/fca/top_predictive_rules.txt`

### Step 7 — Visualisation & Report (`visualize_results.py`)
Generates six charts and a `summary_report.txt`:
- `mobility_trends.png` — UAE 4-category mobility with 7-day rolling average + phase shading
- `sentiment_timeline.png` — compound sentiment + pos/neg fractions with 14-day rolling average
- `features_heatmap.png` — bi-weekly aggregated binary feature activation (readable x-axis, month labels)
- `combined_analysis.png` — 3-panel: mobility index / sentiment / binary key signals with phase bands
- `rules_overview.png` — bubble chart: support vs confidence, bubble size = lift², colour = cross-domain
- `feature_activation.png` — horizontal bar chart of all 29 feature activation rates, colour-coded by domain
- `summary_report.txt` — full LLM-selected rule set with scores, reasoning, and policy recommendations

---

## UAE Policy Timeline (4 Phases)

All charts use background shading to mark these phases:

| Phase | Dates | Policy Context |
|-------|-------|----------------|
| 1 — First lockdown | Mar 1 – May 31 2020 | National lockdown; curfews; schools/businesses closed; grocery demand spikes |
| 2 — Reopening | Jun 1 – Dec 31 2020 | Phased reopening; mask mandates continue; mobility gradually recovers |
| 3 — Vaccine rollout | Jan 1 – Aug 31 2021 | Mass vaccination campaign; restrictions eased in stages; sentiment improves |
| 4 — Endemic transition | Sep 1 – Dec 31 2021 | Near-full reopening; focus shifts from crisis to endemic management |

---

## Key Findings

### Same-Day Rules (top cross-domain, LLM novelty ≥ 7/10, backtest-stable)

| Rule | Finding | Support | Confidence | Lift |
|------|---------|---------|------------|------|
| `health_concern + calm_mobile_baseline → grocery_spike` | Residual health anxiety on otherwise-normal days predicts retail surges — pandemic pantry-loading under surface-level calm | 20% of days | 77% | 1.82 |
| `grocery_spike + high_positive_sentiment → calm_mobile_baseline` | Positive emotional days with grocery activity signal recovery/routine periods | 15% of days | 85% | 2.16 |
| `grocery_spike + vaccine_mentioned → calm_mobile_baseline` | Vaccine-era grocery activity co-occurs with stable mobility — Phase 3 normalisation signal | 26% of days | 80% | 2.01 |

### Predictive Rules (top cross-domain, LLM novelty ≥ 8/10)

| Rule | Finding | Lead time | Confidence | Lift |
|------|---------|-----------|------------|------|
| `grocery_spike + lockdown_mentioned → vaccine_mentioned` (7 days later) | Crisis-adjacent consumer behaviour today forecasts vaccination discourse next week | 7 days | 85% | 1.90 |
| `grocery_spike + high_positive_sentiment_lag1 → calm_mobile_baseline` | Yesterday's positive sentiment + today's grocery activity forecasts routine mobility in a week | 7 days | 81% | 2.09 |
| `high_positive_sentiment + calm_mobile_baseline → grocery_spike` (2 days later) | Stable, positive days reliably precede a consumer activity uptick within 48 hours | 2 days | 89% | 2.08 |

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
python scripts/evaluate_rules_llm.py --llm    # LLM batch selection (same-day + predictive)
```

### Module-level steps
```sh
python -m src.features.lagged_features        # build 671×111 lag/lead matrix
python -m src.validation.temporal_backtest    # 70/30 backtest of same-day rules
python -m src.reporting.policy_briefs         # generate policy_briefs.txt
```

---

## Project Structure

```
capstone-project-covid/
├── data/
│   ├── raw/                         # Source data (large files git-ignored)
│   │   ├── Global_Mobility_Report.csv
│   │   └── reddit_covid_uae_posts_by_day.json
│   └── processed/                   # Generated feature files
│       ├── fca_binary_matrix.csv
│       ├── mobility_data_processed.csv
│       └── reddit_sentiment_features.csv
├── results/
│   ├── fca/
│   │   ├── association_rules.csv
│   │   ├── association_rules_evaluated.csv       # Same-day rules + LLM scores
│   │   ├── association_rules_stable.csv          # Backtest-passing rules
│   │   ├── association_rules_predictive_pruned.csv  # 40 clean predictive rules
│   │   ├── association_rules_predictive_evaluated.csv  # Predictive + LLM scores
│   │   ├── formal_concepts.csv
│   │   ├── crisis_context.cxt                   # Galicia-compatible FCA context
│   │   ├── top_cross_domain_rules.txt            # LLM-selected same-day rules
│   │   └── top_predictive_rules.txt              # LLM-selected predictive rules
│   ├── visualizations/
│   │   ├── mobility_trends.png
│   │   ├── sentiment_timeline.png
│   │   ├── features_heatmap.png
│   │   ├── combined_analysis.png
│   │   ├── rules_overview.png
│   │   └── feature_activation.png
│   ├── policy_briefs.txt
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
├── src/
│   ├── features/
│   │   └── lagged_features.py       # Lag/lead feature matrix (671×111)
│   ├── rules/
│   │   └── prune_rules.py           # 6-stage rule quality filter
│   ├── scoring/
│   │   └── score_rules.py           # Statistical composite scoring
│   ├── validation/
│   │   └── temporal_backtest.py     # 70/30 chronological backtest
│   └── reporting/
│       └── policy_briefs.py         # Structured operational briefs
├── .env.example
├── Makefile
└── requirements.txt
```

---

## Data Sources

- **Google Community Mobility Reports** — [google.com/covid19/mobility](https://www.google.com/covid19/mobility/) — UAE national-level daily % change from baseline across 6 location categories (2020-03-01 to 2021-12-31, 671 days)
- **Reddit** — posts collected via PRAW from UAE-local and global COVID subreddits, filtered to the same date range