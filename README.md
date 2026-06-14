# UAE COVID-19 Behavioral Analysis
### Capstone Project — Cross-Domain Mobility × Emotion Discovery via Formal Concept Analysis

This project investigates how **public mobility patterns** and **online emotional sentiment** co-evolved during the UAE COVID-19 pandemic (March 2020 – December 2021). Using Formal Concept Analysis (FCA), reliability-first rule pruning, and LLM-assisted evaluation, it surfaces non-obvious, policy-relevant associations between physical behavior and public discourse — including short-horizon predictive signals.

---

## Research Question

> *Do changes in UAE population mobility during COVID-19 reliably co-occur with — and forecast — measurable shifts in public emotional sentiment, and if so, which cross-domain patterns are most policy-actionable?*

---

## Headline results (current run)

| Stage | Count |
|---|---|
| Raw same-day rules (support ≥ 12%, conf ≥ 75%, lift ≥ 1.8) | 23 |
| After pruning + subsumption | 8 |
| Stable after 70/30 chronological backtest | **7 (87%)** |
| Clean **predictive** (lag/lead) rules after pruning | **16** |
| — of which cross-domain (mobility × emotion) | **12** |

Predictive rules span 1-, 2-, 3-, and 7-day forecast horizons. Cross-domain rules carry slightly higher mean lift (2.06) than single-domain rules (2.00).

---

## Pipeline overview

```
Raw Data                 Processing               Analysis                Output
─────────                ──────────               ────────                ──────
Google Mobility  ─►  preprocess_and_features ─►  lagged_features.py  ─►  fca_predictive_matrix.csv
Reddit Posts     ─►     (binary matrix)           (lag 1/3/7,
                              │                     lead 2/7)
                              ▼                        │
                        fca_analysis.py ◄─────────────┘  ─►  association_rules.csv
                        (FCA lattice + mining)             ─►  association_rules_predictive.csv
                              │
                        prune_rules.py            ─►  association_rules_predictive_pruned.csv
                        (7-stage filter)               (16 clean predictive rules)
                              │
                        evaluate_rules_llm.py     ─►  association_rules_evaluated.csv
                        (UAE-calibrated LLM)      ─►  association_rules_predictive_evaluated.csv
                              │
                        temporal_backtest.py      ─►  association_rules_stable.csv
                              │
                        policy_briefs.py          ─►  policy_briefs.txt
                              │
                        visualize_results.py      ─►  summary_report.txt + 6 charts
                        (report runs LAST)
```

> **Ordering matters.** The lagged matrix is built **before** `fca_analysis.py` so predictive rules get mined, and the summary report is generated **last** (after scoring/backtest/evaluation) so it reflects the final CSVs rather than stale ones. Both are enforced by `run_pipeline.py`.

---

## Step 1 — Data collection

| Source | Script | Output |
|--------|--------|--------|
| Google Community Mobility Reports (UAE national) | `collect_mobility_data.py` → `process_global_mobility_uae.py` | `mobility_data_processed.csv` |
| Reddit (UAE + COVID subreddits, 2020–2021) | `collect_reddit_data.py` | `reddit_covid_uae_posts_by_day.json` |

Reddit collection uses a multi-subreddit, multi-query, multi-sort strategy across UAE-local (`r/UAE`, `r/dubai`, `r/abudhabi`) and global subreddits (`r/Coronavirus`, `r/worldnews`). Posts are deduplicated by ID; days with fewer than 5 posts are flagged as low-coverage.

## Step 2 — Feature engineering (`preprocess_and_features.py`)

671 daily observations become a **29-column binary matrix**.

**Mobility features** (Google Mobility % change vs. baseline):
- `mobility_drop_retail` (< −25%), `mobility_drop_transit` (< −30%), `mobility_drop_workplace` (< −25%)
- `residential_increase` (> +8%), `grocery_spike` (grocery & pharmacy > +10%)
- `severe_lockdown_behavior` (retail ∧ workplace drop ∧ residential increase)
- `partial_restrictions` (one or more drops, not severe lockdown)

**Social / emotion features** (Reddit via VADER):
- Sentiment: `high_negative_sentiment`, `high_positive_sentiment`, `sentiment_improved`, `sentiment_worsened`, `sentiment_shift_detected`, `dominant_emotion_fear`, `mixed_emotions`
- Keywords: `fear_keywords_present`, `anger_mentioned`, `anxiety_keywords_present`, `sadness_keywords_present`, `solidarity_messages`
- Topics: `covid_topic_detected`, `lockdown_mentioned`, `vaccine_mentioned`, `health_concern`, `compliance_discussed`, `policy_governance_discussion`
- Composite: `emotion_with_mobility_signal`, `emotion_mobility_mismatch`, `calm_mobile_baseline`

`calm_mobile_baseline` marks days with **neither** elevated negative emotion **nor** any mobility-disruption signal — routine/recovery days. Its mobility-disruption test deliberately **excludes** `grocery_spike`, so `calm_mobile_baseline` and `grocery_spike` are independent constructs and rules linking them are empirical, not definitional.

## Step 3 — Lagged feature matrix (`src/features/lagged_features.py`)

Generates a predictive matrix with lagged antecedents (1, 3, 7 days prior) and lead consequents (2, 7 days ahead). Social signals (incl. `vaccine_mentioned`, `high_positive_sentiment`, `health_concern`) are available as **both** lagged predictors and lead targets, so discourse can be forecast forward — not only used to explain the past.

## Step 4 — Formal Concept Analysis & rule mining (`fca_analysis.py`)

- Builds a Galois lattice over the binary matrix (objects > 150 are aggregated to weekly majority-vote for lattice construction; rule mining runs on the full 671 daily rows).
- Mines **same-day** rules (support ≥ 12%, confidence ≥ 75%, lift ≥ 1.8) and, when the predictive matrix exists, **predictive** rules from it.
- A tautology filter removes definitionally-true rules. Rules are tagged `cross_domain = True` when they span mobility and emotion domains — using **base feature names** (suffix-stripped), so `vaccine_mentioned_lag3` is correctly recognised as an emotion feature.

**Same-day result: 23 raw → 8 after pruning → 7 backtest-stable.**

## Step 5 — Predictive rule pruning (`src/rules/prune_rules.py`)

A multi-stage filter removes artefacts before evaluation. The key invariant: **a predictive rule must point forward in time.**

1. **Backward-rule filter** — drops any rule whose consequent carries a `*_lag` suffix (the conclusion would be dated *earlier* than the premise — a lookback, not a forecast). A guard logs loudly if any survive.
2. **Same-day contamination** — removes temporally-unsuffixed rules already covered by same-day analysis.
3. **Leakage** — drops rules with `*_lead` features in the antecedent (future information at prediction time).
4. **Within-variable persistence** — drops rules whose consequent base feature also appears in the antecedent (e.g. `grocery_spike → grocery_spike_lead7` is autocorrelation, not discovery).
5. **Tautology** — suffix-aware removal of definitionally-collinear rules (mobility-drop family, `sentiment_shift_detected ↔ sentiment_improved/worsened`, composite definitions).
6. **Bidirectional deduplication** — keeps the higher-conviction direction of A↔B pairs.
7. **Subsumption + per-consequent cap** — removes rules subsumed by simpler ones; caps at 3 rules per consequent.

**Predictive result: thousands of raw candidates → 16 clean rules (12 cross-domain).**

## Step 6 — LLM rule evaluation (`evaluate_rules_llm.py`)

*Optional — requires `OPENAI_API_KEY`.* Runs **after** statistical filtering; it ranks but cannot introduce rules.

- Sends stratified candidates to `gpt-4o-mini` in a single batch; scores **novelty** (1–10) and **policy relevance** (1–10).
- System prompts embed the UAE policy timeline (4 phases) and require named-agency recommendations (NCEMA, DHA, MoHAP, WAM).
- **`lead_time_days` is computed deterministically from each rule's structure, never supplied by the LLM** — the model scores novelty/policy only.
- Without `--llm`, rules are ranked statistically (cross-domain first, then composite/lift).

## Step 7 — Temporal backtest (`src/validation/temporal_backtest.py`)

70/30 chronological split (train Mar 2020 – Jun 2021, holdout Jul – Dec 2021). Drops rules whose confidence falls > 10 pp on the holdout. **7 of 8 same-day rules stable (87%);** `grocery_spike, weekend → calm_mobile_baseline` dropped (12.2 pp), consistent with fading pandemic-specific weekend routines in Phase 4.

## Step 8 — Policy briefs & report

`policy_briefs.py` emits structured operational briefs (signal window, lead time, action playbook, false-alarm risk). `visualize_results.py` generates six charts and `summary_report.txt`. The report is invoked **last** in the pipeline (via `--report-only`) so it always reflects the freshest CSVs.

---

## UAE policy timeline (chart shading)

| Phase | Dates | Context |
|-------|-------|---------|
| 1 — First lockdown | Mar 1 – May 31 2020 | National lockdown; curfews; grocery demand spikes |
| 2 — Reopening | Jun 1 – Dec 31 2020 | Phased reopening; mask mandates; mobility recovers |
| 3 — Vaccine rollout | Jan 1 – Aug 31 2021 | Mass vaccination; restrictions eased; sentiment improves |
| 4 — Endemic transition | Sep 1 – Dec 31 2021 | Near-full reopening; crisis → endemic management |

---

## Key findings

### Same-day rules (backtest-stable, cross-domain)

| Rule | Finding | Support | Conf. | Lift |
|------|---------|---------|-------|------|
| `grocery_spike + high_positive_sentiment → calm_mobile_baseline` | Positive-sentiment grocery days mark recovery/routine periods | 15% | 85% | 2.16 |
| `grocery_spike + vaccine_mentioned → calm_mobile_baseline` | Vaccine-era grocery activity co-occurs with stable mobility (Phase 3 normalisation) | 26% | 80% | 2.01 |
| `health_concern + calm_mobile_baseline → grocery_spike` | Residual health anxiety on otherwise-calm days predicts retail surges (pantry-loading under surface calm) | 20% | 77% | 1.82 |

### Predictive rules (top cross-domain, LLM novelty ≥ 8/10)

| Rule | Finding | Lead | Conf. | Lift |
|------|---------|------|-------|------|
| `calm_mobile_baseline + high_positive_sentiment_lag1 → grocery_spike_lead7` | Yesterday's positive sentiment + today's calm mobility forecasts a grocery spike a week out | 7 days | 92% | 2.11 |
| `calm_mobile_baseline + sentiment_worsened_lag7 → grocery_spike_lead7` | Worsened sentiment a week prior also forecasts the spike — the response isn't tied to one sentiment direction | 7 days | 92% | 2.11 |
| `high_positive_sentiment + calm_mobile_baseline → grocery_spike_lead2` | Stable, positive days precede a consumer-activity uptick within 48 h (logistics pre-positioning) | 2 days | 89% | 2.08 |

*All predictive rules point forward in time. Earlier drafts of this project surfaced a backward `→ vaccine_mentioned_lag7` "forecast"; that was a lag-direction artefact and is removed by the backward-rule filter.*

---

## Setup

**Prerequisites:** Python 3.10+, Reddit API credentials (collection only), OpenAI API key (LLM evaluation only).

```sh
git clone https://github.com/<your-username>/capstone-project-covid.git
cd capstone-project-covid
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Fill in: REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT
#          OPENAI_API_KEY  (optional — only for --llm evaluation)
```

---

## Usage

### Interactive runner
```sh
python scripts/run_pipeline.py
```
Options: (1) full pipeline, (2) skip data collection, (3) analysis only, (4) post-processing only, (5) custom steps. You are prompted whether to enable LLM scoring.

### Regenerate just the summary report
```sh
python scripts/visualize_results.py --report-only
```

### LLM evaluation only (after FCA has run)
```sh
python scripts/evaluate_rules_llm.py          # statistical ranking
python scripts/evaluate_rules_llm.py --llm    # LLM batch selection (same-day + predictive)
```

### Module-level steps
```sh
python -m src.features.lagged_features        # build lag/lead matrix (BEFORE fca_analysis.py)
python -m src.rules.prune_rules               # prune
python -m src.validation.temporal_backtest    # 70/30 backtest
python -m src.reporting.policy_briefs         # policy_briefs.txt
```

> **Reproducibility:** on a clean checkout, build the lagged matrix before running `fca_analysis.py`, or predictive mining is skipped. `run_pipeline.py` handles this ordering automatically.

---

## Project structure

```
capstone-project-covid/
├── data/
│   ├── raw/                         # Source data (large files git-ignored)
│   │   ├── Global_Mobility_Report.csv
│   │   └── reddit_covid_uae_posts_by_day.json
│   └── processed/
│       ├── fca_binary_matrix.csv
│       ├── fca_predictive_matrix.csv
│       ├── mobility_data_processed.csv
│       └── reddit_sentiment_features.csv
├── results/
│   ├── fca/
│   │   ├── association_rules.csv
│   │   ├── association_rules_evaluated.csv
│   │   ├── association_rules_stable.csv
│   │   ├── association_rules_predictive_pruned.csv      # 16 clean predictive rules
│   │   ├── association_rules_predictive_evaluated.csv   # predictive + LLM scores
│   │   ├── formal_concepts.csv
│   │   ├── crisis_context.cxt
│   │   ├── top_cross_domain_rules.txt
│   │   └── top_predictive_rules.txt
│   ├── visualizations/              # 6 PNG charts
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
│   └── run_pipeline.py
├── src/
│   ├── features/lagged_features.py
│   ├── rules/prune_rules.py
│   ├── scoring/score_rules.py
│   ├── validation/temporal_backtest.py
│   └── reporting/policy_briefs.py
├── .env.example
└── requirements.txt
```

---

## Data sources

- **Google Community Mobility Reports** — UAE national daily % change from baseline, 2020-03-01 to 2021-12-31 (671 days).
- **Reddit** — posts via PRAW from UAE-local and global COVID subreddits, same date range.
