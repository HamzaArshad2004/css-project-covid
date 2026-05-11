"""
LLM-based rule evaluation and prioritization.

Loads association rules from FCA, then optionally uses an OpenAI-compatible
LLM to identify the top 10 most novel and policy-relevant rules in a single
batch call.  Cross-domain rules are always flagged and sorted to the top
regardless of whether the LLM step is run.

Usage (without LLM — statistical ranking only):
    python scripts/evaluate_rules_llm.py

Usage (with LLM selection — requires OPENAI_API_KEY in .env or environment):
    python scripts/evaluate_rules_llm.py --llm

Output:
    results/fca/association_rules_evaluated.csv   (all rules with LLM scores if run)
    results/fca/top_cross_domain_rules.txt        (top 10 human-readable summary)
"""

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Ensure project root is on sys.path so 'src.*' sub-packages are importable
# whether the script is called as 'python scripts/evaluate_rules_llm.py' or
# via 'PYTHONPATH=. python scripts/evaluate_rules_llm.py'.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RULES_FILE = PROJECT_ROOT / "results" / "fca" / "association_rules.csv"
OUT_CSV = PROJECT_ROOT / "results" / "fca" / "association_rules_evaluated.csv"
OUT_TXT = PROJECT_ROOT / "results" / "fca" / "top_cross_domain_rules.txt"

PRED_RULES_FILE = PROJECT_ROOT / "results" / "fca" / "association_rules_predictive_pruned.csv"
OUT_PRED_CSV = PROJECT_ROOT / "results" / "fca" / "association_rules_predictive_evaluated.csv"
OUT_PRED_TXT = PROJECT_ROOT / "results" / "fca" / "top_predictive_rules.txt"

# ---------------------------------------------------------------------------
# Feature domain classification (must stay in sync with fca_analysis.py)
# ---------------------------------------------------------------------------
MOBILITY_FEATURES = frozenset({
    "mobility_drop_retail",
    "mobility_drop_transit",
    "mobility_drop_workplace",
    "residential_increase",
    "grocery_spike",
    "severe_lockdown_behavior",
    "partial_restrictions",
})

EMOTION_FEATURES = frozenset({
    "high_negative_sentiment",
    "dominant_emotion_fear",
    "fear_keywords_present",
    "anger_mentioned",
    "anxiety_keywords_present",
    "sadness_keywords_present",
    "high_positive_sentiment",
    "mixed_emotions",
    "solidarity_messages",
    "sentiment_worsened",
    "sentiment_improved",
    "sentiment_shift_detected",
    "covid_topic_detected",
    "lockdown_mentioned",
    "vaccine_mentioned",
    "health_concern",
    "compliance_discussed",
    "policy_governance_discussion",
})


def _load_env() -> None:
    """Load variables from .env into os.environ (no python-dotenv dependency)."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


def _feature_set(rule: pd.Series) -> frozenset[str]:
    """Return the full set of features referenced in a rule (premise + conclusion)."""
    parts = [p.strip() for p in rule["premise"].split(",")]
    parts.append(rule["conclusion"].strip())
    return frozenset(parts)


def tag_cross_domain(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add / refresh the cross_domain flag based on whether a rule spans
    both the mobility and emotion feature domains.
    """
    def _is_cross(row: pd.Series) -> bool:
        fs = _feature_set(row)
        return bool(fs & MOBILITY_FEATURES) and bool(fs & EMOTION_FEATURES)

    df = df.copy()
    df["cross_domain"] = df.apply(_is_cross, axis=1)
    return df


# ---------------------------------------------------------------------------
# LLM batch selection
# ---------------------------------------------------------------------------

# Number of candidate rules fed to the LLM for comparison.
# The LLM sees all of these and picks the best TOP_N.
CANDIDATE_POOL = 100
TOP_N_LLM = 20

BATCH_SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert data scientist and public health policy advisor reviewing
    association rules mined from crisis behavior data during the COVID-19 period
    in the UAE (March 2020 – December 2021). The dataset combines daily Google
    Community Mobility metrics with Reddit sentiment signals (fear, solidarity,
    compliance discussions, vaccine mentions, etc.).

    ── UAE COVID-19 POLICY TIMELINE (use this to anchor your scores) ──────────
    Phase 1 — Onset & first lockdown (Mar–May 2020):
      - 15 Mar 2020: Schools close, restaurants limited to delivery/takeaway.
      - 26 Mar 2020: National disinfection programme; nightly curfew 8pm–6am.
      - Apr 2020: Strict stay-home order; grocery stores exempted; grocery spikes
        documented. Highest fear sentiment in dataset.

    Phase 2 — Controlled reopening (Jun–Dec 2020):
      - Jun 2020: Partial reopening of retail, restaurants at 30% capacity.
      - Compliance fatigue begins: solidarity discourse drops, compliance_discussed
        declines despite partial_restrictions remaining active.
      - Sep–Oct 2020: Second wave; workplace drop renewed.

    Phase 3 — Vaccine rollout & optimism (Jan–Aug 2021):
      - Jan 2021: UAE launches one of the world's fastest per-capita vaccine
        programmes; vaccine_mentioned spikes on Reddit.
      - Feb–Mar 2021: positive sentiment recovery; grocery_spike normalises.
      - High positive sentiment + vaccine discussion co-occur with reduced fear.

    Phase 4 — Endemic transition (Sep–Dec 2021):
      - Sep 2021: Most restrictions lifted; mobility returns toward baseline.
      - Residual health_concern remains in online discourse even as mobility
        recovers — a decoupling signal.
    ──────────────────────────────────────────────────────────────────────────

    FEATURE DOMAINS:
    - MOBILITY features: mobility_drop_retail, mobility_drop_transit,
      mobility_drop_workplace, residential_increase, grocery_spike,
      severe_lockdown_behavior, partial_restrictions
    - SOCIAL/EMOTION features: high_negative_sentiment, dominant_emotion_fear,
      fear_keywords_present, anger_mentioned, anxiety_keywords_present,
      sadness_keywords_present, high_positive_sentiment, mixed_emotions,
      solidarity_messages, sentiment_worsened, sentiment_improved,
      sentiment_shift_detected, covid_topic_detected, lockdown_mentioned,
      vaccine_mentioned, health_concern, compliance_discussed,
      policy_governance_discussion

    ── NOVELTY SCORING RUBRIC (1–10) ─────────────────────────────────────────
    Score the rule on HOW UNEXPECTED the association is given the UAE timeline above.

    10 — Contradicts or meaningfully extends the known policy narrative.
         Example: positive sentiment predicts INCREASED mobility drop (counter-intuitive
         compliance signal that would not be predicted from policy dates alone).
     8–9 — Cross-domain, directional, and not explainable by a single policy event.
         Example: health_concern persisting AFTER mobility recovery in Phase 4
         (decoupling of discourse from behaviour).
     6–7 — Cross-domain, directional, but timing aligns with an obvious policy event.
         Example: vaccine_mentioned + grocery_spike in Phase 3 (plausible but not trivially
         implied by the vaccine rollout date).
     4–5 — Same domain or explainable by direct policy co-occurrence.
         Example: lockdown_mentioned → mobility_drop during Phase 1 curfew.
     1–3 — Tautological or near-constant feature (covid_topic_detected = 85% of days).
    ──────────────────────────────────────────────────────────────────────────

    ── POLICY RELEVANCE SCORING RUBRIC (1–10) ────────────────────────────────
    Score on WHETHER a UAE official could act on this rule as an early-warning signal.

    10 — Emotion/discourse signal PRECEDES a real-world mobility outcome by a detectable
         lead time AND the signal is specific enough to trigger a named intervention
         (e.g., "if solidarity drops while partial_restrictions active → pre-position
         enforcement resources within 3 days").
     8–9 — Cross-domain leading indicator; action is clear but lead time is implicit.
     6–7 — Useful monitoring signal; action requires additional validation.
     4–5 — Interesting pattern but same-day correlation only; no actionable lead.
     1–3 — Consequent is too broad, too frequent, or too obvious to justify specific action.
    ──────────────────────────────────────────────────────────────────────────

    MANDATORY EXCLUSIONS — do NOT select rules where:
    - residential_increase is the conclusion and any mobility_drop is in the premise.
    - Both premise and conclusion are purely mobility features.
    - covid_topic_detected or dominant_emotion_fear is the conclusion (both active ~85% of days).
    - The rule is a near-duplicate of a higher-ranked rule already selected.
    - Conclusion is trivially expected from premise semantics
      (e.g., lockdown_mentioned → mobility_drop; sentiment_worsened → high_negative_sentiment).
    - MORE THAN 2 rules share the same conclusion feature in your top 10.

    PREFER rules that:
    - Reveal a phase-transition signal (discourse shifts BEFORE mobility shifts, or vice versa).
    - Capture compliance fatigue (positive mobility + reduced solidarity/compliance).
    - Capture vaccine-era decoupling (high positive sentiment while health concern remains).
    - Show weekend vs weekday asymmetry in crisis response.

    Your task: from the numbered candidate rules below, select and rank the
    TOP 20 that best satisfy the above criteria.

    Return ONLY a JSON array of exactly 20 objects, ordered best-first:
    [
      {
        "rule_id": <int — the id field from the candidate list>,
        "novelty_score": <int 1-10 using the rubric above>,
        "policy_score": <int 1-10 using the rubric above>,
        "reasoning": "<2-3 sentences: name the specific UAE policy phase this relates to, explain why the direction is non-obvious, and quantify the expected lead/lag if applicable>",
        "policy_recommendation": "<2 concrete sentences: name a specific UAE agency or programme (e.g., NCEMA, WAM, DHA), describe the exact trigger condition, and state the recommended response action>"
      },
      ...
    ]
    No markdown fences. No extra text. Valid JSON only.
""")


PREDICTIVE_SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert data scientist and public health policy advisor reviewing
    PREDICTIVE association rules mined from UAE COVID-19 crisis behavior data
    (March 2020 – December 2021). These rules use lagged or lead features —
    meaning they describe how today's signals forecast future behaviour, or how
    yesterday's signals explain today's outcome.

    The dataset combines daily Google Community Mobility metrics with Reddit
    sentiment signals. Rules were mined using a lagged/lead feature matrix
    (lags: 1, 3, 7 days; leads: 2, 7 days) and have already passed hard
    statistical filters (support ≥ 10 %, confidence ≥ 75 %, lift ≥ 1.8).

    ── FEATURE SUFFIX CONVENTIONS ────────────────────────────────────────────
    _lag1, _lag3, _lag7  — feature was active 1/3/7 days AGO
    _lead2, _lead7       — feature will be active in 2/7 days FROM NOW
    No suffix            — feature is active TODAY
    ──────────────────────────────────────────────────────────────────────────

    ── UAE COVID-19 POLICY TIMELINE ──────────────────────────────────────────
    Phase 1 — Onset & first lockdown (Mar–May 2020):
      Strict stay-home order; nightly curfew; grocery stores exempted;
      grocery spikes and peak fear sentiment.
    Phase 2 — Controlled reopening (Jun–Dec 2020):
      Partial retail/workplace reopening; compliance fatigue; second wave Sep–Oct.
    Phase 3 — Vaccine rollout & optimism (Jan–Aug 2021):
      UAE launches one of the world's fastest per-capita vaccine programmes;
      vaccine_mentioned spikes; positive sentiment recovery.
    Phase 4 — Endemic transition (Sep–Dec 2021):
      Most restrictions lifted; mobility returns to baseline; residual
      health_concern persists in online discourse.
    ──────────────────────────────────────────────────────────────────────────

    ── NOVELTY SCORING RUBRIC (1–10) ─────────────────────────────────────────
    Focus specifically on whether the LEAD TIME encoded in the rule is useful
    and non-obvious.

    10 — Cross-domain leading indicator with a specific, actionable lead time
         (e.g., an emotion/sentiment signal today predicts a MOBILITY outcome
         in 7 days, or vice versa).  The direction would not be obvious from
         policy dates alone.
     8–9 — Cross-domain with clear lead time; timing aligns with a known phase
         transition but the cross-domain direction is non-trivial.
     6–7 — Within-domain but shows non-trivial temporal persistence or mean
         reversion (e.g., emotional oscillation, compliance decay).
     4–5 — Within-domain persistence that is broadly expected (sentiment tends
         to persist day-over-day, lockdown discourse is sticky).
     1–3 — Near-tautological: the lag/lead merely restates that a slow-changing
         variable is still active (e.g., partial_restrictions_lag7 → partial_restrictions).
    ──────────────────────────────────────────────────────────────────────────

    ── POLICY RELEVANCE SCORING RUBRIC (1–10) ────────────────────────────────
    Score on WHETHER a UAE official could use this rule as an EARLY WARNING.

    10 — Gives ≥ 2 days lead time, the trigger is a monitorable today-signal,
         and the consequence is a real-world actionable outcome.
         Explicitly names a UAE body (NCEMA, DHA, MoHAP, WAM, etc.) and action.
     8–9 — Clear lead time; action is obvious but operationalisation needs one
         additional step.
     6–7 — Useful monitoring signal; action requires further validation before
         deployment.
     4–5 — Same-domain persistence; gives situational awareness but no new
         intervention opportunity.
     1–3 — Consequent is too generic or too slow-changing to drive specific action.
    ──────────────────────────────────────────────────────────────────────────

    MANDATORY EXCLUSIONS — do NOT select rules where:
    - The consequent is a lagged version of a feature already in the antecedent
      at the same time offset (within-variable persistence artefact).
    - Both antecedent and consequent are purely within the mobility domain
      or purely within the sentiment domain AND the lag is 1 day or less
      (too short to be actionable).
    - The consequent is covid_topic_detected or dominant_emotion_fear
      (near-constant, ~85 % of days).
    - MORE THAN 2 rules share the same consequent feature in your top 10.

    PREFER rules where:
    - An EMOTION/DISCOURSE signal today or yesterday PREDICTS a MOBILITY outcome
      in 2–7 days (the most operationally valuable direction).
    - A MOBILITY signal predicts a future SENTIMENT or DISCOURSE shift
      (feedback loop discovery).
    - The rule captures a phase-transition signal (compliance decay, vaccine
      optimism, endemic normalisation).

    Your task: from the numbered candidate rules below, select and rank the
    TOP 10 most valuable predictive rules for UAE crisis management.

    Return ONLY a JSON array of exactly 10 objects, ordered best-first:
    [
      {
        "rule_id": <int>,
        "novelty_score": <int 1-10>,
        "policy_score": <int 1-10>,
        "lead_time_days": <int — the effective forecast horizon in days, based on the _lead or _lag suffix; use 0 if same-day>,
        "reasoning": "<2-3 sentences: name the UAE phase, explain why the temporal direction is non-obvious, state what the lead time enables a decision-maker to do>",
        "policy_recommendation": "<2 sentences: name a specific UAE agency, state the exact monitoring trigger and the recommended pre-emptive action>"
      },
      ...
    ]
    No markdown fences. No extra text. Valid JSON only.
""")


def _build_candidate_block(df: pd.DataFrame) -> str:
    """Format a DataFrame of candidate rules as a numbered text block for the LLM.

    Annotates each rule with conviction and composite_score when available so
    the LLM can weight statistical quality alongside domain relevance.
    """
    lines = []
    for _, row in df.iterrows():
        conviction = row.get("conviction")
        composite  = row.get("composite_score")
        extras = ""
        if conviction is not None and str(conviction) not in ("nan", "inf", ""):
            try:
                extras += f"  conviction={float(conviction):.2f}"
            except (ValueError, TypeError):
                pass
        if composite is not None and str(composite) not in ("nan", ""):
            try:
                extras += f"  composite={float(composite):.3f}"
            except (ValueError, TypeError):
                pass
        lines.append(
            f"[id={row['rule_id']}]  "
            f"IF: {row['premise']}  "
            f"THEN: {row['conclusion']}  "
            f"| support={float(row['support_pct']):.1f}%  "
            f"conf={float(row['confidence']):.1f}%  "
            f"lift={float(row['lift']):.2f}  "
            f"cross_domain={bool(row.get('cross_domain', False))}"
            + extras
        )
    return "\n".join(lines)


def _stratified_candidates(df: pd.DataFrame, candidate_pool: int) -> pd.DataFrame:
    """
    Return a stratified candidate pool so the LLM sees diverse rule types.

    Split the pool evenly between:
    - Rules whose conclusion is a SOCIAL/EMOTION feature (mobility → emotion direction)
    - Rules whose conclusion is a MOBILITY feature (emotion → mobility direction)

    Within each stratum, sort by lift descending.
    """
    mob = MOBILITY_FEATURES
    half = candidate_pool // 2

    social_conclusion = (
        df[df["cross_domain"] & ~df["conclusion"].isin(mob)]
        .sort_values(["lift", "confidence"], ascending=False)
        .head(half)
    )
    mob_conclusion = (
        df[df["cross_domain"] & df["conclusion"].isin(mob)]
        .sort_values(["lift", "confidence"], ascending=False)
        .head(half)
    )
    # Fill any shortfall in one stratum from the other
    combined = pd.concat([social_conclusion, mob_conclusion]).drop_duplicates()
    if len(combined) < candidate_pool:
        remaining = df[~df.index.isin(combined.index)].sort_values(
            ["lift", "confidence"], ascending=False
        ).head(candidate_pool - len(combined))
        combined = pd.concat([combined, remaining])
    return combined.head(candidate_pool)


def select_top_rules_with_llm(
    df: pd.DataFrame,
    model: str = "gpt-4o-mini",
    candidate_pool: int = CANDIDATE_POOL,
    top_n: int = TOP_N_LLM,
    system_prompt: str | None = None,
) -> pd.DataFrame:
    """
    Send the top `candidate_pool` rules to the LLM in a single batch call and
    ask it to select the `top_n` most novel and policy-relevant ones.

    Returns the full df with three new columns on the selected rows:
      novelty_score, policy_score, llm_reasoning
    The returned df is sorted so the LLM-selected top rules appear first (in LLM rank
    order), followed by the remaining rules.
    """
    try:
        from openai import OpenAI  # noqa: PLC0415
    except ImportError:
        print("ERROR: openai package not installed.  Run: pip install openai", file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "ERROR: OPENAI_API_KEY not found. Add it to .env or set it as an environment variable.",
            file=sys.stderr,
        )
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # Assign stable integer IDs so LLM can reference them
    df = df.copy().reset_index(drop=True)
    df["rule_id"] = df.index

    # Pre-sort: use stratified selection so LLM sees both directions of cross-domain rules
    candidates = _stratified_candidates(df, candidate_pool)

    candidate_block = _build_candidate_block(candidates)
    user_msg = (
        f"Select the top {top_n} rules from these {len(candidates)} candidates:\n\n"
        + candidate_block
    )

    print(f"  Sending {len(candidates)} candidate rules to LLM for batch selection ...")
    active_prompt = system_prompt if system_prompt is not None else BATCH_SYSTEM_PROMPT
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": active_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
        max_tokens=2400,
    )

    raw = response.choices[0].message.content.strip()
    try:
        selections = json.loads(raw)
    except json.JSONDecodeError:
        print(f"WARNING: LLM returned non-JSON response:\n{raw}", file=sys.stderr)
        print("Falling back to statistical ranking.", file=sys.stderr)
        return df

    if not isinstance(selections, list):
        print("WARNING: LLM response was not a JSON array; falling back to statistical ranking.",
              file=sys.stderr)
        return df

    # Initialise score columns
    df["novelty_score"] = None
    df["policy_score"] = None
    df["llm_reasoning"] = ""
    df["llm_policy_recommendation"] = ""
    df["llm_rank"] = None
    df["lead_time_days"] = None

    valid_ids = set(df["rule_id"].tolist())
    for rank, entry in enumerate(selections[:top_n], start=1):
        rid = entry.get("rule_id")
        if rid not in valid_ids:
            continue
        idx = df.index[df["rule_id"] == rid][0]
        df.at[idx, "novelty_score"] = entry.get("novelty_score")
        df.at[idx, "policy_score"] = entry.get("policy_score")
        df.at[idx, "llm_reasoning"] = entry.get("reasoning", "")
        df.at[idx, "llm_policy_recommendation"] = entry.get("policy_recommendation", "")
        df.at[idx, "llm_rank"] = rank
        if "lead_time_days" in entry:
            df.at[idx, "lead_time_days"] = entry.get("lead_time_days")

    df["llm_composite"] = df[["novelty_score", "policy_score"]].mean(axis=1)

    print(f"  LLM selected {df['llm_rank'].notna().sum()} rules.")
    return df


# ---------------------------------------------------------------------------
# Human-readable summary writer
# ---------------------------------------------------------------------------

def write_summary(df: pd.DataFrame, out_path: Path, top_n: int = 10) -> None:
    """Write a text summary of the top rules.

    If the df contains an 'llm_rank' column (set by the LLM batch selection),
    the LLM-selected top-10 are shown in LLM rank order.  Otherwise rules are
    ranked statistically (cross-domain first, then lift).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    llm_selected = "llm_rank" in df.columns and df["llm_rank"].notna().any()

    if llm_selected:
        # Show only the LLM-selected rules in their chosen order
        ranked = (
            df[df["llm_rank"].notna()]
            .sort_values("llm_rank")
            .head(top_n)
        )
        header_note = f"  LLM-SELECTED: Top {top_n} Most Novel & Policy-Relevant Rules\n"
    else:
        sort_cols = ["cross_domain", "lift", "confidence"]
        ranked = df.sort_values(sort_cols, ascending=[False, False, False]).head(top_n)
        header_note = f"  Top {top_n} Rules by Statistical Ranking (cross-domain priority)\n"

    with open(out_path, "w") as f:
        f.write("=" * 72 + "\n")
        f.write("  TOP RULES — CROSS-DOMAIN MOBILITY × EMOTION ANALYSIS\n")
        f.write("  UAE COVID-19 PERIOD\n")
        f.write(header_note)
        f.write("=" * 72 + "\n\n")

        for i, (_, rule) in enumerate(ranked.iterrows(), start=1):
            f.write(f"Rule {i}:  {'[CROSS-DOMAIN] ' if rule.get('cross_domain') else ''}\n")
            f.write(f"  IF:   {rule['premise']}\n")
            f.write(f"  THEN: {rule['conclusion']}\n")
            f.write(
                f"  Support: {int(rule['support'])} days ({float(rule['support_pct']):.1f}%)  "
                f"Confidence: {float(rule['confidence']):.1f}%  Lift: {float(rule['lift']):.2f}\n"
            )
            if llm_selected and pd.notna(rule.get("llm_composite")):
                f.write(
                    f"  LLM: novelty={rule['novelty_score']}/10  "
                    f"policy_relevance={rule['policy_score']}/10\n"
                )
                if rule.get("llm_reasoning"):
                    f.write(f"  Why it matters: {rule['llm_reasoning']}\n")
                if rule.get("llm_policy_recommendation"):
                    f.write(f"  Policy recommendation: {rule['llm_policy_recommendation']}\n")
            f.write("\n")

    print(f"Summary written to {out_path}")


# ---------------------------------------------------------------------------
# Predictive rules human-readable summary
# ---------------------------------------------------------------------------

def _write_predictive_summary(df: pd.DataFrame, out_path: Path, top_n: int = 10) -> None:
    """Write a human-readable text summary of the LLM-evaluated predictive rules."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    llm_selected = "llm_rank" in df.columns and df["llm_rank"].notna().any()
    ranked = (
        df[df["llm_rank"].notna()].sort_values("llm_rank").head(top_n)
        if llm_selected
        else df.sort_values("lift", ascending=False).head(top_n)
    )
    with open(out_path, "w") as f:
        f.write("=" * 72 + "\n")
        f.write("  TOP PREDICTIVE RULES — TEMPORAL FORECASTING SIGNALS\n")
        f.write("  UAE COVID-19 PERIOD (March 2020 – December 2021)\n")
        f.write("  LLM-EVALUATED with lead-time and policy recommendations\n")
        f.write("=" * 72 + "\n\n")
        for i, (_, rule) in enumerate(ranked.iterrows(), start=1):
            lead = rule.get("lead_time_days")
            lead_str = f"  Lead time: {int(lead)} days\n" if pd.notna(lead) else ""
            f.write(f"Predictive Rule {i}:\n")
            f.write(f"  IF:   {rule['premise']}\n")
            f.write(f"  THEN: {rule['conclusion']}\n")
            f.write(
                f"  Support: {int(rule['support'])} days ({float(rule['support_pct']):.1f}%)  "
                f"Confidence: {float(rule['confidence']):.1f}%  "
                f"Lift: {float(rule['lift']):.2f}\n"
            )
            if llm_selected and pd.notna(rule.get("llm_rank")):
                f.write(
                    f"  LLM: novelty={rule['novelty_score']}/10  "
                    f"policy_relevance={rule['policy_score']}/10\n"
                )
                f.write(lead_str)
                if rule.get("llm_reasoning"):
                    f.write(f"  Why it matters: {rule['llm_reasoning']}\n")
                if rule.get("llm_policy_recommendation"):
                    f.write(f"  Policy recommendation: {rule['llm_policy_recommendation']}\n")
            f.write("\n")
    print(f"Predictive summary written to {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    _load_env()

    parser = argparse.ArgumentParser(description="Evaluate FCA rules with optional LLM scoring.")
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Enable LLM scoring via OpenAI API (requires OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use for scoring (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=TOP_N_LLM,
        help=f"Number of top rules to select and include in the text summary (default: {TOP_N_LLM}).",
    )
    args = parser.parse_args()

    if not RULES_FILE.exists():
        print(
            f"ERROR: Rules file not found at {RULES_FILE}.\n"
            "Run scripts/fca_analysis.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    df = pd.read_csv(RULES_FILE)
    print(f"Loaded {len(df)} rules from {RULES_FILE}")

    # Refresh / add cross_domain tag (handles files produced before this script existed)
    df = tag_cross_domain(df)

    # ------------------------------------------------------------------
    # Step 1: Prune with hard gates + redundancy removal
    # ------------------------------------------------------------------
    try:
        import importlib.util as _ilu
        _prune_spec = _ilu.find_spec("src.rules.prune_rules")
    except (ModuleNotFoundError, ValueError):
        _prune_spec = None

    if _prune_spec is not None:
        from src.rules.prune_rules import prune_rules, build_prevalence_map  # noqa: PLC0415
        from src.scoring.score_rules import score_rules                       # noqa: PLC0415

        binary_matrix_path = PROJECT_ROOT / "data" / "processed" / "fca_binary_matrix.csv"
        prevalence_map: dict = {}
        if binary_matrix_path.exists():
            import pandas as _pd  # noqa: PLC0415
            bm = _pd.read_csv(binary_matrix_path)
            prevalence_map = build_prevalence_map(bm)

        print(f"\n--- Pruning pass (before: {len(df)} rules) ---")
        df = prune_rules(df, prevalence_map=prevalence_map)
        print(f"--- Scoring {len(df)} surviving rules ---")
        df = score_rules(df)
    else:
        print(
            "NOTE: src.rules.prune_rules not importable "
            "(ensure project root is on PYTHONPATH). Skipping pruning."
        )

    n_cross = int(df["cross_domain"].sum())
    print(f"\n  Cross-domain rules (mobility × emotion): {n_cross}")
    print(f"  Single-domain rules: {len(df) - n_cross}")

    if args.llm:
        print(f"\nRunning LLM batch selection with model '{args.model}' ...")
        df = select_top_rules_with_llm(df, model=args.model, top_n=args.top_n)

    # Final sort: LLM-ranked rows first, then composite_score (if present), then lift
    if "llm_rank" in df.columns and df["llm_rank"].notna().any():
        sort_by = ["llm_rank"]
        sort_asc = [True]
        if "composite_score" in df.columns:
            sort_by.append("composite_score")
            sort_asc.append(False)
        sort_by.append("lift")
        sort_asc.append(False)
        df = df.sort_values(sort_by, ascending=sort_asc, na_position="last").reset_index(drop=True)
    elif "composite_score" in df.columns:
        df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    else:
        df = df.sort_values(
            ["cross_domain", "lift", "confidence", "support"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nEvaluated rules saved to {OUT_CSV}")

    write_summary(df, OUT_TXT, top_n=args.top_n)

    # Also write structured policy briefs if module available
    try:
        from src.reporting.policy_briefs import generate_policy_briefs  # noqa: PLC0415
        briefs_path = PROJECT_ROOT / "results" / "policy_briefs.txt"
        generate_policy_briefs(df, top_n=args.top_n, output_path=briefs_path)
    except ImportError:
        pass

    # ------------------------------------------------------------------
    # Step 4 (optional): LLM evaluation of PREDICTIVE rules
    # ------------------------------------------------------------------
    if args.llm and PRED_RULES_FILE.exists():
        print(f"\n{'='*60}")
        print("PREDICTIVE RULES — LLM evaluation")
        print(f"{'='*60}")
        pred_df = pd.read_csv(PRED_RULES_FILE)
        print(f"Loaded {len(pred_df)} predictive rules from {PRED_RULES_FILE.name}")

        # Score predictive rules statistically if not already done
        if "composite_score" not in pred_df.columns:
            try:
                from src.scoring.score_rules import score_rules  # noqa: PLC0415
                pred_df = score_rules(pred_df)
            except ImportError:
                pass

        print(f"\nRunning LLM evaluation with predictive-rules prompt ...")
        pred_df = select_top_rules_with_llm(
            pred_df,
            model=args.model,
            top_n=10,
            system_prompt=PREDICTIVE_SYSTEM_PROMPT,
        )

        # Sort: LLM-ranked first, then composite_score
        if pred_df["llm_rank"].notna().any():
            sort_by = ["llm_rank"]
            sort_asc = [True]
            if "composite_score" in pred_df.columns:
                sort_by.append("composite_score")
                sort_asc.append(False)
            pred_df = pred_df.sort_values(sort_by, ascending=sort_asc, na_position="last").reset_index(drop=True)

        OUT_PRED_CSV.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(OUT_PRED_CSV, index=False)
        print(f"Predictive rules evaluated and saved to {OUT_PRED_CSV}")

        _write_predictive_summary(pred_df, OUT_PRED_TXT)
    elif args.llm:
        print(f"\nNOTE: Predictive rules file not found at {PRED_RULES_FILE}; skipping predictive LLM step.")

    # Print a quick preview
    print("\nTop 10 rules after evaluation:")
    preview_cols = [c for c in [
        "premise", "conclusion", "cross_domain", "composite_score", "lift", "confidence",
    ] if c in df.columns]
    if "llm_composite" in df.columns:
        preview_cols.append("llm_composite")
    print(df[preview_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
