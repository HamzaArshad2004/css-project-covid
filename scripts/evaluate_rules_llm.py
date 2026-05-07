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
RULES_FILE = PROJECT_ROOT / "results" / "fca" / "association_rules.csv"
OUT_CSV = PROJECT_ROOT / "results" / "fca" / "association_rules_evaluated.csv"
OUT_TXT = PROJECT_ROOT / "results" / "fca" / "top_cross_domain_rules.txt"

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

    WHAT MAKES A RULE GENUINELY VALUABLE:
    - The conclusion domain DIFFERS from the dominant premise domain.
      Best cases: SOCIAL/EMOTION features in premise → MOBILITY conclusion,
      or MOBILITY features in premise → SOCIAL/EMOTION conclusion.
    - The relationship is directional and behaviorally meaningful, not just
      pandemic co-occurrence (e.g., both features being active during the same
      wave does not count as a discovery).
    - The finding would be actionable: it tells an official WHEN to intervene
      or WHAT social signal predicts a real-world mobility change (or vice versa).

    MANDATORY EXCLUSIONS — do NOT select rules where:
    - residential_increase is the conclusion and any mobility_drop is in the premise
      (mechanically implied: less out-of-home time = more time at home).
    - Both premise and conclusion are purely mobility features, even with one emotion feature added
      (e.g., mobility_drop_retail + fear → mobility_drop_transit during the same lockdown period
      is just "lockdown happened twice" — all mobility drops co-occur because the same event causes them).
    - covid_topic_detected or dominant_emotion_fear is the conclusion — both are active
      ~85% of all days in the dataset (near-constant), making any rule predicting them
      statistically inflated but informationally empty.
    - vaccine_mentioned is the conclusion and both premise features are from the same
      pandemic phase (calendar co-occurrence, not a causal signal).
    - The rule is a near-duplicate of a higher-ranked rule; prefer the most specific
      and actionable variant.
    - Conclusion is trivially expected from premise semantics
      (e.g., lockdown_mentioned → mobility drops; sentiment_worsened → high_negative_sentiment).
    - MORE THAN 2 rules in your top 10 share the same conclusion feature. Deliberately
      spread the conclusions across different outcome variables.

    PREFER rules that reveal:
    - Social discourse or emotional state that PREDICTS mobility change (leading indicator).
    - Mobility patterns that PREDICT changes in social sentiment or discussion topics.
    - Weekend or temporal modulation of crisis response.
    - Surprising combinations (e.g., solidarity messages co-occurring with compliance
      drops, or grocery spikes predicting sentiment shifts).

    Your task: from the numbered candidate rules below, select and rank the
    TOP 20 that best satisfy the above criteria.

    Return ONLY a JSON array of exactly 20 objects, ordered best-first:
    [
      {
        "rule_id": <int — the id field from the candidate list>,
        "novelty_score": <int 1-10>,
        "policy_score": <int 1-10>,
        "reasoning": "<1-2 sentences: what makes this finding novel — be specific about the directional insight>",
        "policy_recommendation": "<1-2 concrete sentences: specific action a UAE government or public health official should take>"
      },
      ...
    ]
    No markdown fences. No extra text. Valid JSON only.
""")


def _build_candidate_block(df: pd.DataFrame) -> str:
    """Format a DataFrame of candidate rules as a numbered text block for the LLM."""
    lines = []
    for _, row in df.iterrows():
        lines.append(
            f"[id={row['rule_id']}]  "
            f"IF: {row['premise']}  "
            f"THEN: {row['conclusion']}  "
            f"| support={float(row['support_pct']):.1f}%  "
            f"conf={float(row['confidence']):.1f}%  "
            f"lift={float(row['lift']):.2f}  "
            f"cross_domain={bool(row.get('cross_domain', False))}"
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
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": BATCH_SYSTEM_PROMPT},
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

    n_cross = int(df["cross_domain"].sum())
    print(f"  Cross-domain rules (mobility × emotion): {n_cross}")
    print(f"  Single-domain rules: {len(df) - n_cross}")

    if args.llm:
        print(f"\nRunning LLM batch selection with model '{args.model}' ...")
        df = select_top_rules_with_llm(df, model=args.model, top_n=args.top_n)

    # Final sort: LLM-ranked rows first, then statistical fallback
    if "llm_rank" in df.columns:
        df = df.sort_values(
            ["llm_rank", "cross_domain", "lift"],
            ascending=[True, False, False],
            na_position="last",
        ).reset_index(drop=True)
    else:
        df = df.sort_values(
            ["cross_domain", "lift", "confidence", "support"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nEvaluated rules saved to {OUT_CSV}")

    write_summary(df, OUT_TXT, top_n=args.top_n)

    # Print a quick preview
    print("\nTop 10 rules after evaluation:")
    preview_cols = ["premise", "conclusion", "cross_domain", "lift", "confidence"]
    if "llm_composite" in df.columns:
        preview_cols.append("llm_composite")
    print(df[preview_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
