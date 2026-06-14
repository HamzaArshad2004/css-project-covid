"""
LLM-based rule annotation (NOT selection or ranking).

Loads association rules from FCA. Rule validity is determined ENTIRELY by the
statistical pipeline (support, confidence, lift), multi-stage pruning, and the
70/30 chronological backtest. The optional LLM step is a post-hoc QUALITATIVE
ANNOTATION layer only: it scores each already-valid rule for policy relevance
and writes a plain-language rationale and a named-agency recommendation. It
never selects, ranks, or validates rules. Rule ordering is always statistical.

Usage (without LLM -- statistical ordering only):
    python scripts/evaluate_rules_llm.py

Usage (with LLM annotation -- requires OPENAI_API_KEY in .env or environment):
    python scripts/evaluate_rules_llm.py --llm

Output:
    results/fca/association_rules_evaluated.csv             (all rules, annotated if --llm)
    results/fca/top_cross_domain_rules.txt                  (human-readable summary)
    results/fca/association_rules_predictive_evaluated.csv  (predictive rules, annotated)
    results/fca/top_predictive_rules.txt                    (predictive summary)
"""

import argparse
import json
import os
import re
import sys
import textwrap
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Ensure project root is on sys.path so 'src.*' sub-packages are importable.
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


# ---------------------------------------------------------------------------
# Deterministic temporal helpers (NEVER taken from the LLM)
# ---------------------------------------------------------------------------

def compute_lead_time_days(premise: str, conclusion: str) -> int:
    """Forecast horizon implied by a rule's temporal structure.

    Computed from the rule itself — never from the LLM.
      - ``*_lead{n}`` consequent → +n
      - same-day consequent with ``*_lag{n}`` antecedent → n (largest lag)
      - otherwise → 0
    """
    conclusion = str(conclusion).strip()
    m = re.search(r"_lead(\d+)$", conclusion)
    if m:
        return int(m.group(1))
    if re.search(r"_lag\d+$", conclusion):
        return 0  # backward rule — should not occur post-pruning
    lags = [int(x) for x in re.findall(r"_lag(\d+)", str(premise))]
    return max(lags) if lags else 0


def _is_backward_rule(premise: str, conclusion: str) -> bool:
    """True if the conclusion carries a ``*_lag`` suffix (points to the past)."""
    return bool(re.search(r"_lag\d+$", str(conclusion).strip()))


def _feature_set(rule: pd.Series) -> frozenset:
    """Return the full set of features referenced in a rule (premise + conclusion)."""
    parts = [p.strip() for p in rule["premise"].split(",")]
    parts.append(rule["conclusion"].strip())
    return frozenset(parts)


def tag_cross_domain(df: pd.DataFrame) -> pd.DataFrame:
    """Add / refresh the cross_domain flag. Lag/lead suffixes are stripped
    before domain lookup so e.g. vaccine_mentioned_lead7 still resolves to
    the EMOTION domain."""
    def _base(f: str) -> str:
        return re.sub(r"_(lag|lead)\d+$", "", f.strip())

    def _is_cross(row: pd.Series) -> bool:
        fs = {_base(f) for f in _feature_set(row)}
        return bool(fs & MOBILITY_FEATURES) and bool(fs & EMOTION_FEATURES)

    df = df.copy()
    df["cross_domain"] = df.apply(_is_cross, axis=1)
    return df


# ---------------------------------------------------------------------------
# LLM annotation prompts (annotation only — no selection, no ranking)
# ---------------------------------------------------------------------------

ANNOTATE_SAMEDAY_PROMPT = textwrap.dedent("""\
    You are an expert data scientist and public health policy advisor annotating
    association rules mined from UAE COVID-19 crisis behavior data
    (March 2020 – December 2021). The dataset combines daily Google Community
    Mobility metrics with Reddit sentiment signals.

    IMPORTANT: These rules have ALREADY been validated by a statistical pipeline
    (support, confidence, lift thresholds), multi-stage pruning, and a
    chronological backtest. Your job is NOT to select, rank, validate, or reject
    any rule. Every rule is statistically valid. You provide ONLY a qualitative
    annotation for each: a policy-relevance score, a short rationale, and a
    named-agency recommendation. Annotate EVERY rule; omit none.

    ── UAE COVID-19 POLICY TIMELINE ──────────────────────────────────────────
    Phase 1 — Onset & first lockdown (Mar–May 2020): strict stay-home; curfew;
      grocery stores exempted; grocery spikes; peak fear sentiment.
    Phase 2 — Controlled reopening (Jun–Dec 2020): partial retail/workplace
      reopening; compliance fatigue; second wave Sep–Oct.
    Phase 3 — Vaccine rollout & optimism (Jan–Aug 2021): fast per-capita vaccine
      programme; vaccine_mentioned spikes; positive sentiment recovery.
    Phase 4 — Endemic transition (Sep–Dec 2021): restrictions lifted; mobility
      returns to baseline; residual health_concern persists in discourse.
    ──────────────────────────────────────────────────────────────────────────

    MOBILITY features: mobility_drop_retail, mobility_drop_transit,
      mobility_drop_workplace, residential_increase, grocery_spike,
      severe_lockdown_behavior, partial_restrictions
    SOCIAL/EMOTION features: high_negative_sentiment, dominant_emotion_fear,
      fear_keywords_present, anger_mentioned, anxiety_keywords_present,
      sadness_keywords_present, high_positive_sentiment, mixed_emotions,
      solidarity_messages, sentiment_worsened, sentiment_improved,
      sentiment_shift_detected, covid_topic_detected, lockdown_mentioned,
      vaccine_mentioned, health_concern, compliance_discussed,
      policy_governance_discussion

    ── POLICY RELEVANCE SCORING RUBRIC (1–10) ────────────────────────────────
    This is an annotation aid for the reader; it does NOT determine whether the
    rule is kept (the statistical pipeline already did that).
    10 — Directly actionable by a named UAE body (NCEMA, DHA, MoHAP, WAM) with a
         specific trigger and action.
     8–9 — Operationally useful; named agency and action obvious but needs one
         additional validation step.
     6–7 — Useful monitoring signal; further validation required.
     4–5 — Situational awareness only; no clear intervention opportunity.
     1–3 — Too generic or too slow-changing to drive specific action.
    ──────────────────────────────────────────────────────────────────────────

    For cross-domain rules, note in the rationale which crisis phase the rule
    most relates to and whether the direction is operationally interesting.

    IMPORTANT: Return EXACTLY ONE object per input rule, with the same rule_id.
    The number of objects MUST equal the number of rules provided. Do not omit,
    filter, deduplicate, or reorder by importance.

    Return ONLY a JSON array with one object per rule (any order):
    [
      {
        "rule_id": <int>,
        "policy_score": <int 1-10>,
        "reasoning": "<2-3 sentences: name the UAE phase, note whether the direction is operationally interesting, and what it reveals>",
        "policy_recommendation": "<2 sentences: name a specific UAE agency, the trigger condition, and the recommended action>"
      },
      ...
    ]
    No markdown fences. No extra text. Valid JSON only.
""")


ANNOTATE_PREDICTIVE_PROMPT = textwrap.dedent("""\
    You are an expert data scientist and public health policy advisor annotating
    PREDICTIVE association rules mined from UAE COVID-19 crisis behavior data
    (March 2020 – December 2021). These rules use lagged/lead features and
    describe how today's (or the recent past's) signals forecast future behaviour.

    IMPORTANT: These rules have ALREADY been validated by a statistical pipeline
    and a multi-stage pruning pipeline, and every rule points FORWARD in time.
    Your job is NOT to select, rank, validate, or reject any rule. You provide
    ONLY a qualitative annotation for each: a policy-relevance score, a rationale,
    and a named-agency recommendation. Annotate EVERY rule; omit none.

    ── FEATURE SUFFIX CONVENTIONS ────────────────────────────────────────────
    _lag1, _lag3, _lag7 — feature was active 1/3/7 days AGO
    _lead2, _lead7      — feature will be active in 2/7 days FROM NOW
    No suffix           — feature is active TODAY
    You do NOT report lead time; it is computed separately and deterministically.
    Focus only on POLICY RELEVANCE and the rationale.
    ──────────────────────────────────────────────────────────────────────────

    ── UAE COVID-19 POLICY TIMELINE ──────────────────────────────────────────
    Phase 1 — Onset & first lockdown (Mar–May 2020).
    Phase 2 — Controlled reopening (Jun–Dec 2020); compliance fatigue.
    Phase 3 — Vaccine rollout & optimism (Jan–Aug 2021).
    Phase 4 — Endemic transition (Sep–Dec 2021); residual health_concern.
    ──────────────────────────────────────────────────────────────────────────

    ── POLICY RELEVANCE SCORING RUBRIC (1–10) ────────────────────────────────
    Annotation aid only; does NOT determine whether the rule is kept.
    10 — Monitorable today/past trigger → real-world actionable future outcome;
         names a UAE body (NCEMA, DHA, MoHAP, WAM) and action.
     8–9 — Clear forward signal; operationalisation needs one more step.
     6–7 — Useful monitoring signal; needs further validation.
     4–5 — Same-domain persistence; situational awareness only.
     1–3 — Consequent too generic or slow-changing to drive specific action.
    ──────────────────────────────────────────────────────────────────────────

    Higher relevance: an emotion/discourse signal predicting a MOBILITY outcome
    at a 2–7 day lead, or a mobility signal predicting a future discourse shift.

    IMPORTANT: Return EXACTLY ONE object per input rule, with the same rule_id.
    The number of objects MUST equal the number of rules provided. Do not omit,
    filter, deduplicate, or reorder by importance.

    Return ONLY a JSON array with one object per rule (any order):
    [
      {
        "rule_id": <int>,
        "policy_score": <int 1-10>,
        "reasoning": "<2-3 sentences: name the UAE phase, state what the forecast enables a decision-maker to do>",
        "policy_recommendation": "<2 sentences: name a specific UAE agency, the monitoring trigger, and the pre-emptive action>"
      },
      ...
    ]
    No markdown fences. No extra text. Valid JSON only.
""")


def _build_candidate_block(df: pd.DataFrame) -> str:
    """Format rules as a numbered text block for the LLM. Backward-pointing rules
    (a _lag consequent) that somehow reach here are logged to stderr."""
    backward = df[df.apply(
        lambda r: _is_backward_rule(str(r["premise"]), str(r["conclusion"])), axis=1
    )]
    if not backward.empty:
        print(
            f"\n  *** WARNING: {len(backward)} backward-pointing rule(s) with a "
            f"_lag consequent reached annotation. These are lookbacks, not "
            f"forecasts, and should have been pruned. ***",
            file=sys.stderr,
        )

    lines = []
    for _, row in df.iterrows():
        conviction = row.get("conviction")
        composite = row.get("composite_score")
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


def annotate_all_rules_with_llm(
    df: pd.DataFrame,
    model: str = "gpt-4o-mini",
    system_prompt: str | None = None,
) -> pd.DataFrame:
    """Annotate EVERY rule with the LLM (policy relevance + rationale), dropping none.

    This is an annotation layer, not a selection or ranking step. Rule validity is
    established upstream by the statistical pipeline. Every rule is annotated;
    nothing is filtered. Output ordering is statistical, set by the caller.

    Adds columns: policy_score, llm_reasoning, llm_policy_recommendation,
    lead_time_days. Does NOT add any novelty score or llm_rank.
    """
    try:
        from openai import OpenAI  # noqa: PLC0415
    except ImportError:
        print("ERROR: openai package not installed. Run: pip install openai", file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    df = df.copy().reset_index(drop=True)
    df["rule_id"] = df.index

    candidate_block = _build_candidate_block(df)
    n = len(df)
    user_msg = (
        f"Annotate ALL {n} of the following validated rules. Return exactly {n} "
        f"objects, one per rule_id:\n\n" + candidate_block
    )

    print(f"  Sending ALL {n} rules to LLM for annotation ...")
    active_prompt = system_prompt if system_prompt is not None else ANNOTATE_SAMEDAY_PROMPT
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": active_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
        max_tokens=4000,
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        annotations = json.loads(raw)
    except json.JSONDecodeError:
        print(f"WARNING: LLM returned non-JSON:\n{raw}", file=sys.stderr)
        print("Proceeding with statistical ordering and no annotations.", file=sys.stderr)
        return df
    if not isinstance(annotations, list):
        print("WARNING: LLM response not a JSON array; skipping annotations.", file=sys.stderr)
        return df

    df["policy_score"] = None
    df["llm_reasoning"] = ""
    df["llm_policy_recommendation"] = ""
    df["lead_time_days"] = None

    valid_ids = set(df["rule_id"].tolist())
    seen = set()
    for entry in annotations:
        rid = entry.get("rule_id")
        if rid not in valid_ids:
            continue
        idx = df.index[df["rule_id"] == rid][0]
        df.at[idx, "policy_score"] = entry.get("policy_score")
        df.at[idx, "llm_reasoning"] = entry.get("reasoning", "")
        df.at[idx, "llm_policy_recommendation"] = entry.get("policy_recommendation", "")
        df.at[idx, "lead_time_days"] = compute_lead_time_days(
            str(df.at[idx, "premise"]), str(df.at[idx, "conclusion"])
        )
        seen.add(rid)

    missing = valid_ids - seen
    if missing:
        print(
            f"  WARNING: LLM did not annotate {len(missing)} rule(s): ids {sorted(missing)}. "
            f"They are retained with null annotation.",
            file=sys.stderr,
        )

    print(f"  LLM annotated {df['policy_score'].notna().sum()} / {n} rules.")
    return df


def _statistical_order(df: pd.DataFrame) -> pd.DataFrame:
    """Order rules by objective statistics only. Prefers the pipeline's
    composite_score when present, else cross-domain → lift → confidence.
    The LLM never influences ordering."""
    if "composite_score" in df.columns and df["composite_score"].notna().any():
        return df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    sort_cols = [c for c in ["cross_domain", "lift", "confidence", "support"] if c in df.columns]
    return df.sort_values(sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Human-readable summary writers
# ---------------------------------------------------------------------------

def write_summary(df: pd.DataFrame, out_path: Path, top_n: int | None = None) -> None:
    """Write a text summary of same-day rules, ordered by statistics.

    Inclusion and order are statistical. If the LLM annotation step ran, each
    rule additionally shows a policy annotation — but the LLM never ranks/selects.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    annotated = "policy_score" in df.columns and df["policy_score"].notna().any()

    ranked = _statistical_order(df)
    if top_n is not None:
        ranked = ranked.head(top_n)

    with open(out_path, "w") as f:
        f.write("=" * 72 + "\n")
        f.write("  TOP RULES — CROSS-DOMAIN MOBILITY × EMOTION ANALYSIS\n")
        f.write("  UAE COVID-19 PERIOD\n")
        f.write("  Ordered by statistics; LLM provides policy annotation only.\n")
        f.write("=" * 72 + "\n\n")

        for i, (_, rule) in enumerate(ranked.iterrows(), start=1):
            f.write(f"Rule {i}:  {'[CROSS-DOMAIN] ' if rule.get('cross_domain') else ''}\n")
            f.write(f"  IF:   {rule['premise']}\n")
            f.write(f"  THEN: {rule['conclusion']}\n")
            f.write(
                f"  Support: {int(rule['support'])} days ({float(rule['support_pct']):.1f}%)  "
                f"Confidence: {float(rule['confidence']):.1f}%  Lift: {float(rule['lift']):.2f}\n"
            )
            if annotated and pd.notna(rule.get("policy_score")):
                f.write(f"  Policy relevance (LLM annotation): {int(rule['policy_score'])}/10\n")
                reasoning = rule.get("llm_reasoning", "")
                if reasoning and str(reasoning).strip() not in ("", "nan"):
                    f.write(f"  Insight: {reasoning}\n")
                rec = rule.get("llm_policy_recommendation", "")
                if rec and str(rec).strip() not in ("", "nan"):
                    f.write(f"  Recommendation: {rec}\n")
            f.write("\n")

    print(f"Summary written to {out_path}")


def _write_predictive_summary(df: pd.DataFrame, out_path: Path, top_n: int | None = None) -> None:
    """Write a human-readable summary of predictive rules, ordered by statistics."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    annotated = "policy_score" in df.columns and df["policy_score"].notna().any()

    ranked = _statistical_order(df)
    if top_n is not None:
        ranked = ranked.head(top_n)

    with open(out_path, "w") as f:
        f.write("=" * 72 + "\n")
        f.write("  TOP PREDICTIVE RULES — TEMPORAL FORECASTING SIGNALS\n")
        f.write("  UAE COVID-19 PERIOD (March 2020 – December 2021)\n")
        f.write("  Ordered by statistics; LLM provides policy annotation only.\n")
        f.write("=" * 72 + "\n\n")
        for i, (_, rule) in enumerate(ranked.iterrows(), start=1):
            lead = compute_lead_time_days(str(rule["premise"]), str(rule["conclusion"]))
            lead_str = f"  Lead time: {int(lead)} days\n" if lead else "  Lead time: same-day\n"
            f.write(f"Predictive Rule {i}:\n")
            f.write(f"  IF:   {rule['premise']}\n")
            f.write(f"  THEN: {rule['conclusion']}\n")
            f.write(
                f"  Support: {int(rule['support'])} days ({float(rule['support_pct']):.1f}%)  "
                f"Confidence: {float(rule['confidence']):.1f}%  "
                f"Lift: {float(rule['lift']):.2f}\n"
            )
            f.write(lead_str)
            if annotated and pd.notna(rule.get("policy_score")):
                f.write(f"  Policy relevance (LLM annotation): {int(rule['policy_score'])}/10\n")
                reasoning = rule.get("llm_reasoning", "")
                if reasoning and str(reasoning).strip() not in ("", "nan"):
                    f.write(f"  Insight: {reasoning}\n")
                rec = rule.get("llm_policy_recommendation", "")
                if rec and str(rec).strip() not in ("", "nan"):
                    f.write(f"  Recommendation: {rec}\n")
            f.write("\n")
    print(f"Predictive summary written to {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    _load_env()

    parser = argparse.ArgumentParser(
        description="Annotate FCA rules with optional LLM policy commentary (no selection/ranking)."
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Enable LLM annotation via OpenAI API (requires OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Optionally limit the text summary to the top-N rules by statistics "
             "(default: show all).",
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

    df = tag_cross_domain(df)

    # ------------------------------------------------------------------
    # Step 1: Prune with hard gates + redundancy removal, then score.
    # This is where rule VALIDITY is established — not the LLM.
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
            bm = pd.read_csv(binary_matrix_path)
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

    # ------------------------------------------------------------------
    # Step 2 (optional): LLM ANNOTATION of the validated same-day rules.
    # ------------------------------------------------------------------
    if args.llm:
        print(f"\nAnnotating ALL {len(df)} validated same-day rules with '{args.model}' ...")
        df = annotate_all_rules_with_llm(df, model=args.model, system_prompt=ANNOTATE_SAMEDAY_PROMPT)

    # Always store in statistical order — the LLM never reorders.
    df = _statistical_order(df)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nEvaluated (annotated) rules saved to {OUT_CSV}")

    write_summary(df, OUT_TXT, top_n=args.top_n)

    # Structured policy briefs if module available
    try:
        from src.reporting.policy_briefs import generate_policy_briefs  # noqa: PLC0415
        briefs_path = PROJECT_ROOT / "results" / "policy_briefs.txt"
        generate_policy_briefs(df, top_n=args.top_n, output_path=briefs_path)
    except (ImportError, TypeError):
        pass

    # ------------------------------------------------------------------
    # Step 3 (optional): LLM ANNOTATION of PREDICTIVE rules.
    # ------------------------------------------------------------------
    if args.llm and PRED_RULES_FILE.exists():
        print(f"\n{'='*60}")
        print("PREDICTIVE RULES — LLM annotation")
        print(f"{'='*60}")
        pred_df = pd.read_csv(PRED_RULES_FILE)
        print(f"Loaded {len(pred_df)} predictive rules from {PRED_RULES_FILE.name}")

        pred_df = tag_cross_domain(pred_df)

        if "composite_score" not in pred_df.columns:
            try:
                from src.scoring.score_rules import score_rules  # noqa: PLC0415
                pred_df = score_rules(pred_df)
            except ImportError:
                pass

        print(f"\nAnnotating ALL {len(pred_df)} predictive rules ...")
        pred_df = annotate_all_rules_with_llm(
            pred_df,
            model=args.model,
            system_prompt=ANNOTATE_PREDICTIVE_PROMPT,
        )

        pred_df = _statistical_order(pred_df)

        OUT_PRED_CSV.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(OUT_PRED_CSV, index=False)
        print(f"Predictive rules annotated and saved to {OUT_PRED_CSV}")

        _write_predictive_summary(pred_df, OUT_PRED_TXT, top_n=args.top_n)
    elif args.llm:
        print(f"\nNOTE: Predictive rules file not found at {PRED_RULES_FILE}; skipping predictive step.")

    # Quick preview
    print("\nRules after annotation (statistical order):")
    preview_cols = [c for c in [
        "premise", "conclusion", "cross_domain", "composite_score", "lift", "confidence",
        "policy_score",
    ] if c in df.columns]
    print(df[preview_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()