"""Generate structured operational policy briefs from association rules.

Each brief covers:
  - Signal window    : how long to monitor antecedent conditions
  - Lead time        : expected days before the outcome manifests
  - Threshold        : what level of signal triggers the alert
  - Action playbook  : 1-2 concrete interventions
  - Risk of false alarm

Usage
-----
CLI:
    python -m src.reporting.policy_briefs
    python -m src.reporting.policy_briefs --rules results/fca/association_rules_stable.csv

API:
    from src.reporting.policy_briefs import generate_policy_briefs
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Prefer stable (backtest-validated) rules; fall back to evaluated, then scored.
_STABLE    = PROJECT_ROOT / "results" / "fca" / "association_rules_stable.csv"
_EVALUATED = PROJECT_ROOT / "results" / "fca" / "association_rules_evaluated.csv"
_SCORED    = PROJECT_ROOT / "results" / "fca" / "association_rules_scored.csv"
RULES_IN = _STABLE if _STABLE.exists() else (_EVALUATED if _EVALUATED.exists() else _SCORED)
BRIEFS_OUT = PROJECT_ROOT / "results" / "policy_briefs.txt"


# ---------------------------------------------------------------------------
# Human-readable descriptions for each feature (base name, without _lagN/_leadN)
# ---------------------------------------------------------------------------
FEATURE_DESCRIPTIONS: dict[str, str] = {
    "dominant_emotion_fear":        "dominant fear sentiment in public discourse",
    "high_negative_sentiment":      "high negative sentiment in social media",
    "high_positive_sentiment":      "high positive sentiment in social media",
    "solidarity_messages":          "surge in solidarity / community support messages",
    "compliance_discussed":         "compliance with restrictions actively discussed",
    "health_concern":               "health concern keywords trending",
    "sentiment_shift_detected":     "detectable shift in public sentiment",
    "sentiment_worsened":           "worsening public sentiment",
    "sentiment_improved":           "improving public sentiment",
    "lockdown_mentioned":           "lockdown / restriction terms trending",
    "vaccine_mentioned":            "vaccine-related discourse active",
    "policy_governance_discussion": "policy governance discussion trending",
    "fear_keywords_present":        "fear keywords present in posts",
    "anxiety_keywords_present":     "anxiety keywords present in posts",
    "anger_mentioned":              "anger keywords in social media",
    "sadness_keywords_present":     "sadness keywords present in posts",
    "mixed_emotions":               "mixed emotional tone in discourse",
    "covid_topic_detected":         "COVID-19 topic actively discussed",
    "grocery_spike":                "spike in grocery / retail mobility",
    "mobility_drop_retail":         "drop in retail & recreation mobility",
    "mobility_drop_transit":        "drop in transit station mobility",
    "mobility_drop_workplace":      "drop in workplace mobility",
    "residential_increase":         "increase in residential dwell time",
    "severe_lockdown_behavior":     "severe lockdown-consistent mobility pattern",
    "partial_restrictions":         "partial mobility restriction pattern",
    "calm_mobile_baseline": "calm-sentiment / unimpaired-mobility baseline period",
    "emotion_with_mobility_signal": "emotion–mobility co-signal active",
    "weekend":                      "weekend day",
}


def _describe_feature(feat: str) -> str:
    """Human-readable label for a (possibly _lagN / _leadN suffixed) feature."""
    base = re.sub(r"_(lag\d+|lead\d+)$", "", feat)
    suffix = re.search(r"_(lag(\d+)|lead(\d+))$", feat)
    desc = FEATURE_DESCRIPTIONS.get(base, base.replace("_", " "))
    if suffix:
        if suffix.group(2):   # lag
            desc = f"{desc}  [{suffix.group(2)} days ago]"
        elif suffix.group(3): # lead
            desc = f"{desc}  [in {suffix.group(3)} days]"
    return desc


def _extract_lead_days(conclusion: str) -> int | None:
    m = re.search(r"_lead(\d+)$", conclusion)
    return int(m.group(1)) if m else None


def _extract_max_lag(premise_features: list[str]) -> int | None:
    lags = [int(m.group(1)) for f in premise_features if (m := re.search(r"_lag(\d+)$", f))]
    return max(lags) if lags else None


def _false_alarm_risk(confidence: float, lift: float) -> str:
    if confidence >= 85 and lift >= 2.5:
        return "LOW   — high confidence and lift; primary signal"
    if confidence >= 75 and lift >= 1.8:
        return "MEDIUM — moderate certainty; corroborate with secondary signals"
    return "HIGH  — borderline thresholds; treat as supplementary signal only"


def _default_action(
    conclusion: str,
    is_predictive: bool,
    is_cross_domain: bool,
    lead_days: int | None,
) -> str:
    base = re.sub(r"_(lag\d+|lead\d+)$", "", conclusion)
    lead_str = f" (expected in ~{lead_days} days)" if lead_days else ""
    if "grocery_spike" in base:
        return (
            f"Notify grocery retailers and supply-chain coordinators to build stock "
            f"reserves{lead_str}; stagger opening hours to reduce crowding risk."
        )
    if "mobility_drop" in base or "lockdown" in base or "severe_lockdown" in base:
        return (
            f"Alert mobility-management and logistics teams{lead_str}; "
            "pre-position resources and draft public communication on upcoming restrictions."
        )
    if "sentiment_worsened" in base or "high_negative" in base:
        return (
            f"Activate public-communications team{lead_str}; "
            "prepare reassurance messaging and identify grievance sources on social platforms."
        )
    if "solidarity" in base or "compliance" in base:
        return (
            f"Amplify community solidarity and compliance campaigns{lead_str}; "
            "partner with local influencers to reinforce positive public behaviour."
        )
    if is_predictive and is_cross_domain:
        return (
            f"Activate preparedness protocol{lead_str}: brief operations teams, "
            "prepare public communication, and increase monitoring frequency."
        )
    return (
        "Flag for policy review team; combine with secondary indicators "
        "before triggering a formal response."
    )


def format_rule_brief(rule: pd.Series, rule_number: int) -> str:
    """Format a single rule as an operational policy brief."""
    premise_features = [f.strip() for f in str(rule["premise"]).split(",")]
    conclusion = str(rule["conclusion"]).strip()

    lead_days = _extract_lead_days(conclusion)
    max_lag = _extract_max_lag(premise_features)
    is_lagged = max_lag is not None or lead_days is not None
    is_cross = bool(rule.get("cross_domain", False))

    support_pct = float(rule.get("support_pct", 0))
    confidence = float(rule.get("confidence", 0))
    lift = float(rule.get("lift", 0))
    support_count = int(rule.get("support", support_pct))  # fallback

    # Signal window
    if max_lag:
        signal_window = f"Rolling {max_lag}-day lookback window"
    else:
        signal_window = "Daily monitoring (same-day signal)"

    # Lead time description
    if lead_days is not None:
        lead_str = f"{lead_days} days ahead — PREDICTIVE rule"
    elif is_lagged:
        lead_str = "Lagged antecedent — allows early detection before outcome"
    else:
        lead_str = "Same-day (contemporaneous correlation; not strictly predictive)"

    # Threshold
    threshold_str = (
        f"Alert when antecedent features are co-active on a day "
        f"(base rate: {support_pct:.0f}% of historical days)"
    )

    # Action
    llm_rec = rule.get("llm_policy_recommendation", "")
    if llm_rec and str(llm_rec) not in ("nan", "", "None"):
        action = str(llm_rec)
    else:
        action = _default_action(conclusion, is_lagged, is_cross, lead_days)

    false_alarm = _false_alarm_risk(confidence, lift)

    antecedent_lines = "\n".join(
        f"    • {_describe_feature(f)}" for f in premise_features
    )

    tag = ""
    if is_lagged or lead_days is not None:
        tag += "[PREDICTIVE] "
    if is_cross:
        tag += "[CROSS-DOMAIN] "
    if not tag:
        tag = "[CONTEMPORANEOUS] "

    lines = [
        "=" * 72,
        f"RULE {rule_number}: {tag.strip()}",
        "",
        "  WHEN THESE CONDITIONS ARE OBSERVED:",
        antecedent_lines,
        "",
        f"  EXPECTED OUTCOME:  {_describe_feature(conclusion)}",
        "",
        "  STATISTICS:",
        f"    Support   : {support_pct:.1f}% of analysis days",
        f"    Confidence: {confidence:.1f}%",
        f"    Lift      : {lift:.2f}x  (baseline rate)",
    ]

    for extra in ["conviction", "leverage", "composite_score"]:
        val = rule.get(extra)
        if val is not None and str(val) not in ("nan", "None"):
            try:
                lines.append(f"    {extra.capitalize():<12}: {float(val):.3f}")
            except (ValueError, TypeError):
                pass

    lines += [
        "",
        f"  SIGNAL WINDOW  : {signal_window}",
        f"  LEAD TIME      : {lead_str}",
        f"  TRIGGER        : {threshold_str}",
        "",
        "  ACTION PLAYBOOK:",
        f"    {action}",
        "",
        f"  FALSE ALARM RISK: {false_alarm}",
        "",
    ]
    return "\n".join(lines)


def generate_policy_briefs(
    df: pd.DataFrame,
    top_n: int = 15,
    output_path: Path | None = None,
) -> str:
    """Generate a multi-rule policy brief document.

    Parameters
    ----------
    df:
        Scored (and optionally LLM-evaluated) rules.
    top_n:
        Number of rules to include.
    output_path:
        If given, write the brief to this file in addition to returning the text.

    Returns
    -------
    str
        Full brief text.
    """
    # Sort priority: LLM rank → composite_score → lift
    if "llm_rank" in df.columns and df["llm_rank"].notna().any():
        ranked = df.sort_values("llm_rank", na_position="last").head(top_n)
    elif "composite_score" in df.columns:
        ranked = df.sort_values("composite_score", ascending=False).head(top_n)
    else:
        ranked = df.sort_values(
            ["cross_domain", "lift"], ascending=[False, False]
        ).head(top_n)

    header_lines = [
        "=" * 72,
        "  OPERATIONAL POLICY BRIEFS",
        "  UAE COVID-19 CRISIS BEHAVIORAL ANALYSIS",
        f"  Top {min(top_n, len(ranked))} Rules — Actionable Intelligence for Policymakers",
        "=" * 72,
        "",
        "PURPOSE: Each brief translates a data-mining finding into a structured",
        "operational signal for UAE government and public health officials.",
        "",
    ]

    sections = ["\n".join(header_lines)]
    for i, (_, rule) in enumerate(ranked.iterrows(), start=1):
        sections.append(format_rule_brief(rule, i))

    # --- Sanity dashboard ---
    n = len(ranked)
    n_predictive = int(
        ranked.apply(
            lambda r: "_lag" in str(r.get("premise", ""))
                      or "_lead" in str(r.get("conclusion", "")),
            axis=1,
        ).sum()
    )
    n_cross = int(ranked.get("cross_domain", pd.Series(False)).sum())
    top_consequents = ranked["conclusion"].value_counts().head(5)

    dashboard_lines = [
        "=" * 72,
        "  SANITY DASHBOARD",
        "=" * 72,
        f"  Rules in brief        : {n}",
        f"  Predictive (lagged)   : {n_predictive}  ({100 * n_predictive / max(n, 1):.0f}%)",
        f"  Cross-domain          : {n_cross}  ({100 * n_cross / max(n, 1):.0f}%)",
        f"  Median support        : {ranked['support_pct'].median():.1f}%",
        f"  Median confidence     : {ranked['confidence'].median():.1f}%",
        f"  Median lift           : {ranked['lift'].median():.2f}x",
        "",
        "  Consequent distribution:",
    ]
    for feat, cnt in top_consequents.items():
        pct = 100 * cnt / max(n, 1)
        flag = "  *** consider rebalancing" if pct > 40 else ""
        dashboard_lines.append(f"    {feat}: {cnt} ({pct:.0f}%){flag}")
    dashboard_lines.append("")

    sections.append("\n".join(dashboard_lines))

    full_text = "\n".join(sections)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as fh:
            fh.write(full_text)
        print(f"Policy briefs written to {output_path}")

    return full_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate structured policy briefs.")
    parser.add_argument("--rules", default=str(RULES_IN), help="Input rules CSV.")
    parser.add_argument("--output", default=str(BRIEFS_OUT), help="Output text file.")
    parser.add_argument("--top-n", type=int, default=15)
    args = parser.parse_args()

    rules_path = Path(args.rules)
    if not rules_path.exists():
        # Fall back to scored rules if stable not yet available
        fallback = PROJECT_ROOT / "results" / "fca" / "association_rules_scored.csv"
        if fallback.exists():
            print(f"Note: {rules_path.name} not found; using {fallback.name}")
            rules_path = fallback
        else:
            print(f"ERROR: Rules file not found at {args.rules}")
            return

    df = pd.read_csv(rules_path)
    print(f"Loaded {len(df)} rules from {rules_path}")

    full_text = generate_policy_briefs(df, top_n=args.top_n, output_path=Path(args.output))
    print("\n--- Preview ---")
    print(full_text[:1500])


if __name__ == "__main__":
    main()
