"""Rule quality filtering and redundancy pruning.

Pipeline
--------
1. Hard statistical gates  (support, confidence, lift, leverage, conviction)
2. High base-rate consequent removal
3. Obviousness tagging     (calendar-only antecedent, same-domain, high-prevalence)
4. Subsumption pruning     (drop complex rule if simpler rule has similar confidence)
5. Per-consequent cap      (max N rules per unique consequent)

Usage
-----
CLI:
    python -m src.rules.prune_rules
    python -m src.rules.prune_rules --min-support 15 --min-lift 2.0

API:
    from src.rules.prune_rules import prune_rules
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RULES_IN = PROJECT_ROOT / "results" / "fca" / "association_rules.csv"
RULES_OUT = PROJECT_ROOT / "results" / "fca" / "association_rules_pruned.csv"

# ---------------------------------------------------------------------------
# Default quality-gate thresholds
# ---------------------------------------------------------------------------
MIN_SUPPORT_PCT: float = 12.0       # % of analysis days
MIN_CONFIDENCE: float = 75.0        # % (0-100 scale)
MIN_LIFT: float = 1.8
MIN_LEVERAGE: float = 0.0           # strictly positive association
MIN_CONVICTION: float = 1.1         # conviction = (1 - p_c) / (1 - conf)

MAX_RULES_PER_CONSEQUENT: int = 3
SUBSUMPTION_CONF_TOLERANCE: float = 0.05   # 5 percentage-point band

# Features treated as "calendar-only" antecedents (trigger penalty)
CALENDAR_FEATURES: frozenset[str] = frozenset({"weekend"})

# Consequents active on more than this fraction of days are penalised
HIGH_PREVALENCE_THRESHOLD: float = 0.40

# Feature pairs where one is definitionally derived from the other.
# Predictive rules whose antecedent *and* consequent both contain members of
# the same group are tautological (they describe definitional overlap, not a
# discovered pattern) and are removed.
TAUTOLOGY_GROUPS: list[frozenset[str]] = [
    # All mobility-drop features co-occur during the same lockdown event;
    # any rule predicting one from another (including across lead windows)
    # is a lockdown-persistence artefact, not a discovery.
    frozenset({
        "severe_lockdown_behavior",
        "mobility_drop_workplace",
        "mobility_drop_retail",
        "mobility_drop_transit",
        "residential_increase",
        "partial_restrictions",
    }),
    frozenset({
        "calm_mobile_baseline",
        "emotion_with_mobility_signal",
        "emotion_mobility_mismatch",
    }),
    # sentiment_shift_detected is computed as abs(delta) > 0.2, while
    # sentiment_improved (delta > 0.2) and sentiment_worsened (delta < -0.2)
    # are the two signed halves of that same threshold on the same variable.
    # The shift flag is therefore logically implied by either directional flag,
    # so any rule pairing them is definitional collinearity, not a discovery
    # (e.g. sentiment_shift_detected → sentiment_worsened just reads off the sign).
    frozenset({
        "sentiment_shift_detected",
        "sentiment_improved",
        "sentiment_worsened",
    }),
]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _parse_premise_set(premise_str: str) -> frozenset[str]:
    return frozenset(f.strip() for f in str(premise_str).split(","))


def _lag_level(feat: str) -> tuple[str, int]:
    """Return the temporal kind and magnitude of a feature suffix.

    ('lead', n) for ``*_leadN``  — points to the future
    ('lag',  n) for ``*_lagN``   — points to the past
    ('none', 0) for an unsuffixed (same-day) feature
    """
    m = re.search(r"_(lag|lead)(\d+)$", feat.strip())
    return (m.group(1), int(m.group(2))) if m else ("none", 0)


def is_backward_rule(premise: str, conclusion: str) -> bool:
    """True if a rule's conclusion points backward (or sideways) in time.

    A genuine predictive rule must forecast forward:
      - consequent is ``*_lead``                       (forecast the future), OR
      - consequent is same-day AND >=1 antecedent is ``*_lag``
                                                        (use the past to call today).
    Anything with a ``*_lag`` consequent dates the conclusion *earlier* than the
    premise — a lookback, not a forecast — and is a backward rule.
    Pure same-day rules (no temporal suffix anywhere) are NOT backward; they are
    handled by the same-day filter, so this returns False for them.
    """
    prem_feats = [f.strip() for f in str(premise).split(",")]
    conc_kind, _ = _lag_level(str(conclusion).strip())
    prem_has_temporal = any(_lag_level(f)[0] != "none" for f in prem_feats)

    if conc_kind == "none" and not prem_has_temporal:
        return False                      # same-day rule — not backward
    if conc_kind == "lead":
        return False                      # forward forecast — valid
    if conc_kind == "none" and prem_has_temporal:
        return False                      # past → today — valid
    # conc_kind == "lag"  →  conclusion is dated before the premise → backward
    return True


def warn_on_backward_rules(df: pd.DataFrame, context: str) -> pd.DataFrame:
    """Loudly log any backward (``*_lag`` consequent) rules that reach a stage
    where they should already have been pruned. Returns ``df`` unchanged so it
    can be dropped inline into a pipeline without altering behaviour.
    """
    if df.empty or "premise" not in df.columns or "conclusion" not in df.columns:
        return df
    mask = df.apply(
        lambda r: is_backward_rule(str(r["premise"]), str(r["conclusion"])), axis=1
    )
    n = int(mask.sum())
    if n:
        print(
            f"\n  *** WARNING [{context}]: {n} backward-pointing rule(s) detected "
            f"with a _lag consequent. These are lookbacks, not forecasts, and "
            f"should have been pruned upstream. Listing up to 10: ***",
            file=sys.stderr,
        )
        cols = [c for c in ["premise", "conclusion", "confidence", "lift"] if c in df.columns]
        print(df.loc[mask, cols].head(10).to_string(index=False), file=sys.stderr)
    return df


def compute_conviction(confidence_pct: float, p_consequent: float) -> float:
    """conviction = (1 - P(consequent)) / (1 - confidence).

    Returns ``inf`` when confidence == 1.0 (perfect rule).
    """
    conf = confidence_pct / 100.0
    if conf >= 1.0:
        return float("inf")
    if p_consequent >= 1.0:
        return 1.0
    return (1.0 - p_consequent) / (1.0 - conf)


def add_conviction(df: pd.DataFrame, prevalence_map: dict[str, float]) -> pd.DataFrame:
    """Compute and attach conviction if not already present."""
    if "conviction" in df.columns:
        return df
    df = df.copy()
    df["conviction"] = df.apply(
        lambda r: compute_conviction(
            float(r["confidence"]),
            prevalence_map.get(str(r["conclusion"]).strip(), 0.5),
        ),
        axis=1,
    )
    return df


# ---------------------------------------------------------------------------
# Step 1 — Hard statistical gates
# ---------------------------------------------------------------------------

def apply_hard_filters(
    df: pd.DataFrame,
    min_support_pct: float = MIN_SUPPORT_PCT,
    min_confidence: float = MIN_CONFIDENCE,
    min_lift: float = MIN_LIFT,
    min_leverage: float = MIN_LEVERAGE,
    min_conviction: float = MIN_CONVICTION,
) -> pd.DataFrame:
    """Remove rules below any hard statistical threshold."""
    before = len(df)
    df = df[df["support_pct"].astype(float) >= min_support_pct].copy()
    df = df[df["confidence"].astype(float) >= min_confidence].copy()
    df = df[df["lift"].astype(float) >= min_lift].copy()
    if "leverage" in df.columns:
        df = df[df["leverage"].astype(float) > min_leverage].copy()
    if "conviction" in df.columns:
        finite_conv = df["conviction"].replace(float("inf"), 999.0).astype(float)
        df = df[finite_conv >= min_conviction].copy()
    print(f"  Hard filters    : {before} → {len(df)} rules  ({before - len(df)} removed)")
    return df


# ---------------------------------------------------------------------------
# Step 2 — Obviousness tagging
# ---------------------------------------------------------------------------

def tag_obviousness(
    df: pd.DataFrame,
    prevalence_map: dict[str, float],
) -> pd.DataFrame:
    """Attach an ``obviousness_penalty`` column (0.0 – 1.0).

    Penalties accumulate:
    +0.3  antecedent consists entirely of calendar features (e.g., ``weekend``)
    +0.2  consequent is a high-prevalence feature (active > 40 % of days)
    +0.2  rule is same-domain (cross_domain == False or missing)
    """
    df = df.copy()
    penalties: list[float] = []
    for _, row in df.iterrows():
        penalty = 0.0
        premise_feats = _parse_premise_set(str(row["premise"]))

        if premise_feats.issubset(CALENDAR_FEATURES):
            penalty += 0.3

        p_c = prevalence_map.get(str(row["conclusion"]).strip(), 0.0)
        if p_c > HIGH_PREVALENCE_THRESHOLD:
            penalty += 0.2

        if not bool(row.get("cross_domain", False)):
            penalty += 0.2

        penalties.append(min(penalty, 1.0))
    df["obviousness_penalty"] = penalties
    return df


# ---------------------------------------------------------------------------
# Step 2a — Backward-rule filter (lookbacks masquerading as predictions)
# ---------------------------------------------------------------------------

def remove_same_lag_artefacts(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rules whose conclusion points backward or sideways in time.

    A genuine predictive rule must forecast the future:
    - consequent is ``*_lead``  (forecasting forward), OR
    - consequent is same-day AND at least one antecedent is ``*_lag``
      (using the past to predict today).

    Any rule with a ``*_lag`` consequent is a lookback, not a forecast — the
    conclusion is dated earlier than the premise — and is removed regardless of
    the antecedent lags. (The previous version only removed such a rule when
    *every* antecedent shared the *same* lag as the consequent, which let rules
    like ``grocery_spike, lockdown_mentioned → vaccine_mentioned_lag7`` survive
    with same-day antecedents and a backward conclusion.)

    Pure same-day rules (no temporal suffix anywhere) are left untouched.
    """

    def _is_artefact(row) -> bool:
        return is_backward_rule(str(row["premise"]), str(row["conclusion"]))

    artefact_mask = df.apply(_is_artefact, axis=1)
    before = len(df)
    df = df[~artefact_mask].copy().reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"  Backward filter : {before} → {len(df)} rules  ({removed} lookback/same-lag artefacts removed)")
    return df


# ---------------------------------------------------------------------------
# Step 2b — Data-leakage filter (predictive rules only)
# ---------------------------------------------------------------------------

def remove_leaky_predictive_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rules where the antecedent contains *_lead* features.

    In a lagged/lead predictive matrix the antecedent must be made up of
    present-day or *_lag* features only.  Any rule with a *_lead* term in
    the premise uses future information at prediction time — data leakage.
    Rules that contain no lag/lead features at all (same-day rules) are
    left untouched.
    """
    has_temporal = df["premise"].str.contains(r"_lag\d|_lead\d", regex=True)
    if not has_temporal.any():
        return df  # no temporal rules; nothing to do

    leaky_mask = df["premise"].str.contains(r"_lead\d", regex=True) & has_temporal
    before = len(df)
    df = df[~leaky_mask].copy().reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"  Leakage filter  : {before} → {len(df)} rules  ({removed} antecedents contained _lead features)")
    return df


# ---------------------------------------------------------------------------
# Step 2b2 — Within-variable persistence filter
# ---------------------------------------------------------------------------

def remove_within_variable_persistence(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rules where the consequent's BASE feature also appears (at any
    lag/lead) in the antecedent.

    Example artefacts this catches that string-equality checks miss:
        grocery_spike                     -> grocery_spike_lead7
        vaccine_mentioned_lag1            -> vaccine_mentioned
        sentiment_improved, high_positive_sentiment_lag7 -> high_positive_sentiment

    These describe a single variable's autocorrelation across time (the value
    persists), not a relationship between distinct phenomena. The forecast adds
    no information beyond "this state tends to last." Any *other* antecedent
    feature in such a rule is typically doing no work — confidence is already
    driven by the self-overlap (note the 100%-confidence cases).

    A rule survives only if the consequent's base feature is entirely absent
    from the antecedent's base features.
    """
    def _base(f: str) -> str:
        return re.sub(r"_(lag|lead)\d+$", "", f.strip())

    def _is_persistence(row) -> bool:
        prem_bases = {_base(f) for f in str(row["premise"]).split(",")}
        conc_base = _base(str(row["conclusion"]))
        return conc_base in prem_bases

    mask = df.apply(_is_persistence, axis=1)
    before = len(df)
    df = df[~mask].copy().reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"  Persistence flt : {before} → {len(df)} rules  ({removed} within-variable persistence artefacts removed)")
    return df


# ---------------------------------------------------------------------------
# Step 2c — Tautology filter (definitionally collinear features)
# ---------------------------------------------------------------------------

def remove_tautological_rules(
    df: pd.DataFrame,
    tautology_groups: list[frozenset[str]] | None = None,
) -> pd.DataFrame:
    """Remove rules where antecedent and consequent both draw from the same
    definitionally-derived feature group (e.g. severe_lockdown_behavior and
    mobility_drop_workplace).
    """
    if tautology_groups is None:
        tautology_groups = TAUTOLOGY_GROUPS

    def _base(f: str) -> str:
        return re.sub(r"_(lag|lead)\d+$", "", f.strip())

    tautological: list[bool] = []
    for _, row in df.iterrows():
        premise_feats = _parse_premise_set(str(row["premise"]))
        conclusion_feat = str(row["conclusion"]).strip()
        base_premise = {_base(f) for f in premise_feats}
        base_conclusion = _base(conclusion_feat)
        is_tautological = any(
            base_conclusion in g and bool(base_premise & g)
            for g in tautology_groups
        )
        tautological.append(is_tautological)

    before = len(df)
    df = df[~pd.Series(tautological, index=df.index)].copy().reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"  Tautology filter: {before} → {len(df)} rules  ({removed} removed — definitionally collinear)")
    return df


# ---------------------------------------------------------------------------
# Step 3 — Bidirectional rule deduplication
# ---------------------------------------------------------------------------

def remove_bidirectional_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the weaker direction when both A→B and B→A exist.

    Keeps the rule with higher conviction (more informative direction).
    When conviction is equal (or both are ``inf``), keeps the one with
    higher confidence.  Only applies to single-feature antecedents.
    """
    df = df.copy().reset_index(drop=True)
    to_drop: set[int] = set()

    for i, row_i in df.iterrows():
        if i in to_drop:
            continue
        prem_i = _parse_premise_set(str(row_i["premise"]))
        if len(prem_i) != 1:  # only deduplicate single-feature antecedents
            continue
        conc_i = str(row_i["conclusion"]).strip()
        feat_i = next(iter(prem_i))

        for j, row_j in df.iterrows():
            if j <= i or j in to_drop:
                continue
            prem_j = _parse_premise_set(str(row_j["premise"]))
            conc_j = str(row_j["conclusion"]).strip()
            if len(prem_j) == 1 and next(iter(prem_j)) == conc_i and conc_j == feat_i:
                # Bidirectional pair found — drop the weaker one
                conv_i = float(row_i.get("conviction") or 0)
                conv_j = float(row_j.get("conviction") or 0)
                if conv_i == float("inf") and conv_j == float("inf"):
                    # fall back to confidence
                    conv_i = float(row_i.get("confidence") or 0)
                    conv_j = float(row_j.get("confidence") or 0)
                to_drop.add(j if conv_i >= conv_j else i)

    before = len(df)
    df = df.drop(index=list(to_drop)).reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"  Bidirectional   : {before} → {len(df)} rules  ({removed} weaker-direction duplicates removed)")
    return df


# ---------------------------------------------------------------------------
# Step 4 — Subsumption pruning
# ---------------------------------------------------------------------------

def remove_subsumed(
    df: pd.DataFrame,
    conf_tolerance: float = SUBSUMPTION_CONF_TOLERANCE,
) -> pd.DataFrame:
    """Drop rules subsumed by a simpler rule with similar confidence.

    If  ``A → C``  and  ``{A, B} → C``  both exist and their confidence
    differs by at most ``conf_tolerance`` (as a fraction, not pct), the
    more complex ``{A, B} → C`` is removed.
    """
    df = df.copy().reset_index(drop=True)
    to_drop: set[int] = set()

    for conclusion, group in df.groupby("conclusion"):
        idxs = group.index.tolist()
        for i in range(len(idxs)):
            if idxs[i] in to_drop:
                continue
            premise_i = _parse_premise_set(df.loc[idxs[i], "premise"])
            conf_i = float(df.loc[idxs[i], "confidence"])
            for j in range(len(idxs)):
                if i == j or idxs[j] in to_drop:
                    continue
                premise_j = _parse_premise_set(df.loc[idxs[j], "premise"])
                conf_j = float(df.loc[idxs[j], "confidence"])
                # premise_i ⊂ premise_j  and  conf is within tolerance → drop j
                if premise_i < premise_j and abs(conf_i - conf_j) / 100.0 <= conf_tolerance:
                    to_drop.add(idxs[j])

    before = len(df)
    df = df.drop(index=list(to_drop)).reset_index(drop=True)
    print(f"  Subsumption     : {before} → {len(df)} rules  ({len(to_drop)} removed)")
    return df


# ---------------------------------------------------------------------------
# Step 4 — Per-consequent cap
# ---------------------------------------------------------------------------

def cap_per_consequent(
    df: pd.DataFrame,
    max_rules: int = MAX_RULES_PER_CONSEQUENT,
) -> pd.DataFrame:
    """Keep only the top ``max_rules`` rules per consequent, ranked by score."""
    sort_col = "composite_score" if "composite_score" in df.columns else "lift"
    df = (
        df.sort_values(sort_col, ascending=False)
        .groupby("conclusion", group_keys=False)
        .head(max_rules)
        .reset_index(drop=True)
    )
    return df


# ---------------------------------------------------------------------------
# Full pruning pipeline
# ---------------------------------------------------------------------------

def prune_rules(
    df: pd.DataFrame,
    prevalence_map: dict[str, float] | None = None,
    min_support_pct: float = MIN_SUPPORT_PCT,
    min_confidence: float = MIN_CONFIDENCE,
    min_lift: float = MIN_LIFT,
    min_leverage: float = MIN_LEVERAGE,
    min_conviction: float = MIN_CONVICTION,
    max_rules_per_consequent: int = MAX_RULES_PER_CONSEQUENT,
) -> pd.DataFrame:
    """Full pruning pipeline: hard gates → conviction → obviousness → subsumption → cap.

    Parameters
    ----------
    df:
        Raw association rules produced by ``fca_analysis.py``.
    prevalence_map:
        Maps feature name → fraction of days the feature is active.
        Used for conviction computation and obviousness scoring.
        Pass ``None`` to skip conviction filtering and obviousness tagging.
    """
    if prevalence_map is None:
        prevalence_map = {}

    # 1. Conviction
    df = add_conviction(df, prevalence_map)

    # 2. Hard gates
    df = apply_hard_filters(
        df,
        min_support_pct=min_support_pct,
        min_confidence=min_confidence,
        min_lift=min_lift,
        min_leverage=min_leverage,
        min_conviction=min_conviction,
    )
    if df.empty:
        print("  WARNING: No rules survived hard filters.")
        return df

    # 2a. Backward-rule filter: any _lag consequent is a lookback, not a forecast
    df = remove_same_lag_artefacts(df)
    if df.empty:
        return df

    # 2a2. Pure same-day filter: if the df contains ANY temporal feature, remove
    # rules that have no temporal suffix anywhere — they duplicate same-day analysis.
    has_temporal_col = (
        df["premise"].str.contains(r"_(?:lag|lead)\d+", regex=True)
        | df["conclusion"].str.contains(r"_(?:lag|lead)\d+", regex=True)
    )
    if has_temporal_col.any():
        before = len(df)
        df = df[has_temporal_col].copy().reset_index(drop=True)
        removed = before - len(df)
        if removed:
            print(f"  Same-day filter : {before} → {len(df)} rules  ({removed} same-day rules removed from predictive set)")
    if df.empty:
        return df

    # 2b. Data-leakage filter (removes _lead features from antecedents)
    df = remove_leaky_predictive_rules(df)
    if df.empty:
        return df

    # 2b2. Within-variable persistence filter (e.g. grocery_spike → grocery_spike_lead7)
    df = remove_within_variable_persistence(df)
    if df.empty:
        return df

    # 2c. Tautology filter (collinear composite features)
    df = remove_tautological_rules(df)
    if df.empty:
        return df

    # 3. Obviousness tagging (non-filtering; used by scorer)
    df = tag_obviousness(df, prevalence_map)

    # 3b. Bidirectional deduplication
    df = remove_bidirectional_duplicates(df)

    # 4. Subsumption pruning
    df = remove_subsumed(df)

    # 5. Per-consequent cap
    df = cap_per_consequent(df, max_rules=max_rules_per_consequent)

    # Final safety net: nothing backward should remain. Log loudly if it does.
    df = warn_on_backward_rules(df, context="prune_rules:final")

    print(f"  After pruning   : {len(df)} rules remain")
    return df.reset_index(drop=True)


def build_prevalence_map(binary_df: pd.DataFrame) -> dict[str, float]:
    """Compute feature prevalence from the raw binary matrix (binary columns only)."""
    date_col = binary_df.columns[0]
    features = binary_df.drop(columns=[date_col])
    result: dict[str, float] = {}
    for col in features.columns:
        numeric = pd.to_numeric(features[col], errors="coerce")
        if numeric.isna().any():
            continue
        unique = set(numeric.dropna().unique())
        if unique.issubset({0, 1, 0.0, 1.0}):
            result[col] = float(numeric.mean())
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Prune association rules.")
    parser.add_argument("--input", default=str(RULES_IN))
    parser.add_argument("--output", default=str(RULES_OUT))
    parser.add_argument("--matrix",
                        default=str(PROJECT_ROOT / "data" / "processed" / "fca_binary_matrix.csv"),
                        help="Binary matrix CSV for computing feature prevalence.")
    parser.add_argument("--min-support", type=float, default=MIN_SUPPORT_PCT,
                        help="Minimum support %% (default: %(default)s)")
    parser.add_argument("--min-confidence", type=float, default=MIN_CONFIDENCE,
                        help="Minimum confidence %% (default: %(default)s)")
    parser.add_argument("--min-lift", type=float, default=MIN_LIFT,
                        help="Minimum lift (default: %(default)s)")
    parser.add_argument("--min-conviction", type=float, default=MIN_CONVICTION,
                        help="Minimum conviction (default: %(default)s)")
    parser.add_argument("--max-per-consequent", type=int, default=MAX_RULES_PER_CONSEQUENT,
                        help="Max rules per consequent (default: %(default)s)")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rules from {args.input}")

    prevalence_map: dict[str, float] = {}
    matrix_path = Path(args.matrix)
    if matrix_path.exists():
        binary_df = pd.read_csv(matrix_path)
        prevalence_map = build_prevalence_map(binary_df)
        print(f"Built prevalence map from {len(prevalence_map)} features in {matrix_path.name}")

    pruned = prune_rules(
        df,
        prevalence_map=prevalence_map,
        min_support_pct=args.min_support,
        min_confidence=args.min_confidence,
        min_lift=args.min_lift,
        min_conviction=args.min_conviction,
        max_rules_per_consequent=args.max_per_consequent,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pruned.to_csv(out_path, index=False)
    print(f"\nPruned rules saved to {out_path}")

    if not pruned.empty:
        top_cols = [c for c in ["premise", "conclusion", "support_pct", "confidence", "lift",
                                "conviction", "cross_domain"] if c in pruned.columns]
        print(pruned[top_cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()