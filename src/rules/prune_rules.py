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
]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _parse_premise_set(premise_str: str) -> frozenset[str]:
    return frozenset(f.strip() for f in str(premise_str).split(","))


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
# Step 2a — Same-lag artefact filter
# ---------------------------------------------------------------------------

def remove_same_lag_artefacts(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rules where every antecedent feature and the consequent share the
    same lag/lead suffix, making the rule equivalent to a same-day rule just
    shifted in time (e.g. A_lag1, B_lag1 → C_lag1 is NOT a genuine prediction).

    A genuine predictive rule must either:
    - Have a *_lead* consequent (forecasting the future), or
    - Have a current-day (no suffix) consequent with at least one lagged antecedent
      (using the past to predict today).

    Rules with no temporal suffixes at all (same-day rules) are left untouched.
    """
    import re

    def _lag_level(feat: str):
        m = re.search(r"_(lag|lead)(\d+)$", feat)
        return (m.group(1), int(m.group(2))) if m else ("none", 0)

    def _is_same_lag_artefact(row) -> bool:
        prem_feats = [f.strip() for f in str(row["premise"]).split(",")]
        conc = str(row["conclusion"]).strip()
        conc_kind, conc_n = _lag_level(conc)

        # Same-day rules — no temporal suffix anywhere; keep them
        prem_has_temporal = any(_lag_level(f)[0] != "none" for f in prem_feats)
        if conc_kind == "none" and not prem_has_temporal:
            return False

        # Valid: consequent is _lead (genuine future forecast)
        if conc_kind == "lead":
            return False

        # Valid: consequent is current-day AND at least one antecedent is lagged
        if conc_kind == "none" and prem_has_temporal:
            return False

        # Artefact: consequent is _lag and ALL antecedents share the same lag number
        if conc_kind == "lag":
            prem_lag_ns = [_lag_level(f)[1] for f in prem_feats if _lag_level(f)[0] == "lag"]
            if prem_lag_ns and all(n == conc_n for n in prem_lag_ns):
                return True

        return False

    artefact_mask = df.apply(_is_same_lag_artefact, axis=1)
    before = len(df)
    df = df[~artefact_mask].copy().reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"  Same-lag filter : {before} → {len(df)} rules  ({removed} same-lag artefacts removed)")
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

    tautological: list[bool] = []
    for _, row in df.iterrows():
        premise_feats = _parse_premise_set(str(row["premise"]))
        conclusion_feat = str(row["conclusion"]).strip()
        # Strip lag/lead suffixes to check the base feature name
        def _base(f: str) -> str:
            import re
            return re.sub(r"_(lag|lead)\d+$", "", f)
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

    # 2a. Same-lag artefact filter (e.g. A_lag1, B_lag1 → C_lag1 is not a prediction)
    df = remove_same_lag_artefacts(df)
    if df.empty:
        return df

    # 2a2. Pure same-day filter: if the df contains ANY temporal feature, remove
    # rules that have no temporal suffix anywhere — they duplicate same-day analysis.
    has_temporal_col = (
        df["premise"].str.contains(r"_(lag|lead)\d+", regex=True)
        | df["conclusion"].str.contains(r"_(lag|lead)\d+", regex=True)
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
