"""Composite multi-objective scoring for association rules.

Score formula
-------------
composite_score = w_stat  * stat_strength
                + w_temp  * temporal_validity
                + w_cross * cross_domain
                - w_obv   * obviousness_penalty

Where:
  stat_strength      — min-max normalised average of support, confidence, lift, leverage
  temporal_validity  — 1.0 if antecedent contains lagged features (_lagN),
                       0.5 if same-day cross-domain,
                       0.0 if same-day same-domain
  cross_domain       — 1.0 if rule spans both mobility and emotion domains
  obviousness_penalty— 0.0–1.0 from prune_rules.tag_obviousness

Usage
-----
CLI:
    python -m src.scoring.score_rules
    python -m src.scoring.score_rules --input results/fca/association_rules_pruned.csv

API:
    from src.scoring.score_rules import score_rules
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RULES_IN = PROJECT_ROOT / "results" / "fca" / "association_rules_pruned.csv"
RULES_OUT = PROJECT_ROOT / "results" / "fca" / "association_rules_scored.csv"

# Composite score weights (must sum to 1.0 when obviousness_penalty == 0)
W_STAT_STRENGTH: float = 0.40
W_TEMPORAL_VALIDITY: float = 0.35
W_CROSS_DOMAIN: float = 0.15
W_OBVIOUSNESS_PENALTY: float = 0.10   # subtracted


def _normalize(series: pd.Series) -> pd.Series:
    """Min-max normalise to [0, 1]; returns 0.5 if all values are equal."""
    lo, hi = float(series.min()), float(series.max())
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    return (series.astype(float) - lo) / (hi - lo)


def compute_stat_strength(df: pd.DataFrame) -> pd.Series:
    """Normalised composite of support_pct, confidence, lift, and (if present) leverage."""
    components: list[pd.Series] = []
    for col in ["support_pct", "confidence", "lift"]:
        if col in df.columns:
            components.append(_normalize(df[col]))
    if "leverage" in df.columns:
        components.append(_normalize(df["leverage"].clip(lower=0)))
    if not components:
        return pd.Series(0.5, index=df.index)
    return pd.concat(components, axis=1).mean(axis=1)


def compute_temporal_validity(df: pd.DataFrame) -> pd.Series:
    """Assign temporal validity score per rule.

    1.0 — antecedent contains ``_lagN`` columns  (truly predictive)
    0.5 — same-day cross-domain rule
    0.0 — same-day same-domain rule
    """
    scores: list[float] = []
    for _, row in df.iterrows():
        premise = str(row.get("premise", ""))
        conclusion = str(row.get("conclusion", ""))
        is_lagged = "_lag" in premise or "_lead" in conclusion
        is_cross = bool(row.get("cross_domain", False))
        if is_lagged:
            scores.append(1.0)
        elif is_cross:
            scores.append(0.5)
        else:
            scores.append(0.0)
    return pd.Series(scores, index=df.index)


def score_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ``composite_score`` and intermediate components; return sorted DataFrame.

    New columns added:
      stat_strength, temporal_validity, composite_score
    """
    df = df.copy()

    stat = compute_stat_strength(df)
    temporal = compute_temporal_validity(df)
    cross = df.get("cross_domain", pd.Series(False, index=df.index)).astype(float)
    obv = df.get("obviousness_penalty", pd.Series(0.0, index=df.index)).astype(float)

    df["stat_strength"] = stat
    df["temporal_validity"] = temporal
    df["composite_score"] = (
        W_STAT_STRENGTH * stat
        + W_TEMPORAL_VALIDITY * temporal
        + W_CROSS_DOMAIN * cross
        - W_OBVIOUSNESS_PENALTY * obv
    ).clip(0.0, 1.0)

    return df.sort_values("composite_score", ascending=False).reset_index(drop=True)


def print_sanity_dashboard(df: pd.DataFrame) -> None:
    """Print a distribution summary of key rule metrics."""
    n = len(df)
    if n == 0:
        print("  No rules to summarise.")
        return

    n_lagged = int(
        df.apply(
            lambda r: "_lag" in str(r.get("premise", ""))
                      or "_lead" in str(r.get("conclusion", "")),
            axis=1,
        ).sum()
    )
    n_cross = int(df.get("cross_domain", pd.Series(False)).sum())
    top_consequents = df["conclusion"].value_counts().head(3)

    print("\n--- Sanity Dashboard ---")
    print(f"  Total rules            : {n}")
    print(f"  Predictive (lagged)    : {n_lagged}  ({100 * n_lagged / n:.0f}%)")
    print(f"  Cross-domain           : {n_cross}  ({100 * n_cross / n:.0f}%)")
    print(f"  Median support         : {df['support_pct'].median():.1f}%")
    print(f"  Median confidence      : {df['confidence'].median():.1f}%")
    print(f"  Median lift            : {df['lift'].median():.2f}x")
    print(f"  Top consequents        :")
    for feat, cnt in top_consequents.items():
        pct = 100 * cnt / n
        flag = "  *** dominates >40%" if pct > 40 else ""
        print(f"    {feat}: {cnt} rules ({pct:.0f}%){flag}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score association rules with composite metric.")
    parser.add_argument("--input", default=str(RULES_IN))
    parser.add_argument("--output", default=str(RULES_OUT))
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rules from {args.input}")

    scored = score_rules(df)
    print_sanity_dashboard(scored)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out_path, index=False)
    print(f"\nScored rules saved to {out_path}")

    preview_cols = [c for c in [
        "premise", "conclusion", "composite_score", "temporal_validity",
        "stat_strength", "lift", "confidence", "support_pct",
    ] if c in scored.columns]
    print("\nTop 15 rules by composite_score:")
    print(scored[preview_cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
