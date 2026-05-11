"""Generate lagged and lead (future) binary feature columns for predictive rule mining.

Lagged features capture past states of emotion/social signals (antecedent candidates).
Lead (future) features capture upcoming mobility/outcome states to predict (consequent candidates).

The resulting "predictive matrix" lets the FCA rule miner discover rules of the form:
    IF <social_signal>_lag3 AND <mobility_signal> THEN <outcome>_lead2

Usage
-----
CLI:
    python -m src.features.lagged_features
    python -m src.features.lagged_features --lags 1 3 7 --leads 2 7

API:
    from src.features.lagged_features import build_predictive_matrix, get_lead_feature_names
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BINARY_MATRIX = PROJECT_ROOT / "data" / "processed" / "fca_binary_matrix.csv"
PREDICTIVE_MATRIX_OUT = PROJECT_ROOT / "data" / "processed" / "fca_predictive_matrix.csv"

# ---------------------------------------------------------------------------
# Feature role definitions
# ---------------------------------------------------------------------------

# Emotion/social features — used as LEADING INDICATOR antecedents (lagged)
SOCIAL_PREDICTOR_FEATURES: list[str] = [
    "dominant_emotion_fear",
    "high_negative_sentiment",
    "high_positive_sentiment",
    "solidarity_messages",
    "compliance_discussed",
    "health_concern",
    "sentiment_shift_detected",
    "sentiment_worsened",
    "sentiment_improved",
    "lockdown_mentioned",
    "vaccine_mentioned",
    "policy_governance_discussion",
    "fear_keywords_present",
    "anxiety_keywords_present",
]

# Mobility features — used as OUTCOME TARGETS (shifted forward into lead columns)
MOBILITY_TARGET_FEATURES: list[str] = [
    "grocery_spike",
    "mobility_drop_retail",
    "mobility_drop_transit",
    "mobility_drop_workplace",
    "residential_increase",
    "severe_lockdown_behavior",
    "partial_restrictions",
]

# Social features can also be outcomes (bidirectional: mobility predicts mood)
SOCIAL_TARGET_FEATURES: list[str] = [
    "sentiment_shift_detected",
    "sentiment_worsened",
    "sentiment_improved",
    "high_negative_sentiment",
    "solidarity_messages",
    "compliance_discussed",
]

# Default lag/lead windows (days)
DEFAULT_LAG_DAYS: list[int] = [1, 3, 7]
DEFAULT_LEAD_DAYS: list[int] = [2, 7]


def build_predictive_matrix(
    df: pd.DataFrame,
    lag_features: list[str] | None = None,
    target_features: list[str] | None = None,
    lag_days: list[int] = DEFAULT_LAG_DAYS,
    lead_days: list[int] = DEFAULT_LEAD_DAYS,
) -> pd.DataFrame:
    """Build an expanded binary matrix with lagged predictor and future lead columns.

    Parameters
    ----------
    df:
        Daily binary feature matrix; first column is expected to be 'Date'.
    lag_features:
        Features to generate lagged versions of (antecedent candidates).
        Defaults to ``SOCIAL_PREDICTOR_FEATURES``.
    target_features:
        Features to generate future-lead versions of (consequent candidates).
        Defaults to ``MOBILITY_TARGET_FEATURES + SOCIAL_TARGET_FEATURES``.
    lag_days:
        How many days back to shift predictor features.
    lead_days:
        How many days ahead to shift target features.

    Returns
    -------
    pd.DataFrame
        Expanded matrix with NaN boundary rows dropped.  All feature columns
        are integer (0/1).  Original same-day columns are preserved alongside
        the lagged/lead variants so contemporaneous rules can still be mined.
    """
    if lag_features is None:
        lag_features = SOCIAL_PREDICTOR_FEATURES
    if target_features is None:
        target_features = MOBILITY_TARGET_FEATURES + SOCIAL_TARGET_FEATURES

    result = df.copy()
    date_col = result.columns[0]

    # Resolve only features that actually exist AND are binary (0/1) in this matrix
    all_cols = set(result.columns)

    def _is_binary(col: str) -> bool:
        vals = pd.to_numeric(result[col], errors="coerce")
        if vals.isna().any():
            return False
        return set(vals.dropna().unique()).issubset({0, 1, 0.0, 1.0})

    lag_features = [f for f in lag_features if f in all_cols and _is_binary(f)]
    target_features = [f for f in target_features if f in all_cols and _is_binary(f)]

    # Add lagged versions of predictor features
    for feat in lag_features:
        if feat not in result.columns:
            continue
        for lag in lag_days:
            result[f"{feat}_lag{lag}"] = result[feat].shift(lag)

    # Add future-lead versions of target features
    for feat in target_features:
        if feat not in result.columns:
            continue
        for lead in lead_days:
            result[f"{feat}_lead{lead}"] = result[feat].shift(-lead)

    # Drop rows with NaN (start/end boundaries introduced by shifting)
    result = result.dropna(subset=[c for c in result.columns if "_lag" in c or "_lead" in c]).reset_index(drop=True)

    # Cast only the newly created lag/lead columns to int
    new_cols = [c for c in result.columns if "_lag" in c or "_lead" in c]
    result[new_cols] = result[new_cols].astype(int)

    return result


def get_lead_feature_names(
    target_features: list[str] | None = None,
    lead_days: list[int] = DEFAULT_LEAD_DAYS,
) -> list[str]:
    """Return all ``_leadN`` column names that act as valid predictive consequents."""
    if target_features is None:
        target_features = MOBILITY_TARGET_FEATURES + SOCIAL_TARGET_FEATURES
    return [f"{feat}_lead{lead}" for feat in target_features for lead in lead_days]


def get_lag_feature_names(
    lag_features: list[str] | None = None,
    lag_days: list[int] = DEFAULT_LAG_DAYS,
) -> list[str]:
    """Return all ``_lagN`` column names that act as lagged antecedent candidates."""
    if lag_features is None:
        lag_features = SOCIAL_PREDICTOR_FEATURES
    return [f"{feat}_lag{lag}" for feat in lag_features for lag in lag_days]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build predictive (lagged/lead) binary feature matrix."
    )
    parser.add_argument(
        "--input", default=str(BINARY_MATRIX), help="Input binary matrix CSV."
    )
    parser.add_argument(
        "--output", default=str(PREDICTIVE_MATRIX_OUT), help="Output CSV path."
    )
    parser.add_argument(
        "--lags",
        nargs="+",
        type=int,
        default=DEFAULT_LAG_DAYS,
        help="Lag days to generate (default: 1 3 7).",
    )
    parser.add_argument(
        "--leads",
        nargs="+",
        type=int,
        default=DEFAULT_LEAD_DAYS,
        help="Lead days to generate (default: 2 7).",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded binary matrix: {df.shape[0]} rows x {df.shape[1]} columns")

    pred_df = build_predictive_matrix(df, lag_days=args.lags, lead_days=args.leads)

    n_lag = sum(1 for c in pred_df.columns if "_lag" in c)
    n_lead = sum(1 for c in pred_df.columns if "_lead" in c)
    n_same = pred_df.shape[1] - n_lag - n_lead - 1  # minus Date col

    print(
        f"Predictive matrix: {pred_df.shape[0]} rows x {pred_df.shape[1]} columns\n"
        f"  Same-day features  : {n_same}\n"
        f"  Lagged antecedents : {n_lag}\n"
        f"  Lead targets       : {n_lead}\n"
        f"  (Rows lost to NaN boundaries: {df.shape[0] - pred_df.shape[0]})"
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
