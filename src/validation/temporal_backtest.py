"""Temporal train/holdout backtest for rule stability.

Splits the binary matrix at a configurable cutpoint (default: first 70 % of days),
re-evaluates each rule's confidence and lift on both partitions, and marks rules as
"stable" when the confidence drop stays within a tolerance band.

Usage
-----
CLI:
    python -m src.validation.temporal_backtest
    python -m src.validation.temporal_backtest --train-fraction 0.70 --max-drop 10

API:
    from src.validation.temporal_backtest import run_backtest
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BINARY_MATRIX = PROJECT_ROOT / "data" / "processed" / "fca_binary_matrix.csv"

# Prefer LLM-evaluated rules if present; fall back to composite-scored rules.
_EVALUATED = PROJECT_ROOT / "results" / "fca" / "association_rules_evaluated.csv"
_SCORED    = PROJECT_ROOT / "results" / "fca" / "association_rules_scored.csv"
RULES_IN = _EVALUATED if _EVALUATED.exists() else _SCORED

STABLE_RULES_OUT = PROJECT_ROOT / "results" / "fca" / "association_rules_stable.csv"
BACKTEST_REPORT_OUT = PROJECT_ROOT / "results" / "fca" / "backtest_report.csv"

TRAIN_FRACTION: float = 0.70
MAX_CONFIDENCE_DROP_PCT: float = 10.0   # percentage points


def _evaluate_rule(
    binary_df: pd.DataFrame,
    premise_features: list[str],
    conclusion_feature: str,
) -> dict[str, float]:
    """Compute support/confidence/lift for a single rule on a binary DataFrame.

    Returns an empty dict if any feature is missing or the premise never fires.
    """
    available = set(binary_df.columns)
    missing = [f for f in premise_features if f not in available]
    if missing or conclusion_feature not in available:
        return {}

    n = len(binary_df)
    premise_mask = binary_df[premise_features].all(axis=1)
    support_count = int(premise_mask.sum())

    if support_count == 0:
        return {"support_count": 0, "support_pct": 0.0, "confidence": 0.0, "lift": 0.0}

    conclusion_hits = int(binary_df.loc[premise_mask, conclusion_feature].sum())
    confidence = conclusion_hits / support_count
    p_conclusion = float(binary_df[conclusion_feature].mean())
    lift = confidence / p_conclusion if p_conclusion > 0 else 0.0

    return {
        "support_count": support_count,
        "support_pct": support_count / n * 100.0,
        "confidence": confidence * 100.0,
        "lift": lift,
    }


def run_backtest(
    rules_df: pd.DataFrame,
    binary_df: pd.DataFrame,
    train_fraction: float = TRAIN_FRACTION,
    max_confidence_drop_pct: float = MAX_CONFIDENCE_DROP_PCT,
) -> pd.DataFrame:
    """Evaluate rules on training and holdout splits and flag stable rules.

    Columns added to the returned DataFrame:
    - train_support_pct, train_confidence, train_lift
    - holdout_support_pct, holdout_confidence, holdout_lift
    - confidence_drop   (train_confidence − holdout_confidence, in pp)
    - stable            (True if confidence_drop ≤ max_confidence_drop_pct)

    Parameters
    ----------
    rules_df:
        Scored association rules.
    binary_df:
        Raw binary feature matrix **without** the date column.
    train_fraction:
        Fraction of rows (chronological) to use as training set.
    max_confidence_drop_pct:
        Maximum allowed drop in confidence (in percentage points) from train
        to holdout before a rule is considered unstable.
    """
    n = len(binary_df)
    split_idx = int(n * train_fraction)
    train_df = binary_df.iloc[:split_idx].reset_index(drop=True)
    holdout_df = binary_df.iloc[split_idx:].reset_index(drop=True)

    print(
        f"  Train set  : {len(train_df)} days "
        f"({100 * len(train_df) / n:.0f}% of {n})"
    )
    print(
        f"  Holdout set: {len(holdout_df)} days "
        f"({100 * len(holdout_df) / n:.0f}% of {n})"
    )

    records: list[dict] = []
    for _, row in rules_df.iterrows():
        premise_feats = [f.strip() for f in str(row["premise"]).split(",")]
        conclusion_feat = str(row["conclusion"]).strip()

        train_stats = _evaluate_rule(train_df, premise_feats, conclusion_feat)
        holdout_stats = _evaluate_rule(holdout_df, premise_feats, conclusion_feat)

        train_conf = train_stats.get("confidence", float("nan"))
        holdout_conf = holdout_stats.get("confidence", float("nan"))

        if np.isnan(train_conf) or np.isnan(holdout_conf):
            drop = float("nan")
            stable = False
        else:
            drop = float(train_conf) - float(holdout_conf)
            stable = drop <= max_confidence_drop_pct

        records.append(
            {
                "train_support_pct": train_stats.get("support_pct", float("nan")),
                "train_confidence": train_conf,
                "train_lift": train_stats.get("lift", float("nan")),
                "holdout_support_pct": holdout_stats.get("support_pct", float("nan")),
                "holdout_confidence": holdout_conf,
                "holdout_lift": holdout_stats.get("lift", float("nan")),
                "confidence_drop": drop,
                "stable": stable,
            }
        )

    backtest_cols = pd.DataFrame(records, index=rules_df.index)
    combined = pd.concat([rules_df.reset_index(drop=True), backtest_cols], axis=1)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporal backtest for rule stability.")
    parser.add_argument("--rules", default=str(RULES_IN),
                        help="Input rules CSV (default: evaluated > scored, whichever exists).")
    parser.add_argument("--matrix", default=str(BINARY_MATRIX), help="Binary feature matrix CSV.")
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=TRAIN_FRACTION,
        help="Fraction of days for the training set (default: %(default)s).",
    )
    parser.add_argument(
        "--max-drop",
        type=float,
        default=MAX_CONFIDENCE_DROP_PCT,
        help="Max confidence drop in pp to be considered stable (default: %(default)s).",
    )
    parser.add_argument("--output", default=str(STABLE_RULES_OUT))
    args = parser.parse_args()

    rules_df = pd.read_csv(args.rules)
    binary_df = pd.read_csv(args.matrix)

    # Drop date column
    date_col = binary_df.columns[0]
    binary_df = binary_df.drop(columns=[date_col])

    print(f"Loaded {len(rules_df)} rules | {len(binary_df)} daily observations")

    combined = run_backtest(
        rules_df,
        binary_df,
        train_fraction=args.train_fraction,
        max_confidence_drop_pct=args.max_drop,
    )

    n_stable = int(combined["stable"].sum())
    n_total = len(combined)
    print(
        f"\nBacktest: {n_stable}/{n_total} rules are stable "
        f"(conf drop ≤ {args.max_drop:.0f} pp)"
    )

    # Save full report
    BACKTEST_REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(BACKTEST_REPORT_OUT, index=False)
    print(f"Full backtest report saved to {BACKTEST_REPORT_OUT}")

    # Save stable-only subset
    stable_rules = combined[combined["stable"]].reset_index(drop=True)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stable_rules.to_csv(out_path, index=False)
    print(f"Stable rules saved to {out_path}  ({len(stable_rules)} rules)")

    # Diagnose unstable rules
    unstable = combined[~combined["stable"]].sort_values("confidence_drop", ascending=False)
    if not unstable.empty:
        diag_cols = [c for c in [
            "premise", "conclusion", "train_confidence",
            "holdout_confidence", "confidence_drop",
        ] if c in unstable.columns]
        print(f"\nUnstable rules ({len(unstable)}) — largest confidence drop first:")
        print(unstable[diag_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
