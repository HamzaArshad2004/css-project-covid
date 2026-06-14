"""
Script for Formal Concept Analysis and Galois Lattice generation
Requires: pip install pandas numpy concepts networkx matplotlib
"""

import math
import re
import pandas as pd
from concepts import Context
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BINARY_MATRIX = PROJECT_ROOT / "data" / "processed" / "fca_binary_matrix.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "fca"

# ---------------------------------------------------------------------------
# Tautology definitions: a rule "premise → conclusion" is definitionally true
# (and should be excluded) if the premise contains at least one feature from
# EVERY component group listed for that conclusion.
# ---------------------------------------------------------------------------
TAUTOLOGY_DEFINITIONS: dict[str, list[set[str]]] = {
    # emotion_with_mobility_signal = (any_emotion) AND (any_mobility_signal)
    "emotion_with_mobility_signal": [
        {"dominant_emotion_fear", "fear_keywords_present", "high_negative_sentiment",
         "anger_mentioned", "anxiety_keywords_present", "sadness_keywords_present"},
        {"mobility_drop_retail", "mobility_drop_transit", "mobility_drop_workplace",
         "residential_increase", "grocery_spike", "severe_lockdown_behavior",
         "partial_restrictions"},
    ],
    # emotion_mobility_mismatch = (any_emotion) AND NOT (any_mobility_signal)
    "emotion_mobility_mismatch": [
        {"dominant_emotion_fear", "fear_keywords_present", "high_negative_sentiment",
         "anger_mentioned", "anxiety_keywords_present", "sadness_keywords_present"},
        {"mobility_drop_retail", "mobility_drop_transit", "mobility_drop_workplace",
         "residential_increase", "grocery_spike", "severe_lockdown_behavior",
         "partial_restrictions"},
    ],
    # severe_lockdown_behavior = retail_drop AND workplace_drop AND residential_increase
    "severe_lockdown_behavior": [
        {"mobility_drop_retail", "mobility_drop_workplace", "residential_increase"},
    ],
    # partial_restrictions = (retail OR workplace OR transit drop) AND NOT severe_lockdown
    "partial_restrictions": [
        {"mobility_drop_retail", "mobility_drop_workplace", "mobility_drop_transit"},
    ],
    # SLB requires retail/workplace/residential by definition → concluding them is tautological.
    # Also block concluding retail or workplace from each other + lockdown co-occurrence:
    # all three mobility drops are driven by the same lockdown event.
    "mobility_drop_retail": [
        {"severe_lockdown_behavior", "mobility_drop_transit", "mobility_drop_workplace"},
    ],
    "mobility_drop_workplace": [
        {"severe_lockdown_behavior"},
    ],
    # Google Mobility computes residential as complement of out-of-home time →
    # any mobility drop mechanically raises residential time.
    "residential_increase": [
        {"mobility_drop_retail", "mobility_drop_transit", "mobility_drop_workplace",
         "severe_lockdown_behavior", "emotion_with_mobility_signal"},
    ],
    # mobility_drop_transit co-occurs with other drops in the same lockdown event.
    # Also block emotion_with_mobility_signal (already contains a mobility drop by definition)
    # and residential_increase (Google Mobility computes it as complement of out-of-home time,
    # so any mobility drop raises residential — the same lockdown causes both).
    "mobility_drop_transit": [
        {"mobility_drop_retail", "mobility_drop_workplace",
         "severe_lockdown_behavior", "partial_restrictions",
         "emotion_with_mobility_signal", "residential_increase"},
    ],
    # sentiment_shift_detected = sentiment_improved OR sentiment_worsened
    "sentiment_shift_detected": [
        {"sentiment_improved", "sentiment_worsened"},
    ],
    # calm_mobile_baseline = NOT(negative emotion) AND NOT(mobility disruption).
    # Represents days of routine activity and calm public sentiment — a
    # positive behavioural-state indicator, not a residual catch-all.
    # Cannot hold when any mobility drop feature is in the premise.
    "calm_mobile_baseline": [
        {"mobility_drop_retail", "mobility_drop_transit", "mobility_drop_workplace",
         "severe_lockdown_behavior", "partial_restrictions",
         "emotion_with_mobility_signal"},
    ],
}

MOBILITY_FEATURES = frozenset({
    "mobility_drop_retail", "mobility_drop_transit", "mobility_drop_workplace",
    "residential_increase", "grocery_spike", "severe_lockdown_behavior",
    "partial_restrictions",
})

EMOTION_FEATURES = frozenset({
    "high_negative_sentiment", "dominant_emotion_fear", "fear_keywords_present",
    "anger_mentioned", "anxiety_keywords_present", "sadness_keywords_present",
    "high_positive_sentiment", "mixed_emotions", "solidarity_messages",
    "sentiment_worsened", "sentiment_improved", "sentiment_shift_detected",
    "covid_topic_detected", "lockdown_mentioned", "vaccine_mentioned",
    "health_concern", "compliance_discussed", "policy_governance_discussion",
})


def _base_feature(feature: str) -> str:
    """Strip a trailing _lagN / _leadN suffix to recover the base feature name.

    Cross-domain classification (and any other domain lookup) must operate on
    the base feature: ``vaccine_mentioned_lag3`` is an EMOTION feature just as
    ``vaccine_mentioned`` is.  Without this, every lagged/lead feature in the
    predictive matrix silently fails domain membership and rules are mis-tagged
    as ``cross_domain=False`` even when they clearly span both domains.
    """
    return re.sub(r"_(lag|lead)\d+$", "", str(feature).strip())


def _is_tautological(premise: tuple[str, ...], conclusion: str) -> bool:
    """Return True if conclusion is definitionally implied by the premise.

    Suffixes are stripped so the same-day tautology rules also catch their
    lagged/lead variants (e.g. ``sentiment_improved_lag1`` still counts as a
    member of the ``sentiment_shift_detected`` definition group).
    """
    conclusion_base = _base_feature(conclusion)
    defs = TAUTOLOGY_DEFINITIONS.get(conclusion_base)
    if defs is None:
        return False
    premise_set = {_base_feature(p) for p in premise}
    return all(any(f in premise_set for f in group) for group in defs)


def _is_cross_domain(premise: tuple[str, ...], conclusion: str) -> bool:
    """Return True if the rule spans both mobility and emotion domains.

    Base feature names are used so lagged/lead variants are classified by the
    domain of their underlying feature.
    """
    all_features = {_base_feature(p) for p in premise} | {_base_feature(conclusion)}
    return bool(all_features & MOBILITY_FEATURES) and bool(all_features & EMOTION_FEATURES)


class FCAAnalyzer:
    def __init__(self, binary_matrix_file):
        """
        Initialize FCA analyzer

        Args:
            binary_matrix_file: CSV file with binary features (Date as first column)
        """
        self.binary_matrix_file = Path(binary_matrix_file)
        self.df = pd.read_csv(self.binary_matrix_file)
        self.date_col = self.df.iloc[:, 0]

        candidate_data = self.df.iloc[:, 1:].copy()
        binary_columns = []

        for column in candidate_data.columns:
            numeric_values = pd.to_numeric(candidate_data[column], errors='coerce')
            raw_non_null = candidate_data[column].notna().sum()
            converted_non_null = numeric_values.notna().sum()

            if raw_non_null == 0 or converted_non_null != raw_non_null:
                continue

            unique_values = set(numeric_values.dropna().unique())
            if unique_values.issubset({0, 1}):
                binary_columns.append(column)
                candidate_data[column] = numeric_values.astype(int)

        self.binary_data = candidate_data[binary_columns]
        self.context = None
        self.lattice = None

    def create_formal_context(self):
        """Create formal context for FCA"""
        objects = [f"Day_{i}" for i in range(len(self.binary_data))]
        properties = list(self.binary_data.columns)

        bools = []
        for _, row in self.binary_data.iterrows():
            bools.append(tuple(row.astype(bool)))

        self.context = Context(objects, properties, bools)
        return self.context

    def generate_lattice(self):
        """Generate Galois lattice"""
        if self.context is None:
            self.create_formal_context()

        self.lattice = self.context.lattice
        return self.lattice

    def extract_concepts(self):
        """Extract all formal concepts"""
        if self.lattice is None:
            self.generate_lattice()

        concepts = []
        for concept in self.lattice:
            concepts.append({
                'extent': concept.extent,
                'intent': concept.intent,
                'support': len(concept.extent),
            })

        return pd.DataFrame(concepts)

    def extract_implications(
        self,
        min_support=0.05,
        max_premise_size=2,
        min_confidence=0.8,
        min_lift=1.05,
        max_conclusion_prevalence=0.75,
    ):
        """
        Extract implication rules from the binary matrix with quality filters.

        Args:
            min_support: Minimum support threshold (0.0 to 1.0)
            max_premise_size: Max number of attributes in premise
            min_confidence: Minimum rule confidence (0.0 to 1.0)
            min_lift: Minimum rule lift (>1 means positive association)
            max_conclusion_prevalence: Ignore overly common conclusions
        """
        if self.context is None:
            self.create_formal_context()

        implications = []
        min_support_count = max(2, math.ceil(min_support * len(self.binary_data)))

        feature_names = list(self.binary_data.columns)
        total_rows = len(self.binary_data)

        conclusion_prevalence = {
            col: float(self.binary_data[col].mean()) for col in feature_names
        }

        for premise_size in range(1, min(max_premise_size, len(feature_names)) + 1):
            for premise_tuple in combinations(feature_names, premise_size):
                premise_mask = self.binary_data[list(premise_tuple)].all(axis=1)
                support_count = int(premise_mask.sum())

                if support_count < min_support_count:
                    continue

                for conclusion in feature_names:
                    if conclusion in premise_tuple:
                        continue

                    p_conclusion = conclusion_prevalence[conclusion]
                    if p_conclusion == 0 or p_conclusion > max_conclusion_prevalence:
                        continue

                    conclusion_support = int(self.binary_data.loc[premise_mask, conclusion].sum())
                    confidence = conclusion_support / support_count
                    lift = confidence / p_conclusion
                    leverage = (conclusion_support / total_rows) - (
                        (support_count / total_rows) * p_conclusion
                    )

                    if confidence >= min_confidence and lift >= min_lift:
                        if _is_tautological(premise_tuple, conclusion):
                            continue
                        implications.append({
                            'premise': ', '.join(premise_tuple),
                            'conclusion': conclusion,
                            'support': support_count,
                            'support_pct': support_count / total_rows * 100,
                            'confidence': confidence * 100,
                            'lift': lift,
                            'leverage': leverage,
                            'cross_domain': _is_cross_domain(premise_tuple, conclusion),
                        })

        return pd.DataFrame(implications)

    def visualize_lattice(self, output_file='lattice_visualization.png'):
        """Visualize the Galois lattice"""
        if self.lattice is None:
            self.generate_lattice()

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        G = nx.DiGraph()
        concept_labels = {}
        for i, concept in enumerate(self.lattice):
            G.add_node(i)
            attrs = list(concept.intent)[:3]
            label = f"C{i}\\n{len(concept.extent)} days"
            if attrs:
                label += f"\\n{', '.join(attrs)}"
            concept_labels[i] = label

        concepts_list = list(self.lattice)
        for i, concept in enumerate(concepts_list):
            for j, other_concept in enumerate(concepts_list):
                if i != j:
                    if (
                        set(concept.extent).issubset(set(other_concept.extent))
                        and set(other_concept.intent).issubset(set(concept.intent))
                        and len(set(concept.extent)) < len(set(other_concept.extent))
                    ):
                        G.add_edge(j, i)

        plt.figure(figsize=(16, 12), constrained_layout=True)
        pos = nx.spring_layout(G, k=2, iterations=50)

        nx.draw(
            G,
            pos,
            labels=concept_labels,
            node_color='lightblue',
            node_size=3000,
            font_size=12,
            font_weight='bold',
            arrows=True,
            arrowsize=20,
            edge_color='gray',
            linewidths=2,
            with_labels=True,
        )

        plt.title("Galois Lattice - Crisis Behavior Formal Concept Analysis", fontsize=20, fontweight='bold')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Lattice visualization saved to {output_path}")

    def export_for_galicia(self, output_file='galicia_context.cxt'):
        """Export formal context in Galicia .cxt format"""
        if self.context is None:
            self.create_formal_context()

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write("B\\n\\n")
            f.write(f"{len(self.context.objects)}\\n")
            f.write(f"{len(self.context.properties)}\\n\\n")

            for obj in self.context.objects:
                f.write(f"{obj}\\n")

            for prop in self.context.properties:
                f.write(f"{prop}\\n")

            for row in self.context.bools:
                row_str = ''.join(['X' if val else '.' for val in row])
                f.write(f"{row_str}\\n")

        print(f"Context exported to Galicia format: {output_path}")


def aggregate_to_weekly(df: pd.DataFrame, binary_columns: list[str]) -> pd.DataFrame:
    """
    Reduce a daily binary matrix to ISO-week snapshots using majority vote.

    A feature is 1 for a given week if it was active on more than half the
    days in that week.  This keeps the FCA context tractable for the
    concepts library while preserving the temporal signal.
    """
    work = df.copy()
    work["Date"] = pd.to_datetime(work["Date"])
    work["week"] = work["Date"].dt.to_period("W")

    weekly = (
        work.groupby("week")[binary_columns]
        .mean()
        .ge(0.5)              # majority vote → True/False
        .astype(int)
        .reset_index()
    )
    weekly["week"] = weekly["week"].astype(str)
    return weekly


# ---------------------------------------------------------------------------
# Maximum number of objects the concepts library handles comfortably.
# Above this we fall back to a weekly-aggregated context.
# ---------------------------------------------------------------------------
MAX_LATTICE_OBJECTS = 150


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Association rule mining — always run on the full daily matrix
    # Thresholds tightened: support>=12%, lift>=1.8, conviction added.
    # ------------------------------------------------------------------
    daily_analyzer = FCAAnalyzer(DEFAULT_BINARY_MATRIX)
    rules_df = daily_analyzer.extract_implications(
        min_support=0.12,
        max_premise_size=2,
        min_confidence=0.75,
        min_lift=1.8,
        max_conclusion_prevalence=0.75,
    )

    # Compute conviction for every rule
    if not rules_df.empty:
        feature_prevalence = {
            col: float(daily_analyzer.binary_data[col].mean())
            for col in daily_analyzer.binary_data.columns
        }

        def _conviction(row) -> float:
            conf = float(row["confidence"]) / 100.0
            p_c = feature_prevalence.get(str(row["conclusion"]).strip(), 0.5)
            if conf >= 1.0:
                return float("inf")
            if p_c >= 1.0:
                return 1.0
            return (1.0 - p_c) / (1.0 - conf)

        rules_df["conviction"] = rules_df.apply(_conviction, axis=1)
    if not rules_df.empty:
        rules_df = rules_df.sort_values(
            ['cross_domain', 'lift', 'confidence', 'support'],
            ascending=[False, False, False, False],
        )
    print(f"Mined {len(rules_df)} rules at tightened thresholds "
          f"(support>=12%, confidence>=75%, lift>=1.8)")
    rules_df.to_csv(RESULTS_DIR / 'association_rules.csv', index=False)
    print(f"Association rules saved to {RESULTS_DIR / 'association_rules.csv'}")
    print(f"Daily objects: {len(daily_analyzer.binary_data)}")
    print("\nTop 10 Association Rules:")
    print(rules_df.head(10))

    # ------------------------------------------------------------------
    # Also mine from the predictive (lagged/lead) matrix if it exists.
    # This produces rules like: IF fear_lag3 THEN grocery_spike_lead2
    # ------------------------------------------------------------------
    predictive_matrix = PROJECT_ROOT / "data" / "processed" / "fca_predictive_matrix.csv"
    if predictive_matrix.exists():
        print("\nMining predictive rules from lagged/lead matrix ...")
        pred_analyzer = FCAAnalyzer(predictive_matrix)
        pred_rules = pred_analyzer.extract_implications(
            min_support=0.10,      # slightly looser: fewer rows after NaN trim
            max_premise_size=2,
            min_confidence=0.75,
            min_lift=1.8,
            max_conclusion_prevalence=0.80,
        )
        if not pred_rules.empty:
            pred_rules.to_csv(RESULTS_DIR / 'association_rules_predictive.csv', index=False)
            print(f"  Mined {len(pred_rules)} predictive rules → "
                  f"{RESULTS_DIR / 'association_rules_predictive.csv'}")

            # Prune predictive rules immediately — 3k+ raw rules are too many for the LLM
            try:
                import sys as _sys
                if str(PROJECT_ROOT) not in _sys.path:
                    _sys.path.insert(0, str(PROJECT_ROOT))
                from src.rules.prune_rules import prune_rules, build_prevalence_map  # noqa: PLC0415
                pred_bm = pd.read_csv(DEFAULT_BINARY_MATRIX)
                prev_map = build_prevalence_map(pred_bm)
                print(f"  Pruning {len(pred_rules)} predictive rules ...")
                pred_pruned = prune_rules(
                    pred_rules,
                    prevalence_map=prev_map,
                    min_support_pct=10.0,
                    min_confidence=75.0,
                    min_lift=1.8,
                    min_conviction=1.1,
                    max_rules_per_consequent=3,
                )
                pred_pruned.to_csv(RESULTS_DIR / 'association_rules_predictive_pruned.csv', index=False)
                print(f"  Pruned to {len(pred_pruned)} predictive rules → "
                      f"{RESULTS_DIR / 'association_rules_predictive_pruned.csv'}")
            except ImportError:
                print("  NOTE: src.rules.prune_rules not importable; skipping predictive rule pruning.")
    else:
        print("\nPredictive matrix not found; run 'python -m src.features.lagged_features' first.")

    # ------------------------------------------------------------------
    # Formal context / lattice — use weekly aggregation when the daily
    # matrix is too large for the concepts library
    # ------------------------------------------------------------------
    n_objects = len(daily_analyzer.binary_data)
    binary_cols = list(daily_analyzer.binary_data.columns)

    if n_objects <= MAX_LATTICE_OBJECTS:
        print(f"\nBuilding formal context from {n_objects} daily objects (within limit).")
        context_analyzer = daily_analyzer
    else:
        print(
            f"\nDaily matrix has {n_objects} objects (limit {MAX_LATTICE_OBJECTS})."
            " Aggregating to weekly snapshots for formal context / lattice."
        )
        full_df = pd.read_csv(DEFAULT_BINARY_MATRIX)
        weekly_df = aggregate_to_weekly(full_df, binary_cols)
        print(f"Weekly context: {len(weekly_df)} objects × {len(binary_cols)} attributes")

        # Write the weekly context to a temp location so FCAAnalyzer can read it
        weekly_path = RESULTS_DIR / "fca_weekly_context.csv"
        weekly_df.rename(columns={"week": "Date"}).to_csv(weekly_path, index=False)
        context_analyzer = FCAAnalyzer(weekly_path)

    context = context_analyzer.create_formal_context()
    print(f"Created formal context: {len(context.objects)} objects × {len(context.properties)} properties")

    lattice = context_analyzer.generate_lattice()
    print(f"Generated lattice with {len(list(lattice))} concepts")

    concepts_df = context_analyzer.extract_concepts()
    concepts_df.to_csv(RESULTS_DIR / 'formal_concepts.csv', index=False)
    print(f"Extracted {len(concepts_df)} formal concepts")

    context_analyzer.visualize_lattice(RESULTS_DIR / 'galois_lattice.png')
    context_analyzer.export_for_galicia(RESULTS_DIR / 'crisis_context.cxt')