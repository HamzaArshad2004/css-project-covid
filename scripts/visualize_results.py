"""Results visualization for COVID mobility-sentiment-FCA outputs."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ========= CONFIG =========
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOBILITY_FILE = PROJECT_ROOT / "data" / "processed" / "mobility_data_processed.csv"
FCA_FILE = PROJECT_ROOT / "data" / "processed" / "fca_binary_matrix.csv"
REDDIT_JSON = PROJECT_ROOT / "data" / "raw" / "reddit_covid_uae_posts_by_day.json"
RULES_FILE = PROJECT_ROOT / "results" / "fca" / "association_rules.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "visualizations"
# ==========================

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


class ResultsVisualizer:
    def __init__(self):
        """Load all data files"""
        print("📂 Loading data files...")

        self.mobility_df = pd.read_csv(MOBILITY_FILE)
        if 'Date' not in self.mobility_df.columns and 'date' in self.mobility_df.columns:
            self.mobility_df = self.mobility_df.rename(columns={'date': 'Date'})
        self.mobility_df['Date'] = pd.to_datetime(self.mobility_df['Date'])

        self.features_df = pd.read_csv(FCA_FILE)
        self.features_df['Date'] = pd.to_datetime(self.features_df['Date'])

        if REDDIT_JSON.exists():
            with open(REDDIT_JSON, 'r', encoding='utf-8') as f:
                data = json.load(f)
            posts_by_day = data.get('posts_by_day', data)
        else:
            posts_by_day = {}

        daily_rows = []
        for day, posts in posts_by_day.items():
            if posts:
                daily_rows.append({
                    'Date': pd.to_datetime(day),
                    'num_posts': len(posts),
                    'avg_score': np.mean([p.get('score', 0) for p in posts]),
                })
        self.reddit_daily = pd.DataFrame(daily_rows)

        print(f"  ✓ Loaded {len(self.mobility_df)} days of mobility data")
        print(f"  ✓ Loaded {len(self.features_df)} days of features")
        print(f"  ✓ Loaded {len(self.reddit_daily)} days of Reddit data")

    @staticmethod
    def binary_columns(df, exclude=None):
        exclude = exclude or set()
        cols = []
        for col in df.columns:
            if col in exclude:
                continue
            vals = pd.to_numeric(df[col], errors='coerce')
            if vals.notna().sum() != df[col].notna().sum() or vals.notna().sum() == 0:
                continue
            unique_vals = set(vals.dropna().unique())
            if unique_vals.issubset({0, 1}):
                cols.append(col)
        return cols

    def plot_mobility_trends(self):
        """Plot mobility metrics over time"""
        print("\n📈 Creating mobility trends plot...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('UAE COVID Mobility Trends', fontsize=16, fontweight='bold')

        metrics = [
            ('retail_recreation_change', '#1f77b4', 'Retail & Recreation'),
            ('transit_change', '#ff7f0e', 'Transit Stations'),
            ('workplaces_change', '#2ca02c', 'Workplaces'),
            ('residential_change', '#d62728', 'Residential'),
        ]

        for idx, (metric, color, label) in enumerate(metrics):
            if metric and metric in self.mobility_df.columns:
                ax = axes[idx // 2, idx % 2]
                ax.plot(self.mobility_df['Date'], self.mobility_df[metric], color=color, linewidth=2, marker='o', markersize=4)
                ax.set_title(label, fontsize=12, fontweight='bold')
                ax.set_xlabel('Date', fontsize=10)
                ax.set_ylabel(label, fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        output_file = OUTPUT_DIR / 'mobility_trends.png'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved to {output_file}")

    def plot_sentiment_timeline(self):
        """Plot sentiment over time"""
        print("💭 Creating sentiment timeline...")

        fig, ax = plt.subplots(figsize=(14, 6))

        if 'avg_compound' in self.features_df.columns:
            ax.plot(self.features_df['Date'], self.features_df['avg_compound'], label='Compound Sentiment', linewidth=2.5, marker='o', color='purple', markersize=5)

            if 'neg_fraction' in self.features_df.columns:
                ax.plot(self.features_df['Date'], self.features_df['neg_fraction'], label='Negative Fraction', linewidth=2, marker='s', color='red', alpha=0.7, markersize=4)

            if 'pos_fraction' in self.features_df.columns:
                ax.plot(self.features_df['Date'], self.features_df['pos_fraction'], label='Positive Fraction', linewidth=2, marker='^', color='green', alpha=0.7, markersize=4)

        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax.set_title('Daily COVID Sentiment from Reddit Posts', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Sentiment Score', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        plt.tight_layout()
        output_file = OUTPUT_DIR / 'sentiment_timeline.png'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved to {output_file}")

    def plot_features_heatmap(self):
        """Create heatmap of binary features"""
        print("🔥 Creating features heatmap...")

        binary_cols = self.binary_columns(self.features_df, exclude={'Date', 'num_posts'})

        if len(binary_cols) == 0:
            print("  ⚠️  No binary columns found, skipping heatmap")
            return

        preferred_order = [
            'high_negative_sentiment', 'fear_keywords_present', 'anger_mentioned',
            'mobility_drop_retail', 'mobility_drop_transit', 'mobility_drop_workplace',
            'severe_lockdown_behavior', 'policy_governance_discussion',
            'solidarity_messages', 'sentiment_shift_detected',
            'emotion_with_mobility_signal', 'emotion_mobility_mismatch',
            'vaccine_mentioned', 'weekend',
        ]
        binary_cols = [c for c in preferred_order if c in binary_cols]

        if len(binary_cols) == 0:
            print("  ⚠️  No selected binary columns found, skipping heatmap")
            return

        binary_matrix = self.features_df[binary_cols]

        plt.figure(figsize=(14, 10))
        sns.heatmap(
            binary_matrix.T,
            cmap='RdYlGn_r',
            cbar_kws={'label': 'Feature Active'},
            yticklabels=binary_cols,
            xticklabels=[f"D{i + 1}" for i in range(len(binary_matrix))],
        )

        plt.title('Binary Feature Activation Heatmap', fontsize=14, fontweight='bold')
        plt.xlabel('Days', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.tight_layout()

        output_file = OUTPUT_DIR / 'features_heatmap.png'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved to {output_file}")

    def plot_combined_mobility_sentiment(self):
        """Plot mobility and sentiment together"""
        print("🔗 Creating combined mobility-sentiment plot...")

        merged = self.features_df[['Date', 'avg_compound']].merge(
            self.mobility_df[['Date', 'workplaces_change']],
            on='Date',
            how='outer',
        )

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        color1 = '#1f77b4'
        ax1.plot(merged['Date'], merged['workplaces_change'], color=color1, linewidth=2.5, marker='o', markersize=5)
        ax1.set_ylabel('Workplace Mobility Change (%)', color=color1, fontsize=12, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Mobility vs Sentiment During COVID', fontsize=14, fontweight='bold')

        color2 = '#d62728'
        ax2.plot(merged['Date'], merged['avg_compound'], color=color2, linewidth=2.5, marker='s', markersize=5)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax2.set_ylabel('Sentiment (Compound)', color=color2, fontsize=12, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.grid(True, alpha=0.3)

        plt.xticks(rotation=45)
        plt.tight_layout()

        output_file = OUTPUT_DIR / 'combined_analysis.png'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved to {output_file}")

    def create_summary_report(self):
        """Generate text summary"""
        print("📝 Creating summary report...")

        output_file = OUTPUT_DIR.parent / 'summary_report.txt'
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Prefer LLM-evaluated rules if available, fall back to raw rules
        evaluated_file = RULES_FILE.parent / "association_rules_evaluated.csv"
        try:
            rules_df = pd.read_csv(evaluated_file)
            llm_available = "llm_rank" in rules_df.columns and rules_df["llm_rank"].notna().any()
        except Exception:
            try:
                rules_df = pd.read_csv(RULES_FILE)
            except Exception:
                rules_df = pd.DataFrame()
            llm_available = False

        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("CRISIS BEHAVIORAL ANALYSIS - SUMMARY REPORT\n")
            f.write("UAE COVID-19 ANALYSIS\n")
            f.write("=" * 80 + "\n\n")

            f.write("DATA OVERVIEW:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Analysis Period: {self.features_df['Date'].min().date()} to {self.features_df['Date'].max().date()}\n")
            f.write(f"Total Days Analyzed: {len(self.features_df)}\n")
            binary_cols = self.binary_columns(self.features_df, exclude={'Date', 'num_posts'})
            f.write(f"Binary Features: {len(binary_cols)}\n\n")

            f.write("FEATURE ACTIVATION SUMMARY:\n")
            f.write("-" * 80 + "\n")
            for col in sorted(binary_cols):
                activation_pct = (pd.to_numeric(self.features_df[col], errors='coerce').fillna(0).sum() / len(self.features_df)) * 100
                if activation_pct > 0:
                    f.write(f"  {col:45s} {activation_pct:5.1f}%\n")

            if len(rules_df) > 0:
                if llm_available:
                    top_rules = (
                        rules_df[rules_df["llm_rank"].notna()]
                        .sort_values("llm_rank")
                    )
                    n = len(top_rules)
                    f.write(f"\n\nTOP {n} ASSOCIATION RULES (LLM-SELECTED — most novel & policy-relevant):\n")
                    f.write("-" * 80 + "\n")
                    for rank_i, (_, row) in enumerate(top_rules.iterrows(), start=1):
                        cross = " [CROSS-DOMAIN]" if row.get("cross_domain") else ""
                        f.write(f"\nRule {rank_i}:{cross}\n")
                        f.write(f"  IF:   {row['premise']}\n")
                        f.write(f"  THEN: {row['conclusion']}\n")
                        f.write(f"  Support: {row['support']} days ({row['support_pct']:.1f}%)\n")
                        f.write(f"  Confidence: {row['confidence']:.1f}%\n")
                        f.write(f"  Lift: {row['lift']:.2f}\n")
                        novelty = row.get("novelty_score")
                        policy = row.get("policy_score")
                        if pd.notna(novelty) and pd.notna(policy):
                            f.write(f"  LLM: novelty={int(novelty)}/10  policy_relevance={int(policy)}/10\n")
                        reasoning = row.get("llm_reasoning", "")
                        if reasoning and str(reasoning).strip():
                            f.write(f"  Why it matters: {reasoning}\n")
                        rec = row.get("llm_policy_recommendation", "")
                        if rec and str(rec).strip():
                            f.write(f"  Policy recommendation: {rec}\n")
                else:
                    top_rules = rules_df.sort_values(
                        ["cross_domain", "lift", "confidence"],
                        ascending=[False, False, False],
                    ).head(10)
                    f.write("\n\nTOP 10 ASSOCIATION RULES (statistical ranking):\n")
                    f.write("-" * 80 + "\n")
                    for rank_i, (_, row) in enumerate(top_rules.iterrows(), start=1):
                        f.write(f"\nRule {rank_i}:\n")
                        f.write(f"  IF:   {row['premise']}\n")
                        f.write(f"  THEN: {row['conclusion']}\n")
                        f.write(f"  Support: {row['support']} days ({row['support_pct']:.1f}%)\n")
                        if 'confidence' in row:
                            f.write(f"  Confidence: {row['confidence']:.1f}%\n")
                        if 'lift' in row:
                            f.write(f"  Lift: {row['lift']:.2f}\n")

            f.write("\n" + "=" * 80 + "\n")

        print(f"  ✓ Saved to {output_file}")


def main():
    print("=" * 70)
    print("RESULTS VISUALIZATION")
    print("=" * 70)

    visualizer = ResultsVisualizer()
    visualizer.plot_mobility_trends()
    visualizer.plot_sentiment_timeline()
    visualizer.plot_features_heatmap()
    visualizer.plot_combined_mobility_sentiment()
    visualizer.create_summary_report()

    print("\n" + "=" * 70)
    print("✓ VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"\nAll visualizations saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
