"""Results visualization for COVID mobility-sentiment-FCA outputs."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# UAE COVID policy phase boundaries
_PHASES = [
    ("2020-03-01", "2020-05-31", "#ffd6d6", "Phase 1\nFirst lockdown"),
    ("2020-06-01", "2020-12-31", "#fff3cd", "Phase 2\nReopening"),
    ("2021-01-01", "2021-08-31", "#d6f5d6", "Phase 3\nVaccine rollout"),
    ("2021-09-01", "2021-12-31", "#d6eaff", "Phase 4\nEndemic"),
]


def _add_phase_bands(ax, alpha: float = 0.18) -> None:
    """Shade UAE pandemic phases on a date-axis Axes."""
    for start, end, color, label in _PHASES:
        ax.axvspan(pd.to_datetime(start), pd.to_datetime(end),
                   color=color, alpha=alpha, zorder=0)


def _month_locator(ax) -> None:
    """Apply quarterly major ticks and monthly minor ticks to a date x-axis."""
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.tick_params(axis='x', rotation=30, labelsize=12)

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
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
})


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
        """Plot mobility metrics over time with 7-day rolling average and phase shading."""
        print("\n📈 Creating mobility trends plot...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('UAE COVID-19 Mobility Trends (Mar 2020 – Dec 2021)',
                     fontsize=20, fontweight='bold', y=1.01)

        metrics = [
            ('retail_recreation_change', '#1f77b4', 'Retail & Recreation (% vs baseline)'),
            ('transit_change', '#ff7f0e', 'Transit Stations (% vs baseline)'),
            ('workplaces_change', '#2ca02c', 'Workplaces (% vs baseline)'),
            ('residential_change', '#d62728', 'Residential (% vs baseline)'),
        ]

        for idx, (metric, color, label) in enumerate(metrics):
            if metric not in self.mobility_df.columns:
                continue
            ax = axes[idx // 2, idx % 2]
            series = self.mobility_df.set_index('Date')[metric]
            roll7 = series.rolling(7, center=True, min_periods=3).mean()

            _add_phase_bands(ax)
            ax.plot(series.index, series.values,
                    color=color, linewidth=0.6, alpha=0.35, label='Daily')
            ax.plot(roll7.index, roll7.values,
                    color=color, linewidth=2.2, label='7-day avg')
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.4)

            ax.set_title(label, fontsize=15, fontweight='bold')
            ax.set_ylabel('% change', fontsize=13)
            ax.legend(fontsize=12, loc='lower right')
            ax.grid(True, alpha=0.25, axis='y')
            _month_locator(ax)

        # Phase legend in last subplot if any panel is empty
        phase_patches = [mpatches.Patch(color=c, alpha=0.5, label=l.replace('\n', ' '))
                         for _, _, c, l in _PHASES]
        fig.legend(handles=phase_patches, loc='lower center', ncol=4,
                   fontsize=12, framealpha=0.9, bbox_to_anchor=(0.5, -0.03))

        plt.tight_layout()
        output_file = OUTPUT_DIR / 'mobility_trends.png'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved to {output_file}")

    def plot_sentiment_timeline(self):
        """Plot sentiment with 14-day rolling average and phase shading."""
        print("💭 Creating sentiment timeline...")

        if 'avg_compound' not in self.features_df.columns:
            print("  ⚠️  avg_compound column not found; skipping")
            return

        fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
        fig.suptitle('Reddit Sentiment During UAE COVID-19 (Mar 2020 – Dec 2021)',
                     fontsize=19, fontweight='bold')

        df = self.features_df.set_index('Date').sort_index()

        # ── Panel 1: Compound sentiment ────────────────────────────────────
        ax1 = axes[0]
        compound = df['avg_compound']
        roll14 = compound.rolling(14, center=True, min_periods=5).mean()

        _add_phase_bands(ax1)
        ax1.fill_between(compound.index, compound.values, 0,
                         where=compound.values >= 0, alpha=0.15, color='green')
        ax1.fill_between(compound.index, compound.values, 0,
                         where=compound.values < 0, alpha=0.15, color='red')
        ax1.plot(compound.index, compound.values,
                 color='slategray', linewidth=0.5, alpha=0.4, label='Daily')
        ax1.plot(roll14.index, roll14.values,
                 color='purple', linewidth=2.2, label='14-day avg')
        ax1.axhline(0, color='black', linewidth=0.9, linestyle='--', alpha=0.5)
        ax1.set_ylabel('Compound Score', fontsize=14)
        ax1.set_ylim(-1.1, 1.1)
        ax1.legend(fontsize=12, loc='lower right')
        ax1.grid(True, alpha=0.2, axis='y')
        ax1.set_title('Compound Sentiment', fontsize=14)

        # ── Panel 2: Positive vs Negative fractions ────────────────────────
        ax2 = axes[1]
        _add_phase_bands(ax2)
        if 'pos_fraction' in df.columns:
            pos_roll = df['pos_fraction'].rolling(14, center=True, min_periods=5).mean()
            ax2.plot(df.index, df['pos_fraction'].values,
                     color='green', linewidth=0.5, alpha=0.3)
            ax2.plot(pos_roll.index, pos_roll.values,
                     color='green', linewidth=2, label='Positive (14-day avg)')
        if 'neg_fraction' in df.columns:
            neg_roll = df['neg_fraction'].rolling(14, center=True, min_periods=5).mean()
            ax2.plot(df.index, df['neg_fraction'].values,
                     color='red', linewidth=0.5, alpha=0.3)
            ax2.plot(neg_roll.index, neg_roll.values,
                     color='red', linewidth=2, label='Negative (14-day avg)')
        ax2.set_ylabel('Fraction of Posts', fontsize=14)
        ax2.legend(fontsize=12, loc='upper right')
        ax2.grid(True, alpha=0.2, axis='y')
        ax2.set_title('Positive vs Negative Post Fraction', fontsize=14)

        _month_locator(ax2)
        ax2.set_xlabel('Date', fontsize=14)

        # Phase legend
        phase_patches = [mpatches.Patch(color=c, alpha=0.5, label=l.replace('\n', ' '))
                         for _, _, c, l in _PHASES]
        fig.legend(handles=phase_patches, loc='lower center', ncol=4,
                   fontsize=12, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

        plt.tight_layout()
        output_file = OUTPUT_DIR / 'sentiment_timeline.png'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved to {output_file}")

    def plot_features_heatmap(self):
        """Heatmap with monthly x-axis ticks and 28-day aggregation to avoid clutter."""
        print("🔥 Creating features heatmap...")

        binary_cols = self.binary_columns(self.features_df, exclude={'Date', 'num_posts'})

        preferred_order = [
            'severe_lockdown_behavior', 'mobility_drop_retail',
            'mobility_drop_workplace', 'mobility_drop_transit',
            'residential_increase', 'grocery_spike', 'partial_restrictions',
            'dominant_emotion_fear', 'high_negative_sentiment', 'fear_keywords_present',
            'anger_mentioned', 'anxiety_keywords_present',
            'health_concern', 'lockdown_mentioned', 'compliance_discussed',
            'policy_governance_discussion', 'covid_topic_detected',
            'solidarity_messages', 'vaccine_mentioned',
            'sentiment_shift_detected', 'sentiment_improved', 'sentiment_worsened',
            'high_positive_sentiment', 'calm_mobile_baseline',
            'emotion_with_mobility_signal', 'emotion_mobility_mismatch', 'weekend',
        ]
        binary_cols = [c for c in preferred_order if c in binary_cols]
        if not binary_cols:
            binary_cols = self.binary_columns(self.features_df, exclude={'Date', 'num_posts'})
        if not binary_cols:
            print("  ⚠️  No binary columns found, skipping heatmap")
            return

        # Resample to bi-weekly mean to reduce noise and width
        df_indexed = self.features_df.set_index('Date')[binary_cols].sort_index()
        df_resampled = df_indexed.resample('14D').mean()

        fig, ax = plt.subplots(figsize=(16, 9))
        im = ax.imshow(
            df_resampled.T.values,
            aspect='auto',
            cmap='RdYlGn_r',
            vmin=0, vmax=1,
            interpolation='nearest',
        )

        # Y-axis: feature names
        ax.set_yticks(range(len(binary_cols)))
        ax.set_yticklabels(binary_cols, fontsize=11)

        # X-axis: bi-weekly period dates → show monthly labels
        period_dates = df_resampled.index
        month_positions, month_labels = [], []
        prev_month = None
        for i, d in enumerate(period_dates):
            if d.month != prev_month:
                month_positions.append(i)
                month_labels.append(d.strftime("%b\n%Y"))
                prev_month = d.month
        ax.set_xticks(month_positions)
        ax.set_xticklabels(month_labels, fontsize=11)

        # Phase boundary lines
        for start, _, _, label in _PHASES[1:]:
            phase_ts = pd.Timestamp(start)
            # Find nearest bi-weekly bin
            diffs = [(abs((d - phase_ts).days), i) for i, d in enumerate(period_dates)]
            closest_i = min(diffs)[1]
            ax.axvline(closest_i - 0.5, color='white', linewidth=1.5, alpha=0.7)

        cbar = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
        cbar.set_label('Activation rate (2-week window)', fontsize=13)
        ax.set_title(
            'Feature Activation Heatmap — UAE COVID-19 (bi-weekly aggregation)',
            fontsize=17, fontweight='bold'
        )
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Feature', fontsize=14)

        plt.tight_layout()
        output_file = OUTPUT_DIR / 'features_heatmap.png'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved to {output_file}")

    def plot_combined_mobility_sentiment(self):
        """Three-panel combined plot: mobility 4-metric average, compound sentiment,
           and grocery-spike binary signal — with phase shading and rolling averages."""
        print("🔗 Creating combined mobility-sentiment plot...")

        mob = self.mobility_df.set_index('Date').sort_index()
        feat = self.features_df.set_index('Date').sort_index()

        # Build a composite mobility index (mean of 4 categories, flip residential)
        mob_cols = [c for c in
                    ['retail_recreation_change', 'transit_change',
                     'workplaces_change', 'grocery_change']
                    if c in mob.columns]
        if mob_cols:
            mob_index = mob[mob_cols].mean(axis=1)
        else:
            mob_index = mob['workplaces_change'] if 'workplaces_change' in mob.columns else None

        fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
        fig.suptitle(
            'UAE COVID-19: Mobility, Sentiment & Key Signals (Mar 2020 – Dec 2021)',
            fontsize=19, fontweight='bold'
        )

        # Panel 1: Mobility index
        ax1 = axes[0]
        if mob_index is not None:
            roll7 = mob_index.rolling(7, center=True, min_periods=3).mean()
            _add_phase_bands(ax1)
            ax1.plot(mob_index.index, mob_index.values,
                     color='#1f77b4', linewidth=0.6, alpha=0.3)
            ax1.plot(roll7.index, roll7.values,
                     color='#1f77b4', linewidth=2.2, label='Mobility index (7-day avg)')
            ax1.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.4)
        ax1.set_ylabel('% vs baseline', fontsize=14)
        ax1.legend(fontsize=12, loc='lower right')
        ax1.grid(True, alpha=0.2, axis='y')
        ax1.set_title('Composite Mobility Index', fontsize=14)

        # Panel 2: Compound sentiment
        ax2 = axes[1]
        if 'avg_compound' in feat.columns:
            compound = feat['avg_compound']
            roll14 = compound.rolling(14, center=True, min_periods=5).mean()
            _add_phase_bands(ax2)
            ax2.fill_between(compound.index, compound.values, 0,
                             where=compound.values >= 0, alpha=0.12, color='green')
            ax2.fill_between(compound.index, compound.values, 0,
                             where=compound.values < 0, alpha=0.12, color='red')
            ax2.plot(compound.index, compound.values,
                     color='slategray', linewidth=0.5, alpha=0.35)
            ax2.plot(roll14.index, roll14.values,
                     color='purple', linewidth=2.2, label='Compound sentiment (14-day avg)')
            ax2.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.4)
        ax2.set_ylabel('Sentiment score', fontsize=14)
        ax2.set_ylim(-1.1, 1.1)
        ax2.legend(fontsize=12, loc='lower right')
        ax2.grid(True, alpha=0.2, axis='y')
        ax2.set_title('Reddit Compound Sentiment', fontsize=14)

        # Panel 3: Grocery spike + vaccine binary signals
        ax3 = axes[2]
        _add_phase_bands(ax3)
        binary_signals = [
            ('grocery_spike', '#e07b39', 'Grocery spike'),
            ('vaccine_mentioned', '#2ca02c', 'Vaccine mentioned'),
            ('severe_lockdown_behavior', '#d62728', 'Severe lockdown'),
        ]
        offsets = [0.7, 0.4, 0.1]
        for (col, color, label), offset in zip(binary_signals, offsets):
            if col in feat.columns:
                active_days = feat.index[feat[col] == 1]
                ax3.scatter(active_days,
                            [offset] * len(active_days),
                            color=color, s=6, alpha=0.7, label=label)
        ax3.set_yticks(offsets)
        ax3.set_yticklabels(['Grocery\nspike', 'Vaccine\nmentioned', 'Severe\nlockdown'],
                            fontsize=11)
        ax3.set_ylim(0, 1)
        ax3.legend(fontsize=12, loc='lower right')
        ax3.grid(True, alpha=0.2, axis='y')
        ax3.set_title('Key Binary Signals', fontsize=14)

        _month_locator(ax3)
        ax3.set_xlabel('Date', fontsize=14)

        # Phase legend
        phase_patches = [mpatches.Patch(color=c, alpha=0.5, label=l.replace('\n', ' '))
                         for _, _, c, l in _PHASES]
        fig.legend(handles=phase_patches, loc='lower center', ncol=4,
                   fontsize=12, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

        plt.tight_layout()
        output_file = OUTPUT_DIR / 'combined_analysis.png'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved to {output_file}")

    def plot_rules_overview(self):
        """Bubble chart of FCA rules: x=support, y=confidence, size=lift, colour=cross-domain."""
        print("🔵 Creating rules overview chart...")

        fca_dir = RULES_FILE.parent
        rules_path_options = [
            fca_dir / "association_rules_evaluated.csv",
            fca_dir / "association_rules_stable.csv",
            RULES_FILE,
        ]
        rules_df = None
        for p in rules_path_options:
            if p.exists():
                rules_df = pd.read_csv(p)
                break
        if rules_df is None or len(rules_df) == 0:
            print("  ⚠️  No rules file found; skipping rules overview")
            return

        fig, ax = plt.subplots(figsize=(11, 7))
        for cross_domain, group in rules_df.groupby(
                rules_df['cross_domain'].astype(str).str.lower().isin(['true', '1', 'yes'])):
            color = '#e07b39' if cross_domain else '#5b9bd5'
            label = 'Cross-domain' if cross_domain else 'Same-domain'
            sizes = (group['lift'].clip(lower=1) ** 2) * 60
            ax.scatter(
                group['support_pct'] if 'support_pct' in group.columns else group['support'] * 100,
                group['confidence'],
                s=sizes,
                c=color, alpha=0.7, edgecolors='white', linewidths=0.5,
                label=label,
            )

        # Annotate top rules by lift
        top = rules_df.nlargest(8, 'lift')
        for _, row in top.iterrows():
            x = row.get('support_pct', row['support'] * 100)
            y = row['confidence']
            premise_short = str(row['premise'])[:30] + ('…' if len(str(row['premise'])) > 30 else '')
            ax.annotate(premise_short, (x, y),
                        fontsize=10, ha='left', va='bottom',
                        xytext=(4, 3), textcoords='offset points',
                        color='#333')

        ax.set_xlabel('Support (%)', fontsize=14)
        ax.set_ylabel('Confidence (%)', fontsize=14)
        ax.set_title('FCA Association Rules — Support vs Confidence\n(bubble size = lift²)',
                     fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.25)

        # Lift size legend
        for lift_val, label in [(1.8, 'lift=1.8'), (2.5, 'lift=2.5'), (3.5, 'lift=3.5')]:
            ax.scatter([], [], s=(lift_val ** 2) * 60, c='gray', alpha=0.5, label=label)
        ax.legend(fontsize=12, loc='lower right')

        plt.tight_layout()
        output_file = OUTPUT_DIR / 'rules_overview.png'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved to {output_file}")

    def plot_feature_activation(self):
        """Horizontal bar chart of binary feature activation rates."""
        print("📊 Creating feature activation chart...")

        binary_cols = self.binary_columns(self.features_df, exclude={'Date', 'num_posts'})
        if not binary_cols:
            print("  ⚠️  No binary columns found; skipping")
            return

        rates = self.features_df[binary_cols].mean().sort_values(ascending=True) * 100

        # Colour by domain
        mobility_features = {
            'mobility_drop_retail', 'mobility_drop_transit', 'mobility_drop_workplace',
            'residential_increase', 'grocery_spike', 'severe_lockdown_behavior', 'partial_restrictions',
        }
        emotion_features = {
            'high_negative_sentiment', 'high_positive_sentiment', 'sentiment_improved',
            'sentiment_worsened', 'sentiment_shift_detected', 'dominant_emotion_fear',
            'mixed_emotions', 'fear_keywords_present', 'anger_mentioned',
            'anxiety_keywords_present', 'sadness_keywords_present', 'solidarity_messages',
        }
        topic_features = {
            'covid_topic_detected', 'lockdown_mentioned', 'vaccine_mentioned',
            'health_concern', 'compliance_discussed', 'policy_governance_discussion',
        }
        colors = []
        for feat in rates.index:
            if feat in mobility_features:
                colors.append('#1f77b4')
            elif feat in emotion_features:
                colors.append('#d62728')
            elif feat in topic_features:
                colors.append('#2ca02c')
            else:
                colors.append('#9467bd')

        fig, ax = plt.subplots(figsize=(10, max(6, len(rates) * 0.35)))
        bars = ax.barh(rates.index, rates.values, color=colors, alpha=0.8, edgecolor='white')

        # Value labels
        for bar, val in zip(bars, rates.values):
            ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                    f'{val:.0f}%', va='center', fontsize=11)

        ax.set_xlabel('% of days active', fontsize=14)
        ax.set_title('Feature Activation Rates — 671 days (Mar 2020 – Dec 2021)',
                     fontsize=16, fontweight='bold')
        ax.set_xlim(0, 105)
        ax.grid(True, alpha=0.2, axis='x')
        ax.axvline(50, color='black', linewidth=0.8, linestyle='--', alpha=0.4)

        legend_patches = [
            mpatches.Patch(color='#1f77b4', label='Mobility'),
            mpatches.Patch(color='#d62728', label='Emotion/Sentiment'),
            mpatches.Patch(color='#2ca02c', label='Topic/Discourse'),
            mpatches.Patch(color='#9467bd', label='Composite'),
        ]
        ax.legend(handles=legend_patches, fontsize=12, loc='lower right')

        plt.tight_layout()
        output_file = OUTPUT_DIR / 'feature_activation.png'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved to {output_file}")

    def create_summary_report(self):
        """Generate text summary"""
        print("Creating summary report...")

        output_file = OUTPUT_DIR.parent / 'summary_report.txt'
        output_file.parent.mkdir(parents=True, exist_ok=True)

        fca_dir = RULES_FILE.parent

        # ---- Load rule files (preference order) ----
        def _try_load(path):
            try:
                return pd.read_csv(path)
            except Exception:
                return None

        stable_df   = _try_load(fca_dir / "association_rules_stable.csv")
        evaluated_df = _try_load(fca_dir / "association_rules_evaluated.csv")
        raw_df      = _try_load(RULES_FILE)

        # Primary display: stable > evaluated > raw
        if stable_df is not None and len(stable_df) > 0:
            rules_df = stable_df
            rules_source = "BACKTEST-STABLE + LLM-EVALUATED"
        elif evaluated_df is not None and len(evaluated_df) > 0:
            rules_df = evaluated_df
            rules_source = "LLM-EVALUATED"
        elif raw_df is not None:
            rules_df = raw_df
            rules_source = "STATISTICAL (raw)"
        else:
            rules_df = pd.DataFrame()
            rules_source = "none"

        llm_available = (
            "llm_rank" in rules_df.columns and rules_df["llm_rank"].notna().any()
            if not rules_df.empty else False
        )

        # Predictive rules
        pred_df = _try_load(fca_dir / "association_rules_predictive_pruned.csv")

        # Backtest report
        backtest_df = _try_load(fca_dir / "backtest_report.csv")

        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("CRISIS BEHAVIORAL ANALYSIS - SUMMARY REPORT\n")
            f.write("UAE COVID-19 ANALYSIS\n")
            f.write("=" * 80 + "\n\n")

            # ---- Data overview ----
            f.write("DATA OVERVIEW:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Analysis Period: {self.features_df['Date'].min().date()} to {self.features_df['Date'].max().date()}\n")
            f.write(f"Total Days Analyzed: {len(self.features_df)}\n")
            binary_cols = self.binary_columns(self.features_df, exclude={'Date', 'num_posts'})
            f.write(f"Binary Features: {len(binary_cols)}\n\n")

            # ---- Feature activation ----
            f.write("FEATURE ACTIVATION SUMMARY:\n")
            f.write("-" * 80 + "\n")
            for col in sorted(binary_cols):
                activation_pct = (pd.to_numeric(self.features_df[col], errors='coerce').fillna(0).sum() / len(self.features_df)) * 100
                if activation_pct > 0:
                    f.write(f"  {col:45s} {activation_pct:5.1f}%\n")

            # ---- Rule pipeline quality summary ----
            f.write("\n\nRULE PIPELINE QUALITY SUMMARY:\n")
            f.write("-" * 80 + "\n")
            if raw_df is not None:
                f.write(f"  Raw rules mined (support>=12%, lift>=1.8):   {len(raw_df)}\n")
            scored_df = _try_load(fca_dir / "association_rules_scored.csv")
            if scored_df is not None:
                f.write(f"  After pruning & subsumption removal:         {len(scored_df)}\n")
            if evaluated_df is not None:
                n_llm = int(evaluated_df.get("llm_rank", pd.Series(dtype=float)).notna().sum())
                f.write(f"  After LLM selection:                         {n_llm}\n")
            if stable_df is not None:
                f.write(f"  After temporal backtest (stable only):       {len(stable_df)}\n")
            if pred_df is not None:
                f.write(f"  Predictive (lagged/lead) rules (pruned):     {len(pred_df)}\n")

            # ---- Backtest stability ----
            if backtest_df is not None and "stable" in backtest_df.columns:
                n_total = len(backtest_df)
                n_stable = int(backtest_df["stable"].sum())
                f.write(f"\n\nTEMPORAL BACKTEST (70/30 chronological split):\n")
                f.write("-" * 80 + "\n")
                f.write(f"  Rules tested:  {n_total}\n")
                f.write(f"  Stable rules:  {n_stable}  ({100*n_stable//max(n_total,1)}%)\n")
                f.write(f"  Unstable rules: {n_total - n_stable}  (confidence drop > 10pp on holdout)\n")
                if "confidence_drop" in backtest_df.columns:
                    unstable = backtest_df[~backtest_df["stable"]].sort_values("confidence_drop", ascending=False)
                    for _, r in unstable.iterrows():
                        f.write(f"    DROPPED: {r['premise']} → {r['conclusion']}"
                                f"  (drop={r['confidence_drop']:.1f}pp)\n")

            # ---- Main association rules ----
            if len(rules_df) > 0:
                if llm_available:
                    top_rules = (
                        rules_df[rules_df["llm_rank"].notna()]
                        .sort_values("llm_rank")
                    )
                    n = len(top_rules)
                    f.write(f"\n\nTOP {n} ASSOCIATION RULES ({rules_source}):\n")
                else:
                    sort_col = "composite_score" if "composite_score" in rules_df.columns else "lift"
                    top_rules = rules_df.sort_values(sort_col, ascending=False).head(15)
                    n = len(top_rules)
                    f.write(f"\n\nTOP {n} ASSOCIATION RULES ({rules_source}):\n")
                f.write("-" * 80 + "\n")

                for rank_i, (_, row) in enumerate(top_rules.iterrows(), start=1):
                    cross = " [CROSS-DOMAIN]" if row.get("cross_domain") else ""
                    f.write(f"\nRule {rank_i}:{cross}\n")
                    f.write(f"  IF:   {row['premise']}\n")
                    f.write(f"  THEN: {row['conclusion']}\n")
                    f.write(f"  Support: {int(row['support'])} days ({float(row['support_pct']):.1f}%)\n")
                    f.write(f"  Confidence: {float(row['confidence']):.1f}%\n")
                    f.write(f"  Lift: {float(row['lift']):.2f}\n")
                    for extra_col, label in [
                        ("conviction",       "Conviction"),
                        ("leverage",         "Leverage"),
                        ("composite_score",  "Composite Score"),
                        ("temporal_validity","Temporal Validity"),
                    ]:
                        val = row.get(extra_col)
                        if val is not None and str(val) not in ("nan", "None", ""):
                            try:
                                f.write(f"  {label}: {float(val):.3f}\n")
                            except (ValueError, TypeError):
                                pass
                    novelty = row.get("novelty_score")
                    policy  = row.get("policy_score")
                    if pd.notna(novelty) and pd.notna(policy):
                        f.write(f"  LLM: novelty={int(novelty)}/10  policy_relevance={int(policy)}/10\n")
                    reasoning = row.get("llm_reasoning", "")
                    if reasoning and str(reasoning).strip() not in ("", "nan"):
                        f.write(f"  Why it matters: {reasoning}\n")
                    rec = row.get("llm_policy_recommendation", "")
                    if rec and str(rec).strip() not in ("", "nan"):
                        f.write(f"  Policy recommendation: {rec}\n")

            # ---- Predictive rules section ----
            if pred_df is not None and len(pred_df) > 0:
                # Prefer LLM-evaluated predictive rules if available
                pred_eval_path = fca_dir / "association_rules_predictive_evaluated.csv"
                if pred_eval_path.exists():
                    pred_display = pd.read_csv(pred_eval_path)
                    pred_has_llm = "llm_rank" in pred_display.columns and pred_display["llm_rank"].notna().any()
                else:
                    pred_display = pred_df
                    pred_has_llm = False

                f.write(f"\n\nTOP PREDICTIVE (LAGGED/LEAD) RULES — {len(pred_df)} total after pruning:\n")
                f.write("-" * 80 + "\n")
                if pred_has_llm:
                    top_pred = pred_display[pred_display["llm_rank"].notna()].sort_values("llm_rank").head(10)
                else:
                    sort_col = "composite_score" if "composite_score" in pred_display.columns else "lift"
                    top_pred = pred_display.sort_values(sort_col, ascending=False).head(10)

                for rank_i, (_, row) in enumerate(top_pred.iterrows(), start=1):
                    f.write(f"\nPredictive Rule {rank_i}:\n")
                    f.write(f"  IF:   {row['premise']}\n")
                    f.write(f"  THEN: {row['conclusion']}\n")
                    f.write(f"  Support: {int(row['support'])} days ({float(row['support_pct']):.1f}%)"
                            f"  Confidence: {float(row['confidence']):.1f}%"
                            f"  Lift: {float(row['lift']):.2f}\n")
                    if pred_has_llm and pd.notna(row.get("llm_rank")):
                        f.write(f"  LLM: novelty={row['novelty_score']}/10  policy_relevance={row['policy_score']}/10\n")
                        lead = row.get("lead_time_days")
                        if pd.notna(lead):
                            f.write(f"  Lead time: {int(lead)} days\n")
                        reasoning = row.get("llm_reasoning", "")
                        if reasoning and str(reasoning).strip() not in ("", "nan"):
                            f.write(f"  Why it matters: {reasoning}\n")
                        rec = row.get("llm_policy_recommendation", "")
                        if rec and str(rec).strip() not in ("", "nan"):
                            f.write(f"  Policy recommendation: {rec}\n")

            # ---- Pointer to policy briefs ----
            briefs_path = output_file.parent / "policy_briefs.txt"
            pred_briefs_path = fca_dir.parent / "top_predictive_rules.txt"
            if briefs_path.exists():
                f.write("\n\n" + "=" * 80 + "\n")
                f.write("OPERATIONAL POLICY BRIEFS:\n")
                f.write("-" * 80 + "\n")
                f.write("Full operational policy briefs (signal windows, lead times, action\n")
                f.write("playbooks, false-alarm risk ratings) are in:\n")
                f.write("  results/policy_briefs.txt\n")
                if pred_briefs_path.exists():
                    f.write("  results/fca/top_predictive_rules.txt  (predictive rules)\n")

            f.write("\n" + "=" * 80 + "\n")

        print(f"  Saved to {output_file}")


def main():
    print("=" * 70)
    print("RESULTS VISUALIZATION")
    print("=" * 70)

    visualizer = ResultsVisualizer()
    visualizer.plot_mobility_trends()
    visualizer.plot_sentiment_timeline()
    visualizer.plot_features_heatmap()
    visualizer.plot_combined_mobility_sentiment()
    visualizer.plot_rules_overview()
    visualizer.plot_feature_activation()
    visualizer.create_summary_report()

    print("\n" + "=" * 70)
    print("✓ VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"\nAll visualizations saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
