"""COVID-19 UAE feature engineering for FCA input matrix generation."""

import json
from pathlib import Path

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOBILITY_FILE = PROJECT_ROOT / "data" / "processed" / "mobility_data_processed.csv"
REDDIT_JSON = PROJECT_ROOT / "data" / "raw" / "reddit_covid_uae_posts_by_day.json"
REDDIT_FEATURES_OUT = PROJECT_ROOT / "data" / "processed" / "reddit_sentiment_features.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "fca_binary_matrix.csv"

analyzer = SentimentIntensityAnalyzer()

COVID_WORDS = ["covid", "coronavirus", "corona", "pandemic", "sars-cov-2", "virus"]
LOCKDOWN_WORDS = ["lockdown", "curfew", "restriction", "shutdown", "stay home", "movement"]
VACCINE_WORDS = ["vaccine", "vaccination", "pfizer", "moderna", "sinopharm", "booster", "jab"]
HEALTH_WORDS = ["hospital", "icu", "cases", "positive", "infected", "symptoms", "death"]
COMPLIANCE_WORDS = ["mask", "social distancing", "sanitizer", "fine", "penalty", "rules"]
POLICY_WORDS = ["government", "authorities", "ministry", "moh", "policy", "regulation"]

FEAR_WORDS = ["afraid", "scared", "worried", "panic", "anxiety", "terrified", "nervous"]
ANGER_WORDS = ["angry", "frustrated", "unfair", "blame", "outrage"]
SADNESS_WORDS = ["sad", "heartbroken", "grief", "depressed", "hopeless"]
ANXIETY_WORDS = ["anxious", "uncertain", "stress", "stressed", "overwhelmed"]
SOLIDARITY_WORDS = ["together", "support", "help", "thank you", "heroes", "stay safe"]


def contains_keywords(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in keywords)


def load_posts_by_day(path: Path) -> dict:
    if not path.exists():
        print(f"Warning: Reddit source file not found at {path}")
        print("Proceeding with mobility-only feature matrix")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("posts_by_day", data)


def build_social_features(posts_by_day: dict) -> pd.DataFrame:
    rows = []
    prev_avg = None

    for day in sorted(posts_by_day.keys()):
        posts = posts_by_day[day]
        if not posts:
            continue

        scores = []
        neg_count = 0
        pos_count = 0
        covid_count = 0
        lockdown_count = 0
        vaccine_count = 0
        health_count = 0
        compliance_count = 0
        policy_count = 0
        fear_count = 0
        anger_count = 0
        sadness_count = 0
        anxiety_count = 0
        solidarity_count = 0

        for post in posts:
            text = f"{post.get('title', '')} {post.get('text', '')}".strip()
            sentiment = analyzer.polarity_scores(text)
            compound = sentiment["compound"]
            scores.append(compound)

            if compound <= -0.3:
                neg_count += 1
            if compound >= 0.3:
                pos_count += 1

            covid_count += int(contains_keywords(text, COVID_WORDS))
            lockdown_count += int(contains_keywords(text, LOCKDOWN_WORDS))
            vaccine_count += int(contains_keywords(text, VACCINE_WORDS))
            health_count += int(contains_keywords(text, HEALTH_WORDS))
            compliance_count += int(contains_keywords(text, COMPLIANCE_WORDS))
            policy_count += int(contains_keywords(text, POLICY_WORDS))
            fear_count += int(contains_keywords(text, FEAR_WORDS))
            anger_count += int(contains_keywords(text, ANGER_WORDS))
            sadness_count += int(contains_keywords(text, SADNESS_WORDS))
            anxiety_count += int(contains_keywords(text, ANXIETY_WORDS))
            solidarity_count += int(contains_keywords(text, SOLIDARITY_WORDS))

        if not scores:
            continue

        avg_compound = sum(scores) / len(scores)
        sentiment_shift_detected = 0
        sentiment_improved = 0
        sentiment_worsened = 0
        if prev_avg is not None:
            delta = avg_compound - prev_avg
            sentiment_shift_detected = int(abs(delta) > 0.2)
            sentiment_improved = int(delta > 0.2)
            sentiment_worsened = int(delta < -0.2)
        prev_avg = avg_compound

        dominant_emotion = "fear"
        emotion_counts = {
            "fear": fear_count,
            "anger": anger_count,
            "sadness": sadness_count,
            "anxiety": anxiety_count,
        }
        if any(v > 0 for v in emotion_counts.values()):
            dominant_emotion = max(emotion_counts, key=emotion_counts.get)

        rows.append(
            {
                "Date": pd.to_datetime(day),
                "num_posts": len(posts),
                "avg_compound": avg_compound,
                "neg_fraction": neg_count / len(posts),
                "pos_fraction": pos_count / len(posts),
                "high_negative_sentiment": int(avg_compound < -0.3),
                "high_positive_sentiment": int(avg_compound > 0.3),
                "sentiment_shift_detected": sentiment_shift_detected,
                "sentiment_improved": sentiment_improved,
                "sentiment_worsened": sentiment_worsened,
                "sentiment_volatility": float(pd.Series(scores).std(ddof=0)),
                "fear_keywords_present": int(fear_count > 0),
                "anger_mentioned": int(anger_count > 0),
                "sadness_keywords_present": int(sadness_count > 0),
                "anxiety_keywords_present": int(anxiety_count > 0),
                "dominant_emotion_fear": int(dominant_emotion == "fear"),
                "dominant_emotion": dominant_emotion,
                "mixed_emotions": int(sum(v > 0 for v in emotion_counts.values()) > 1),
                "covid_topic_detected": int(covid_count > 0),
                "lockdown_mentioned": int(lockdown_count > 0),
                "vaccine_mentioned": int(vaccine_count > 0),
                "health_concern": int(health_count > 0),
                "compliance_discussed": int(compliance_count > 0),
                "policy_governance_discussion": int(policy_count > 0),
                "solidarity_messages": int(solidarity_count > 0),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["Date"])

    return pd.DataFrame(rows).sort_values("Date")


def build_mobility_features(df: pd.DataFrame) -> pd.DataFrame:
    mobility = df.copy()
    if "Date" not in mobility.columns and "date" in mobility.columns:
        mobility = mobility.rename(columns={"date": "Date"})

    mobility["Date"] = pd.to_datetime(mobility["Date"])

    mobility["mobility_drop_retail"] = (mobility["retail_recreation_change"] < -25).astype(int)
    mobility["mobility_drop_transit"] = (mobility["transit_change"] < -30).astype(int)
    mobility["mobility_drop_workplace"] = (mobility["workplaces_change"] < -25).astype(int)
    mobility["residential_increase"] = (mobility["residential_change"] > 8).astype(int)
    mobility["grocery_spike"] = (mobility["grocery_pharmacy_change"] > 10).astype(int)

    mobility["severe_lockdown_behavior"] = (
        (mobility["mobility_drop_retail"] == 1)
        & (mobility["mobility_drop_workplace"] == 1)
        & (mobility["residential_increase"] == 1)
    ).astype(int)

    mobility["partial_restrictions"] = (
        (
            (mobility["mobility_drop_retail"] == 1)
            | (mobility["mobility_drop_workplace"] == 1)
            | (mobility["mobility_drop_transit"] == 1)
        )
        & (mobility["severe_lockdown_behavior"] == 0)
    ).astype(int)

    return mobility


def main() -> None:
    print("=" * 70)
    print("COVID-19 UAE FEATURE ENGINEERING")
    print("=" * 70)

    if not MOBILITY_FILE.exists():
        raise FileNotFoundError(
            f"Missing mobility input at {MOBILITY_FILE}. Run scripts/collect_mobility_data.py first."
        )

    posts_by_day = load_posts_by_day(REDDIT_JSON)
    social_df = build_social_features(posts_by_day)
    print(f"Built social features for {len(social_df)} days")

    mobility_df = pd.read_csv(MOBILITY_FILE)
    mobility_df = build_mobility_features(mobility_df)
    print(f"Built mobility features for {len(mobility_df)} days")

    combined = mobility_df.merge(social_df, on="Date", how="left")
    combined = combined.sort_values("Date")
    combined = combined.fillna(0)

    required_social_cols = [
        "num_posts",
        "avg_compound",
        "neg_fraction",
        "pos_fraction",
        "high_negative_sentiment",
        "high_positive_sentiment",
        "sentiment_shift_detected",
        "sentiment_improved",
        "sentiment_worsened",
        "sentiment_volatility",
        "fear_keywords_present",
        "anger_mentioned",
        "sadness_keywords_present",
        "anxiety_keywords_present",
        "dominant_emotion_fear",
        "dominant_emotion",
        "mixed_emotions",
        "covid_topic_detected",
        "lockdown_mentioned",
        "vaccine_mentioned",
        "health_concern",
        "compliance_discussed",
        "policy_governance_discussion",
        "solidarity_messages",
    ]
    for col in required_social_cols:
        if col not in combined.columns:
            combined[col] = 0

    combined["weekend"] = combined["Date"].dt.dayofweek.isin([5, 6]).astype(int)

    emotional_signal = (
        (combined["high_negative_sentiment"] == 1)
        | (combined["fear_keywords_present"] == 1)
        | (combined["anger_mentioned"] == 1)
    )
    mobility_signal = (
        (combined["mobility_drop_retail"] == 1)
        | (combined["mobility_drop_transit"] == 1)
        | (combined["mobility_drop_workplace"] == 1)
        | (combined["severe_lockdown_behavior"] == 1)
    )

    combined["emotion_with_mobility_signal"] = (emotional_signal & mobility_signal).astype(int)
    combined["emotion_mobility_mismatch"] = (emotional_signal & (~mobility_signal)).astype(int)
    # calm_mobile_baseline: days where NEITHER elevated negative emotion NOR any
    # mobility-disruption signal is present.  Operationally, these are "routine"
    # or recovery days — the public is going about normal activities without
    # detectable crisis stress in online discourse.  Distinct from mere absence
    # of lockdown: it requires simultaneously calm sentiment AND unimpaired
    # mobility, making it a genuine positive behavioural-state indicator rather
    # than a residual catch-all.
    combined["calm_mobile_baseline"] = ((~emotional_signal) & (~mobility_signal)).astype(int)

    REDDIT_FEATURES_OUT.parent.mkdir(parents=True, exist_ok=True)
    if not social_df.empty:
        social_df.to_csv(REDDIT_FEATURES_OUT, index=False)
        print(f"Saved Reddit sentiment features to {REDDIT_FEATURES_OUT}")
    else:
        pd.DataFrame(columns=["Date"]).to_csv(REDDIT_FEATURES_OUT, index=False)
        print(f"Saved empty Reddit sentiment features file to {REDDIT_FEATURES_OUT}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved FCA matrix to {OUTPUT_FILE}")
    print(f"Rows: {len(combined)}, Columns: {len(combined.columns)}")

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()