"""
Reddit Data Collection Script — UAE COVID-19
Collects COVID-19 related UAE posts and saves them grouped by day.

Strategy:
  - UAE-local subreddits are searched with COVID-only terms (UAE context is implicit).
  - Global/news subreddits are searched with "UAE <covid-term>" combined queries so
    only UAE-relevant posts are returned.
  - Each (subreddit, query) pair is tried with multiple sort orders (relevance, top,
    new) to maximise coverage across the 2020-2021 date range.
  - Posts are deduplicated by post_id before saving.

Required environment variables (set in .env or shell):
    REDDIT_CLIENT_ID     - Reddit app client ID
    REDDIT_CLIENT_SECRET - Reddit app client secret
    REDDIT_USER_AGENT    - Descriptive user-agent string
"""

import os
import praw
import json
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path
import time
import sys

# ========= CONFIG =========

# Subreddits where UAE context is implicit — search with COVID terms only
UAE_SUBREDDITS = ["UAE", "dubai", "abudhabi"]

# Global/news subreddits — queries must include a UAE location term
GLOBAL_SUBREDDITS = ["Coronavirus", "COVID19", "worldnews", "news", "middleeast"]

# Queries used against UAE-local subreddits
UAE_SUB_QUERIES = [
    "covid",
    "coronavirus",
    "pandemic",
    "lockdown",
    "vaccine",
    "curfew",
    "quarantine",
    "mask mandate",
]

# Queries used against global subreddits (location + topic)
GLOBAL_QUERIES = [
    "UAE covid",
    "UAE coronavirus",
    "UAE lockdown",
    "UAE vaccine",
    "UAE curfew",
    "UAE quarantine",
    "Dubai covid",
    "Dubai coronavirus",
    "Dubai lockdown",
    "Dubai vaccine",
    "Abu Dhabi covid",
    "Abu Dhabi coronavirus",
    "Emirates pandemic",
    "Emirates vaccine",
]

# Sort strategies tried per (subreddit, query) pair to maximise coverage
SORT_STRATEGIES = ["relevance", "top", "new"]

# Reddit allows up to 1000 results per search call
SEARCH_LIMIT = 1000

START_DATE = datetime(2020, 3, 1, tzinfo=timezone.utc)
END_DATE = datetime(2021, 12, 31, tzinfo=timezone.utc)

# Keywords used for final relevance filtering of returned posts
FILTER_KEYWORDS = [
    "covid", "coronavirus", "corona", "pandemic", "lockdown", "vaccine",
    "quarantine", "curfew", "uae", "dubai", "abu dhabi", "emirates",
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = PROJECT_ROOT / "data" / "raw" / "reddit_covid_uae_posts_by_day.json"
# ==========================


def load_project_env(env_path: Path):
    """Load KEY=VALUE pairs from .env into process environment if missing."""
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


load_project_env(PROJECT_ROOT / ".env")

# Credentials loaded from environment variables.
# Copy .env.example to .env and fill in your values, or export them in your shell.
CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID", "")
CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET", "")
USER_AGENT = os.environ.get("REDDIT_USER_AGENT", "crisis-analysis-bot/1.0")

def matches_keywords(text: str) -> bool:
    """Return True if the text contains at least one relevance-filter keyword."""
    if not text:
        return False
    lowered = text.lower()
    return any(k in lowered for k in FILTER_KEYWORDS)


def search_with_retry(subreddit, query: str, sort: str, max_retries: int = 3) -> list:
    """Search a subreddit with exponential backoff on rate-limit errors."""
    for attempt in range(max_retries):
        try:
            return list(
                subreddit.search(
                    query=query,
                    sort=sort,
                    time_filter="all",
                    limit=SEARCH_LIMIT,
                )
            )
        except Exception as e:
            error_msg = str(e).lower()
            if ("rate" in error_msg or "429" in error_msg) and attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 5
                print(f"    ⚠️  Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"    ❌ Search error ({sort}): {e}")
                return []
    return []


def collect_from_subreddit(
    reddit,
    subreddit_name: str,
    queries: list[str],
    seen_ids: set,
) -> tuple[int, int, dict]:
    """
    Run every query × sort strategy against one subreddit.
    Returns (scanned, matched, posts_by_day_fragment).
    """
    subreddit = reddit.subreddit(subreddit_name)
    local_posts: dict[str, list] = defaultdict(list)
    scanned = 0
    matched = 0

    for query in queries:
        for sort in SORT_STRATEGIES:
            results = search_with_retry(subreddit, query, sort)
            scanned += len(results)

            for post in results:
                if post.id in seen_ids:
                    continue

                created = datetime.fromtimestamp(post.created_utc, tz=timezone.utc)
                if created < START_DATE or created > END_DATE:
                    continue

                text = f"{post.title or ''} {post.selftext or ''}"
                if not matches_keywords(text):
                    continue

                seen_ids.add(post.id)
                matched += 1
                day_str = created.strftime("%Y-%m-%d")

                local_posts[day_str].append(
                    {
                        "subreddit": subreddit_name,
                        "post_id": post.id,
                        "title": post.title,
                        "text": post.selftext,
                        "author": str(post.author),
                        "created_utc": created.isoformat(),
                        "score": post.score,
                        "num_comments": post.num_comments,
                        "url": post.url,
                        "upvote_ratio": getattr(post, "upvote_ratio", None),
                        "is_self": post.is_self,
                    }
                )

            time.sleep(1)  # stay well within rate limits between calls

    return scanned, matched, local_posts

def main():
    print("=" * 70)
    print("REDDIT DATA COLLECTION - UAE COVID")
    print("=" * 70)

    # Validate credentials
    if not CLIENT_ID or not CLIENT_SECRET:
        print("❌ Reddit credentials not found.")
        print("   Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET environment variables.")
        print("   See .env.example for details.")
        sys.exit(1)

    # Initialize Reddit API
    try:
        reddit = praw.Reddit(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            user_agent=USER_AGENT,
        )
        reddit.user.me()
        print("✓ Reddit API connected successfully\n")
    except Exception as e:
        print(f"❌ Failed to connect to Reddit API: {e}")
        sys.exit(1)

    posts_by_day: dict[str, list] = defaultdict(list)
    seen_ids: set[str] = set()
    total_scanned = 0
    total_matched = 0

    # --- UAE-local subreddits (COVID queries, location implicit) ---
    print("── UAE subreddits (COVID queries) " + "─" * 35)
    for sub_name in UAE_SUBREDDITS:
        print(f"📡 r/{sub_name}  ({len(UAE_SUB_QUERIES)} queries × {len(SORT_STRATEGIES)} sorts)")
        scanned, matched, fragment = collect_from_subreddit(
            reddit, sub_name, UAE_SUB_QUERIES, seen_ids
        )
        total_scanned += scanned
        total_matched += matched
        for day, plist in fragment.items():
            posts_by_day[day].extend(plist)
        print(f"   ✓ {matched} unique posts matched")
        time.sleep(2)

    # --- Global subreddits (UAE + COVID combined queries) ---
    print("\n── Global subreddits (UAE+COVID queries) " + "─" * 28)
    for sub_name in GLOBAL_SUBREDDITS:
        print(f"📡 r/{sub_name}  ({len(GLOBAL_QUERIES)} queries × {len(SORT_STRATEGIES)} sorts)")
        scanned, matched, fragment = collect_from_subreddit(
            reddit, sub_name, GLOBAL_QUERIES, seen_ids
        )
        total_scanned += scanned
        total_matched += matched
        for day, plist in fragment.items():
            posts_by_day[day].extend(plist)
        print(f"   ✓ {matched} unique posts matched")
        time.sleep(2)

    # Sort each day's posts by score (descending) for quality ordering
    for day in posts_by_day:
        posts_by_day[day].sort(key=lambda p: p.get("score", 0), reverse=True)

    metadata = {
        "collection_date": datetime.now(timezone.utc).isoformat(),
        "date_range": {
            "start": START_DATE.isoformat(),
            "end": END_DATE.isoformat(),
        },
        "uae_subreddits": UAE_SUBREDDITS,
        "global_subreddits": GLOBAL_SUBREDDITS,
        "sort_strategies": SORT_STRATEGIES,
        "search_limit_per_call": SEARCH_LIMIT,
        "total_scanned": total_scanned,
        "total_matched": total_matched,
        "days_with_data": len(posts_by_day),
    }

    print("\n" + "=" * 70)
    print("COLLECTION SUMMARY")
    print("=" * 70)
    print(f"Posts scanned (incl. duplicates): {total_scanned}")
    print(f"Unique posts matched:             {total_matched}")
    print(f"Days with data:                   {len(posts_by_day)}")
    if posts_by_day:
        print(f"Avg posts/day:                    {total_matched / len(posts_by_day):.1f}")

    output = {"metadata": metadata, "posts_by_day": dict(posts_by_day)}
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Saved to {OUTPUT_FILE}")
    print("\n" + "=" * 70)
    print("SAMPLE DATA (First 3 days)")
    print("=" * 70)
    for day, plist in sorted(posts_by_day.items())[:3]:
        print(f"\n📅 {day} ({len(plist)} posts):")
        for p in plist[:3]:
            title = p["title"][:65] + "..." if len(p["title"]) > 65 else p["title"]
            print(f"  • [{p['subreddit']}] {title}")


if __name__ == "__main__":
    main()