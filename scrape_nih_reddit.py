"""
Reddit r/nih Keyword Analyzer
==============================
Scrapes posts from r/nih over the past 30 days and identifies
the 20 most common keywords. Skips promoted/sponsored posts.

SETUP (one-time):
1. Go to https://www.reddit.com/prefs/apps
2. Click "Create App" or "Create Another App"
3. Choose "script" as the app type
4. Fill in any name (e.g. "NIH Keyword Analyzer") and redirect URI: http://localhost:8080
5. Copy the CLIENT_ID (under the app name) and CLIENT_SECRET
6. Paste them into the config section below

INSTALL DEPENDENCIES:
    pip install praw nltk

Then run:
    python scrape_nih_reddit.py
"""

import praw
import nltk
import re
import json
from collections import Counter
from datetime import datetime, timezone, timedelta

# ─────────────────────────────────────────────
# CONFIG — fill in your Reddit API credentials
# ─────────────────────────────────────────────
CLIENT_ID     = "YOUR_CLIENT_ID"       # From reddit.com/prefs/apps
CLIENT_SECRET = "YOUR_CLIENT_SECRET"   # From reddit.com/prefs/apps
USER_AGENT    = "nih_keyword_analyzer/1.0 by YOUR_REDDIT_USERNAME"

SUBREDDIT     = "nih"
DAYS_BACK     = 30   # How far back to look (in days)
TOP_N         = 20   # Number of top keywords to show
# ─────────────────────────────────────────────


def download_nltk_data():
    """Download required NLTK datasets if not already present."""
    packages = ["stopwords", "punkt", "punkt_tab"]
    for pkg in packages:
        try:
            nltk.data.find(f"tokenizers/{pkg}" if pkg.startswith("punkt") else f"corpora/{pkg}")
        except LookupError:
            print(f"Downloading NLTK data: {pkg}...")
            nltk.download(pkg, quiet=True)


def get_stop_words():
    """Return an expanded set of stop words to filter out noise."""
    from nltk.corpus import stopwords

    base_stops = set(stopwords.words("english"))

    # Add Reddit-specific and common filler words
    extra_stops = {
        # Reddit noise
        "www", "http", "https", "com", "reddit", "post", "comment",
        "thread", "sub", "subreddit", "upvote", "downvote", "edit",
        "update", "deleted", "removed", "bot", "mod", "moderator",
        # Common filler
        "like", "just", "also", "would", "could", "one", "know",
        "think", "get", "got", "getting", "want", "wanted", "going",
        "really", "much", "many", "well", "see", "said", "say",
        "make", "made", "use", "used", "using", "even", "still",
        "since", "new", "good", "great", "thank", "thanks", "please",
        "anyone", "someone", "everyone", "something", "anything",
        "people", "way", "thing", "things", "time", "year", "years",
        "day", "days", "month", "months", "week", "weeks",
        # Single letters and numbers that slip through
        "s", "t", "ve", "re", "ll", "d", "m",
    }

    return base_stops | extra_stops


def is_sponsored(post):
    """
    Detect promoted/sponsored posts.
    Reddit marks these via post_hint, distinguished, or whitelist_status fields.
    """
    if getattr(post, "promoted", False):
        return True
    if getattr(post, "distinguished", None) == "admin":
        return True
    # Whitelist status appears on promoted posts
    if getattr(post, "whitelist_status", None) in ("promo_adult_nsfw", "promo_all"):
        return True
    # Some promoted posts have no author or are link posts with specific domains
    if getattr(post, "author", None) is None:
        return True
    return False


def extract_words(text):
    """Tokenize text and return clean lowercase alphabetic words (length ≥ 3)."""
    if not text:
        return []
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    # Remove special characters, keep letters and spaces
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    words = text.lower().split()
    # Keep only words that are purely alphabetic and ≥ 3 characters
    return [w for w in words if w.isalpha() and len(w) >= 3]


def fetch_posts(reddit, subreddit_name, days_back):
    """Fetch all posts from the subreddit within the past `days_back` days."""
    subreddit = reddit.subreddit(subreddit_name)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    cutoff_ts = cutoff.timestamp()

    posts = []
    skipped_sponsored = 0

    print(f"\nFetching posts from r/{subreddit_name} (past {days_back} days)...")
    print("This may take a minute depending on post volume.\n")

    # Reddit API returns posts newest-first; we stop when we pass the cutoff
    for post in subreddit.new(limit=None):
        if post.created_utc < cutoff_ts:
            break  # Older than our window — stop

        if is_sponsored(post):
            skipped_sponsored += 1
            continue

        posts.append(post)

    print(f"  Found {len(posts)} posts within the past {days_back} days")
    if skipped_sponsored:
        print(f"  Skipped {skipped_sponsored} sponsored/promoted post(s)")

    return posts


def analyze_keywords(posts, stop_words, top_n):
    """Extract all words from posts and count frequency, excluding stop words."""
    word_counts = Counter()

    for post in posts:
        # Include title + selftext (body text for text posts)
        combined = f"{post.title} {post.selftext}"
        words = extract_words(combined)
        filtered = [w for w in words if w not in stop_words]
        word_counts.update(filtered)

    return word_counts.most_common(top_n)


def print_results(top_keywords, subreddit_name, days_back):
    """Print a formatted results table."""
    print("\n" + "=" * 50)
    print(f"  TOP {len(top_keywords)} KEYWORDS — r/{subreddit_name} (past {days_back} days)")
    print("=" * 50)
    print(f"  {'RANK':<6} {'KEYWORD':<25} {'COUNT':>6}")
    print("-" * 50)
    for rank, (word, count) in enumerate(top_keywords, start=1):
        print(f"  {rank:<6} {word:<25} {count:>6}")
    print("=" * 50)


def save_results(top_keywords, subreddit_name, days_back, posts):
    """Save results to a JSON file for further use."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"nih_keywords_{timestamp}.json"

    output = {
        "subreddit": subreddit_name,
        "days_analyzed": days_back,
        "posts_analyzed": len(posts),
        "generated_at": datetime.now().isoformat(),
        "top_keywords": [{"rank": i + 1, "keyword": w, "count": c}
                         for i, (w, c) in enumerate(top_keywords)],
    }

    with open(filename, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved to: {filename}")
    return filename


def main():
    # Validate config
    if CLIENT_ID == "YOUR_CLIENT_ID" or CLIENT_SECRET == "YOUR_CLIENT_SECRET":
        print("\n⚠️  Please fill in your Reddit API credentials in the CONFIG section at the top of this script.")
        print("   Instructions: https://www.reddit.com/prefs/apps\n")
        return

    # Download NLTK data
    download_nltk_data()
    stop_words = get_stop_words()

    # Connect to Reddit API (read-only, no login required)
    reddit = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent=USER_AGENT,
    )
    reddit.read_only = True

    # Fetch posts
    posts = fetch_posts(reddit, SUBREDDIT, DAYS_BACK)

    if not posts:
        print(f"\nNo posts found in r/{SUBREDDIT} within the past {DAYS_BACK} days.")
        return

    # Analyze keywords
    print(f"\nAnalyzing keywords across {len(posts)} posts...")
    top_keywords = analyze_keywords(posts, stop_words, TOP_N)

    # Display results
    print_results(top_keywords, SUBREDDIT, DAYS_BACK)

    # Save to JSON
    save_results(top_keywords, SUBREDDIT, DAYS_BACK, posts)


if __name__ == "__main__":
    main()
