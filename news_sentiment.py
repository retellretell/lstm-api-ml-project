import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict

import requests

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except Exception:
    TEXTBLOB_AVAILABLE = False

logger = logging.getLogger(__name__)

NEWS_API_KEY = os.getenv("NEWS_API_KEY")


def _analyze_sentiment(text: str) -> str:
    """Return sentiment label for given text."""
    if TEXTBLOB_AVAILABLE:
        polarity = TextBlob(text).sentiment.polarity
        if polarity > 0.1:
            return "positive"
        if polarity < -0.1:
            return "negative"
        return "neutral"
    # Fallback heuristic
    lower = text.lower()
    positive_words = ["gain", "rise", "surge", "beat", "profit", "growth"]
    negative_words = ["fall", "drop", "loss", "decline", "slump", "fear"]
    score = 0
    for w in positive_words:
        if w in lower:
            score += 1
    for w in negative_words:
        if w in lower:
            score -= 1
    if score > 0:
        return "positive"
    if score < 0:
        return "negative"
    return "neutral"


def get_recent_news(days_back: int = 1) -> List[Dict]:
    """Fetch recent market news and return sentiment scored articles."""
    start_date = datetime.utcnow() - timedelta(days=days_back)
    if NEWS_API_KEY:
        try:
            params = {
                "q": "stock market",
                "from": start_date.strftime("%Y-%m-%d"),
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 20,
                "apiKey": NEWS_API_KEY,
            }
            resp = requests.get(
                "https://newsapi.org/v2/everything", params=params, timeout=10
            )
            resp.raise_for_status()
            data = resp.json()
            articles = []
            for art in data.get("articles", []):
                title = art.get("title")
                url = art.get("url")
                published_at = art.get("publishedAt")
                if not title or not published_at:
                    continue
                try:
                    date = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                except ValueError:
                    date = datetime.utcnow()
                sentiment = _analyze_sentiment(title)
                articles.append({
                    "title": title,
                    "date": date,
                    "sentiment": sentiment,
                    "url": url,
                })
            return articles
        except Exception as e:
            logger.error("Failed to fetch news: %s", e)
    else:
        logger.warning("NEWS_API_KEY not set, returning dummy news")

    # Dummy data fallback
    return [
        {
            "title": "Market is quiet due to missing API key",
            "date": datetime.utcnow(),
            "sentiment": "neutral",
            "url": "#",
        }
    ]
