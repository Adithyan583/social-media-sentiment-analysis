"""Shared, dependency-free sentiment and review-theme analysis."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any


POSITIVE_WORDS = {
    "adore", "amazing", "awesome", "best", "brilliant", "comfortable",
    "delicious", "easy", "enjoyable", "excellent", "fantastic", "fast", "friendly",
    "good", "great", "happy", "helpful", "impressive", "love", "loved",
    "lovely", "perfect", "quickly", "recommend", "reliable", "resolved",
    "satisfied", "smooth", "superb", "useful", "wonderful", "worthwhile", "works",
}

NEGATIVE_WORDS = {
    "annoying", "awful", "broken", "buggy", "cold", "confusing", "crash",
    "crashes", "crashing", "damaged", "disappointing", "disappointed", "expensive",
    "hate", "hated", "horrible", "late", "poor", "problem", "refund", "slow",
    "terrible", "uncomfortable", "unhelpful", "unusable", "useless", "waste",
    "worst", "wrong",
}

POSITIVE_PHRASES = {
    "no complaints", "works perfectly", "value for money", "easy to use",
    "high quality", "would recommend", "above expectations",
}
NEGATIVE_PHRASES = {
    "does not work", "doesn't work", "did not work", "not worth",
    "waste of money", "poor quality", "customer service",  # service is judged by nearby words
    "would not recommend", "never again", "below expectations",
}
NEGATORS = {"not", "no", "never", "hardly", "rarely", "without", "isn't", "wasn't", "doesn't", "didn't", "can't", "cannot"}
INTENSIFIERS = {"very": 1.35, "really": 1.25, "extremely": 1.7, "absolutely": 1.6, "so": 1.2, "slightly": 0.65, "somewhat": 0.75}
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|[!?]+", re.IGNORECASE)

THEMES = {
    "Quality": {"quality", "build", "material", "durable", "sturdy", "cheap", "defect", "broken", "damaged", "packaging"},
    "Price & value": {"price", "cost", "expensive", "cheap", "value", "money", "worth", "affordable"},
    "Delivery": {"delivery", "shipping", "shipped", "arrived", "late", "package", "packaging", "courier", "order"},
    "Customer service": {"service", "support", "staff", "agent", "response", "responsive", "helpful", "refund", "representative"},
    "Usability": {"easy", "difficult", "simple", "setup", "install", "interface", "use", "usable", "confusing", "app"},
    "Performance": {"fast", "slow", "speed", "performance", "battery", "crash", "crashes", "crashing", "lag", "works", "working", "function"},
    "Experience": {"experience", "enjoyable", "comfortable", "happy", "satisfied", "disappointed", "recommend", "love", "hate"},
}


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "")]


def analyze_sentiment(text: Any) -> dict[str, Any]:
    """Return a sentiment label, normalized score, and explainable signal counts."""
    raw = "" if text is None else str(text)
    tokens = _tokens(raw)
    words = [t for t in tokens if t[0].isalpha()]
    pos = neg = 0.0
    evidence: list[str] = []

    normalized = re.sub(r"\s+", " ", raw.lower()).strip()
    for phrase in POSITIVE_PHRASES:
        if phrase in normalized:
            pos += 1.5
            evidence.append(phrase)
    for phrase in NEGATIVE_PHRASES:
        if phrase in normalized and phrase not in {"customer service"}:
            neg += 1.5
            evidence.append(phrase)

    for index, word in enumerate(words):
        if word not in POSITIVE_WORDS and word not in NEGATIVE_WORDS:
            continue
        valence = 1.0 if word in POSITIVE_WORDS else -1.0
        previous = words[max(0, index - 3):index]
        multiplier = next((INTENSIFIERS[w] for w in reversed(previous) if w in INTENSIFIERS), 1.0)
        if any(w in NEGATORS for w in previous[-3:]):
            valence *= -0.8
        # Contrast clauses typically carry the main opinion after "but".
        if "but" in words[:index]:
            multiplier *= 1.25
        if valence > 0:
            pos += multiplier
        else:
            neg += multiplier
        evidence.append(word)

    exclamations = raw.count("!")
    if exclamations and (pos or neg):
        if pos > neg:
            pos += min(exclamations * 0.15, 0.45)
        elif neg > pos:
            neg += min(exclamations * 0.15, 0.45)

    total = pos + neg
    score = 0.0 if total == 0 else round((pos - neg) / max(total, 2.0), 3)
    if total < 0.7 or abs(score) < 0.16:
        label = "Neutral"
    else:
        label = "Positive" if score > 0 else "Negative"
    strength = "Low" if abs(score) < 0.35 else "Moderate" if abs(score) < 0.7 else "Strong"
    return {
        "sentiment": label,
        "score": score,
        "strength": strength,
        "positive_signals": round(pos, 2),
        "negative_signals": round(neg, 2),
        "evidence": ", ".join(dict.fromkeys(evidence)) or "No clear sentiment terms",
    }


def analyze_review(text: Any) -> dict[str, Any]:
    """Add explainable aspect/theme labels to the shared sentiment result."""
    raw = "" if text is None else str(text)
    tokens = set(_tokens(raw))
    sentiment = analyze_sentiment(raw)
    themes = [name for name, terms in THEMES.items() if tokens.intersection(terms)]
    return {**sentiment, "themes": themes or ["Other / general"]}


def sentiment_summary(values: list[str]) -> dict[str, int]:
    counts = Counter(values)
    return {name: counts.get(name, 0) for name in ("Positive", "Neutral", "Negative")}
