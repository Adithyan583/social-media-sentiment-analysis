"""A local MapReduce-style demonstration using the shared review analyzer."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from sentiment_analysis import analyze_review


def main() -> None:
    filename = Path(__file__).resolve().parent / "social_media_posts.csv"
    # MAP: each review emits sentiment and theme keys.
    mapped = []
    with filename.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or "text" not in reader.fieldnames:
            raise ValueError("CSV must contain a 'text' column.")
        for row in reader:
            result = analyze_review(row.get("text", ""))
            mapped.append((row, result))

    # REDUCE: aggregate counts by sentiment and theme.
    sentiment_counts = Counter(result["sentiment"] for _, result in mapped)
    theme_counts = Counter(theme for _, result in mapped for theme in result["themes"])

    print("\n===== MAPREDUCE REVIEW ANALYSIS =====")
    print("\nIndividual review results:")
    for row, result in mapped:
        identity = row.get("post_id") or row.get("review_id") or "?"
        print(f"Review {identity}: {result['sentiment']} ({result['score']:+.2f}) | {', '.join(result['themes'])}")
    print("\n===== REDUCE: SENTIMENT SUMMARY =====")
    for label in ("Positive", "Neutral", "Negative"):
        print(f"{label}: {sentiment_counts[label]} reviews")
    print("\n===== REDUCE: THEME SUMMARY =====")
    for theme, count in theme_counts.most_common():
        print(f"{theme}: {count} mentions")
    print(f"\nTotal reviews processed: {len(mapped)}")


if __name__ == "__main__":
    main()
