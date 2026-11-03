from __future__ import annotations

import csv
from pathlib import Path

from sentiment_analysis import analyze_review, sentiment_summary


def main() -> None:
    filename = Path(__file__).resolve().parent / "social_media_posts.csv"
    results = []
    with filename.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or "text" not in reader.fieldnames:
            raise ValueError("CSV must contain a 'text' column.")
        for row in reader:
            results.append({**row, **analyze_review(row.get("text", ""))})

    print("\n=== REVIEW SENTIMENT ANALYSIS ===\n")
    for review in results:
        identity = review.get("post_id") or review.get("review_id") or "?"
        print(f"Review {identity}: {review['sentiment']} ({review['score']:+.2f}, {review['strength']}) | {', '.join(review['themes'])}")
    print("\n=== SENTIMENT SUMMARY ===")
    for label, count in sentiment_summary([item["sentiment"] for item in results]).items():
        print(f"{label}: {count}")
    print(f"Total reviews: {len(results)}")


if __name__ == "__main__":
    main()
