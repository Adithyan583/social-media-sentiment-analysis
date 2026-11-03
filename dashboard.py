from __future__ import annotations

from collections import Counter
from pathlib import Path
import re

import pandas as pd
import streamlit as st

from sentiment_analysis import analyze_review


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = BASE_DIR / "social_media_posts.csv"
STOPWORDS = {
    "about", "after", "again", "also", "and", "are", "because", "been", "but",
    "for", "from", "had", "has", "have", "into", "its", "just", "like", "not",
    "our", "really", "the", "their", "there", "this", "that", "they", "them",
    "then", "very", "was", "were", "with", "would", "you", "your",
}

st.set_page_config(page_title="Review Insights", page_icon="📊", layout="wide")
st.title("📊 Review & Feedback Insights")
st.caption("Explore sentiment, recurring themes, and the feedback behind each result.")

uploaded = st.file_uploader("Upload a CSV of reviews or posts", type=["csv"])
try:
    if uploaded is not None:
        source_df = pd.read_csv(uploaded)
        source_name = uploaded.name
    else:
        source_df = pd.read_csv(DEFAULT_DATA)
        source_name = DEFAULT_DATA.name
except FileNotFoundError:
    st.error("The sample dataset is missing. Upload a CSV to continue.")
    st.stop()
except Exception as exc:
    st.error(f"Could not read this CSV: {exc}")
    st.stop()

source_df.columns = [str(c).strip() for c in source_df.columns]
if "sample_source" in source_df.columns:
    origins = source_df["sample_source"].value_counts()
    st.info(
        "The included dataset is illustrative: "
        f"{origins.get('Illustrative seed', 0):,} seed examples and "
        f"{origins.get('Synthetic example', 0):,} generated examples. "
        "It is for demonstrating analysis and charts, not a source of real customer feedback."
    )
text_col = next((c for c in source_df.columns if c.lower() in {"text", "review", "feedback", "comment"}), None)
if not text_col:
    st.error("Your CSV needs a text column named text, review, feedback, or comment.")
    st.stop()

df = source_df.copy()
df[text_col] = df[text_col].fillna("").astype(str)
df = df[df[text_col].str.strip().ne("")].copy()
if df.empty:
    st.warning("There are no non-empty reviews to analyze.")
    st.stop()

analysis = df[text_col].map(analyze_review).apply(pd.Series)
df = pd.concat([df.reset_index(drop=True), analysis.reset_index(drop=True)], axis=1)

with st.sidebar:
    st.header("Filter results")
    st.caption(f"Source: {source_name} · {len(df):,} reviews")
    selected_sentiments = st.multiselect("Sentiment", ["Positive", "Neutral", "Negative"], default=["Positive", "Neutral", "Negative"])
    theme_options = sorted({theme for themes in df["themes"] for theme in themes})
    selected_themes = st.multiselect("Theme", theme_options, default=theme_options)
    product_col = next((c for c in df.columns if c.lower() in {"product", "product_name", "post", "category", "source"}), None)
    if product_col:
        product_values = sorted(df[product_col].dropna().astype(str).unique())
        selected_products = st.multiselect(product_col.replace("_", " ").title(), product_values, default=product_values)
    else:
        selected_products = None

filtered = df[df["sentiment"].isin(selected_sentiments)]
filtered = filtered[filtered["themes"].map(lambda themes: bool(set(themes).intersection(selected_themes)))]
if product_col and selected_products is not None:
    filtered = filtered[filtered[product_col].astype(str).isin(selected_products)]

if filtered.empty:
    st.info("No reviews match the current filters.")
    st.stop()

counts = filtered["sentiment"].value_counts()
total = len(filtered)
positive_share = counts.get("Positive", 0) / total * 100
c1, c2, c3, c4 = st.columns(4)
c1.metric("Reviews analyzed", f"{total:,}")
c2.metric("Positive", f"{counts.get('Positive', 0):,}", f"{positive_share:.0f}% of filtered reviews")
c3.metric("Neutral", f"{counts.get('Neutral', 0):,}")
c4.metric("Negative", f"{counts.get('Negative', 0):,}")

left, right = st.columns(2)
with left:
    st.subheader("Sentiment distribution")
    st.bar_chart(counts.reindex(["Positive", "Neutral", "Negative"], fill_value=0), color="#5087e8")
with right:
    st.subheader("Themes mentioned")
    theme_counts = Counter(theme for themes in filtered["themes"] for theme in themes)
    st.bar_chart(pd.Series(theme_counts).sort_values(ascending=False), color="#43a982")

st.subheader("What people talk about")
theme_rows = []
for theme in sorted(theme_counts):
    reviews = filtered[filtered["themes"].map(lambda themes: theme in themes)]
    sentiment_counts = reviews["sentiment"].value_counts()
    theme_rows.append({
        "Theme": theme,
        "Reviews": len(reviews),
        "Positive": int(sentiment_counts.get("Positive", 0)),
        "Neutral": int(sentiment_counts.get("Neutral", 0)),
        "Negative": int(sentiment_counts.get("Negative", 0)),
        "Average sentiment score": round(reviews["score"].mean(), 2),
    })
st.dataframe(pd.DataFrame(theme_rows), use_container_width=True, hide_index=True)

rating_col = next((c for c in df.columns if c.lower() in {"rating", "stars", "score"}), None)
if rating_col:
    rated = filtered.copy()
    rated["_numeric_rating"] = pd.to_numeric(rated[rating_col], errors="coerce")
    rating_summary = rated.dropna(subset=["_numeric_rating"]).groupby("sentiment")["_numeric_rating"].mean()
    if not rating_summary.empty:
        st.subheader("Average rating by detected sentiment")
        st.bar_chart(rating_summary.reindex(["Positive", "Neutral", "Negative"]).dropna(), color="#e6a23c")

date_col = next((c for c in df.columns if c.lower() in {"date", "created_at", "timestamp", "published_at"}), None)
if date_col:
    dated = filtered.copy()
    dated[date_col] = pd.to_datetime(dated[date_col], errors="coerce")
    dated = dated.dropna(subset=[date_col])
    if not dated.empty:
        dated["Period"] = dated[date_col].dt.to_period("M").astype(str)
        trend = dated.pivot_table(index="Period", columns="sentiment", values=text_col, aggfunc="count", fill_value=0)
        for label in ["Positive", "Neutral", "Negative"]:
            if label not in trend:
                trend[label] = 0
        st.subheader("Sentiment over time")
        st.line_chart(trend[["Positive", "Neutral", "Negative"]])

st.subheader("Frequently used words")
terms = Counter(
    word.lower() for text in filtered[text_col]
    for word in re.findall(r"\b[a-zA-Z]{3,}\b", text)
    if word.lower() not in STOPWORDS
)
if terms:
    st.write(" · ".join(f"**{word}** ({count})" for word, count in terms.most_common(12)))
else:
    st.caption("No recurring terms found in the selected reviews.")

st.subheader("Review-level results")
display_cols = [c for c in ["post_id", "review_id", product_col, date_col, text_col, "rating", "sample_source", "sentiment", "score", "strength", "themes", "evidence"] if c and c in filtered.columns]
st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)
st.download_button(
    "Download analyzed reviews as CSV",
    data=filtered.drop(columns=["themes"]).assign(themes=filtered["themes"].map("; ".join)).to_csv(index=False).encode("utf-8"),
    file_name="analyzed_reviews.csv",
    mime="text/csv",
)
st.caption("Sentiment is an automated estimate. Theme labels are keyword-based and should be reviewed for important decisions.")
