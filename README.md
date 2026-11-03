# Review & Feedback Insights

A small review analysis project with a Streamlit dashboard, a command-line summary, and local MapReduce-style and Apache Spark demonstrations. All entry points share the same analyzer so their sentiment labels remain consistent.

## Run the dashboard

1. Install Python 3.10 or newer.
2. Install the packages listed in `requirements.txt` with `python -m pip install -r requirements.txt`.
3. Start the dashboard with `streamlit run dashboard.py`.
4. Use the included sample data or upload your own CSV.

The CSV must include one text column named `text`, `review`, `feedback`, or `comment`. Optional columns such as `product`, `category`, `date`, `rating`, and `review_id` are retained in the results. Date columns named `date`, `created_at`, `timestamp`, or `published_at` can be used for monthly trend charts.

The included CSV contains 560 illustrative reviews: 20 seed examples and 540 generated examples spanning six products, six themes, three sentiment levels, dates, and ratings. The generated records are synthetic and should only be used to demonstrate the analysis and dashboard. Real conclusions require representative, real review data.

## Other entry points

- `python main.py` prints review-level sentiment and a summary.
- `python mapreduce_sentiment.py` demonstrates mapping review results and reducing sentiment and theme counts.
- `python spark_sentiment.py` performs the same analysis through a local Spark session.

## Analysis notes

The shared analyzer in `sentiment_analysis.py` uses an explainable word and phrase lexicon, simple negation and intensity rules, and keyword-based review themes. It reports Positive, Neutral, or Negative sentiment, a score from -1 to +1, strength, evidence terms, and themes. It does not understand every context, sarcasm, domain-specific language, or languages other than English. Theme detection is keyword-based, and sentiment should be treated as an estimate rather than a ground-truth label.
