# Social Media Sentiment Analysis Using Big Data

## Project Overview
This project analyzes social media posts and classifies them into three sentiment categories: Positive, Negative, and Neutral.

It demonstrates how Big Data technologies can be used to process and analyze text data through Python, MapReduce, Apache Spark, and Streamlit.

## Technologies Used
- Python
- Apache Spark (PySpark)
- MapReduce
- Streamlit
- Pandas

## Project Features
- Sentiment classification of social media posts.
- Positive, Negative, and Neutral sentiment analysis.
- MapReduce-based sentiment processing.
- Apache Spark-based data processing.
- Interactive Streamlit dashboard.
- Sentiment distribution visualization.

## Project Structure
- `main.py` – Main project entry point.
- `sentiment_analysis.py` – Sentiment analysis logic.
- `mapreduce_sentiment.py` – MapReduce implementation.
- `spark_sentiment.py` – Spark implementation.
- `dashboard.py` – Interactive Streamlit dashboard.
- `social_media_posts.csv` – Sample social media dataset.
- `requirements.txt` – Required Python packages.

## Sample Results
The sample dataset contains 15 social media posts.

| Sentiment | Posts |
|---|---:|
| Positive | 5 |
| Negative | 5 |
| Neutral | 5 |
| **Total** | **15** |

## How to Run

1. Install Python 3.10 or newer.
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Run the dashboard:

```bash
streamlit run dashboard.py
```

4. Run the MapReduce implementation:

```bash
python mapreduce_sentiment.py
```

5. Run the Spark implementation:

```bash
python spark_sentiment.py
```

## Learning Outcomes
- Understanding Big Data processing concepts.
- Practical exposure to Apache Spark and MapReduce.
- Working with text data using Python.
- Building an interactive data visualization dashboard.

## Project Type
KTU Big Data Processing Project
