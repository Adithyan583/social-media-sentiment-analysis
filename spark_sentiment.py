"""Apache Spark demonstration using the shared review analyzer."""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, explode, split, udf
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

from sentiment_analysis import analyze_review


def main() -> None:
    filename = Path(__file__).resolve().parent / "social_media_posts.csv"
    spark = (
        SparkSession.builder
        .appName("ReviewFeedbackAnalysis")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    try:
        df = spark.read.option("header", True).option("inferSchema", True).csv(str(filename))
        if "text" not in df.columns:
            raise ValueError("CSV must contain a 'text' column.")

        analysis_schema = StructType([
            StructField("sentiment", StringType(), False),
            StructField("score", DoubleType(), False),
            StructField("themes", StringType(), False),
        ])

        def analyze_for_spark(text):
            result = analyze_review(text)
            return {
                "sentiment": result["sentiment"],
                "score": float(result["score"]),
                "themes": ",".join(result["themes"]),
            }

        analysis_udf = udf(analyze_for_spark, analysis_schema)
        results = (
            df.withColumn("analysis", analysis_udf(col("text")))
            .withColumn("sentiment", col("analysis.sentiment"))
            .withColumn("sentiment_score", col("analysis.score"))
            .withColumn("themes", col("analysis.themes"))
            .drop("analysis")
        )
        print("\n===== SPARK REVIEW ANALYSIS =====")
        results.show(truncate=False)
        print("\n===== SENTIMENT SUMMARY =====")
        results.groupBy("sentiment").agg(count("*").alias("reviews")).show()
        print("===== THEME SUMMARY =====")
        results.withColumn("theme", explode(split(col("themes"), ","))) \
            .groupBy("theme").agg(count("*").alias("reviews")).orderBy(col("reviews").desc()).show(truncate=False)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
