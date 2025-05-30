# data_lake_manager.py
import os
import json
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from delta import *
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLakeManager:
    """
    Manages the multi-layer data lake architecture
    Bronze: Raw data as ingested
    Silver: Cleaned and standardized data
    Gold: Feature-engineered data ready for ML
    """
    
    def __init__(self, base_path: str, spark_session: Optional[SparkSession] = None):
        self.base_path = base_path
        self.bronze_path = os.path.join(base_path, "bronze")
        self.silver_path = os.path.join(base_path, "silver")
        self.gold_path = os.path.join(base_path, "gold")
        
        # Initialize paths
        for path in [self.bronze_path, self.silver_path, self.gold_path]:
            os.makedirs(path, exist_ok=True)
        
        # Initialize Spark with Delta Lake
        if spark_session:
            self.spark = spark_session
        else:
            self.spark = SparkSession.builder \
                .appName("FakeNewsDataLake") \
                .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0") \
                .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
                .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
                .getOrCreate()
    
    def write_to_bronze(self, df: pd.DataFrame, source_type: str):
        """Write raw data to bronze layer with partitioning"""
        try:
            # Add metadata
            df['bronze_timestamp'] = datetime.now().isoformat()
            df['bronze_date'] = datetime.now().strftime('%Y-%m-%d')
            
            # Convert to Spark DataFrame
            spark_df = self.spark.createDataFrame(df)
            
            # Write to Delta Lake with partitioning
            output_path = os.path.join(self.bronze_path, source_type)
            
            spark_df.write \
                .mode("append") \
                .partitionBy("bronze_date", "source") \
                .format("delta") \
                .save(output_path)
            
            logger.info(f"Written {len(df)} records to bronze layer: {output_path}")
            
        except Exception as e:
            logger.error(f"Error writing to bronze layer: {str(e)}")
    
    def process_to_silver(self):
        """Process bronze data to silver layer with cleaning and standardization"""
        try:
            # Read from bronze
            bronze_df = self.spark.read.format("delta").load(self.bronze_path)
            
            # Data cleaning and standardization
            silver_df = bronze_df \
                .filter(col("content").isNotNull()) \
                .filter(length(col("content")) > 100) \
                .withColumn("content_cleaned", 
                    regexp_replace(col("content"), r'<[^>]+>', '')) \
                .withColumn("title_cleaned", 
                    regexp_replace(col("title"), r'<[^>]+>', '')) \
                .withColumn("word_count", 
                    size(split(col("content_cleaned"), " "))) \
                .withColumn("char_count", 
                    length(col("content_cleaned"))) \
                .withColumn("has_author", 
                    when(col("author").isNotNull() & (col("author") != ""), 1).otherwise(0)) \
                .withColumn("silver_timestamp", current_timestamp()) \
                .select(
                    "id", "source", "source_type", 
                    "title_cleaned", "content_cleaned",
                    "url", "published_date", "author",
                    "word_count", "char_count", "has_author",
                    "label", "silver_timestamp"
                )
            
            # Deduplicate based on content hash
            silver_df = silver_df.dropDuplicates(["id"])
            
            # Write to silver layer
            silver_df.write \
                .mode("overwrite") \
                .format("delta") \
                .save(self.silver_path)
            
            # Optimize silver table
            delta_table = DeltaTable.forPath(self.spark, self.silver_path)
            delta_table.optimize().executeCompaction()
            
            logger.info("Processed data to silver layer")
            
        except Exception as e:
            logger.error(f"Error processing to silver layer: {str(e)}")
    
    def engineer_features_to_gold(self):
        """Feature engineering for ML - creates gold layer"""
        try:
            # Read from silver
            silver_df = self.spark.read.format("delta").load(self.silver_path)
            
            # Feature engineering
            gold_df = silver_df \
                .withColumn("title_length", length(col("title_cleaned"))) \
                .withColumn("exclamation_count", 
                    length(col("content_cleaned")) - 
                    length(regexp_replace(col("content_cleaned"), "!", ""))) \
                .withColumn("question_count",
                    length(col("content_cleaned")) - 
                    length(regexp_replace(col("content_cleaned"), "\\?", ""))) \
                .withColumn("caps_ratio",
                    length(regexp_replace(col("content_cleaned"), "[^A-Z]", "")) / 
                    col("char_count")) \
                .withColumn("avg_word_length",
                    col("char_count") / col("word_count")) \
                .withColumn("numeric_count",
                    size(split(col("content_cleaned"), "[0-9]+"))) \
                .withColumn("url_count",
                    size(split(col("content_cleaned"), "https?://\\S+"))) \
                .withColumn("gold_timestamp", current_timestamp())
            
            # Create TF-IDF features using Spark ML
            from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StopWordsRemover
            from pyspark.ml import Pipeline
            
            # Text processing pipeline
            tokenizer = Tokenizer(inputCol="content_cleaned", outputCol="words")
            remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
            hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=1000)
            idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
            
            pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf])
            model = pipeline.fit(gold_df)
            gold_df = model.transform(gold_df)
            
            # Select final features
            feature_columns = [
                "id", "source", "source_type", "label",
                "word_count", "char_count", "has_author",
                "title_length", "exclamation_count", "question_count",
                "caps_ratio", "avg_word_length", "numeric_count", "url_count",
                "tfidf_features", "gold_timestamp"
            ]
            
            gold_df = gold_df.select(feature_columns)
            
            # Write to gold layer with partitioning by label (for stratified sampling)
            gold_df.write \
                .mode("overwrite") \
                .partitionBy("label") \
                .format("delta") \
                .save(self.gold_path)
            
            # Create feature statistics
            self._create_feature_statistics(gold_df)
            
            logger.info("Created gold layer with engineered features")
            
        except Exception as e:
            logger.error(f"Error creating gold layer: {str(e)}")
    
    def _create_feature_statistics(self, df):
        """Create and store feature statistics for monitoring"""
        stats_df = df.select(
            "word_count", "char_count", "title_length",
            "exclamation_count", "question_count", "caps_ratio",
            "avg_word_length", "numeric_count", "url_count"
        ).describe()
        
        stats_path = os.path.join(self.gold_path, "feature_statistics")
        stats_df.write.mode("overwrite").parquet(stats_path)
    
    def get_data_quality_metrics(self):
        """Calculate data quality metrics across layers"""
        metrics = {}
        
        # Bronze layer metrics
        if os.path.exists(self.bronze_path):
            bronze_df = self.spark.read.format("delta").load(self.bronze_path)
            metrics['bronze'] = {
                'total_records': bronze_df.count(),
                'unique_sources': bronze_df.select("source").distinct().count(),
                'date_range': bronze_df.select(
                    min("bronze_date").alias("min_date"),
                    max("bronze_date").alias("max_date")
                ).collect()[0].asDict()
            }
        
        # Silver layer metrics
        if os.path.exists(self.silver_path):
            silver_df = self.spark.read.format("delta").load(self.silver_path)
            metrics['silver'] = {
                'total_records': silver_df.count(),
                'avg_word_count': silver_df.agg(avg("word_count")).collect()[0][0],
                'null_content_ratio': silver_df.filter(col("content_cleaned").isNull()).count() / silver_df.count()
            }
        
        # Gold layer metrics
        if os.path.exists(self.gold_path):
            gold_df = self.spark.read.format("delta").load(self.gold_path)
            if 'label' in gold_df.columns:
                label_dist = gold_df.groupBy("label").count().collect()
                metrics['gold'] = {
                    'total_records': gold_df.count(),
                    'label_distribution': {row['label']: row['count'] for row in label_dist}
                }
        
        return metrics
    
    def create_ml_ready_dataset(self, train_ratio: float = 0.8):
        """Create train/test splits from gold layer"""
        gold_df = self.spark.read.format("delta").load(self.gold_path)
        
        # Filter only labeled data
        labeled_df = gold_df.filter(col("label").isNotNull())
        
        # Stratified split
        train_df, test_df = labeled_df.randomSplit([train_ratio, 1-train_ratio], seed=42)
        
        # Save splits
        train_path = os.path.join(self.gold_path, "train")
        test_path = os.path.join(self.gold_path, "test")
        
        train_df.write.mode("overwrite").format("delta").save(train_path)
        test_df.write.mode("overwrite").format("delta").save(test_path)
        
        logger.info(f"Created ML datasets - Train: {train_df.count()}, Test: {test_df.count()}")
        
        return train_path, test_path


# Example usage and data lake initialization
if __name__ == "__main__":
    # Initialize data lake
    lake_manager = DataLakeManager("/data/fake-news-lake")
    
    # Example: Process streaming data from Kafka to bronze
    from pyspark.sql.functions import from_json
    
    # Define schema for incoming news data
    news_schema = StructType([
        StructField("id", StringType()),
        StructField("source", StringType()),
        StructField("source_type", StringType()),
        StructField("title", StringType()),
        StructField("content", StringType()),
        StructField("url", StringType()),
        StructField("published_date", StringType()),
        StructField("author", StringType()),
        StructField("label", StringType()),
        StructField("raw_data", StringType())
    ])
    
    # Read from Kafka stream
    kafka_df = lake_manager.spark \
        .readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", "raw-news-feed") \
        .load()
    
    # Parse JSON data
    parsed_df = kafka_df.select(
        from_json(col("value").cast("string"), news_schema).alias("data")
    ).select("data.*")
    
    # Write to bronze layer using streaming
    query = parsed_df.writeStream \
        .outputMode("append") \
        .format("delta") \
        .option("checkpointLocation", "/data/checkpoints/bronze") \
        .partitionBy("source_type") \
        .trigger(processingTime='5 minutes') \
        .start(os.path.join(lake_manager.bronze_path, "streaming"))
    
    # Run batch processing periodically
    lake_manager.process_to_silver()
    lake_manager.engineer_features_to_gold()
    lake_manager.create_ml_ready_dataset()