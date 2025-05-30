# data_ingestion.py
import json
import feedparser
import requests
from kafka import KafkaProducer
from datetime import datetime
import hashlib
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsIngestionPipeline:
    """
    Handles ingestion from multiple sources into Kafka for the data lake
    """
    
    def __init__(self, kafka_config: Dict):
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_config['bootstrap_servers'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None
        )
        self.topic = kafka_config.get('topic', 'raw-news-feed')
        
    def ingest_rss_feed(self, feed_url: str, source_name: str):
        """Ingest news from RSS feeds"""
        try:
            feed = feedparser.parse(feed_url)
            
            for entry in feed.entries:
                # Create unique ID for deduplication
                content_hash = hashlib.md5(
                    f"{entry.title}{entry.get('summary', '')}".encode()
                ).hexdigest()
                
                article = {
                    'id': content_hash,
                    'source': source_name,
                    'source_type': 'rss',
                    'title': entry.title,
                    'content': entry.get('summary', ''),
                    'url': entry.link,
                    'published_date': entry.get('published', ''),
                    'ingestion_timestamp': datetime.now().isoformat(),
                    'raw_data': dict(entry)  # Store complete raw data
                }
                
                # Send to Kafka
                self.producer.send(
                    self.topic,
                    key=content_hash,
                    value=article
                )
                
            logger.info(f"Ingested {len(feed.entries)} articles from {source_name}")
            
        except Exception as e:
            logger.error(f"Error ingesting RSS feed {feed_url}: {str(e)}")
    
    def ingest_news_api(self, api_endpoint: str, api_key: str, params: Dict):
        """Ingest from news APIs like NewsAPI"""
        headers = {'Authorization': f'Bearer {api_key}'}
        
        try:
            response = requests.get(api_endpoint, headers=headers, params=params)
            data = response.json()
            
            for article in data.get('articles', []):
                content_hash = hashlib.md5(
                    f"{article['title']}{article.get('description', '')}".encode()
                ).hexdigest()
                
                news_item = {
                    'id': content_hash,
                    'source': article.get('source', {}).get('name', 'unknown'),
                    'source_type': 'news_api',
                    'title': article['title'],
                    'content': article.get('content', article.get('description', '')),
                    'url': article['url'],
                    'published_date': article.get('publishedAt', ''),
                    'author': article.get('author', ''),
                    'image_url': article.get('urlToImage', ''),
                    'ingestion_timestamp': datetime.now().isoformat(),
                    'raw_data': article
                }
                
                self.producer.send(
                    self.topic,
                    key=content_hash,
                    value=news_item
                )
                
            logger.info(f"Ingested {len(data.get('articles', []))} articles from API")
            
        except Exception as e:
            logger.error(f"Error ingesting from news API: {str(e)}")
    
    def ingest_social_media_post(self, post_data: Dict, platform: str):
        """Ingest social media posts (Twitter, Reddit, etc.)"""
        content_hash = hashlib.md5(
            f"{platform}{post_data.get('id', '')}".encode()
        ).hexdigest()
        
        social_post = {
            'id': content_hash,
            'source': platform,
            'source_type': 'social_media',
            'title': post_data.get('title', ''),  # For Reddit
            'content': post_data.get('text', post_data.get('body', '')),
            'url': post_data.get('url', ''),
            'author': post_data.get('author', post_data.get('user', {}).get('username', '')),
            'engagement': {
                'likes': post_data.get('likes', 0),
                'shares': post_data.get('shares', 0),
                'comments': post_data.get('comments', 0)
            },
            'published_date': post_data.get('created_at', ''),
            'ingestion_timestamp': datetime.now().isoformat(),
            'raw_data': post_data
        }
        
        self.producer.send(
            self.topic,
            key=content_hash,
            value=social_post
        )
    
    def ingest_labeled_dataset(self, file_path: str, label_column: str = 'label'):
        """Ingest pre-labeled datasets for training"""
        import pandas as pd
        
        try:
            df = pd.read_csv(file_path)
            
            for _, row in df.iterrows():
                content_hash = hashlib.md5(
                    f"{row.get('title', '')}{row.get('text', '')}".encode()
                ).hexdigest()
                
                labeled_item = {
                    'id': content_hash,
                    'source': 'training_dataset',
                    'source_type': 'labeled_data',
                    'title': row.get('title', ''),
                    'content': row.get('text', row.get('content', '')),
                    'label': row[label_column],  # 'fake' or 'real'
                    'ingestion_timestamp': datetime.now().isoformat(),
                    'raw_data': row.to_dict()
                }
                
                self.producer.send(
                    self.topic,
                    key=content_hash,
                    value=labeled_item
                )
                
            logger.info(f"Ingested {len(df)} labeled articles from dataset")
            
        except Exception as e:
            logger.error(f"Error ingesting labeled dataset: {str(e)}")
    
    def close(self):
        """Flush and close Kafka producer"""
        self.producer.flush()
        self.producer.close()


# Airflow DAG for orchestration
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import timedelta

default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'news_ingestion_pipeline',
    default_args=default_args,
    description='Ingest news from multiple sources',
    schedule_interval=timedelta(hours=1),
    catchup=False
)

def run_rss_ingestion():
    kafka_config = {
        'bootstrap_servers': 'localhost:9092',
        'topic': 'raw-news-feed'
    }
    
    pipeline = NewsIngestionPipeline(kafka_config)
    
    # List of RSS feeds to monitor
    rss_feeds = [
        ('https://rss.cnn.com/rss/cnn_topstories.rss', 'CNN'),
        ('https://feeds.bbci.co.uk/news/rss.xml', 'BBC'),
        ('https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml', 'NYTimes')
    ]
    
    for feed_url, source in rss_feeds:
        pipeline.ingest_rss_feed(feed_url, source)
    
    pipeline.close()

rss_task = PythonOperator(
    task_id='ingest_rss_feeds',
    python_callable=run_rss_ingestion,
    dag=dag
)