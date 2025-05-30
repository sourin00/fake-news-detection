# 🔍 Fake News Detection Platform

A scalable, real-time fake news detection system built using Big Data technologies and Machine Learning. The platform ingests news from multiple sources, processes them through a data lake architecture, and uses distributed ML models to classify content as fake or real.

## Overview

The Fake News Detection Platform is designed to tackle the growing problem of misinformation by:
- Automatically collecting news from multiple sources (RSS feeds, APIs, social media)
- Processing data through a medallion architecture data lake (Bronze → Silver → Gold)
- Training distributed ML models using Apache Spark MLlib
- Providing real-time predictions via REST API
- Offering comprehensive monitoring and analytics dashboards

### Key Capabilities

- **Real-time Detection**: Process and classify news articles within seconds
- **Multi-Source Ingestion**: Support for RSS feeds, News APIs, and social media
- **Scalable Architecture**: Handle millions of articles using distributed computing
- **High Accuracy**: Ensemble ML models achieving >90% accuracy
- **Data Lake Storage**: Efficient storage and processing of structured/unstructured data

## Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                   Fake News Detection Platform                │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  Data Sources          Ingestion           Data Lake          │
│  ┌──────────┐        ┌─────────┐        ┌─────────────┐       │
│  │ RSS Feed │───┐    │         │        │   Bronze    │       │
│  │ News API │───┼───▶│  Kafka  │───────▶│   Silver    │       │
│  │ Social   │───┘    │         │        │   Gold      │       │
│  └──────────┘        └─────────┘        └──────┬──────┘       │
│                                                │              │
│                      Processing & ML           │              │
│                    ┌─────────────────┐         │              │
│                    │   Spark MLlib   │◀────────┘              │
│                    │  ┌───────────┐  │                        │
│                    │  │ Random    │  │      Serving           │
│                    │  │ Forest    │  │    ┌─────────┐         │
│                    │  ├───────────┤  │    │   API   │         │
│                    │  │ Logistic  │  │───▶│ Service │         │
│                    │  │ Regression│  │    └─────────┘         │
│                    │  ├───────────┤  │                        │
│                    │  │ Gradient  │  │    ┌─────────┐         │
│                    │  │ Boosting  │  │    │Dashboard│         │
│                    │  └───────────┘  │    └─────────┘         │
│                    └─────────────────┘                        │
└───────────────────────────────────────────────────────────────┘
```

## Features

### Data Ingestion
- **Multi-source Support**: RSS feeds, News APIs, social media platforms
- **Real-time Streaming**: Apache Kafka for high-throughput data ingestion
- **Deduplication**: Content-based hashing to avoid duplicate processing
- **Schema Evolution**: Flexible data ingestion supporting changing formats

### Data Lake Architecture
- **Bronze Layer**: Raw data storage with full fidelity
- **Silver Layer**: Cleaned, deduplicated, and standardized data
- **Gold Layer**: Feature-engineered data ready for ML
- **Delta Lake**: ACID transactions and time-travel capabilities

### Machine Learning
- **Multiple Algorithms**: Random Forest, Logistic Regression, Gradient Boosting
- **Ensemble Learning**: Combine multiple models for better accuracy
- **Feature Engineering**: 
  - Content features (word count, capitalization ratio)
  - Stylistic features (exclamation marks, questions)
  - TF-IDF text vectorization
- **Distributed Training**: Leverage Spark MLlib for scalable training

### Real-time Serving
- **REST API**: Simple HTTP endpoints for predictions
- **Caching**: Redis integration for low-latency responses
- **Batch Processing**: Support for bulk article analysis
- **Feature Importance**: Explainable AI insights

### Monitoring & Analytics
- **Real-time Dashboard**: Streamlit-based monitoring interface
- **Performance Metrics**: Throughput, latency, accuracy tracking
- **Data Quality**: Automated quality checks across all layers
- **System Health**: Component status and resource utilization

## Technology Stack

### Core Technologies
- **Apache Spark 3.4+**: Distributed data processing and ML
- **Apache Kafka**: Real-time data streaming
- **Delta Lake**: ACID-compliant data lake storage
- **Python 3.8+**: Primary programming language

### Storage & Databases
- **HDFS/MinIO**: Distributed file storage
- **Redis**: In-memory caching
- **PostgreSQL**: Metadata storage

### ML & Analytics
- **PySpark MLlib**: Distributed machine learning
- **scikit-learn**: Additional ML utilities
- **pandas**: Data manipulation
- **NLTK/TextBlob**: Natural language processing

### Orchestration & Monitoring
- **Apache Airflow**: Workflow orchestration
- **Streamlit**: Interactive dashboards
- **MLflow**: ML experiment tracking
- **Docker**: Containerization

## Prerequisites

### System Requirements
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: At least 50GB free space
- **CPU**: 4+ cores recommended
- **OS**: Linux/macOS (Windows via WSL2)

### Software Requirements
```bash
# Required software versions
- Python 3.8+
- Java 8 or 11 (for Spark)
- Docker & Docker Compose
- Git
```

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/sourin00/fake-news-detection.git
cd fake-news-detection
```

### 2. Set Up Python Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables
```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your configurations
# - API keys for news sources
# - Kafka connection settings
# - Spark configuration
```

### 4. Start Infrastructure Services
```bash
# Start all services using Docker Compose
docker-compose up -d

# Verify services are running
docker-compose ps
```

### 5. Initialize Data Lake
```bash
# Create data lake structure
python scripts/init_data_lake.py

# Verify structure
hdfs dfs -ls /data/fake-news-lake/
```

### 6. Download Pre-trained Models (Optional)
```bash
# Download pre-trained models
wget https://example.com/models/fake-news-model.zip
unzip fake-news-model.zip -d models/
```

## Usage

### Starting the Data Ingestion Pipeline

```bash
# Start Kafka consumer for data ingestion
python src/ingestion/kafka_consumer.py

# In another terminal, start the ingestion sources
python src/ingestion/run_ingestion.py --sources rss,api,social
```

### Training the ML Models

```bash
# Run the training pipeline
python src/ml/train_models.py \
  --input /data/fake-news-lake/gold/train \
  --output /models/fake-news-detector \
  --algorithms rf,lr,gbt
```

### Starting the API Service

```bash
# Start the Flask API server
python src/api/app.py

# The API will be available at http://localhost:5000
```

### Launching the Monitoring Dashboard

```bash
# Start Streamlit dashboard
streamlit run src/dashboard/app.py

# Access at http://localhost:8501
```

## Data Pipeline

### 1. Data Sources Configuration

```python
# config/sources.yaml
rss_feeds:
  - url: "https://rss.cnn.com/rss/cnn_topstories.rss"
    name: "CNN"
  - url: "https://feeds.bbci.co.uk/news/rss.xml"
    name: "BBC"

news_api:
  endpoint: "https://newsapi.org/v2/top-headlines"
  api_key: "${NEWS_API_KEY}"
  
social_media:
  twitter:
    enabled: false  # Requires Twitter API access
  reddit:
    enabled: true
    subreddits: ["news", "worldnews"]
```

### 2. Data Flow

```
Raw Data → Kafka → Bronze Layer → Silver Layer → Gold Layer → ML Models
   ↓                     ↓              ↓             ↓           ↓
Ingestion          Deduplication   Cleaning    Feature Eng.  Training
```

### 3. Scheduling with Airflow

```python
# DAG runs every hour
schedule_interval = "@hourly"

# Key tasks:
# 1. Ingest from sources
# 2. Process to Silver
# 3. Engineer features to Gold
# 4. Retrain models (daily)
```

## Machine Learning

### How Spark MLlib Detects Fake News

The fake news detection process in our system works through a combination of feature engineering, pattern recognition, and machine learning algorithms. Here's a detailed explanation:

#### 1. **Feature Extraction Process**

The system analyzes multiple characteristics of news articles that research has shown to correlate with fake news:

```python
features = {
    # Content features
    'word_count': len(text.split()),           # Fake news often shorter
    'char_count': len(text),                   # Character count
    'avg_word_length': char_count / word_count, # Complexity indicator
    
    # Style features (fake news tends to be sensational)
    'exclamation_count': text.count('!'),      # Emotional emphasis
    'question_count': text.count('?'),         # Clickbait questions
    'caps_ratio': sum(1 for c in text if c.isupper()) / len(text),  # SHOUTING
    
    # Credibility features
    'has_author': 1 if author else 0,         # Missing authors suspicious
    'url_count': len(re.findall(r'http[s]?://', text)),  # Source citations
    'numeric_count': count_numbers(text),      # Statistics usage
    
    # Text features (TF-IDF)
    'tfidf_vector': vectorizer.transform([text])  # Word patterns
}
```

#### 2. **Why These Features Work**

Research has identified common patterns in fake news:

- **Sensationalism**: Excessive use of exclamation marks (!!! BREAKING!!!)
- **Emotional Language**: High caps ratio, emotional word choices
- **Lack of Sources**: Few or no URLs, missing author attribution
- **Clickbait Style**: Questions in titles, short content
- **Specific Vocabulary**: Certain words appear more frequently in fake news

#### 3. **The ML Pipeline**

Our system uses three complementary algorithms:

**Random Forest Classifier:**
```python
# Creates 100 decision trees that vote on the outcome
RandomForestClassifier(numTrees=100, maxDepth=10)

# Example decision path:
# IF exclamation_count > 3 AND caps_ratio > 0.2 → 80% fake
# IF has_author = 0 AND url_count < 2 → 75% fake
```

**Logistic Regression:**
```python
# Learns optimal weights for each feature
LogisticRegression(maxIter=100)

# Example learned weights:
# score = 0.8 * exclamation_count + 0.6 * caps_ratio - 0.5 * has_author ...
# IF score > threshold → classify as fake
```

**Gradient Boosting Trees:**
```python
# Builds trees sequentially, each correcting previous errors
GBTClassifier(maxIter=50)

# Captures complex patterns like:
# "High caps + many questions + no author = very likely fake"
```

#### 4. **Real Example Walkthrough**

Let's trace how the system analyzes a suspicious article:

```python
# Input article
title = "SHOCKING!!! You Won't BELIEVE What Scientists Discovered!!!"
content = "Scientists HATE this one simple trick..."

# Step 1: Feature extraction
features = {
    'exclamation_count': 6,      # Very high
    'caps_ratio': 0.35,          # 35% capitals (extremely high)
    'question_count': 0,
    'word_count': 8,             # Very short
    'has_author': 0,             # No author
    'url_count': 0,              # No sources
    'avg_word_length': 5.2
}

# Step 2: Each model makes prediction
random_forest_prediction = {
    'prediction': 'fake',
    'confidence': 0.95,  # 95% sure it's fake
    'reason': 'Multiple red flags: excessive punctuation, caps, no sources'
}

logistic_regression_prediction = {
    'prediction': 'fake',
    'confidence': 0.92,
    'reason': 'High weights on exclamation and caps features'
}

gbt_prediction = {
    'prediction': 'fake', 
    'confidence': 0.97,
    'reason': 'Pattern matches known clickbait structure'
}

# Step 3: Ensemble decision (majority voting)
final_prediction = 'FAKE' with 94.7% average confidence
```

#### 5. **TF-IDF Text Analysis**

Beyond simple features, the system analyzes word patterns:

```python
# TF-IDF identifies important words
# Words common in fake news get high scores:
fake_news_indicators = [
    'shocking', 'unbelievable', 'hate', 'simple trick',
    'won\'t believe', 'doctors hate', 'banned'
]

# Words common in real news:
real_news_indicators = [
    'according to', 'research shows', 'published in',
    'university', 'peer-reviewed', 'methodology'
]
```

#### 6. **Continuous Learning**

The model improves over time by:
- Retraining on newly labeled data
- Adjusting feature weights based on performance
- Adding new features as fake news tactics evolve

### Feature Engineering

The complete feature set extracted from each article:

```python
# All 15+ features used by the models
features = {
    # Content metrics
    'word_count': len(text.split()),
    'char_count': len(text),
    'avg_word_length': char_count / word_count,
    'title_length': len(title),
    
    # Style indicators
    'exclamation_count': text.count('!'),
    'question_count': text.count('?'),
    'caps_ratio': sum(1 for c in text if c.isupper()) / len(text),
    
    # Credibility markers
    'has_author': 1 if author else 0,
    'url_count': len(re.findall(r'http[s]?://', text)),
    'numeric_count': len(re.findall(r'\d+', text)),
    
    # Text features (TF-IDF)
    'tfidf_vector': vectorizer.transform([text])  # 1000-dimensional vector
}
```

### Model Training

```python
# Example training code showing the complete pipeline
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier

# Assemble features
assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="raw_features"
)

# Scale features for better performance
scaler = StandardScaler(
    inputCol="raw_features",
    outputCol="features"
)

# Create classifier
rf = RandomForestClassifier(
    numTrees=100,
    maxDepth=10,
    labelCol="label_index",
    featuresCol="features"
)

# Build pipeline
pipeline = Pipeline(stages=[assembler, scaler, rf])

# Train model on distributed data
model = pipeline.fit(training_data)
```
### Model Evaluation

```
Model Performance Metrics:
├── Random Forest
│   ├── Accuracy: 92.5%
│   ├── Precision: 91.3%
│   ├── Recall: 93.8%
│   └── F1-Score: 92.5%
├── Logistic Regression
│   ├── Accuracy: 89.7%
│   └── AUC: 0.945
└── Ensemble Model
    └── Accuracy: 94.2%
```

## API Reference

### POST /predict
Analyze a single article.

**Request:**
```json
{
  "title": "Breaking News: Major Event Happens",
  "content": "Article content here...",
  "author": "John Doe",
  "source": "example.com"
}
```

**Response:**
```json
{
  "prediction": "fake",
  "confidence": 0.875,
  "analyzed_at": "2024-10-20T15:30:00Z",
  "features": {
    "exclamation_count": 0,
    "caps_ratio": 0.05,
    "word_count": 250
  }
}
```

### POST /predict_batch
Analyze multiple articles.

**Request:**
```json
{
  "articles": [
    {"title": "...", "content": "..."},
    {"title": "...", "content": "..."}
  ]
}
```

### GET /health
Check API health status.

### GET /feature_importance
Get model feature importance scores.

## Monitoring

### Dashboard Features

1. **Real-time Metrics**
   - Predictions per minute/hour
   - Fake vs Real distribution
   - Average response time

2. **Data Lake Status**
   - Records in each layer
   - Data quality metrics
   - Storage utilization

3. **System Health**
   - Kafka lag monitoring
   - Spark job status
   - API response times

### Accessing Metrics

```bash
# Prometheus metrics
curl http://localhost:5000/metrics

# Grafana dashboards
http://localhost:3000

# Spark UI
http://localhost:8080
```

## Performance

### Benchmarks

| Metric | Value |
|--------|-------|
| Ingestion Throughput | 10,000 articles/min |
| API Response Time | < 100ms (cached) |
| Model Training Time | ~30 min (1M articles) |
| Prediction Accuracy | 94.2% |
| Data Lake Storage | 0.5 TB/month |

### Optimization Tips

1. **Caching**: Enable Redis caching for frequent predictions
2. **Partitioning**: Partition data by date for faster queries
3. **Model Serving**: Use model quantization for faster inference
4. **Resource Allocation**: Adjust Spark executor memory based on workload

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black src/
flake8 src/
```

### Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
