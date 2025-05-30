# ml_pipeline.py
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.functions import col, when, udf
from pyspark.sql.types import FloatType
import mlflow
import mlflow.spark
from datetime import datetime
import json
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FakeNewsMLPipeline:
    """
    Distributed ML pipeline for fake news detection using Spark MLlib
    """
    
    def __init__(self, spark_session: SparkSession):
        self.spark = spark_session
        self.models = {}
        self.best_model = None
        self.pipeline = None
        
        # Initialize MLflow
        mlflow.set_experiment("fake-news-detection")
    
    def prepare_features(self, df):
        """Prepare features for ML training"""
        # Convert label to numeric
        label_indexer = StringIndexer(
            inputCol="label",
            outputCol="label_index",
            handleInvalid="keep"
        )
        
        # Assemble numeric features
        numeric_features = [
            "word_count", "char_count", "has_author",
            "title_length", "exclamation_count", "question_count",
            "caps_ratio", "avg_word_length", "numeric_count", "url_count"
        ]
        
        # Vector assembler for numeric features
        numeric_assembler = VectorAssembler(
            inputCols=numeric_features,
            outputCol="numeric_features"
        )
        
        # Combine with TF-IDF features
        feature_assembler = VectorAssembler(
            inputCols=["numeric_features", "tfidf_features"],
            outputCol="raw_features"
        )
        
        # Scale features
        scaler = StandardScaler(
            inputCol="raw_features",
            outputCol="features",
            withStd=True,
            withMean=False
        )
        
        # Create preprocessing pipeline
        preprocessing = Pipeline(stages=[
            label_indexer,
            numeric_assembler,
            feature_assembler,
            scaler
        ])
        
        return preprocessing
    
    def train_models(self, train_df, test_df):
        """Train multiple models and select the best one"""
        preprocessing = self.prepare_features(train_df)
        
        # Define models to train
        models = {
            "random_forest": RandomForestClassifier(
                labelCol="label_index",
                featuresCol="features",
                numTrees=100,
                maxDepth=10
            ),
            "logistic_regression": LogisticRegression(
                labelCol="label_index",
                featuresCol="features",
                maxIter=100
            ),
            "gradient_boosting": GBTClassifier(
                labelCol="label_index",
                featuresCol="features",
                maxIter=50
            )
        }
        
        evaluator = BinaryClassificationEvaluator(labelCol="label_index")
        
        best_score = 0
        best_model_name = None
        
        for model_name, model in models.items():
            with mlflow.start_run(run_name=f"train_{model_name}"):
                logger.info(f"Training {model_name}...")
                
                # Create pipeline
                pipeline = Pipeline(stages=preprocessing.getStages() + [model])
                
                # Cross validation
                paramGrid = ParamGridBuilder().build()
                
                crossval = CrossValidator(
                    estimator=pipeline,
                    estimatorParamMaps=paramGrid,
                    evaluator=evaluator,
                    numFolds=3
                )
                
                # Train model
                cv_model = crossval.fit(train_df)
                
                # Evaluate on test set
                predictions = cv_model.transform(test_df)
                auc = evaluator.evaluate(predictions)
                
                # Calculate additional metrics
                accuracy_eval = MulticlassClassificationEvaluator(
                    labelCol="label_index",
                    predictionCol="prediction",
                    metricName="accuracy"
                )
                accuracy = accuracy_eval.evaluate(predictions)
                
                # Log metrics
                mlflow.log_metric("auc", auc)
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_param("model_type", model_name)
                
                # Log model
                mlflow.spark.log_model(cv_model, f"model_{model_name}")
                
                logger.info(f"{model_name} - AUC: {auc:.4f}, Accuracy: {accuracy:.4f}")
                
                # Track best model
                if auc > best_score:
                    best_score = auc
                    best_model_name = model_name
                    self.best_model = cv_model
                
                self.models[model_name] = {
                    "model": cv_model,
                    "auc": auc,
                    "accuracy": accuracy
                }
        
        logger.info(f"Best model: {best_model_name} with AUC: {best_score:.4f}")
        return self.best_model
    
    def create_ensemble_model(self, train_df, test_df):
        """Create an ensemble of multiple models"""
        preprocessing = self.prepare_features(train_df)
        
        # Train base models
        rf = RandomForestClassifier(numTrees=50, maxDepth=8)
        lr = LogisticRegression(maxIter=50)
        gbt = GBTClassifier(maxIter=30)
        
        # Create pipelines for each model
        pipelines = []
        for idx, model in enumerate([rf, lr, gbt]):
            pipeline = Pipeline(stages=preprocessing.getStages() + [model])
            pipelines.append(pipeline.fit(train_df))
        
        # Create ensemble predictions
        def ensemble_predict(models, test_data):
            predictions = []
            for model in models:
                pred_df = model.transform(test_data)
                predictions.append(pred_df.select("prediction").toPandas()["prediction"].values)
            
            # Majority voting
            import numpy as np
            ensemble_preds = np.round(np.mean(predictions, axis=0))
            
            # Convert back to DataFrame
            pred_df = test_data.select("label_index").toPandas()
            pred_df["ensemble_prediction"] = ensemble_preds
            
            return self.spark.createDataFrame(pred_df)
        
        # Evaluate ensemble
        ensemble_predictions = ensemble_predict(pipelines, test_df)
        
        # Calculate ensemble metrics
        from pyspark.sql import functions as F
        accuracy = ensemble_predictions.filter(
            col("label_index") == col("ensemble_prediction")
        ).count() / ensemble_predictions.count()
        
        logger.info(f"Ensemble accuracy: {accuracy:.4f}")
        
        return pipelines
    
    def analyze_feature_importance(self):
        """Analyze feature importance from the best model"""
        if not self.best_model:
            logger.error("No model trained yet")
            return None
        
        # Extract the model from pipeline
        stages = self.best_model.bestModel.stages
        model_stage = stages[-1]
        
        if hasattr(model_stage, 'featureImportances'):
            importances = model_stage.featureImportances.toArray()
            
            # Get feature names
            feature_names = [
                "word_count", "char_count", "has_author",
                "title_length", "exclamation_count", "question_count",
                "caps_ratio", "avg_word_length", "numeric_count", "url_count"
            ]
            
            # Create importance dataframe
            importance_df = self.spark.createDataFrame(
                [(name, float(imp)) for name, imp in zip(feature_names, importances[:len(feature_names)])],
                ["feature", "importance"]
            ).orderBy(col("importance").desc())
            
            return importance_df
        
        return None
    
    def predict_single(self, text_data: dict):
        """Predict fake news for a single article"""
        if not self.best_model:
            logger.error("No model trained yet")
            return None
        
        # Create DataFrame from input
        df = self.spark.createDataFrame([text_data])
        
        # Apply the model
        prediction = self.best_model.transform(df)
        
        # Extract prediction and probability
        result = prediction.select("prediction", "probability").collect()[0]
        
        return {
            "prediction": "fake" if result["prediction"] == 1 else "real",
            "confidence": float(result["probability"][int(result["prediction"])])
        }
    
    def save_model(self, path: str):
        """Save the best model to disk"""
        if self.best_model:
            self.best_model.save(path)
            
            # Save model metadata
            metadata = {
                "model_type": "ensemble" if len(self.models) > 1 else "single",
                "models": list(self.models.keys()),
                "best_model_metrics": self.models,
                "saved_date": datetime.now().isoformat()
            }
            
            with open(f"{path}/metadata.json", "w") as f:
                json.dump(metadata, f)
            
            logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a saved model"""
        from pyspark.ml.pipeline import PipelineModel
        self.best_model = PipelineModel.load(path)
        logger.info(f"Model loaded from {path}")


# Model serving API
from flask import Flask, request, jsonify
from pyspark.sql import SparkSession
import redis
import hashlib

class ModelServingAPI:
    """
    REST API for serving fake news predictions
    """
    
    def __init__(self, model_path: str):
        self.app = Flask(__name__)
        
        # Initialize Spark session
        self.spark = SparkSession.builder \
            .appName("FakeNewsServing") \
            .config("spark.sql.shuffle.partitions", "8") \
            .getOrCreate()
        
        # Load model
        self.ml_pipeline = FakeNewsMLPipeline(self.spark)
        self.ml_pipeline.load_model(model_path)
        
        # Initialize Redis cache
        self.cache = redis.Redis(host='localhost', port=6379, db=0)
        self.cache_ttl = 3600  # 1 hour
        
        # Define routes
        self._define_routes()
    
    def _define_routes(self):
        @self.app.route('/health', methods=['GET'])
        def health():
            return jsonify({"status": "healthy", "model": "loaded"})
        
        @self.app.route('/predict', methods=['POST'])
        def predict():
            try:
                data = request.json
                
                # Validate input
                if not data or 'content' not in data:
                    return jsonify({"error": "Missing 'content' field"}), 400
                
                # Check cache
                cache_key = hashlib.md5(data['content'].encode()).hexdigest()
                cached_result = self.cache.get(cache_key)
                
                if cached_result:
                    return json.loads(cached_result)
                
                # Prepare input data
                input_data = {
                    "title": data.get('title', ''),
                    "content": data['content'],
                    "url": data.get('url', ''),
                    "author": data.get('author', ''),
                    "source": data.get('source', 'api')
                }
                
                # Feature engineering (simplified version)
                input_data.update({
                    "word_count": len(input_data['content'].split()),
                    "char_count": len(input_data['content']),
                    "has_author": 1 if input_data['author'] else 0,
                    "title_length": len(input_data['title']),
                    "exclamation_count": input_data['content'].count('!'),
                    "question_count": input_data['content'].count('?'),
                    "caps_ratio": sum(1 for c in input_data['content'] if c.isupper()) / len(input_data['content']),
                    "avg_word_length": len(input_data['content'].replace(' ', '')) / max(len(input_data['content'].split()), 1),
                    "numeric_count": sum(1 for c in input_data['content'] if c.isdigit()),
                    "url_count": input_data['content'].lower().count('http')
                })
                
                # Get prediction
                result = self.ml_pipeline.predict_single(input_data)
                
                # Add additional context
                result['analyzed_at'] = datetime.now().isoformat()
                result['content_preview'] = input_data['content'][:200] + '...'
                
                # Cache result
                self.cache.setex(cache_key, self.cache_ttl, json.dumps(result))
                
                return jsonify(result)
                
            except Exception as e:
                logger.error(f"Prediction error: {str(e)}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/predict_batch', methods=['POST'])
        def predict_batch():
            try:
                data = request.json
                
                if not data or 'articles' not in data:
                    return jsonify({"error": "Missing 'articles' field"}), 400
                
                results = []
                for article in data['articles']:
                    # Process each article
                    result = predict_single_article(article)
                    results.append(result)
                
                return jsonify({"predictions": results})
                
            except Exception as e:
                logger.error(f"Batch prediction error: {str(e)}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/feature_importance', methods=['GET'])
        def feature_importance():
            try:
                importance_df = self.ml_pipeline.analyze_feature_importance()
                if importance_df:
                    importance_data = importance_df.toPandas().to_dict('records')
                    return jsonify({"feature_importance": importance_data})
                else:
                    return jsonify({"error": "Feature importance not available"}), 404
                    
            except Exception as e:
                logger.error(f"Feature importance error: {str(e)}")
                return jsonify({"error": str(e)}), 500
    
    def run(self, host='0.0.0.0', port=5000):
        self.app.run(host=host, port=port, debug=False)


# Training script
if __name__ == "__main__":
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("FakeNewsTraining") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()
    
    # Initialize ML pipeline
    ml_pipeline = FakeNewsMLPipeline(spark)
    
    # Load data from data lake
    train_df = spark.read.format("delta").load("/data/fake-news-lake/gold/train")
    test_df = spark.read.format("delta").load("/data/fake-news-lake/gold/test")
    
    # Train models
    best_model = ml_pipeline.train_models(train_df, test_df)
    
    # Analyze feature importance
    importance_df = ml_pipeline.analyze_feature_importance()
    if importance_df:
        importance_df.show()
    
    # Save model
    ml_pipeline.save_model("/models/fake-news-detector")
    
    # Start serving API
    api = ModelServingAPI("/models/fake-news-detector")
    api.run()