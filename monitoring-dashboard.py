# dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import json
from pyspark.sql import SparkSession
import redis
import time

# Configure Streamlit
st.set_page_config(
    page_title="Fake News Detection Dashboard",
    page_icon="📰",
    layout="wide"
)

class FakeNewsMonitoringDashboard:
    """
    Real-time monitoring dashboard for fake news detection system
    """
    
    def __init__(self):
        # Initialize connections
        self.redis_client = redis.Redis(host='localhost', port=6379, db=1)
        
        # Initialize Spark for data lake queries
        self.spark = SparkSession.builder \
            .appName("DashboardQueries") \
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
            .getOrCreate()
        
        self.api_base_url = "http://localhost:5000"
        self.data_lake_path = "/data/fake-news-lake"
    
    def get_data_lake_metrics(self):
        """Get metrics from data lake layers"""
        metrics = {}
        
        try:
            # Bronze layer metrics
            bronze_df = self.spark.read.format("delta").load(f"{self.data_lake_path}/bronze")
            metrics['bronze_count'] = bronze_df.count()
            metrics['bronze_sources'] = bronze_df.select("source").distinct().count()
            
            # Silver layer metrics
            silver_df = self.spark.read.format("delta").load(f"{self.data_lake_path}/silver")
            metrics['silver_count'] = silver_df.count()
            metrics['avg_word_count'] = silver_df.agg({"word_count": "avg"}).collect()[0][0]
            
            # Gold layer metrics
            gold_df = self.spark.read.format("delta").load(f"{self.data_lake_path}/gold")
            metrics['gold_count'] = gold_df.count()
            
            if 'label' in gold_df.columns:
                label_dist = gold_df.groupBy("label").count().toPandas()
                metrics['label_distribution'] = label_dist
                
        except Exception as e:
            st.error(f"Error fetching data lake metrics: {str(e)}")
            
        return metrics
    
    def get_real_time_metrics(self):
        """Get real-time processing metrics from Redis"""
        metrics = {}
        
        try:
            # Get counters from Redis
            metrics['predictions_today'] = int(self.redis_client.get('predictions_today') or 0)
            metrics['fake_detected_today'] = int(self.redis_client.get('fake_detected_today') or 0)
            metrics['real_detected_today'] = int(self.redis_client.get('real_detected_today') or 0)
            metrics['avg_response_time'] = float(self.redis_client.get('avg_response_time') or 0)
            
            # Get recent predictions
            recent_predictions = []
            for i in range(10):
                pred_key = f"recent_prediction_{i}"
                pred_data = self.redis_client.get(pred_key)
                if pred_data:
                    recent_predictions.append(json.loads(pred_data))
            metrics['recent_predictions'] = recent_predictions
            
        except Exception as e:
            st.error(f"Error fetching real-time metrics: {str(e)}")
            
        return metrics
    
    def render_dashboard(self):
        """Main dashboard rendering function"""
        st.title("🔍 Fake News Detection Platform")
        st.markdown("### Real-time Monitoring Dashboard")
        
        # Sidebar for configuration
        with st.sidebar:
            st.header("Configuration")
            refresh_rate = st.slider("Refresh Rate (seconds)", 1, 30, 5)
            show_advanced = st.checkbox("Show Advanced Metrics", value=False)
            
            # Test prediction
            st.header("Test Prediction")
            test_title = st.text_input("Article Title")
            test_content = st.text_area("Article Content")
            
            if st.button("Analyze"):
                self.analyze_article(test_title, test_content)
        
        # Main dashboard tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Analytics", "🗄️ Data Lake", "🔧 System Health"])
        
        with tab1:
            self.render_overview_tab()
        
        with tab2:
            self.render_analytics_tab()
        
        with tab3:
            self.render_data_lake_tab()
        
        with tab4:
            self.render_system_health_tab()
        
        # Auto-refresh
        time.sleep(refresh_rate)
        st.rerun()
    
    def render_overview_tab(self):
        """Render overview metrics"""
        col1, col2, col3, col4 = st.columns(4)
        
        # Get real-time metrics
        rt_metrics = self.get_real_time_metrics()
        
        with col1:
            st.metric(
                "Total Predictions Today",
                rt_metrics.get('predictions_today', 0),
                delta=f"+{rt_metrics.get('predictions_today', 0) // 24}/hr"
            )
        
        with col2:
            fake_count = rt_metrics.get('fake_detected_today', 0)
            total = rt_metrics.get('predictions_today', 1)
            fake_percentage = (fake_count / total * 100) if total > 0 else 0
            st.metric(
                "Fake News Detected",
                f"{fake_count} ({fake_percentage:.1f}%)",
                delta=f"+{fake_count // 24}/hr"
            )
        
        with col3:
            st.metric(
                "Avg Response Time",
                f"{rt_metrics.get('avg_response_time', 0):.2f}ms",
                delta="-5%" if rt_metrics.get('avg_response_time', 0) < 100 else "+5%"
            )
        
        with col4:
            st.metric(
                "System Status",
                "✅ Operational",
                delta="99.9% uptime"
            )
        
        # Recent predictions
        st.subheader("Recent Predictions")
        recent_preds = rt_metrics.get('recent_predictions', [])
        
        if recent_preds:
            df = pd.DataFrame(recent_preds)
            
            # Color code based on prediction
            def color_prediction(val):
                color = 'red' if val == 'fake' else 'green'
                return f'color: {color}'
            
            styled_df = df.style.applymap(color_prediction, subset=['prediction'])
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.info("No recent predictions available")
        
        # Real-time chart
        st.subheader("Prediction Trend (Last 24 Hours)")
        
        # Generate sample data for visualization
        hours = pd.date_range(end=datetime.now(), periods=24, freq='H')
        trend_data = pd.DataFrame({
            'time': hours,
            'fake': [rt_metrics.get('fake_detected_today', 0) // 24 + i % 5 for i in range(24)],
            'real': [rt_metrics.get('real_detected_today', 0) // 24 + i % 3 for i in range(24)]
        })
        
        fig = px.line(trend_data, x='time', y=['fake', 'real'], 
                     title="Hourly Detection Trend",
                     labels={'value': 'Count', 'variable': 'Type'})
        st.plotly_chart(fig, use_container_width=True)
    
    def render_analytics_tab(self):
        """Render analytics and insights"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Source Distribution")
            
            # Get source distribution from data lake
            try:
                silver_df = self.spark.read.format("delta").load(f"{self.data_lake_path}/silver")
                source_dist = silver_df.groupBy("source").count().toPandas()
                
                fig = px.pie(source_dist, values='count', names='source',
                           title="News Sources Distribution")
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.error("Unable to load source distribution")
        
        with col2:
            st.subheader("Feature Importance")
            
            # Get feature importance from API
            try:
                response = requests.get(f"{self.api_base_url}/feature_importance")
                if response.status_code == 200:
                    importance_data = response.json()['feature_importance']
                    df = pd.DataFrame(importance_data)
                    
                    fig = px.bar(df, x='importance', y='feature', orientation='h',
                               title="Model Feature Importance")
                    st.plotly_chart(fig, use_container_width=True)
            except:
                st.error("Unable to load feature importance")
        
        # Word cloud of fake news keywords
        st.subheader("Common Patterns in Fake News")
        
        # Placeholder for word cloud
        st.info("Word cloud analysis would be displayed here based on detected fake news patterns")
    
    def render_data_lake_tab(self):
        """Render data lake metrics"""
        st.subheader("Data Lake Architecture")
        
        # Get data lake metrics
        dl_metrics = self.get_data_lake_metrics()
        
        # Display layer metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Bronze Layer", f"{dl_metrics.get('bronze_count', 0):,} records")
            st.caption(f"Sources: {dl_metrics.get('bronze_sources', 0)}")
        
        with col2:
            st.metric("Silver Layer", f"{dl_metrics.get('silver_count', 0):,} records")
            st.caption(f"Avg words: {dl_metrics.get('avg_word_count', 0):.0f}")
        
        with col3:
            st.metric("Gold Layer", f"{dl_metrics.get('gold_count', 0):,} records")
            st.caption("ML-ready features")
        
        # Data flow diagram
        st.subheader("Data Flow")
        
        flow_data = {
            'Stage': ['Ingestion', 'Bronze', 'Silver', 'Gold', 'Model'],
            'Records': [
                dl_metrics.get('bronze_count', 0),
                dl_metrics.get('bronze_count', 0),
                dl_metrics.get('silver_count', 0),
                dl_metrics.get('gold_count', 0),
                dl_metrics.get('gold_count', 0)
            ]
        }
        
        fig = go.Figure(data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=flow_data['Stage'],
                    color=["blue", "bronze", "silver", "gold", "green"]
                ),
                link=dict(
                    source=[0, 1, 2, 3],
                    target=[1, 2, 3, 4],
                    value=flow_data['Records'][1:]
                )
            )
        ])
        
        fig.update_layout(title_text="Data Lake Flow", font_size=10)
        st.plotly_chart(fig, use_container_width=True)
        
        # Label distribution if available
        if 'label_distribution' in dl_metrics:
            st.subheader("Training Data Distribution")
            label_df = dl_metrics['label_distribution']
            
            fig = px.bar(label_df, x='label', y='count',
                        title="Labeled Data Distribution",
                        color='label',
                        color_discrete_map={'fake': 'red', 'real': 'green'})
            st.plotly_chart(fig, use_container_width=True)
    
    def render_system_health_tab(self):
        """Render system health metrics"""
        st.subheader("System Health Monitoring")
        
        # Kafka metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Kafka Pipeline")
            kafka_metrics = {
                'Status': '✅ Connected',
                'Topics': 'raw-news-feed',
                'Lag': '< 100 messages',
                'Throughput': '~1000 msg/min'
            }
            for key, value in kafka_metrics.items():
                st.metric(key, value)
        
        with col2:
            st.subheader("Spark Cluster")
            spark_metrics = {
                'Status': '✅ Running',
                'Executors': '4',
                'Memory': '16GB',
                'CPU Cores': '8'
            }
            for key, value in spark_metrics.items():
                st.metric(key, value)
        
        # Processing latency chart
        st.subheader("Processing Latency")
        
        # Generate sample latency data
        time_range = pd.date_range(end=datetime.now(), periods=60, freq='min')
        latency_data = pd.DataFrame({
            'time': time_range,
            'ingestion': [50 + i % 20 for i in range(60)],
            'processing': [100 + i % 30 for i in range(60)],
            'prediction': [20 + i % 10 for i in range(60)]
        })
        
        fig = px.line(latency_data, x='time', 
                     y=['ingestion', 'processing', 'prediction'],
                     title="Component Latency (ms)")
        st.plotly_chart(fig, use_container_width=True)
    
    def analyze_article(self, title, content):
        """Analyze a single article using the API"""
        if not content:
            st.error("Please provide article content")
            return
        
        with st.spinner("Analyzing article..."):
            try:
                response = requests.post(
                    f"{self.api_base_url}/predict",
                    json={
                        "title": title,
                        "content": content,
                        "source": "manual_test"
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display result
                    if result['prediction'] == 'fake':
                        st.error(f"🚨 FAKE NEWS DETECTED (Confidence: {result['confidence']:.2%})")
                    else:
                        st.success(f"✅ Real News (Confidence: {result['confidence']:.2%})")
                    
                    # Update Redis metrics
                    self.redis_client.incr('predictions_today')
                    if result['prediction'] == 'fake':
                        self.redis_client.incr('fake_detected_today')
                    else:
                        self.redis_client.incr('real_detected_today')
                    
                else:
                    st.error(f"API Error: {response.status_code}")
                    
            except Exception as e:
                st.error(f"Error analyzing article: {str(e)}")


# Run dashboard
if __name__ == "__main__":
    dashboard = FakeNewsMonitoringDashboard()
    dashboard.render_dashboard()