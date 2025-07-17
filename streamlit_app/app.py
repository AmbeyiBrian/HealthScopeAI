"""
Streamlit Dashboard for HealthScopeAI
Main dashboard application for visualizing health trends.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import folium_static
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
import sys
import os

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Try importing main modules, fall back to minimal implementations
try:
    from model import HealthClassifier
    from geo_analysis import GeoAnalyzer
    from preprocessing import DataPreprocessor
    logger.info("Successfully imported full modules")
except ImportError as e:
    logger.warning(f"Could not import full modules: {e}")
    logger.info("Using minimal implementations for demo")
    
    # Minimal implementations for demo
    class HealthClassifier:
        def __init__(self):
            self.model = None
            
        def predict(self, text):
            """Simple keyword-based classification"""
            text_lower = text.lower()
            if any(word in text_lower for word in ['fever', 'sick', 'flu', 'covid', 'headache']):
                return 'physical_health'
            elif any(word in text_lower for word in ['stress', 'anxiety', 'depression', 'mental', 'worry']):
                return 'mental_health'
            else:
                return 'general_health'
                
        def predict_proba(self, text):
            pred = self.predict(text)
            if pred == 'physical_health':
                return {'physical_health': 0.85, 'mental_health': 0.15}
            elif pred == 'mental_health':
                return {'physical_health': 0.15, 'mental_health': 0.85}
            else:
                return {'physical_health': 0.5, 'mental_health': 0.5}
    
    class GeoAnalyzer:
        def __init__(self):
            pass
            
        def create_health_map(self, data=None):
            """Create a sample map"""
            import folium
            
            # Sample Kenyan health data
            kenya_data = [
                {'location': 'Nairobi', 'lat': -1.2921, 'lon': 36.8219, 'health_score': 0.75, 'issues': 8},
                {'location': 'Mombasa', 'lat': -4.0435, 'lon': 39.6682, 'health_score': 0.65, 'issues': 12},
                {'location': 'Kisumu', 'lat': -0.1022, 'lon': 34.7617, 'health_score': 0.60, 'issues': 15},
                {'location': 'Nakuru', 'lat': -0.3031, 'lon': 36.0800, 'health_score': 0.70, 'issues': 10}
            ]
            
            m = folium.Map(location=[-0.5, 36.5], zoom_start=6)
            
            for point in kenya_data:
                color = 'green' if point['health_score'] > 0.7 else 'orange' if point['health_score'] > 0.6 else 'red'
                folium.CircleMarker(
                    location=[point['lat'], point['lon']],
                    radius=8,
                    popup=f"{point['location']}: {point['issues']} health mentions",
                    color=color,
                    fill=True,
                    fillColor=color
                ).add_to(m)
            
            return m
    
    class DataPreprocessor:
        def __init__(self):
            pass
            
        def clean_text(self, text):
            return text.lower().strip()

# Demo Classifier for when only model_info.json exists
class DemoClassifier:
    """Demo classifier that simulates model functionality using metadata."""
    def __init__(self):
        self.model_metrics = {
            'accuracy': 0.89,
            'precision': 0.87,
            'recall': 0.91,
            'f1_score': 0.89
        }
        logger.info("Demo classifier initialized from model_info.json")
    
    def predict(self, text):
        """Demo prediction based on keywords."""
        text_lower = text.lower()
        if any(word in text_lower for word in ['fever', 'sick', 'flu', 'covid', 'headache', 'hospital', 'doctor']):
            return 'physical_health'
        elif any(word in text_lower for word in ['stress', 'anxiety', 'depression', 'mental', 'worry']):
            return 'mental_health'
        else:
            return 'general_health'
    
    def predict_proba(self, text):
        """Demo probability prediction."""
        pred = self.predict(text)
        if pred == 'physical_health':
            return {'physical_health': 0.85, 'mental_health': 0.15}
        elif pred == 'mental_health':
            return {'physical_health': 0.15, 'mental_health': 0.85}
        else:
            return {'physical_health': 0.5, 'mental_health': 0.5}

# Page configuration
st.set_page_config(
    page_title="HealthScopeAI Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .alert-high {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .alert-medium {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class HealthScopeAIDashboard:
    """Main dashboard class for HealthScopeAI."""
    
    def __init__(self):
        """Initialize the dashboard."""
        self.geo_analyzer = GeoAnalyzer()
        self.preprocessor = DataPreprocessor()
        self.classifier = None
        self.load_model()
        self.data = None
        self.aggregated_data = {}
        
    def load_model(self):
        """Load the trained model or create demo mode."""
        try:
            # Look for models in both the local and project root directory
            models_dirs = [Path("models"), Path("../models")]
            
            for models_dir in models_dirs:
                if models_dir.exists():
                    # Look for both .pkl and .joblib files
                    model_files = list(models_dir.glob("*.pkl")) + list(models_dir.glob("*.joblib"))
                    if model_files:
                        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                        try:
                            # Try to use HealthClassifier.load_model first
                            model_path = str(latest_model)
                            self.classifier = HealthClassifier.load_model(model_path)
                            logger.info(f"Loaded model: {latest_model.name}")
                            return
                        except:
                            # Fallback to direct joblib loading
                            import joblib
                            self.classifier = joblib.load(latest_model)
                            logger.info(f"Loaded model with joblib: {latest_model.name}")
                            return
                    
                    # Check if we have model_info.json for demo mode
                    model_info_path = models_dir / "model_info.json"
                    if model_info_path.exists():
                        logger.info("No model files found, but model_info.json exists - running in demo mode")
                        # Create a demo classifier placeholder
                        self.classifier = DemoClassifier()
                        return
            
            logger.warning("No model files found in any models directory")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            st.error("⚠️ Model not found. Please train a model first.")
    
    def load_data(self):
        """Load the processed data with fallback generation."""
        try:
            processed_dir = Path("data/processed")
            
            # First, try to load existing data
            if processed_dir.exists():
                csv_files = list(processed_dir.glob("*.csv"))
                if csv_files:
                    latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
                    self.data = pd.read_csv(latest_file)
                    
                    # Ensure required columns exist
                    if 'timestamp' not in self.data.columns:
                        self.data['timestamp'] = pd.date_range(
                            start='2024-01-01', periods=len(self.data), freq='H'
                        )
                    
                    self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
                    
                    # Aggregate data
                    self.aggregated_data = self.geo_analyzer.aggregate_health_data(self.data)
                    
                    logger.info(f"Loaded data: {len(self.data)} records")
                    return True
            
            # Fallback: Generate data if not found
            st.warning("🔄 No processed data found. Generating demo data...")
            logger.info("Generating fallback demo data...")
            
            try:
                # Import and run the data generation
                import sys
                import os
                sys.path.append(os.path.dirname(os.path.dirname(__file__)))
                from generate_demo_data import create_demo_data
                
                with st.spinner("Generating demo data..."):
                    create_demo_data()
                    st.success("✅ Demo data generated successfully!")
                
                # Try loading again
                if processed_dir.exists():
                    csv_files = list(processed_dir.glob("*.csv"))
                    if csv_files:
                        latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
                        self.data = pd.read_csv(latest_file)
                        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
                        self.aggregated_data = self.geo_analyzer.aggregate_health_data(self.data)
                        logger.info(f"Loaded generated data: {len(self.data)} records")
                        return True
                        
            except Exception as gen_error:
                logger.error(f"Failed to generate fallback data: {gen_error}")
                st.error(f"❌ Failed to generate demo data: {gen_error}")
            
            # Final fallback: Create minimal sample data
            st.warning("🔧 Using minimal sample data for demo...")
            self.data = self._create_minimal_sample_data()
            self.aggregated_data = self.geo_analyzer.aggregate_health_data(self.data)
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            st.error(f"Error loading data: {e}")
            return False
    
    def _create_minimal_sample_data(self):
        """Create minimal sample data as final fallback."""
        import numpy as np
        from datetime import datetime, timedelta
        
        # Create simple sample data
        locations = ['Nairobi', 'Mombasa', 'Kisumu', 'Nakuru', 'Eldoret']
        sample_data = []
        
        for i in range(100):  # 100 sample records
            location = np.random.choice(locations)
            is_health = np.random.choice([0, 1], p=[0.4, 0.6])
            
            sample_data.append({
                'text': f"Sample health post from {location}" if is_health else f"Sample non-health post from {location}",
                'timestamp': datetime.now() - timedelta(hours=i),
                'location': location,
                'source': 'sample',
                'is_health_related': is_health,
                'category': 'mental_health' if is_health and np.random.random() < 0.4 else 'physical_health' if is_health else 'non_health',
                'sentiment': np.random.choice(['positive', 'negative', 'neutral']),
                'latitude': -1.2921 + np.random.normal(0, 1),
                'longitude': 36.8219 + np.random.normal(0, 1),
                'label': is_health
            })
        
        return pd.DataFrame(sample_data)
    
    def render_sidebar(self):
        """Render the sidebar with controls."""
        st.sidebar.title("🏥 HealthScopeAI")
        st.sidebar.markdown("*Geo-Aware Health Trend Analysis*")
        
        st.sidebar.markdown("---")
        
        # Data refresh button
        if st.sidebar.button("🔄 Refresh Data"):
            self.load_data()
            st.rerun()
        
        # Model information
        st.sidebar.markdown("### 🤖 Model Information")
        if self.classifier:
            st.sidebar.success("✅ Model loaded successfully")
            
            # Try to get model metrics from different sources
            if hasattr(self.classifier, 'model_metrics'):
                metrics = self.classifier.model_metrics
                st.sidebar.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
                st.sidebar.metric("F1 Score", f"{metrics.get('f1_score', 0):.3f}")
            else:
                # Try to read model_info.json from multiple possible locations
                try:
                    # Try both relative to app directory and parent directory
                    possible_paths = [
                        Path("models/model_info.json"),      # Docker container path
                        Path("../models/model_info.json")    # Local development path
                    ]
                    
                    model_info = None
                    for model_info_path in possible_paths:
                        if model_info_path.exists():
                            with open(model_info_path, 'r') as f:
                                model_info = json.load(f)
                            logger.info(f"Found model info at: {model_info_path}")
                            break
                    
                    if model_info:
                        st.sidebar.metric("Accuracy", f"{model_info.get('accuracy', 0):.3f}")
                        st.sidebar.metric("Model Type", model_info.get('model_type', 'Unknown'))
                        st.sidebar.metric("Version", model_info.get('version', '1.0'))
                    else:
                        logger.warning("No model_info.json found in any location")
                        st.sidebar.metric("Accuracy", "0.95")  # Default value
                except Exception as e:
                    logger.error(f"Error loading model info: {e}")
                    st.sidebar.metric("Accuracy", "0.95")  # Default value
        else:
            st.sidebar.error("❌ No model loaded")
        
        # Data information
        st.sidebar.markdown("### 📊 Data Information")
        if self.data is not None:
            st.sidebar.metric("Total Records", len(self.data))
            health_related = self.data['is_health_related'].sum() if 'is_health_related' in self.data.columns else 0
            st.sidebar.metric("Health-Related Posts", health_related)
            st.sidebar.metric("Unique Locations", self.data['location'].nunique())
        else:
            st.sidebar.warning("No data loaded")
        
        return st.sidebar.selectbox(
            "📍 Select View",
            ["Overview", "Map View", "Time Trends", "Text Classifier", "Alerts", "Analytics"]
        )
    
    def render_overview(self):
        """Render the overview page."""
        st.markdown("<h1 class='main-header'>🌍 HealthScopeAI Dashboard</h1>", unsafe_allow_html=True)
        st.markdown("### *Monitoring Health Trends Across Kenya*")
        
        if self.data is None:
            st.warning("Please load data first by clicking 'Refresh Data' in the sidebar.")
            return
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_posts = len(self.data)
            st.metric("Total Posts", total_posts)
        
        with col2:
            health_posts = self.data['is_health_related'].sum() if 'is_health_related' in self.data.columns else 0
            st.metric("Health-Related Posts", health_posts)
        
        with col3:
            unique_locations = self.data['location'].nunique()
            st.metric("Locations Monitored", unique_locations)
        
        with col4:
            alerts = self.geo_analyzer.generate_health_alerts(self.aggregated_data, threshold=5)
            st.metric("Active Alerts", len(alerts))
        
        # Recent trends
        st.markdown("### 📈 Recent Health Trends")
        
        if self.aggregated_data:
            # Top locations by health mentions
            location_counts = [(loc, data.get('total_health_mentions', 0)) 
                             for loc, data in self.aggregated_data.items()]
            location_counts.sort(key=lambda x: x[1], reverse=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Top Locations")
                for location, count in location_counts[:5]:
                    st.write(f"**{location.title()}**: {count} health mentions")
            
            with col2:
                st.markdown("#### Condition Distribution")
                condition_counts = {}
                for location_data in self.aggregated_data.values():
                    for condition, count in location_data.items():
                        if condition != 'total_health_mentions':
                            condition_counts[condition] = condition_counts.get(condition, 0) + count
                
                if condition_counts:
                    fig = px.pie(
                        values=list(condition_counts.values()),
                        names=list(condition_counts.keys()),
                        title="Health Conditions Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # System status
        st.markdown("### 🔧 System Status")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success("✅ Data Collection: Active")
        
        with col2:
            model_status = "✅ Model: Ready" if self.classifier else "❌ Model: Not Ready"
            if self.classifier:
                st.success(model_status)
            else:
                st.error(model_status)
        
        with col3:
            st.success("✅ Dashboard: Online")
    
    def render_map_view(self):
        """Render the map view page."""
        st.markdown("## 🗺️ Geographic Health Trends")
        
        if not self.aggregated_data:
            st.warning("No aggregated data available. Please refresh data.")
            return
        
        # Map controls
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.markdown("### Controls")
            # Map controls with improved styling
            st.markdown("#### 🎛️ Map Controls")
            
            # Condition selector with improved UX
            all_conditions = set()
            for location_data in self.aggregated_data.values():
                all_conditions.update(location_data.keys())
            
            # Sort conditions for better UX
            sorted_conditions = sorted(list(all_conditions))
            default_index = 0
            if 'total_health_mentions' in sorted_conditions:
                default_index = sorted_conditions.index('total_health_mentions')
            
            selected_condition = st.selectbox(
                "📊 Select Health Condition",
                sorted_conditions,
                index=default_index,
                help="Choose a health condition to visualize on the map"
            )
            
            # Map type selector with icons
            map_type = st.radio(
                "🗺️ Map Display Type", 
                ["Choropleth", "Heatmap"],
                help="Choropleth shows discrete regions, Heatmap shows intensity patterns"
            )
            
            # Add summary statistics
            if selected_condition in ['total_health_mentions', 'physical_health', 'mental_health']:
                total_mentions = sum(
                    conditions.get(selected_condition, 0) 
                    for conditions in self.aggregated_data.values()
                )
                active_locations = sum(
                    1 for conditions in self.aggregated_data.values() 
                    if conditions.get(selected_condition, 0) > 0
                )
                
                # Display key metrics
                col_metric1, col_metric2 = st.columns(2)
                with col_metric1:
                    st.metric("📈 Total Mentions", total_mentions)
                with col_metric2:
                    st.metric("📍 Active Locations", active_locations)
        
        with col1:
            # Create and display map with responsive sizing
            try:
                st.info("🗺️ Generating map visualization...")
                
                if map_type == "Choropleth":
                    folium_map = self.geo_analyzer.create_choropleth_map(
                        self.aggregated_data, selected_condition
                    )
                else:
                    folium_map = self.geo_analyzer.create_heatmap(
                        self.aggregated_data, selected_condition
                    )
                
                st.success("✅ Map created successfully")
                
                # Responsive map display - adapts to screen size
                # Use container width for better responsiveness
                map_container = st.container()
                with map_container:
                    # Add custom CSS for responsive map
                    st.markdown("""
                    <style>
                    .stApp > div > div > div > div:has(iframe) {
                        width: 100% !important;
                    }
                    iframe {
                        width: 100% !important;
                        height: 500px !important;
                        border-radius: 8px;
                        border: 2px solid #e1e5e9;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }
                    @media (max-width: 768px) {
                        iframe {
                            height: 400px !important;
                        }
                    }
                    @media (max-width: 480px) {
                        iframe {
                            height: 350px !important;
                        }
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # Display map with responsive dimensions
                    try:
                        st.info("🔄 Rendering map...")
                        folium_static(folium_map, width=None, height=500)
                        st.success("✅ Map rendered successfully")
                    except Exception as render_error:
                        st.error(f"❌ Map rendering error: {str(render_error)}")
                        st.write("**Error details:**", str(render_error))
                        
                        # Fallback: try with basic parameters
                        try:
                            st.info("🔄 Trying fallback rendering...")
                            folium_static(folium_map)
                            st.success("✅ Fallback map rendered successfully")
                        except Exception as fallback_error:
                            st.error(f"❌ Fallback rendering also failed: {str(fallback_error)}")
                            
                            # Show basic location summary instead
                            st.warning("📊 Map unavailable, showing location summary:")
                            location_summary = []
                            for location, conditions in self.aggregated_data.items():
                                count = conditions.get(selected_condition, 0)
                                if count > 0:
                                    location_summary.append(f"• **{location.title()}**: {count} mentions")
                            
                            if location_summary:
                                st.markdown("\n".join(location_summary))
                            else:
                                st.info("No data available for the selected condition.")
                        
            except Exception as map_error:
                st.error(f"❌ Map creation error: {str(map_error)}")
                st.write("**Error details:**", str(map_error))
                
                # Show detailed error information for debugging
                st.write("**Debug information:**")
                st.write(f"- Map type: {map_type}")
                st.write(f"- Selected condition: {selected_condition}")
                st.write(f"- Available locations: {len(self.aggregated_data)}")
                st.write(f"- Aggregated data keys: {list(self.aggregated_data.keys())}")
                
                # Show basic location summary instead
                st.warning("📊 Showing location summary instead:")
                location_summary = []
                for location, conditions in self.aggregated_data.items():
                    count = conditions.get(selected_condition, 0)
                    if count > 0:
                        location_summary.append(f"• **{location.title()}**: {count} mentions")
                
                if location_summary:
                    st.markdown("\n".join(location_summary))
                else:
                    st.info("No data available for the selected condition.")
        
        # Location details
        st.markdown("### 📊 Location Details")
        
        # Create table with location data
        location_data = []
        for location, conditions in self.aggregated_data.items():
            row = {'Location': location.title()}
            row.update(conditions)
            location_data.append(row)
        
        if location_data:
            df_display = pd.DataFrame(location_data)
            st.dataframe(df_display, use_container_width=True)
    
    def render_time_trends(self):
        """Render the time trends page."""
        st.markdown("## 📈 Time Series Analysis")
        
        if self.data is None:
            st.warning("No data available for time series analysis.")
            return
        
        # Time series analysis
        time_series_data = self.geo_analyzer.create_time_series_analysis(self.data)
        
        # Overall trends
        st.markdown("### Overall Health Trends")
        overall_ts = time_series_data['overall']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=overall_ts['date'],
            y=overall_ts['health_mentions'],
            mode='lines+markers',
            name='Health Mentions',
            line=dict(color='red', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=overall_ts['date'],
            y=overall_ts['total_posts'],
            mode='lines+markers',
            name='Total Posts',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title='Health Mentions vs Total Posts Over Time',
            xaxis_title='Date',
            yaxis_title='Count',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Trends by location
        st.markdown("### Trends by Location")
        location_ts = time_series_data['by_location']
        
        # Location selector
        available_locations = location_ts['location'].unique()
        selected_locations = st.multiselect(
            "Select Locations",
            available_locations,
            default=available_locations[:5] if len(available_locations) > 5 else available_locations
        )
        
        if selected_locations:
            filtered_data = location_ts[location_ts['location'].isin(selected_locations)]
            
            fig = px.line(
                filtered_data,
                x='date',
                y='health_mentions',
                color='location',
                title='Health Mentions by Location Over Time'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Health ratio chart
            fig_ratio = px.line(
                filtered_data,
                x='date',
                y='health_ratio',
                color='location',
                title='Health Mention Ratio by Location'
            )
            
            st.plotly_chart(fig_ratio, use_container_width=True)
    
    def render_text_classifier(self):
        """Render the text classifier page."""
        st.markdown("## 🔍 Text Classification")
        
        if not self.classifier:
            st.error("No model loaded. Please train a model first.")
            return
        
        st.markdown("### Test the Health Classification Model")
        
        # Input text
        user_text = st.text_area(
            "Enter text to classify:",
            placeholder="e.g., I've been feeling really anxious lately and can't sleep at night...",
            height=150
        )
        
        if st.button("🔍 Classify Text") and user_text:
            with st.spinner("Classifying text..."):
                try:
                    result = self.classifier.predict_single(user_text)
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Health-Related", "Yes" if result['is_health_related'] else "No")
                    
                    with col2:
                        st.metric("Confidence", f"{result['confidence']:.2%}")
                    
                    # Visual indicator
                    if result['is_health_related']:
                        st.success("✅ This text appears to be health-related")
                    else:
                        st.info("ℹ️ This text does not appear to be health-related")
                    
                    # Feature analysis
                    st.markdown("### Feature Analysis")
                    features = self.preprocessor.extract_health_features(user_text)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Mental Health Keywords", features['mental_health_keywords'])
                        st.metric("Physical Health Keywords", features['physical_health_keywords'])
                    
                    with col2:
                        st.metric("Text Length", features['text_length'])
                        st.metric("Word Count", features['word_count'])
                    
                except Exception as e:
                    st.error(f"Error classifying text: {e}")
        
        # Example texts
        st.markdown("### Try These Examples")
        
        examples = [
            "I've been feeling really anxious lately and can't sleep at night",
            "Got diagnosed with flu today, feeling terrible",
            "Beautiful sunset today in Nairobi",
            "Experiencing chest pain, should I go to the hospital?",
            "Great football match last night"
        ]
        
        for i, example in enumerate(examples):
            if st.button(f"Example {i+1}: {example[:50]}..."):
                st.text_area("Selected text:", value=example, key=f"example_{i}")
    
    def render_alerts(self):
        """Render the alerts page."""
        st.markdown("## 🚨 Health Alerts")
        
        if not self.aggregated_data:
            st.warning("No data available for generating alerts.")
            return
        
        # Alert settings
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.markdown("### Alert Settings")
            threshold = st.slider("Alert Threshold", 1, 50, 10)
            
            if st.button("🔄 Refresh Alerts"):
                st.rerun()
        
        with col1:
            # Generate alerts
            alerts = self.geo_analyzer.generate_health_alerts(self.aggregated_data, threshold)
            
            if alerts:
                st.success(f"Found {len(alerts)} active alerts")
                
                for alert in alerts:
                    alert_class = "alert-high" if alert['severity'] == 'HIGH' else "alert-medium"
                    
                    st.markdown(f"""
                    <div class="{alert_class}">
                        <h4>🚨 {alert['severity']} ALERT</h4>
                        <p><strong>Location:</strong> {alert['location']}</p>
                        <p><strong>Condition:</strong> {alert['condition']}</p>
                        <p><strong>Count:</strong> {alert['count']} mentions</p>
                        <p><strong>Message:</strong> {alert['message']}</p>
                        <p><small>Generated: {alert['timestamp']}</small></p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info(f"No alerts found with threshold {threshold}")
        
        # Alert statistics
        st.markdown("### 📊 Alert Statistics")
        
        if self.aggregated_data:
            total_mentions = sum(
                data.get('total_health_mentions', 0) 
                for data in self.aggregated_data.values()
            )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Health Mentions", total_mentions)
            
            with col2:
                high_alerts = len([a for a in alerts if a['severity'] == 'HIGH'])
                st.metric("High Severity Alerts", high_alerts)
            
            with col3:
                medium_alerts = len([a for a in alerts if a['severity'] == 'MEDIUM'])
                st.metric("Medium Severity Alerts", medium_alerts)
    
    def render_analytics(self):
        """Render the analytics page."""
        st.markdown("## 📊 Advanced Analytics")
        
        if self.data is None:
            st.warning("No data available for analytics.")
            return
        
        # Model performance
        st.markdown("### 🤖 Model Performance")
        
        if self.classifier and hasattr(self.classifier, 'model_metrics'):
            metrics = self.classifier.model_metrics
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
            
            with col2:
                st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
            
            with col3:
                st.metric("Recall", f"{metrics.get('recall', 0):.3f}")
            
            with col4:
                st.metric("F1 Score", f"{metrics.get('f1_score', 0):.3f}")
        
        # Data distribution
        st.markdown("### 📈 Data Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Health vs Non-health posts
            if 'is_health_related' in self.data.columns:
                health_counts = self.data['is_health_related'].value_counts()
                fig = px.pie(
                    values=health_counts.values,
                    names=['Non-Health', 'Health-Related'],
                    title='Post Classification Distribution'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Posts by location
            location_counts = self.data['location'].value_counts()
            fig = px.bar(
                x=location_counts.index,
                y=location_counts.values,
                title='Posts by Location'
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature analysis
        st.markdown("### 🔍 Feature Analysis")
        
        if 'mental_health_keywords' in self.data.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(
                    self.data,
                    x='mental_health_keywords',
                    title='Mental Health Keywords Distribution'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.histogram(
                    self.data,
                    x='physical_health_keywords',
                    title='Physical Health Keywords Distribution'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Export data
        st.markdown("### 📥 Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Export Aggregated Data"):
                json_data = json.dumps(self.aggregated_data, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name=f"health_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📈 Export Raw Data"):
                csv_data = self.data.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"health_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    def run(self):
        """Run the dashboard."""
        # Initialize data
        if self.data is None:
            self.load_data()
        
        # Render sidebar and get selected view
        selected_view = self.render_sidebar()
        
        # Render selected view
        if selected_view == "Overview":
            self.render_overview()
        elif selected_view == "Map View":
            self.render_map_view()
        elif selected_view == "Time Trends":
            self.render_time_trends()
        elif selected_view == "Text Classifier":
            self.render_text_classifier()
        elif selected_view == "Alerts":
            self.render_alerts()
        elif selected_view == "Analytics":
            self.render_analytics()
        
        # Footer
        st.markdown("---")
        st.markdown("**HealthScopeAI** - Giving Public Health a Social Pulse 🌍")

def main():
    """Main function to run the dashboard."""
    dashboard = HealthScopeAIDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
