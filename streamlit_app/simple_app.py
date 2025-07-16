"""
Simplified HealthScopeAI Dashboard - Fast Startup Version
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import folium_static
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="HealthScopeAI Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

def create_sample_data():
    """Create sample health data for demo"""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    
    # Sample Kenyan locations
    locations = [
        {'name': 'Nairobi', 'lat': -1.2921, 'lon': 36.8219},
        {'name': 'Mombasa', 'lat': -4.0435, 'lon': 39.6682},
        {'name': 'Kisumu', 'lat': -0.1022, 'lon': 34.7617},
        {'name': 'Nakuru', 'lat': -0.3031, 'lon': 36.0800},
        {'name': 'Eldoret', 'lat': 0.5143, 'lon': 35.2697}
    ]
    
    data = []
    for date in dates[-30:]:  # Last 30 days
        for location in locations:
            # Generate random health mentions
            physical_health = np.random.poisson(10)
            mental_health = np.random.poisson(5)
            
            data.append({
                'date': date,
                'location': location['name'],
                'lat': location['lat'],
                'lon': location['lon'],
                'physical_health_mentions': physical_health,
                'mental_health_mentions': mental_health,
                'total_mentions': physical_health + mental_health
            })
    
    return pd.DataFrame(data)

def classify_text(text):
    """Simple text classification"""
    text_lower = text.lower()
    if any(word in text_lower for word in ['fever', 'sick', 'flu', 'covid', 'headache', 'pain']):
        return 'Physical Health'
    elif any(word in text_lower for word in ['stress', 'anxiety', 'depression', 'mental', 'worry', 'sad']):
        return 'Mental Health'
    else:
        return 'General Health'

def create_health_map(data):
    """Create health trend map"""
    # Center on Kenya
    m = folium.Map(location=[-0.5, 36.5], zoom_start=6)
    
    # Add markers for each location
    for _, row in data.groupby('location').agg({
        'lat': 'first',
        'lon': 'first',
        'total_mentions': 'sum',
        'physical_health_mentions': 'sum',
        'mental_health_mentions': 'sum'
    }).reset_index().iterrows():
        
        # Color based on total mentions
        total = row['total_mentions']
        if total > 300:
            color = 'red'
        elif total > 200:
            color = 'orange'
        else:
            color = 'green'
        
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=min(total/20, 20),
            popup=f"""
            <b>{row['location']}</b><br>
            Total: {total}<br>
            Physical: {row['physical_health_mentions']}<br>
            Mental: {row['mental_health_mentions']}
            """,
            color=color,
            fill=True,
            fillColor=color,
            weight=2
        ).add_to(m)
    
    return m

def main():
    """Main dashboard function"""
    try:
        # Header
        st.markdown(
            '<h1 style="text-align: center; color: #1f77b4;">🏥 HealthScopeAI Dashboard</h1>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<p style="text-align: center; color: #666;">Giving Public Health a Social Pulse</p>',
            unsafe_allow_html=True
        )
        
        # Sidebar
        st.sidebar.title("🔧 Controls")
        
        # Demo notice
        st.sidebar.info("🔬 **Demo Mode**: Using simulated data for demonstration")
        
        # Load sample data
        with st.spinner("Loading health data..."):
            data = create_sample_data()
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_mentions = data['total_mentions'].sum()
            st.metric("Total Health Mentions", f"{total_mentions:,}")
        
        with col2:
            physical_mentions = data['physical_health_mentions'].sum()
            st.metric("Physical Health", f"{physical_mentions:,}")
        
        with col3:
            mental_mentions = data['mental_health_mentions'].sum()
            st.metric("Mental Health", f"{mental_mentions:,}")
        
        with col4:
            avg_daily = data.groupby('date')['total_mentions'].sum().mean()
            st.metric("Avg Daily Mentions", f"{avg_daily:.0f}")
        
        # Two columns layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📊 Health Trends Over Time")
            
            # Time series chart
            daily_data = data.groupby('date').agg({
                'physical_health_mentions': 'sum',
                'mental_health_mentions': 'sum'
            }).reset_index()
            
            fig = px.line(
                daily_data.melt(id_vars=['date'], var_name='type', value_name='mentions'),
                x='date',
                y='mentions',
                color='type',
                title="Daily Health Mentions"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📍 Geographic Distribution")
            
            # Health map
            health_map = create_health_map(data)
            folium_static(health_map, width=500, height=400)
        
        # Text classification demo
        st.subheader("🤖 AI Health Classification Demo")
        
        sample_texts = [
            "I have been feeling very anxious lately and can't sleep",
            "My head hurts and I have a fever",
            "The weather is nice today",
            "I'm stressed about work and feeling depressed",
            "I think I caught the flu, body aches everywhere"
        ]
        
        user_text = st.selectbox("Try sample texts or enter your own:", 
                                [""] + sample_texts)
        
        if not user_text:
            user_text = st.text_area("Enter text to classify:", placeholder="e.g., I have a headache and feel sick")
        
        if user_text:
            classification = classify_text(user_text)
            
            if classification == 'Physical Health':
                st.success(f"Classification: {classification} 🏥")
            elif classification == 'Mental Health':
                st.warning(f"Classification: {classification} 🧠")
            else:
                st.info(f"Classification: {classification} ℹ️")
        
        # Footer
        st.markdown("---")
        st.markdown(
            "**HealthScopeAI** - Revolutionizing Public Health Monitoring in Africa | "
            "Built with Streamlit, Plotly, and Folium"
        )
        
    except Exception as e:
        logger.error(f"App error: {e}")
        st.error(f"Application error: {e}")
        st.info("Please refresh the page or contact support.")

if __name__ == "__main__":
    main()
