#!/usr/bin/env python3
"""
Generate demo data for HealthScopeAI deployment.
This creates the essential data files needed by the Streamlit dashboard.
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def create_demo_data():
    """Create demo health data for the HealthScopeAI dashboard."""
    print("🚀 Generating demo data for HealthScopeAI...")
    
    # Ensure directories exist
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)
    
    # Generate synthetic health data
    print("📊 Creating dashboard data...")
    
    # Kenyan counties for realistic geographic distribution
    kenyan_counties = [
        "Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret", "Thika", "Malindi",
        "Kitale", "Garissa", "Kakamega", "Meru", "Nyeri", "Machakos", "Kericho",
        "Embu", "Migori", "Homa Bay", "Turkana", "West Pokot", "Marsabit"
    ]
    
    # Health-related text samples
    health_texts = [
        "Feeling unwell with fever and headache",
        "Hospital visit due to respiratory issues",
        "Mental health support needed in community",
        "Diabetes management is challenging",
        "Maternal health services are important",
        "Child vaccination program update",
        "Nutrition awareness campaign",
        "Stress and anxiety levels increasing",
        "COVID symptoms reported",
        "Depression support group meeting"
    ]
    
    # Non-health text samples
    non_health_texts = [
        "Beautiful weather today",
        "Traffic update on main road",
        "New restaurant opening",
        "Football match results",
        "School graduation ceremony",
        "Market prices update",
        "Music concert announcement",
        "Road construction progress",
        "Cultural festival celebration",
        "Election campaign rally"
    ]
    
    # Generate time series data for the last 30 days
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=30),
        end=datetime.now(),
        freq='H'  # Hourly data for more granular time series
    )
    
    # Create comprehensive dashboard data
    data_records = []
    
    for timestamp in dates:
        for county in kenyan_counties:
            # Generate multiple posts per hour per location
            num_posts = np.random.poisson(lam=3)  # Average 3 posts per hour per location
            
            for _ in range(num_posts):
                # Determine if health-related (60% health-related posts)
                is_health_related = np.random.random() < 0.6
                
                # Choose text based on health status
                if is_health_related:
                    text = np.random.choice(health_texts)
                    category = np.random.choice(['mental_health', 'physical_health'], p=[0.4, 0.6])
                    sentiment = np.random.choice(['negative', 'neutral', 'positive'], p=[0.5, 0.3, 0.2])
                else:
                    text = np.random.choice(non_health_texts)
                    category = 'non_health'
                    sentiment = np.random.choice(['negative', 'neutral', 'positive'], p=[0.2, 0.3, 0.5])
                
                data_records.append({
                    "text": text,
                    "timestamp": timestamp.isoformat(),
                    "location": county,
                    "source": np.random.choice(['twitter', 'reddit', 'news', 'survey']),
                    "is_health_related": 1 if is_health_related else 0,
                    "category": category,
                    "sentiment": sentiment,
                    "latitude": -1.2921 + np.random.normal(0, 2),  # Kenya latitude range
                    "longitude": 36.8219 + np.random.normal(0, 3),  # Kenya longitude range
                    "label": 1 if is_health_related else 0  # For model compatibility
                })
    
    # Create DataFrame
    dashboard_data = pd.DataFrame(data_records)
    
    # Convert timestamp to datetime for additional columns
    dashboard_data['timestamp'] = pd.to_datetime(dashboard_data['timestamp'])
    dashboard_data['date'] = dashboard_data['timestamp'].dt.date
    dashboard_data['hour'] = dashboard_data['timestamp'].dt.hour
    dashboard_data['day_of_week'] = dashboard_data['timestamp'].dt.day_name()
    
    # Save dashboard data
    dashboard_data.to_csv("data/processed/dashboard_data.csv", index=False)
    print(f"✅ Dashboard data saved: {len(dashboard_data)} records")
    
    # Generate GeoJSON data
    print("🗺️  Creating geographic health data...")
    
    # Create county-level health summaries
    county_summaries = dashboard_data.groupby('location').agg({
        'is_health_related': 'sum',
        'text': 'count',
        'latitude': 'first',
        'longitude': 'first'
    }).reset_index()
    
    county_summaries.columns = ['location', 'health_mentions', 'total_posts', 'latitude', 'longitude']
    county_summaries['health_ratio'] = county_summaries['health_mentions'] / county_summaries['total_posts']
    
    # Create GeoJSON structure
    features = []
    for _, row in county_summaries.iterrows():
        feature = {
            "type": "Feature",
            "properties": {
                "name": row['location'],
                "health_mentions": int(row['health_mentions']),
                "total_posts": int(row['total_posts']),
                "health_ratio": round(row['health_ratio'], 3),
                "risk_level": "high" if row['health_ratio'] > 0.7 else "medium" if row['health_ratio'] > 0.5 else "low"
            },
            "geometry": {
                "type": "Point",
                "coordinates": [float(row['longitude']), float(row['latitude'])]
            }
        }
        features.append(feature)
    
    geojson_data = {
        "type": "FeatureCollection",
        "features": features
    }
    
    # Save GeoJSON data
    with open("data/processed/health_data.geojson", "w") as f:
        json.dump(geojson_data, f, indent=2)
    
    print(f"✅ Geographic data saved: {len(features)} counties")
    
    # Create a basic model info file
    model_info = {
        "model_type": "demo_classifier",
        "version": "1.0.0",
        "created_date": datetime.now().isoformat(),
        "accuracy": 0.89,
        "precision": 0.87,
        "recall": 0.91,
        "f1_score": 0.89,
        "features": [
            "mental_health_keywords",
            "physical_health_keywords", 
            "text_length",
            "word_count",
            "sentiment_score"
        ],
        "description": "Demo health classifier for HealthScopeAI deployment"
    }
    
    os.makedirs("models", exist_ok=True)
    with open("models/model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    
    print("✅ Model info created")
    print("🎉 Demo data generation completed successfully!")
    
    return True

if __name__ == "__main__":
    try:
        create_demo_data()
        print("✅ All demo data files generated successfully!")
    except Exception as e:
        print(f"❌ Error generating demo data: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
