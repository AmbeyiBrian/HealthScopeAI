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
    
    # Health categories and sample data
    health_conditions = [
        "Mental Health", "Respiratory", "Cardiovascular", "Diabetes", 
        "Infectious Disease", "Maternal Health", "Child Health", "Nutrition"
    ]
    
    # Generate time series data for the last 30 days
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=30),
        end=datetime.now(),
        freq='D'
    )
    
    # Create comprehensive dashboard data
    data_records = []
    
    for date in dates:
        for county in kenyan_counties:
            for condition in health_conditions:
                # Generate realistic patterns
                base_count = np.random.poisson(lam=10)
                
                # Add some realistic patterns
                if condition == "Mental Health":
                    base_count *= 1.5  # Higher mental health mentions
                elif condition == "Respiratory" and date.month in [6, 7, 8]:
                    base_count *= 2  # Higher during cold season
                elif condition == "Maternal Health" and county in ["Nairobi", "Mombasa", "Kisumu"]:
                    base_count *= 1.3  # Urban areas
                
                sentiment = np.random.choice(
                    ["negative", "neutral", "positive"], 
                    p=[0.4, 0.4, 0.2]  # More negative health mentions
                )
                
                urgency = np.random.choice(
                    ["low", "medium", "high"],
                    p=[0.5, 0.3, 0.2]
                )
                
                data_records.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "timestamp": date.isoformat(),
                    "location": county,
                    "health_category": condition,
                    "mention_count": int(base_count),
                    "sentiment": sentiment,
                    "urgency_level": urgency,
                    "latitude": -1.2921 + np.random.normal(0, 2),  # Kenya latitude range
                    "longitude": 36.8219 + np.random.normal(0, 3),  # Kenya longitude range
                    "population": np.random.randint(50000, 2000000),
                    "health_score": np.random.uniform(0.3, 0.9)
                })
    
    # Create DataFrame
    dashboard_data = pd.DataFrame(data_records)
    
    # Add some computed columns
    dashboard_data['week'] = pd.to_datetime(dashboard_data['date']).dt.isocalendar().week
    dashboard_data['month'] = pd.to_datetime(dashboard_data['date']).dt.month
    dashboard_data['day_of_week'] = pd.to_datetime(dashboard_data['date']).dt.day_name()
    
    # Save dashboard data
    dashboard_data.to_csv("data/processed/dashboard_data.csv", index=False)
    print(f"✅ Dashboard data saved: {len(dashboard_data)} records")
    
    # Generate GeoJSON data
    print("🗺️  Creating geographic health data...")
    
    # Create county-level health summaries
    county_summaries = dashboard_data.groupby('location').agg({
        'mention_count': 'sum',
        'health_score': 'mean',
        'latitude': 'first',
        'longitude': 'first',
        'population': 'first'
    }).reset_index()
    
    # Create GeoJSON structure
    features = []
    for _, row in county_summaries.iterrows():
        feature = {
            "type": "Feature",
            "properties": {
                "name": row['location'],
                "total_mentions": int(row['mention_count']),
                "avg_health_score": round(row['health_score'], 3),
                "population": int(row['population']),
                "risk_level": "high" if row['health_score'] < 0.5 else "medium" if row['health_score'] < 0.7 else "low"
            },
            "geometry": {
                "type": "Point",
                "coordinates": [row['longitude'], row['latitude']]
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
