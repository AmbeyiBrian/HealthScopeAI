#!/usr/bin/env python3
"""
Quick diagnostic script to test data loading and app functionality.
"""

import pandas as pd
import json
import os
from pathlib import Path

def test_data_loading():
    """Test if data files can be loaded properly."""
    print("🔍 HealthScopeAI Data Loading Test")
    print("=" * 50)
    
    # Test CSV data
    print("\n📊 Testing CSV Data...")
    try:
        csv_path = "data/processed/dashboard_data.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f"✅ CSV loaded successfully: {len(df)} records")
            print(f"📋 Columns: {list(df.columns)}")
            print(f"🏥 Health related posts: {df['is_health_related'].sum()}")
            print(f"📝 Non-health posts: {len(df) - df['is_health_related'].sum()}")
            print(f"🗺️ Unique locations: {df['location'].nunique()}")
            print(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            # Check required columns for the app
            required_cols = ['text', 'timestamp', 'location', 'is_health_related', 'label']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"❌ Missing required columns: {missing_cols}")
            else:
                print(f"✅ All required columns present")
        else:
            print(f"❌ CSV file not found: {csv_path}")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
    
    # Test GeoJSON data
    print("\n🗺️ Testing GeoJSON Data...")
    try:
        geojson_path = "data/processed/health_data.geojson"
        if os.path.exists(geojson_path):
            with open(geojson_path, 'r') as f:
                geo_data = json.load(f)
            print(f"✅ GeoJSON loaded successfully")
            print(f"📍 Features: {len(geo_data['features'])}")
            print(f"🏷️ Type: {geo_data['type']}")
            
            # Show sample feature
            if geo_data['features']:
                sample = geo_data['features'][0]
                print(f"📋 Sample feature properties: {list(sample['properties'].keys())}")
        else:
            print(f"❌ GeoJSON file not found: {geojson_path}")
    except Exception as e:
        print(f"❌ Error loading GeoJSON: {e}")
    
    # Test model info
    print("\n🤖 Testing Model Info...")
    try:
        model_path = "models/model_info.json"
        if os.path.exists(model_path):
            with open(model_path, 'r') as f:
                model_info = json.load(f)
            print(f"✅ Model info loaded successfully")
            print(f"🏷️ Model type: {model_info.get('model_type', 'Unknown')}")
            print(f"📊 Accuracy: {model_info.get('accuracy', 'Unknown')}")
        else:
            print(f"❌ Model info not found: {model_path}")
    except Exception as e:
        print(f"❌ Error loading model info: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Data Loading Test Complete!")
    
    return True

if __name__ == "__main__":
    test_data_loading()
