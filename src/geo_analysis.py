"""
Geospatial Analysis Module for HealthScopeAI
Handles geographic analysis and visualization of health trends.
"""

import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, Counter

# Geospatial libraries
import geopandas as gpd
import folium
from folium import plugins
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeoAnalyzer:
    """
    Main class for geospatial analysis of health trends.
    """
    
    def __init__(self):
        """Initialize the GeoAnalyzer."""
        self.location_coordinates = {
            'nairobi': {'lat': -1.2921, 'lon': 36.8219},
            'mombasa': {'lat': -4.0435, 'lon': 39.6682},
            'kisumu': {'lat': -0.1022, 'lon': 34.7617},
            'nakuru': {'lat': -0.3031, 'lon': 36.0800},
            'eldoret': {'lat': 0.5143, 'lon': 35.2698},
            'thika': {'lat': -1.0332, 'lon': 37.0690},
            'malindi': {'lat': -3.2194, 'lon': 40.1169},
            'kitale': {'lat': 1.0177, 'lon': 35.0062},
            'garissa': {'lat': -0.4536, 'lon': 39.6401},
            'kakamega': {'lat': 0.2827, 'lon': 34.7519},
            'nyeri': {'lat': -0.4209, 'lon': 36.9483},
            'machakos': {'lat': -1.5177, 'lon': 37.2634},
            'meru': {'lat': 0.0500, 'lon': 37.6500},
            'embu': {'lat': -0.5312, 'lon': 37.4512},
            'lamu': {'lat': -2.2717, 'lon': 40.9020}
        }
        
        self.health_conditions = {
            'mental_health': ['depression', 'anxiety', 'stress', 'panic', 'mental health'],
            'physical_health': ['flu', 'fever', 'pain', 'headache', 'chest pain', 'stomach'],
            'general_health': ['sick', 'hospital', 'doctor', 'medicine', 'symptoms']
        }
        
    def aggregate_health_data(self, df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
        """
        Aggregate health data by location and condition.
        
        Args:
            df: DataFrame with health data
            
        Returns:
            Dictionary with aggregated health data by location
        """
        logger.info("Aggregating health data by location...")
        
        # Initialize result dictionary
        location_health_data = defaultdict(lambda: defaultdict(int))
        
        # Process each row
        for _, row in df.iterrows():
            location = row.get('location', '').lower()
            text = row.get('text', '').lower()
            
            if location in self.location_coordinates:
                # Check for health conditions
                for condition_category, keywords in self.health_conditions.items():
                    for keyword in keywords:
                        if keyword in text:
                            location_health_data[location][keyword] += 1
                
                # Count general health mentions
                if row.get('is_health_related', 0) == 1:
                    location_health_data[location]['total_health_mentions'] += 1
        
        # Convert to regular dict for JSON serialization
        result = {}
        for location, conditions in location_health_data.items():
            result[location] = dict(conditions)
        
        return result
    
    def create_choropleth_map(self, aggregated_data: Dict[str, Dict[str, int]], 
                            condition: str = 'total_health_mentions') -> folium.Map:
        """
        Create a choropleth map showing health trends by location.
        
        Args:
            aggregated_data: Aggregated health data
            condition: Specific condition to visualize
            
        Returns:
            Folium map object
        """
        logger.info(f"Creating choropleth map for condition: {condition}")
        
        # Create base map centered on Kenya
        m = folium.Map(
            location=[-1.2921, 36.8219],  # Nairobi coordinates
            zoom_start=6,
            tiles='OpenStreetMap'
        )
        
        # Add markers for each location
        for location, conditions in aggregated_data.items():
            if location in self.location_coordinates:
                coords = self.location_coordinates[location]
                count = conditions.get(condition, 0)
                
                # Determine marker color based on count
                if count >= 10:
                    color = 'red'
                elif count >= 5:
                    color = 'orange'
                elif count >= 1:
                    color = 'yellow'
                else:
                    color = 'green'
                
                # Create popup text
                popup_text = f"""
                <b>{location.title()}</b><br>
                {condition.replace('_', ' ').title()}: {count}<br>
                """
                
                # Add other conditions to popup
                for cond, cnt in conditions.items():
                    if cond != condition and cnt > 0:
                        popup_text += f"{cond.replace('_', ' ').title()}: {cnt}<br>"
                
                # Add marker
                folium.CircleMarker(
                    location=[coords['lat'], coords['lon']],
                    radius=max(5, count * 2),
                    popup=popup_text,
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.6
                ).add_to(m)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 150px; height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; ">
        <p style="margin: 10px;"><b>Health Mentions</b></p>
        <p style="margin: 10px;"><i class="fa fa-circle" style="color:red;"></i> High (10+)</p>
        <p style="margin: 10px;"><i class="fa fa-circle" style="color:orange;"></i> Medium (5-9)</p>
        <p style="margin: 10px;"><i class="fa fa-circle" style="color:yellow;"></i> Low (1-4)</p>
        <p style="margin: 10px;"><i class="fa fa-circle" style="color:green;"></i> None (0)</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        return m
    
    def create_heatmap(self, aggregated_data: Dict[str, Dict[str, int]], 
                      condition: str = 'total_health_mentions') -> folium.Map:
        """
        Create a heatmap showing health trends intensity.
        
        Args:
            aggregated_data: Aggregated health data
            condition: Specific condition to visualize
            
        Returns:
            Folium map with heatmap
        """
        logger.info(f"Creating heatmap for condition: {condition}")
        
        # Create base map
        m = folium.Map(
            location=[-1.2921, 36.8219],
            zoom_start=6,
            tiles='OpenStreetMap'
        )
        
        # Prepare data for heatmap
        heat_data = []
        for location, conditions in aggregated_data.items():
            if location in self.location_coordinates:
                coords = self.location_coordinates[location]
                count = conditions.get(condition, 0)
                
                if count > 0:
                    # Add multiple points for higher intensity
                    for _ in range(count):
                        heat_data.append([coords['lat'], coords['lon']])
        
        # Add heatmap layer
        if heat_data:
            plugins.HeatMap(heat_data).add_to(m)
        
        return m
    
    def create_time_series_analysis(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Create time series analysis of health trends.
        
        Args:
            df: DataFrame with timestamp and health data
            
        Returns:
            Dictionary with time series data
        """
        logger.info("Creating time series analysis...")
        
        # Ensure timestamp column exists and is datetime
        if 'timestamp' not in df.columns:
            logger.warning("No timestamp column found. Creating dummy timestamps.")
            df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='H')
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Group by date and location
        df['date'] = df['timestamp'].dt.date
        
        # Time series by location
        location_ts = df.groupby(['date', 'location']).agg({
            'is_health_related': 'sum',
            'text': 'count'
        }).reset_index()
        
        location_ts.columns = ['date', 'location', 'health_mentions', 'total_posts']
        location_ts['health_ratio'] = location_ts['health_mentions'] / location_ts['total_posts']
        
        # Overall time series
        overall_ts = df.groupby('date').agg({
            'is_health_related': 'sum',
            'text': 'count'
        }).reset_index()
        
        overall_ts.columns = ['date', 'health_mentions', 'total_posts']
        overall_ts['health_ratio'] = overall_ts['health_mentions'] / overall_ts['total_posts']
        
        return {
            'by_location': location_ts,
            'overall': overall_ts
        }
    
    def create_plotly_dashboard(self, df: pd.DataFrame, aggregated_data: Dict[str, Dict[str, int]]) -> go.Figure:
        """
        Create an interactive Plotly dashboard.
        
        Args:
            df: DataFrame with health data
            aggregated_data: Aggregated health data
            
        Returns:
            Plotly figure object
        """
        logger.info("Creating Plotly dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Health Mentions by Location', 'Condition Distribution',
                          'Time Series Trends', 'Geographic Scatter'),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # 1. Bar chart - Health mentions by location
        locations = list(aggregated_data.keys())
        health_counts = [data.get('total_health_mentions', 0) for data in aggregated_data.values()]
        
        fig.add_trace(
            go.Bar(x=locations, y=health_counts, name='Health Mentions'),
            row=1, col=1
        )
        
        # 2. Pie chart - Condition distribution
        condition_counts = defaultdict(int)
        for location_data in aggregated_data.values():
            for condition, count in location_data.items():
                if condition != 'total_health_mentions':
                    condition_counts[condition] += count
        
        if condition_counts:
            fig.add_trace(
                go.Pie(labels=list(condition_counts.keys()), 
                      values=list(condition_counts.values()),
                      name="Conditions"),
                row=1, col=2
            )
        
        # 3. Time series
        time_series = self.create_time_series_analysis(df)
        overall_ts = time_series['overall']
        
        fig.add_trace(
            go.Scatter(x=overall_ts['date'], y=overall_ts['health_mentions'],
                      mode='lines+markers', name='Health Mentions Over Time'),
            row=2, col=1
        )
        
        # 4. Geographic scatter
        lat_vals = []
        lon_vals = []
        sizes = []
        location_names = []
        
        for location, data in aggregated_data.items():
            if location in self.location_coordinates:
                coords = self.location_coordinates[location]
                lat_vals.append(coords['lat'])
                lon_vals.append(coords['lon'])
                sizes.append(max(5, data.get('total_health_mentions', 0) * 3))
                location_names.append(location.title())
        
        fig.add_trace(
            go.Scatter(x=lon_vals, y=lat_vals, 
                      mode='markers',
                      marker=dict(size=sizes, color='red', opacity=0.6),
                      text=location_names,
                      name='Geographic Distribution'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="HealthScopeAI - Geographic Health Trends Dashboard",
            height=800,
            showlegend=False
        )
        
        return fig
    
    def generate_health_alerts(self, aggregated_data: Dict[str, Dict[str, int]], 
                             threshold: int = 10) -> List[Dict[str, str]]:
        """
        Generate health alerts based on threshold values.
        
        Args:
            aggregated_data: Aggregated health data
            threshold: Threshold for generating alerts
            
        Returns:
            List of alert dictionaries
        """
        logger.info(f"Generating health alerts with threshold: {threshold}")
        
        alerts = []
        
        for location, conditions in aggregated_data.items():
            for condition, count in conditions.items():
                if count >= threshold:
                    alert = {
                        'location': location.title(),
                        'condition': condition.replace('_', ' ').title(),
                        'count': count,
                        'severity': 'HIGH' if count >= 20 else 'MEDIUM',
                        'timestamp': datetime.now().isoformat(),
                        'message': f"High number of {condition.replace('_', ' ')} mentions ({count}) detected in {location.title()}"
                    }
                    alerts.append(alert)
        
        # Sort by count (descending)
        alerts.sort(key=lambda x: x['count'], reverse=True)
        
        return alerts
    
    def save_analysis_results(self, aggregated_data: Dict[str, Dict[str, int]], 
                            filename: str = None) -> str:
        """
        Save analysis results to JSON file.
        
        Args:
            aggregated_data: Aggregated health data
            filename: Optional filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"geo_analysis_{timestamp}.json"
        
        # Create results directory
        results_dir = Path("data/processed")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save data
        output_path = results_dir / filename
        with open(output_path, 'w') as f:
            json.dump(aggregated_data, f, indent=2)
        
        logger.info(f"Analysis results saved to {output_path}")
        return str(output_path)

def main():
    """Main function to run geospatial analysis."""
    # Initialize analyzer
    analyzer = GeoAnalyzer()
    
    # Load processed data
    processed_dir = Path("data/processed")
    csv_files = list(processed_dir.glob("*.csv"))
    
    if not csv_files:
        logger.error("No processed CSV files found. Run preprocessing first.")
        return
    
    # Load the most recent processed file
    latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"Loading data from {latest_file}")
    
    df = pd.read_csv(latest_file)
    
    # Aggregate health data
    aggregated_data = analyzer.aggregate_health_data(df)
    
    # Save results
    analyzer.save_analysis_results(aggregated_data)
    
    # Create visualizations
    choropleth_map = analyzer.create_choropleth_map(aggregated_data)
    heatmap = analyzer.create_heatmap(aggregated_data)
    plotly_dashboard = analyzer.create_plotly_dashboard(df, aggregated_data)
    
    # Generate alerts
    alerts = analyzer.generate_health_alerts(aggregated_data, threshold=5)
    
    # Save maps
    maps_dir = Path("screenshots")
    maps_dir.mkdir(parents=True, exist_ok=True)
    
    choropleth_map.save(maps_dir / "choropleth_map.html")
    heatmap.save(maps_dir / "heatmap.html")
    plotly_dashboard.write_html(maps_dir / "dashboard.html")
    
    # Display results
    print("Geospatial analysis complete!")
    print(f"Analyzed {len(df)} records")
    print(f"Found data for {len(aggregated_data)} locations")
    
    print("\nTop locations by health mentions:")
    sorted_locations = sorted(aggregated_data.items(), 
                            key=lambda x: x[1].get('total_health_mentions', 0), 
                            reverse=True)
    
    for location, data in sorted_locations[:5]:
        print(f"  {location.title()}: {data.get('total_health_mentions', 0)} mentions")
    
    print(f"\nGenerated {len(alerts)} health alerts")
    if alerts:
        print("Top alerts:")
        for alert in alerts[:3]:
            print(f"  - {alert['message']}")
    
    print(f"\nMaps saved to {maps_dir}")

if __name__ == "__main__":
    main()
