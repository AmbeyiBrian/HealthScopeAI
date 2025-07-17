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
        Create a responsive choropleth map showing health trends by location.
        
        Args:
            aggregated_data: Aggregated health data
            condition: Specific condition to visualize
            
        Returns:
            Folium map object
        """
        logger.info(f"Creating choropleth map for condition: {condition}")
        
        try:
            # Create base map centered on Kenya with responsive styling
            m = folium.Map(
                location=[-1.2921, 36.8219],  # Nairobi coordinates
                zoom_start=6,
                tiles='CartoDB positron',  # Clean, responsive tile style
                width='100%',
                height='100%'
            )
            
            # Prepare data for visualization
            locations = []
            values = []
            colors = []
            
            # Get all values to determine color scale
            all_values = []
            for location, conditions in aggregated_data.items():
                if location in self.location_coordinates:
                    count = conditions.get(condition, 0)
                    all_values.append(count)
            
            # Calculate quantiles for better color distribution
            if all_values:
                import numpy as np
                q25, q50, q75, q95 = np.percentile(all_values, [25, 50, 75, 95])
            else:
                q25, q50, q75, q95 = 0, 0, 0, 0
                logger.warning("No data available for quantile calculation")
            
            # Add styled markers for each location
            markers_added = 0
            for location, conditions in aggregated_data.items():
                if location in self.location_coordinates:
                    try:
                        coords = self.location_coordinates[location]
                        count = conditions.get(condition, 0)
                        
                        # Determine color based on quantiles
                        if count >= q95:
                            color = '#d73027'  # Dark red
                            risk_level = 'Very High'
                        elif count >= q75:
                            color = '#fc8d59'  # Orange-red
                            risk_level = 'High'
                        elif count >= q50:
                            color = '#fee08b'  # Yellow
                            risk_level = 'Medium'
                        elif count >= q25:
                            color = '#e0f3f8'  # Light blue
                            risk_level = 'Low'
                        else:
                            color = '#91bfdb'  # Blue
                            risk_level = 'Very Low'
                        
                        # Calculate radius based on count (responsive sizing)
                        base_radius = 8
                        max_radius = 25
                        if max(all_values) > 0:
                            radius = base_radius + (count / max(all_values)) * (max_radius - base_radius)
                        else:
                            radius = base_radius
                        
                        # Create comprehensive popup with better styling
                        popup_text = f"""
                        <div style="font-family: Arial, sans-serif; min-width: 200px;">
                            <h4 style="margin: 0 0 10px 0; color: #2c3e50; border-bottom: 2px solid {color}; padding-bottom: 5px;">
                                📍 {location.title()}
                            </h4>
                            <div style="margin-bottom: 8px;">
                                <strong style="color: {color};">🏥 {condition.replace('_', ' ').title()}:</strong> 
                                <span style="font-size: 16px; font-weight: bold;">{count}</span>
                            </div>
                            <div style="margin-bottom: 8px;">
                                <strong>📊 Risk Level:</strong> 
                                <span style="color: {color}; font-weight: bold;">{risk_level}</span>
                            </div>
                        """
                        
                        # Add other conditions to popup
                        other_conditions = []
                        for cond, cnt in conditions.items():
                            if cond != condition and cnt > 0:
                                other_conditions.append(f"• {cond.replace('_', ' ').title()}: {cnt}")
                        
                        if other_conditions:
                            popup_text += f"""
                            <div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid #eee;">
                                <strong>🔍 Other Health Mentions:</strong><br>
                                <div style="font-size: 12px; margin-top: 5px;">
                                    {'<br>'.join(other_conditions)}
                                </div>
                            </div>
                            """
                        
                        popup_text += "</div>"
                        
                        # Add responsive circle marker
                        folium.CircleMarker(
                            location=[coords['lat'], coords['lon']],
                            radius=radius,
                            popup=folium.Popup(popup_text, max_width=250),
                            tooltip=f"{location.title()}: {count} mentions",
                            color='white',
                            weight=2,
                            fill=True,
                            fillColor=color,
                            fillOpacity=0.8
                        ).add_to(m)
                        
                        markers_added += 1
                        
                    except Exception as marker_error:
                        logger.error(f"Error adding marker for {location}: {str(marker_error)}")
                        continue
            
            logger.info(f"Added {markers_added} markers to choropleth map")
            
            # Add responsive legend with better styling - only if we have data
            if all_values and max(all_values) > 0:
                try:
                    legend_html = f'''
                    <div style="position: fixed; 
                                bottom: 20px; right: 20px; 
                                min-width: 180px; 
                                background-color: rgba(255, 255, 255, 0.95); 
                                border: 2px solid #34495e;
                                border-radius: 8px;
                                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                                z-index: 9999; 
                                font-family: Arial, sans-serif;
                                font-size: 12px;">
                        <div style="background-color: #34495e; color: white; padding: 8px; border-radius: 6px 6px 0 0; margin: 0;">
                            <strong>🗺️ Health Risk Levels</strong>
                        </div>
                        <div style="padding: 10px;">
                            <div style="margin: 5px 0; display: flex; align-items: center;">
                                <div style="width: 15px; height: 15px; background-color: #d73027; border-radius: 50%; margin-right: 8px;"></div>
                                <span>Very High ({q95:.0f}+)</span>
                            </div>
                            <div style="margin: 5px 0; display: flex; align-items: center;">
                                <div style="width: 15px; height: 15px; background-color: #fc8d59; border-radius: 50%; margin-right: 8px;"></div>
                                <span>High ({q75:.0f}-{q95:.0f})</span>
                            </div>
                            <div style="margin: 5px 0; display: flex; align-items: center;">
                                <div style="width: 15px; height: 15px; background-color: #fee08b; border-radius: 50%; margin-right: 8px;"></div>
                                <span>Medium ({q50:.0f}-{q75:.0f})</span>
                            </div>
                            <div style="margin: 5px 0; display: flex; align-items: center;">
                                <div style="width: 15px; height: 15px; background-color: #e0f3f8; border-radius: 50%; margin-right: 8px;"></div>
                                <span>Low ({q25:.0f}-{q50:.0f})</span>
                            </div>
                            <div style="margin: 5px 0; display: flex; align-items: center;">
                                <div style="width: 15px; height: 15px; background-color: #91bfdb; border-radius: 50%; margin-right: 8px;"></div>
                                <span>Very Low (0-{q25:.0f})</span>
                            </div>
                        </div>
                    </div>
                    '''
                    m.get_root().html.add_child(folium.Element(legend_html))
                except Exception as legend_error:
                    logger.warning(f"Could not add legend: {str(legend_error)}")
            
            # Add responsive CSS for mobile devices
            try:
                responsive_css = '''
                <style>
                @media (max-width: 768px) {
                    .leaflet-container {
                        height: 400px !important;
                    }
                    
                    .legend {
                        bottom: 10px !important;
                        right: 10px !important;
                        min-width: 150px !important;
                        font-size: 11px !important;
                    }
                    
                    .leaflet-popup-content {
                        font-size: 12px !important;
                        min-width: 180px !important;
                    }
                }
                
                @media (max-width: 480px) {
                    .leaflet-container {
                        height: 350px !important;
                    }
                    
                    .legend {
                        min-width: 140px !important;
                        font-size: 10px !important;
                    }
                    
                    .leaflet-popup-content {
                        font-size: 11px !important;
                        min-width: 160px !important;
                    }
                }
                </style>
                '''
                m.get_root().html.add_child(folium.Element(responsive_css))
            except Exception as css_error:
                logger.warning(f"Could not add responsive CSS: {str(css_error)}")
            
            logger.info("Choropleth map creation completed successfully")
            return m
            
        except Exception as e:
            logger.error(f"Error creating choropleth map: {str(e)}")
            # Return a basic map as fallback
            basic_map = folium.Map(
                location=[-1.2921, 36.8219],
                zoom_start=6,
                tiles='OpenStreetMap'
            )
            
            # Add a simple marker for Nairobi as fallback
            folium.Marker(
                [-1.2921, 36.8219],
                popup="HealthScopeAI - Map Error Fallback",
                tooltip="Error occurred while creating detailed map"
            ).add_to(basic_map)
            
            return basic_map
    
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
        
        try:
            # Create base map
            m = folium.Map(
                location=[-1.2921, 36.8219],
                zoom_start=6,
                tiles='OpenStreetMap'
            )
            
            # Prepare data for heatmap
            heat_data = []
            total_points = 0
            
            for location, conditions in aggregated_data.items():
                if location in self.location_coordinates:
                    try:
                        coords = self.location_coordinates[location]
                        count = conditions.get(condition, 0)
                        
                        if count > 0:
                            # Add multiple points for higher intensity (but limit to prevent overcrowding)
                            points_to_add = min(count, 50)  # Limit to 50 points per location
                            for _ in range(points_to_add):
                                # Add slight random variation to coordinates for better heatmap effect
                                lat_variation = np.random.uniform(-0.01, 0.01)
                                lon_variation = np.random.uniform(-0.01, 0.01)
                                heat_data.append([
                                    coords['lat'] + lat_variation, 
                                    coords['lon'] + lon_variation
                                ])
                                total_points += 1
                    except Exception as point_error:
                        logger.warning(f"Error processing heatmap point for {location}: {str(point_error)}")
                        continue
            
            logger.info(f"Generated {total_points} heat points for heatmap")
            
            # Add heatmap layer
            if heat_data:
                try:
                    # Configure heatmap with appropriate settings
                    heatmap = plugins.HeatMap(
                        heat_data,
                        min_opacity=0.3,
                        max_zoom=10,
                        radius=25,
                        blur=15,
                        gradient={
                            0.2: 'blue',
                            0.4: 'cyan', 
                            0.6: 'lime',
                            0.8: 'yellow',
                            1.0: 'red'
                        }
                    )
                    heatmap.add_to(m)
                    logger.info("Heatmap layer added successfully")
                except Exception as heatmap_error:
                    logger.error(f"Error adding heatmap layer: {str(heatmap_error)}")
                    # Fallback: add simple markers instead
                    for location, conditions in aggregated_data.items():
                        if location in self.location_coordinates:
                            coords = self.location_coordinates[location]
                            count = conditions.get(condition, 0)
                            if count > 0:
                                folium.CircleMarker(
                                    [coords['lat'], coords['lon']],
                                    radius=min(count / 2, 20),
                                    popup=f"{location.title()}: {count}",
                                    color='red',
                                    fill=True,
                                    fillOpacity=0.6
                                ).add_to(m)
            else:
                logger.warning("No heat data available for heatmap")
                # Add informational marker
                folium.Marker(
                    [-1.2921, 36.8219],
                    popup="No data available for heatmap visualization",
                    icon=folium.Icon(color='gray', icon='info-sign')
                ).add_to(m)
            
            return m
            
        except Exception as e:
            logger.error(f"Error creating heatmap: {str(e)}")
            # Return a basic map as fallback
            basic_map = folium.Map(
                location=[-1.2921, 36.8219],
                zoom_start=6,
                tiles='OpenStreetMap'
            )
            
            # Add a simple marker for Nairobi as fallback
            folium.Marker(
                [-1.2921, 36.8219],
                popup="HealthScopeAI - Heatmap Error Fallback",
                tooltip="Error occurred while creating heatmap",
                icon=folium.Icon(color='red', icon='exclamation-sign')
            ).add_to(basic_map)
            
            return basic_map
    
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
