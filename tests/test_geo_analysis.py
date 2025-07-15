"""
Unit tests for geospatial analysis module.

This test suite validates the GeoAnalyzer class which processes geographic data
for health trend mapping and visualization. It handles location detection,
coordinate mapping, and spatial analysis for the HealthScopeAI dashboard.

Test Coverage:
- Geographic coordinate mapping for Kenyan cities
- Location detection from text content
- Health hotspot identification and clustering
- Distance calculations using Haversine formula
- Choropleth map data preparation
- GeoJSON file generation and validation
- Spatial aggregation and statistics
- Map visualization data formatting
- Geographic boundary validation
"""

import pytest
import pandas as pd
import numpy as np
import folium
import json
import tempfile
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys
import os

# Import the module to test
from geo_analysis import GeoAnalyzer

class TestGeoAnalyzer:
    """
    Test cases for GeoAnalyzer class.
    
    This class validates the geospatial analysis functionality that maps
    health trends across geographic locations in Kenya for HealthScopeAI.
    """
    
    def test_init(self):
        """
        Test GeoAnalyzer initialization.
        
        Verifies that the geo analyzer initializes with proper location coordinates
        and health condition mappings for Kenyan cities.
        """
        analyzer = GeoAnalyzer()
        
        assert hasattr(analyzer, 'location_coordinates')
        assert hasattr(analyzer, 'health_conditions')
        assert len(analyzer.location_coordinates) > 0
        assert 'nairobi' in analyzer.location_coordinates
        assert 'mombasa' in analyzer.location_coordinates
        
        # Check coordinate format and validity
        nairobi_coords = analyzer.location_coordinates['nairobi']
        assert 'lat' in nairobi_coords
        assert 'lon' in nairobi_coords
        assert isinstance(nairobi_coords['lat'], (int, float))
        assert isinstance(nairobi_coords['lon'], (int, float))
    
    def test_location_coordinates_completeness(self):
        """
        Test that location coordinates are complete.
        
        Ensures all major Kenyan cities have proper coordinate mappings
        within the correct geographic boundaries for Kenya.
        """
        analyzer = GeoAnalyzer()
        
        # Major Kenyan cities that should be mapped
        required_cities = ['nairobi', 'mombasa', 'kisumu', 'nakuru', 'eldoret']
        
        for city in required_cities:
            assert city in analyzer.location_coordinates
            coords = analyzer.location_coordinates[city]
            # Validate coordinates are within Kenya's geographic boundaries
            assert -5 <= coords['lat'] <= 5  # Kenya latitude range
            assert 33 <= coords['lon'] <= 42  # Kenya longitude range
    
    def test_health_conditions_categories(self):
        """Test health conditions categorization."""
        analyzer = GeoAnalyzer()
        
        assert 'mental_health' in analyzer.health_conditions
        assert 'physical_health' in analyzer.health_conditions
        
        # Check mental health keywords
        mental_keywords = analyzer.health_conditions['mental_health']
        assert 'depression' in mental_keywords
        assert 'anxiety' in mental_keywords
        assert 'stress' in mental_keywords
        
        # Check physical health keywords
        physical_keywords = analyzer.health_conditions['physical_health']
        assert 'fever' in physical_keywords
        assert 'pain' in physical_keywords
        assert 'flu' in physical_keywords
    
    def test_aggregate_health_data(self, sample_health_data):
        """Test health data aggregation by location."""
        analyzer = GeoAnalyzer()
        
        # Test with sample data
        aggregated = analyzer.aggregate_health_data(sample_health_data)
        
        assert isinstance(aggregated, dict)
        assert len(aggregated) > 0
        
        # Check that known locations are present
        known_locations = set(sample_health_data['location'].str.lower()) & set(analyzer.location_coordinates.keys())
        for location in known_locations:
            assert location in aggregated
            assert isinstance(aggregated[location], dict)
    
    def test_aggregate_health_data_empty(self):
        """Test aggregation with empty data."""
        analyzer = GeoAnalyzer()
        
        empty_df = pd.DataFrame()
        result = analyzer.aggregate_health_data(empty_df)
        
        assert isinstance(result, dict)
        assert len(result) == 0
    
    def test_aggregate_health_data_keyword_detection(self):
        """Test keyword detection in health data aggregation."""
        analyzer = GeoAnalyzer()
        
        # Create test data with specific keywords
        test_data = pd.DataFrame({
            'text': [
                'I have depression and anxiety',  # Mental health
                'Suffering from fever and flu',   # Physical health
                'Beautiful weather today',        # Non-health
                'Feeling stressed about work'     # Mental health
            ],
            'location': ['Nairobi', 'Mombasa', 'Kisumu', 'Nairobi'],
            'is_health_related': [1, 1, 0, 1]
        })
        
        aggregated = analyzer.aggregate_health_data(test_data)
        
        # Check Nairobi has mental health keywords
        if 'nairobi' in aggregated:
            nairobi_data = aggregated['nairobi']
            assert any(keyword in nairobi_data for keyword in ['depression', 'anxiety', 'stress'])
        
        # Check Mombasa has physical health keywords
        if 'mombasa' in aggregated:
            mombasa_data = aggregated['mombasa']
            assert any(keyword in mombasa_data for keyword in ['fever', 'flu'])
    
    def test_create_choropleth_map(self, sample_geo_data):
        """Test choropleth map creation."""
        analyzer = GeoAnalyzer()
        
        # Create a map
        map_obj = analyzer.create_choropleth_map(sample_geo_data)
        
        assert isinstance(map_obj, folium.Map)
        
        # Check map properties
        assert hasattr(map_obj, 'location')
        assert hasattr(map_obj, 'zoom_start')
    
    def test_create_heatmap(self, sample_health_data):
        """Test heatmap creation."""
        analyzer = GeoAnalyzer()
        
        # Create heatmap
        heat_map = analyzer.create_heatmap(sample_health_data)
        
        assert isinstance(heat_map, folium.Map)
        
        # Should have heat map layer
        assert any('HeatMap' in str(child) for child in heat_map._children.values())
    
    def test_calculate_health_density(self, sample_health_data):
        """Test health density calculation."""
        analyzer = GeoAnalyzer()
        
        densities = analyzer.calculate_health_density(sample_health_data)
        
        assert isinstance(densities, dict)
        assert len(densities) > 0
        
        # Check density values are numeric
        for location, density in densities.items():
            assert isinstance(density, (int, float))
            assert density >= 0
    
    def test_find_health_hotspots(self, sample_health_data):
        """Test health hotspot identification."""
        analyzer = GeoAnalyzer()
        
        hotspots = analyzer.find_health_hotspots(
            sample_health_data, 
            threshold=2,  # Low threshold for testing
            top_n=3
        )
        
        assert isinstance(hotspots, list)
        assert len(hotspots) <= 3
        
        # Each hotspot should have required fields
        for hotspot in hotspots:
            assert isinstance(hotspot, dict)
            assert 'location' in hotspot
            assert 'count' in hotspot
            assert 'coordinates' in hotspot
    
    def test_analyze_temporal_trends(self, sample_health_data):
        """Test temporal trend analysis."""
        analyzer = GeoAnalyzer()
        
        trends = analyzer.analyze_temporal_trends(sample_health_data)
        
        assert isinstance(trends, dict)
        assert 'daily_trends' in trends or 'weekly_trends' in trends
        
        # Check trend data structure
        if 'daily_trends' in trends:
            daily = trends['daily_trends']
            assert isinstance(daily, dict)
    
    def test_calculate_distance_between_locations(self):
        """Test distance calculation between locations."""
        analyzer = GeoAnalyzer()
        
        # Calculate distance between Nairobi and Mombasa
        distance = analyzer.calculate_distance('nairobi', 'mombasa')
        
        assert isinstance(distance, (int, float))
        assert distance > 0
        assert distance < 1000  # Should be reasonable distance in km
    
    def test_calculate_distance_invalid_locations(self):
        """Test distance calculation with invalid locations."""
        analyzer = GeoAnalyzer()
        
        # Test with invalid location
        distance = analyzer.calculate_distance('invalid_city', 'nairobi')
        assert distance is None or distance == 0
        
        distance = analyzer.calculate_distance('nairobi', 'another_invalid')
        assert distance is None or distance == 0
    
    def test_get_nearby_locations(self):
        """Test finding nearby locations."""
        analyzer = GeoAnalyzer()
        
        # Find locations near Nairobi
        nearby = analyzer.get_nearby_locations('nairobi', radius_km=200)
        
        assert isinstance(nearby, list)
        assert len(nearby) >= 0
        
        # Each nearby location should have distance info
        for location in nearby:
            assert isinstance(location, dict)
            assert 'location' in location
            assert 'distance' in location
            assert location['distance'] <= 200
    
    def test_create_cluster_analysis(self, sample_health_data):
        """Test cluster analysis of health data."""
        analyzer = GeoAnalyzer()
        
        clusters = analyzer.create_cluster_analysis(
            sample_health_data,
            n_clusters=3
        )
        
        assert isinstance(clusters, dict)
        assert 'cluster_centers' in clusters
        assert 'cluster_labels' in clusters
        assert 'cluster_summary' in clusters
        
        # Check cluster labels
        labels = clusters['cluster_labels']
        assert len(labels) == len(sample_health_data)
        assert all(isinstance(label, (int, np.integer)) for label in labels)
    
    def test_generate_health_report(self, sample_health_data):
        """Test health report generation."""
        analyzer = GeoAnalyzer()
        
        report = analyzer.generate_health_report(sample_health_data)
        
        assert isinstance(report, dict)
        assert 'summary' in report
        assert 'top_locations' in report
        assert 'health_categories' in report
        assert 'recommendations' in report
        
        # Check summary statistics
        summary = report['summary']
        assert 'total_records' in summary
        assert 'health_related_count' in summary
        assert 'unique_locations' in summary
    
    def test_export_geojson(self, sample_health_data, temp_directory):
        """Test GeoJSON export functionality."""
        analyzer = GeoAnalyzer()
        
        # Export to GeoJSON
        output_path = temp_directory / "test_export.geojson"
        analyzer.export_geojson(sample_health_data, str(output_path))
        
        assert output_path.exists()
        
        # Verify GeoJSON format
        with open(output_path, 'r') as f:
            geojson_data = json.load(f)
        
        assert 'type' in geojson_data
        assert geojson_data['type'] == 'FeatureCollection'
        assert 'features' in geojson_data
    
    def test_load_geojson(self, temp_directory):
        """Test GeoJSON loading functionality."""
        analyzer = GeoAnalyzer()
        
        # Create sample GeoJSON
        sample_geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [36.8219, -1.2921]
                    },
                    "properties": {
                        "location": "Nairobi",
                        "health_count": 10
                    }
                }
            ]
        }
        
        geojson_path = temp_directory / "sample.geojson"
        with open(geojson_path, 'w') as f:
            json.dump(sample_geojson, f)
        
        # Load GeoJSON
        loaded_data = analyzer.load_geojson(str(geojson_path))
        
        assert isinstance(loaded_data, dict)
        assert 'type' in loaded_data
        assert loaded_data['type'] == 'FeatureCollection'
    
    def test_create_interactive_dashboard(self, sample_health_data):
        """Test interactive dashboard creation."""
        analyzer = GeoAnalyzer()
        
        # Create dashboard components
        dashboard_data = analyzer.create_interactive_dashboard(sample_health_data)
        
        assert isinstance(dashboard_data, dict)
        assert 'map' in dashboard_data
        assert 'charts' in dashboard_data
        
        # Check map component
        map_component = dashboard_data['map']
        assert isinstance(map_component, (folium.Map, dict))
    
    def test_geographical_boundaries(self):
        """Test geographical boundary validation."""
        analyzer = GeoAnalyzer()
        
        # All coordinates should be within Kenya's boundaries
        for location, coords in analyzer.location_coordinates.items():
            lat, lon = coords['lat'], coords['lon']
            
            # Kenya's approximate boundaries
            assert -5 <= lat <= 5, f"Latitude out of bounds for {location}: {lat}"
            assert 33 <= lon <= 42, f"Longitude out of bounds for {location}: {lon}"
    
    def test_health_condition_coverage(self):
        """Test health condition keyword coverage."""
        analyzer = GeoAnalyzer()
        
        # Should cover major health categories
        all_keywords = []
        for category, keywords in analyzer.health_conditions.items():
            all_keywords.extend(keywords)
        
        # Check for mental health coverage
        mental_terms = ['depression', 'anxiety', 'stress']
        assert any(term in all_keywords for term in mental_terms)
        
        # Check for physical health coverage  
        physical_terms = ['fever', 'pain', 'flu', 'headache']
        assert any(term in all_keywords for term in physical_terms)
    
    @pytest.mark.parametrize("location", ['nairobi', 'mombasa', 'kisumu', 'nakuru'])
    def test_major_cities_coordinates(self, location):
        """Parametrized test for major city coordinates."""
        analyzer = GeoAnalyzer()
        
        assert location in analyzer.location_coordinates
        coords = analyzer.location_coordinates[location]
        
        # Coordinates should be reasonable for Kenya
        assert isinstance(coords['lat'], (int, float))
        assert isinstance(coords['lon'], (int, float))
        assert -5 <= coords['lat'] <= 5
        assert 33 <= coords['lon'] <= 42
    
    def test_error_handling(self):
        """Test error handling in geo analysis."""
        analyzer = GeoAnalyzer()
        
        # Test with invalid data types
        result = analyzer.aggregate_health_data(None)
        assert isinstance(result, dict)
        
        # Test with missing columns
        invalid_df = pd.DataFrame({'wrong_column': [1, 2, 3]})
        result = analyzer.aggregate_health_data(invalid_df)
        assert isinstance(result, dict)
    
    def test_performance_with_large_dataset(self, sample_health_data):
        """Test performance with larger datasets."""
        analyzer = GeoAnalyzer()
        
        # Create larger dataset
        large_data = pd.concat([sample_health_data] * 100, ignore_index=True)
        
        import time
        start_time = time.time()
        
        # Test aggregation performance
        result = analyzer.aggregate_health_data(large_data)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process reasonably quickly
        assert processing_time < 30  # Under 30 seconds for 800 rows
        assert isinstance(result, dict)
    
    def test_coordinate_precision(self):
        """Test coordinate precision for mapping accuracy."""
        analyzer = GeoAnalyzer()
        
        for location, coords in analyzer.location_coordinates.items():
            lat, lon = coords['lat'], coords['lon']
            
            # Coordinates should have reasonable precision (not too rounded)
            lat_str = str(lat)
            lon_str = str(lon)
            
            # Should have at least 2 decimal places for city-level accuracy
            if '.' in lat_str:
                lat_decimals = len(lat_str.split('.')[1])
                assert lat_decimals >= 2, f"Insufficient precision for {location} latitude"
            
            if '.' in lon_str:
                lon_decimals = len(lon_str.split('.')[1])
                assert lon_decimals >= 2, f"Insufficient precision for {location} longitude"
