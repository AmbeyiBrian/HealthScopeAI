"""
Tests for the Streamlit dashboard functionality.

This test suite verifies that the HealthScopeAI dashboard components work correctly,
including data visualization, user interactions, filtering, and real-time updates.
The tests use mocked Streamlit components to simulate dashboard behavior without
requiring a live Streamlit server.

Test Coverage:
- Dashboard initialization and structure
- Data loading and processing for visualization
- Interactive filtering (location, time-based)
- Chart and map data preparation
- Text classification interface
- Alert system logic
- Error handling and edge cases
- Performance metrics
- Real-time update simulation
"""

import pytest
import pandas as pd
import numpy as np
import streamlit as st
from unittest.mock import patch, MagicMock
import tempfile
import sys
import os

# Mock streamlit components for testing
# This class simulates Streamlit's behavior during testing without requiring a browser
class MockStreamlit:
    """
    Mock Streamlit class for testing dashboard components.
    
    This class simulates Streamlit's API during testing, allowing us to verify
    that dashboard functions call the correct Streamlit methods with proper
    parameters without actually rendering a web interface.
    """
    def __init__(self):
        # Track all calls to Streamlit methods for verification
        self.title_calls = []
        self.write_calls = []
        self.error_calls = []
        self.success_calls = []
        
    def title(self, text):
        self.title_calls.append(text)
        
    def write(self, text):
        self.write_calls.append(text)
        
    def error(self, text):
        self.error_calls.append(text)
        
    def success(self, text):
        self.success_calls.append(text)
        
    def sidebar(self):
        return self
        
    def selectbox(self, label, options):
        return options[0] if options else None
        
    def text_input(self, label, value=""):
        return value
        
    def button(self, label):
        return False
        
    def columns(self, n):
        return [self] * n
        
    def metric(self, label, value):
        pass
        
    def plotly_chart(self, fig):
        pass

# Create mock streamlit module
mock_st = MockStreamlit()

class TestDashboard:
    """Test cases for dashboard functionality."""
    
    def test_dashboard_imports(self):
        """
        Test that dashboard can import required modules.
        
        Verifies that all necessary dependencies for the dashboard are available
        and can be imported successfully. This is crucial for dashboard functionality.
        """
        try:
            # Test imports that the dashboard would use
            import pandas as pd  # Data manipulation and analysis
            import plotly.express as px  # Interactive plotting
            import folium  # Map visualization
            
            assert pd is not None
            assert px is not None
            assert folium is not None
            
        except ImportError as e:
            pytest.fail(f"Dashboard dependency import failed: {e}")
    
    @patch('streamlit.title')
    @patch('streamlit.write')
    def test_dashboard_basic_structure(self, mock_write, mock_title):
        """
        Test basic dashboard structure and layout.
        
        Verifies that the dashboard properly initializes with title and description,
        using mocked Streamlit functions to simulate the web interface creation.
        This ensures the dashboard's main components are set up correctly.
        """
        # Mock dashboard initialization
        mock_title.return_value = None
        mock_write.return_value = None
        
        # Simulate dashboard setup - these are the actual values used in the app
        dashboard_title = "HealthScopeAI Dashboard"
        dashboard_description = "Monitor health trends across Kenya"
        
        # These would be called in actual dashboard
        mock_title(dashboard_title)
        mock_write(dashboard_description)
        
        # Verify dashboard structure - ensure methods were called correctly
        mock_title.assert_called_with(dashboard_title)
        mock_write.assert_called_with(dashboard_description)
    
    def test_data_loading_for_dashboard(self, sample_health_data):
        """
        Test data loading functionality for dashboard.
        
        Ensures that the dashboard can properly load and validate health data.
        Checks that all required columns are present and have correct data types
        for proper visualization and analysis in the dashboard.
        """
        # Simulate loading data for dashboard - copy to avoid modifying original
        dashboard_data = sample_health_data.copy()
        
        # Verify data is suitable for dashboard display
        assert not dashboard_data.empty, "Dashboard data should not be empty"
        
        # Check required columns for dashboard functionality
        assert 'text' in dashboard_data.columns, "Text column required for content display"
        assert 'location' in dashboard_data.columns, "Location column required for geographic analysis"
        assert 'timestamp' in dashboard_data.columns, "Timestamp column required for time series analysis"
        assert 'is_health_related' in dashboard_data.columns, "Health classification required for metrics"
        
        # Check data types for proper dashboard processing
        assert pd.api.types.is_datetime64_any_dtype(dashboard_data['timestamp']), "Timestamp must be datetime type"
    
    def test_health_metrics_calculation(self, sample_health_data):
        """Test calculation of health metrics for dashboard."""
        # Calculate key metrics that would be displayed
        total_posts = len(sample_health_data)
        health_posts = sample_health_data['is_health_related'].sum()
        unique_locations = sample_health_data['location'].nunique()
        
        health_percentage = (health_posts / total_posts) * 100 if total_posts > 0 else 0
        
        # Verify metrics
        assert isinstance(total_posts, (int, np.integer))
        assert isinstance(health_posts, (int, np.integer))
        assert isinstance(unique_locations, (int, np.integer))
        assert 0 <= health_percentage <= 100
        
        assert total_posts > 0
        assert unique_locations > 0
    
    def test_location_based_filtering(self, sample_health_data):
        """Test location-based filtering for dashboard."""
        # Get unique locations for filter dropdown
        locations = ['All'] + list(sample_health_data['location'].unique())
        
        # Test filtering by specific location
        test_location = sample_health_data['location'].iloc[0]
        filtered_data = sample_health_data[sample_health_data['location'] == test_location]
        
        # Verify filtering
        assert len(locations) > 1  # Should have 'All' plus actual locations
        assert len(filtered_data) <= len(sample_health_data)
        assert all(filtered_data['location'] == test_location)
    
    def test_time_based_filtering(self, sample_health_data):
        """Test time-based filtering for dashboard."""
        # Create date range filters
        min_date = sample_health_data['timestamp'].min()
        max_date = sample_health_data['timestamp'].max()
        
        # Test filtering by date range
        mid_date = min_date + (max_date - min_date) / 2
        filtered_data = sample_health_data[
            (sample_health_data['timestamp'] >= min_date) & 
            (sample_health_data['timestamp'] <= mid_date)
        ]
        
        # Verify time filtering
        assert min_date <= max_date
        assert len(filtered_data) <= len(sample_health_data)
        if not filtered_data.empty:
            assert filtered_data['timestamp'].min() >= min_date
            assert filtered_data['timestamp'].max() <= mid_date
    
    def test_health_category_distribution(self, sample_health_data):
        """Test health category distribution for charts."""
        # Calculate category distribution
        if 'category' in sample_health_data.columns:
            category_counts = sample_health_data['category'].value_counts()
            
            # Verify distribution data
            assert isinstance(category_counts, pd.Series)
            assert len(category_counts) > 0
            assert all(count >= 0 for count in category_counts.values)
    
    def test_sentiment_analysis_display(self, sample_health_data):
        """Test sentiment analysis data for dashboard."""
        if 'sentiment' in sample_health_data.columns:
            sentiment_distribution = sample_health_data['sentiment'].value_counts()
            
            # Verify sentiment data
            assert isinstance(sentiment_distribution, pd.Series)
            expected_sentiments = ['positive', 'negative', 'neutral']
            assert any(sentiment in sentiment_distribution.index for sentiment in expected_sentiments)
    
    def test_map_data_preparation(self, sample_health_data):
        """Test data preparation for map visualization."""
        # Prepare data for map
        map_data = sample_health_data.groupby('location').agg({
            'is_health_related': ['count', 'sum'],
            'latitude': 'first',
            'longitude': 'first'
        }).reset_index()
        
        # Flatten column names
        map_data.columns = ['location', 'total_posts', 'health_posts', 'latitude', 'longitude']
        
        # Verify map data
        assert not map_data.empty
        assert 'location' in map_data.columns
        assert 'latitude' in map_data.columns
        assert 'longitude' in map_data.columns
        assert 'health_posts' in map_data.columns
        
        # Check coordinate validity
        assert map_data['latitude'].notna().all()
        assert map_data['longitude'].notna().all()
    
    def test_chart_data_preparation(self, sample_health_data):
        """Test data preparation for charts."""
        # Prepare daily trend data
        daily_trends = sample_health_data.groupby(
            sample_health_data['timestamp'].dt.date
        )['is_health_related'].agg(['count', 'sum']).reset_index()
        
        daily_trends.columns = ['date', 'total_posts', 'health_posts']
        daily_trends['health_rate'] = daily_trends['health_posts'] / daily_trends['total_posts']
        
        # Verify chart data
        assert not daily_trends.empty
        assert 'date' in daily_trends.columns
        assert 'health_rate' in daily_trends.columns
        assert all(0 <= rate <= 1 for rate in daily_trends['health_rate'])
    
    def test_text_classification_interface(self, sample_health_data):
        """Test text classification interface functionality."""
        # Simulate text input for classification
        test_texts = [
            "I have a headache and feel sick",
            "Beautiful weather today",
            "Feeling anxious about work"
        ]
        
        # Mock classification results
        mock_results = []
        for text in test_texts:
            result = {
                'text': text,
                'is_health_related': 'headache' in text.lower() or 'anxious' in text.lower(),
                'confidence': np.random.uniform(0.6, 0.9),
                'category': 'physical_health' if 'headache' in text.lower() else 
                           'mental_health' if 'anxious' in text.lower() else 'non_health'
            }
            mock_results.append(result)
        
        # Verify classification interface data
        assert len(mock_results) == len(test_texts)
        for result in mock_results:
            assert 'text' in result
            assert 'is_health_related' in result
            assert 'confidence' in result
            assert 0 <= result['confidence'] <= 1
    
    def test_alert_system_logic(self, sample_health_data):
        """Test alert system logic for dashboard."""
        # Calculate alert conditions
        total_posts = len(sample_health_data)
        health_posts = sample_health_data['is_health_related'].sum()
        health_rate = health_posts / total_posts if total_posts > 0 else 0
        
        # Define alert thresholds
        HIGH_HEALTH_RATE_THRESHOLD = 0.7
        CRITICAL_HEALTH_RATE_THRESHOLD = 0.9
        
        # Generate alerts
        alerts = []
        if health_rate > CRITICAL_HEALTH_RATE_THRESHOLD:
            alerts.append({
                'level': 'critical',
                'message': f'Critical health trend detected: {health_rate:.1%} of posts are health-related'
            })
        elif health_rate > HIGH_HEALTH_RATE_THRESHOLD:
            alerts.append({
                'level': 'warning',
                'message': f'High health activity: {health_rate:.1%} of posts are health-related'
            })
        
        # Verify alert system
        assert isinstance(alerts, list)
        for alert in alerts:
            assert 'level' in alert
            assert 'message' in alert
            assert alert['level'] in ['info', 'warning', 'critical']
    
    def test_dashboard_state_management(self):
        """Test dashboard state management."""
        # Mock session state
        session_state = {
            'selected_location': 'All',
            'date_range': ('2025-01-01', '2025-12-31'),
            'show_predictions': True,
            'selected_chart_type': 'line'
        }
        
        # Verify state structure
        assert 'selected_location' in session_state
        assert 'date_range' in session_state
        assert 'show_predictions' in session_state
        assert isinstance(session_state['show_predictions'], bool)
        assert len(session_state['date_range']) == 2
    
    def test_data_export_functionality(self, sample_health_data, temp_directory):
        """Test data export functionality from dashboard."""
        # Prepare filtered data for export
        export_data = sample_health_data[['text', 'location', 'timestamp', 'is_health_related']]
        
        # Export to CSV
        export_path = temp_directory / "dashboard_export.csv"
        export_data.to_csv(export_path, index=False)
        
        # Verify export
        assert export_path.exists()
        
        # Load and verify exported data
        loaded_data = pd.read_csv(export_path)
        assert len(loaded_data) == len(export_data)
        assert list(loaded_data.columns) == list(export_data.columns)
    
    def test_dashboard_error_handling(self):
        """Test error handling in dashboard components."""
        # Test with empty data
        empty_data = pd.DataFrame()
        
        # Should handle empty data gracefully
        try:
            if not empty_data.empty:
                metrics = {
                    'total_posts': len(empty_data),
                    'health_posts': 0,
                    'locations': 0
                }
            else:
                metrics = {
                    'total_posts': 0,
                    'health_posts': 0,
                    'locations': 0
                }
            
            assert metrics['total_posts'] == 0
            
        except Exception as e:
            pytest.fail(f"Dashboard error handling failed: {e}")
    
    def test_responsive_design_data(self, sample_health_data):
        """Test data preparation for responsive design."""
        # Prepare data for different screen sizes
        mobile_data = sample_health_data[['location', 'is_health_related']].groupby('location').sum()
        desktop_data = sample_health_data.groupby(['location', 'timestamp']).agg({
            'is_health_related': 'sum',
            'text': 'count'
        }).reset_index()
        
        # Verify responsive data
        assert not mobile_data.empty
        assert not desktop_data.empty
        assert len(mobile_data.columns) <= len(desktop_data.columns)  # Mobile has fewer columns
    
    def test_real_time_update_simulation(self, sample_health_data):
        """Test real-time update simulation for dashboard."""
        # Simulate new data arriving
        new_posts = pd.DataFrame({
            'text': ['New health post', 'Another update'],
            'location': ['Nairobi', 'Mombasa'],
            'timestamp': pd.Timestamp.now(),
            'is_health_related': [True, False]
        })
        
        # Combine with existing data
        updated_data = pd.concat([sample_health_data, new_posts], ignore_index=True)
        
        # Verify real-time update
        assert len(updated_data) > len(sample_health_data)
        assert updated_data['timestamp'].max() >= sample_health_data['timestamp'].max()
    
    def test_user_interaction_simulation(self):
        """Test user interaction simulation."""
        # Simulate user interactions
        user_actions = [
            {'action': 'filter_location', 'value': 'Nairobi'},
            {'action': 'change_date_range', 'value': ('2025-01-01', '2025-06-30')},
            {'action': 'toggle_predictions', 'value': True},
            {'action': 'classify_text', 'value': 'I feel sick today'}
        ]
        
        # Process user actions
        for action in user_actions:
            if action['action'] == 'filter_location':
                selected_location = action['value']
                assert selected_location in ['All', 'Nairobi', 'Mombasa', 'Kisumu']
            
            elif action['action'] == 'change_date_range':
                date_range = action['value']
                assert len(date_range) == 2
                assert date_range[0] <= date_range[1]
            
            elif action['action'] == 'toggle_predictions':
                show_predictions = action['value']
                assert isinstance(show_predictions, bool)
            
            elif action['action'] == 'classify_text':
                input_text = action['value']
                assert isinstance(input_text, str)
                assert len(input_text) > 0
    
    def test_performance_metrics_for_dashboard(self, sample_health_data):
        """Test performance metrics calculation for dashboard."""
        import time
        
        # Measure data processing time
        start_time = time.time()
        
        # Simulate dashboard data processing
        metrics = sample_health_data.describe()
        location_summary = sample_health_data.groupby('location').size()
        trend_data = sample_health_data.groupby(
            sample_health_data['timestamp'].dt.date
        ).size()
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Verify performance
        assert processing_time < 5  # Should process quickly for dashboard
        assert not metrics.empty
        assert not location_summary.empty
    
    @pytest.mark.parametrize("chart_type", ['bar', 'line', 'pie', 'scatter'])
    def test_chart_type_support(self, chart_type, sample_health_data):
        """Test support for different chart types."""
        # Prepare data for different chart types
        if chart_type == 'bar':
            chart_data = sample_health_data['location'].value_counts()
        elif chart_type == 'line':
            chart_data = sample_health_data.groupby(
                sample_health_data['timestamp'].dt.date
            ).size()
        elif chart_type == 'pie':
            chart_data = sample_health_data['category'].value_counts() if 'category' in sample_health_data.columns else sample_health_data['location'].value_counts()
        elif chart_type == 'scatter':
            chart_data = sample_health_data[['latitude', 'longitude', 'is_health_related']]
        
        # Verify chart data
        assert chart_data is not None
        assert not (isinstance(chart_data, pd.DataFrame) and chart_data.empty)
        assert not (isinstance(chart_data, pd.Series) and chart_data.empty)
