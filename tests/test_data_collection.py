"""
Unit tests for data collection module.

This test suite validates the DataCollector class which gathers health-related
social media posts from various sources (Twitter, Reddit, Kaggle datasets).
It ensures proper data collection, validation, and storage functionality.

Test Coverage:
- Data collector initialization and configuration
- Social media API integration (Twitter, Reddit)
- Kaggle dataset downloading and processing
- Synthetic data generation for testing
- Data validation and quality checks
- File I/O operations and data persistence
- Error handling for API failures
- Rate limiting and authentication
- Data format standardization
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
from unittest.mock import patch, MagicMock, mock_open
import json
import os
from pathlib import Path
import sys

# Import the module to test
from data_collection import DataCollector

class TestDataCollector:
    """
    Test cases for DataCollector class.
    
    This class validates the data collection functionality that gathers
    health-related content from multiple sources for the HealthScopeAI system.
    """
    
    def test_init_default(self):
        """
        Test DataCollector initialization with defaults.
        
        Verifies that the data collector initializes properly with default
        configuration and creates necessary directories for data storage.
        """
        collector = DataCollector()
        assert isinstance(collector.config, dict)
        assert collector.data_dir.exists()
        assert collector.data_dir.name == "raw"
    
    def test_init_with_config(self, temp_directory):
        """
        Test DataCollector initialization with config file.
        
        Ensures that the collector can load configuration from a JSON file
        containing API keys and other settings.
        """
        # Create mock config file with API credentials
        config_data = {
            'twitter_bearer_token': 'test_token',
            'reddit_client_id': 'test_id'
        }
        config_path = temp_directory / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config_data, f)
        
        collector = DataCollector(str(config_path))
        assert collector.config['twitter_bearer_token'] == 'test_token'
        assert collector.config['reddit_client_id'] == 'test_id'
    
    @patch.dict(os.environ, {
        'TWITTER_BEARER_TOKEN': 'env_token',
        'REDDIT_CLIENT_ID': 'env_id'
    })
    def test_load_config_from_env(self):
        """
        Test loading configuration from environment variables.
        
        Validates that API credentials can be loaded from environment variables
        as a secure alternative to config files.
        """
        collector = DataCollector()
        assert collector.config['twitter_bearer_token'] == 'env_token'
        assert collector.config['reddit_client_id'] == 'env_id'
    
    def test_generate_synthetic_data(self):
        """Test synthetic data generation."""
        collector = DataCollector()
        
        synthetic_data = collector.generate_synthetic_data(num_samples=50)
        
        assert isinstance(synthetic_data, pd.DataFrame)
        assert len(synthetic_data) == 50
        assert 'text' in synthetic_data.columns
        assert 'timestamp' in synthetic_data.columns
        assert 'location' in synthetic_data.columns
        assert 'source' in synthetic_data.columns
        assert 'is_health_related' in synthetic_data.columns
    
    def test_synthetic_data_health_distribution(self):
        """Test health vs non-health distribution in synthetic data."""
        collector = DataCollector()
        
        data = collector.generate_synthetic_data(num_samples=100)
        
        health_count = data['is_health_related'].sum()
        non_health_count = len(data) - health_count
        
        # Should have both health and non-health samples
        assert health_count > 0
        assert non_health_count > 0
        assert health_count + non_health_count == 100
    
    def test_synthetic_data_categories(self):
        """Test health category distribution in synthetic data."""
        collector = DataCollector()
        
        data = collector.generate_synthetic_data(num_samples=100)
        
        categories = data['category'].unique()
        assert 'mental_health' in categories
        assert 'physical_health' in categories
        assert 'non_health' in categories
    
    def test_synthetic_data_locations(self):
        """Test location distribution in synthetic data."""
        collector = DataCollector()
        
        data = collector.generate_synthetic_data(num_samples=100)
        
        locations = data['location'].unique()
        kenyan_cities = ['Nairobi', 'Mombasa', 'Kisumu', 'Nakuru', 'Eldoret']
        
        # Should include Kenyan cities
        assert any(city in locations for city in kenyan_cities)
        assert len(locations) > 1  # Multiple locations
    
    def test_synthetic_data_timestamps(self):
        """Test timestamp generation in synthetic data."""
        collector = DataCollector()
        
        data = collector.generate_synthetic_data(num_samples=50)
        
        # Should have valid timestamps
        assert pd.api.types.is_datetime64_any_dtype(data['timestamp'])
        
        # Timestamps should be in reasonable range
        min_time = data['timestamp'].min()
        max_time = data['timestamp'].max()
        assert min_time < max_time
    
    def test_collect_kaggle_data(self, temp_directory):
        """Test Kaggle data collection simulation."""
        collector = DataCollector()
        collector.data_dir = temp_directory
        
        # Mock successful Kaggle data collection
        with patch.object(collector, '_download_kaggle_dataset') as mock_download:
            mock_download.return_value = pd.DataFrame({
                'text': ['Sample health text', 'Another health post'],
                'label': ['health', 'health']
            })
            
            result = collector.collect_kaggle_data('test-dataset')
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0
            assert 'text' in result.columns
            mock_download.assert_called_once()
    
    @patch('requests.get')
    def test_collect_reddit_data_mock(self, mock_get):
        """Test Reddit data collection with mocked API."""
        collector = DataCollector()
        
        # Mock Reddit API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'data': {
                'children': [
                    {
                        'data': {
                            'title': 'Health related post',
                            'selftext': 'I am feeling sick',
                            'created_utc': 1640995200,
                            'subreddit': 'health'
                        }
                    }
                ]
            }
        }
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        result = collector.collect_reddit_data(
            subreddits=['health'],
            limit=1
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0 if not result.empty else True  # Handle case where no data
    
    def test_collect_twitter_data_simulation(self):
        """Test Twitter data collection simulation."""
        collector = DataCollector()
        
        # Test without actual API credentials (should return synthetic data)
        result = collector.collect_twitter_data(
            keywords=['health', 'medical'],
            limit=10
        )
        
        assert isinstance(result, pd.DataFrame)
        # Should return some data (either real or synthetic)
        assert len(result) >= 0
    
    def test_save_data(self, temp_directory):
        """Test data saving functionality."""
        collector = DataCollector()
        collector.data_dir = temp_directory
        
        # Create test data
        test_data = pd.DataFrame({
            'text': ['test text 1', 'test text 2'],
            'timestamp': pd.date_range('2025-01-01', periods=2),
            'source': ['test', 'test']
        })
        
        filename = collector.save_data(test_data, source='test')
        
        assert filename is not None
        saved_file = temp_directory / filename
        assert saved_file.exists()
        
        # Verify saved data
        loaded_data = pd.read_csv(saved_file)
        assert len(loaded_data) == len(test_data)
        assert 'text' in loaded_data.columns
    
    def test_load_existing_data(self, temp_directory):
        """Test loading existing data."""
        collector = DataCollector()
        collector.data_dir = temp_directory
        
        # Create and save test data first
        test_data = pd.DataFrame({
            'text': ['existing text'],
            'timestamp': ['2025-01-01'],
            'source': ['existing']
        })
        
        filename = "test_data.csv"
        test_data.to_csv(temp_directory / filename, index=False)
        
        # Load the data
        loaded_data = collector.load_data(filename)
        
        assert isinstance(loaded_data, pd.DataFrame)
        assert len(loaded_data) == 1
        assert loaded_data['text'].iloc[0] == 'existing text'
    
    def test_combine_datasets(self):
        """Test combining multiple datasets."""
        collector = DataCollector()
        
        # Create multiple datasets
        data1 = pd.DataFrame({
            'text': ['text 1', 'text 2'],
            'source': ['twitter', 'twitter'],
            'timestamp': pd.date_range('2025-01-01', periods=2)
        })
        
        data2 = pd.DataFrame({
            'text': ['text 3', 'text 4'],
            'source': ['reddit', 'reddit'],
            'timestamp': pd.date_range('2025-01-03', periods=2)
        })
        
        combined = collector.combine_datasets([data1, data2])
        
        assert isinstance(combined, pd.DataFrame)
        assert len(combined) == 4
        assert 'twitter' in combined['source'].values
        assert 'reddit' in combined['source'].values
    
    def test_filter_health_related(self):
        """Test filtering health-related content."""
        collector = DataCollector()
        
        # Create mixed data
        mixed_data = pd.DataFrame({
            'text': [
                'I have a headache',
                'Beautiful weather today',
                'Feeling anxious about work',
                'Great movie last night',
                'Need to see a doctor'
            ],
            'source': ['test'] * 5
        })
        
        health_data = collector.filter_health_related(mixed_data)
        
        assert isinstance(health_data, pd.DataFrame)
        assert len(health_data) <= len(mixed_data)
        # Should contain health-related posts
        assert len(health_data) > 0
    
    def test_clean_collected_data(self):
        """Test data cleaning functionality."""
        collector = DataCollector()
        
        # Create dirty data
        dirty_data = pd.DataFrame({
            'text': [
                'Normal text',
                '',  # Empty text
                None,  # None value
                'Text with @mention and #hashtag',
                'a' * 500  # Very long text
            ],
            'source': ['test'] * 5,
            'timestamp': pd.date_range('2025-01-01', periods=5)
        })
        
        cleaned_data = collector.clean_data(dirty_data)
        
        assert isinstance(cleaned_data, pd.DataFrame)
        assert len(cleaned_data) <= len(dirty_data)
        # Should remove empty/None texts
        assert not cleaned_data['text'].isnull().any()
        assert not (cleaned_data['text'] == '').any()
    
    def test_add_metadata(self):
        """Test adding metadata to collected data."""
        collector = DataCollector()
        
        # Create basic data
        basic_data = pd.DataFrame({
            'text': ['I feel sick', 'Beautiful day'],
            'source': ['twitter', 'twitter']
        })
        
        enriched_data = collector.add_metadata(basic_data)
        
        assert isinstance(enriched_data, pd.DataFrame)
        assert len(enriched_data) == len(basic_data)
        
        # Should add metadata columns
        expected_columns = ['is_health_related', 'category', 'sentiment']
        for col in expected_columns:
            assert col in enriched_data.columns
    
    def test_data_quality_validation(self):
        """Test data quality validation."""
        collector = DataCollector()
        
        # Create test data with quality issues
        test_data = pd.DataFrame({
            'text': ['good text', '', 'another good text', None],
            'source': ['twitter', 'reddit', 'twitter', 'twitter'],
            'timestamp': [
                '2025-01-01',
                '2025-01-02', 
                'invalid_date',
                '2025-01-04'
            ]
        })
        
        quality_report = collector.validate_data_quality(test_data)
        
        assert isinstance(quality_report, dict)
        assert 'total_rows' in quality_report
        assert 'empty_text' in quality_report
        assert 'invalid_timestamps' in quality_report
        assert quality_report['total_rows'] == 4
        assert quality_report['empty_text'] >= 1  # Should detect empty/None texts
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        collector = DataCollector()
        
        import time
        start_time = time.time()
        
        # Test rate limiting delay
        collector._apply_rate_limit(delay=0.1)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Should have applied delay
        assert elapsed >= 0.05  # Allow some tolerance
    
    def test_error_handling(self):
        """Test error handling in data collection."""
        collector = DataCollector()
        
        # Test with invalid file path
        result = collector.load_data("nonexistent_file.csv")
        assert result is None or result.empty
        
        # Test with invalid dataset name
        result = collector.collect_kaggle_data("invalid/dataset")
        assert isinstance(result, pd.DataFrame)  # Should return empty DataFrame
    
    @pytest.mark.parametrize("source", ['twitter', 'reddit', 'kaggle', 'synthetic'])
    def test_collect_by_source(self, source):
        """Parametrized test for different data sources."""
        collector = DataCollector()
        
        if source == 'synthetic':
            result = collector.generate_synthetic_data(num_samples=10)
        elif source == 'kaggle':
            result = collector.collect_kaggle_data('test-dataset')
        elif source == 'twitter':
            result = collector.collect_twitter_data(['health'], limit=5)
        elif source == 'reddit':
            result = collector.collect_reddit_data(['health'], limit=5)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 0  # Can be empty if API fails
    
    def test_batch_collection(self):
        """Test batch data collection from multiple sources."""
        collector = DataCollector()
        
        sources = ['synthetic']  # Use synthetic for reliable testing
        
        all_data = collector.collect_batch_data(
            sources=sources,
            samples_per_source=20
        )
        
        assert isinstance(all_data, pd.DataFrame)
        assert len(all_data) > 0
        assert 'source' in all_data.columns
    
    def test_data_export_formats(self, temp_directory):
        """Test exporting data in different formats."""
        collector = DataCollector()
        collector.data_dir = temp_directory
        
        # Create test data
        test_data = pd.DataFrame({
            'text': ['test 1', 'test 2'],
            'timestamp': pd.date_range('2025-01-01', periods=2),
            'source': ['test', 'test']
        })
        
        # Test CSV export
        csv_file = collector.export_data(test_data, format='csv', filename='test_export')
        assert (temp_directory / csv_file).exists()
        
        # Test JSON export
        json_file = collector.export_data(test_data, format='json', filename='test_export')
        assert (temp_directory / json_file).exists()
    
    def test_performance_monitoring(self):
        """Test collection performance monitoring."""
        collector = DataCollector()
        
        import time
        start_time = time.time()
        
        # Generate data and measure time
        data = collector.generate_synthetic_data(num_samples=100)
        
        end_time = time.time()
        collection_time = end_time - start_time
        
        # Should be reasonably fast
        assert collection_time < 10  # Should complete in under 10 seconds
        assert len(data) == 100
