"""
Test configuration and fixtures for HealthScopeAI test suite.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

@pytest.fixture
def sample_health_data():
    """Create sample health data for testing."""
    return pd.DataFrame({
        'text': [
            "I've been feeling really anxious lately",
            "Having flu symptoms today",
            "Beautiful weather in Nairobi",
            "Experiencing chest pain and difficulty breathing",
            "Mental health awareness is important",
            "Just finished a great workout",
            "Dealing with depression and stress",
            "Hospital visit for checkup"
        ],
        'timestamp': pd.date_range('2025-01-01', periods=8, freq='D'),
        'location': ['Nairobi', 'Mombasa', 'Kisumu', 'Nairobi', 'Eldoret', 'Thika', 'Nakuru', 'Mombasa'],
        'source': ['twitter', 'reddit', 'twitter', 'survey', 'news', 'twitter', 'reddit', 'survey'],
        'is_health_related': [True, True, False, True, True, False, True, True],
        'category': ['mental_health', 'physical_health', 'non_health', 'physical_health', 
                    'mental_health', 'non_health', 'mental_health', 'physical_health'],
        'sentiment': ['negative', 'negative', 'positive', 'negative', 'neutral', 'positive', 'negative', 'neutral'],
        'severity': [3, 5, 0, 8, 2, 0, 6, 3],
        'latitude': [-1.286, -4.043, -0.091, -1.286, 0.514, -1.033, -0.303, -4.043],
        'longitude': [36.818, 39.658, 34.768, 36.818, 35.270, 37.069, 36.066, 39.658]
    })

@pytest.fixture
def sample_raw_texts():
    """Sample raw text data for preprocessing tests."""
    return [
        "I'm feeling so stressed about work and life 😔 #mentalhealth",
        "Been having fever and headaches for 3 days now... anyone else?",
        "LOVE THIS BEAUTIFUL DAY IN NAIROBI!!! ❤️🌞",
        "Chest pain is getting worse, might need to see a doctor",
        "@someone check this link: https://example.com/health-tips",
        "Hii homa imeanza tena... siwezi kupata dawa 😢",  # Swahili
        "Just finished my morning run! Feeling great 💪",
        "Depression is real guys, let's talk about it more"
    ]

@pytest.fixture
def temp_directory():
    """Create a temporary directory for file tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture
def mock_model_data():
    """Mock model training data."""
    np.random.seed(42)
    X = np.random.rand(100, 10)  # 100 samples, 10 features
    y = np.random.randint(0, 2, 100)  # Binary classification
    return X, y

@pytest.fixture
def sample_geo_data():
    """Sample geospatial data for testing."""
    return {
        'Nairobi': {'lat': -1.286, 'lon': 36.818, 'health_count': 15, 'mental_health': 8, 'physical_health': 7},
        'Mombasa': {'lat': -4.043, 'lon': 39.658, 'health_count': 12, 'mental_health': 5, 'physical_health': 7},
        'Kisumu': {'lat': -0.091, 'lon': 34.768, 'health_count': 8, 'mental_health': 3, 'physical_health': 5},
        'Eldoret': {'lat': 0.514, 'lon': 35.270, 'health_count': 6, 'mental_health': 2, 'physical_health': 4}
    }

@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'model': {
            'type': 'logistic_regression',
            'features': 'tfidf',
            'test_size': 0.2,
            'random_state': 42
        },
        'data': {
            'min_text_length': 10,
            'max_features': 5000,
            'remove_stopwords': True
        },
        'geo': {
            'default_location': 'Unknown',
            'coordinate_precision': 3
        }
    }
