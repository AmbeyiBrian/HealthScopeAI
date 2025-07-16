"""
Minimal Data Preprocessing Module for HealthScopeAI
Temporary minimal implementation to get the app running.
"""

import pandas as pd
import numpy as np
from typing import Dict, List


class DataPreprocessor:
    """
    Minimal DataPreprocessor class for basic functionality.
    """

    def __init__(self):
        """Initialize the DataPreprocessor."""
        self.is_fitted = False

    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning.
        """
        if not isinstance(text, str):
            return ""
        
        # Basic cleaning
        text = text.lower().strip()
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text

    def extract_health_features(self, text: str) -> Dict[str, float]:
        """
        Extract basic health-related features from text.
        """
        if not isinstance(text, str):
            text = ""
        
        text = text.lower()
        
        # Count health-related keywords
        health_keywords = ["sick", "pain", "fever", "headache", "tired", "stress", "anxiety"]
        symptom_keywords = ["hurt", "ache", "feel", "symptom", "problem"]
        medical_keywords = ["doctor", "hospital", "medicine", "treatment", "diagnosis"]
        
        health_count = sum(1 for keyword in health_keywords if keyword in text)
        symptom_count = sum(1 for keyword in symptom_keywords if keyword in text)
        medical_count = sum(1 for keyword in medical_keywords if keyword in text)
        
        return {
            "health_keyword_count": health_count,
            "symptom_keyword_count": symptom_count,
            "medical_keyword_count": medical_count,
            "text_length": len(text),
            "word_count": len(text.split()),
            "uppercase_ratio": sum(1 for c in text if c.isupper()) / max(len(text), 1),
            "exclamation_count": text.count('!')
        }

    def preprocess_data(self, data: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
        """
        Basic data preprocessing.
        """
        if data is None or data.empty:
            return self._generate_sample_data()
        
        # Basic preprocessing
        if text_column in data.columns:
            data[text_column] = data[text_column].apply(self.clean_text)
        
        self.is_fitted = True
        return data

    def _generate_sample_data(self) -> pd.DataFrame:
        """Generate sample data for demonstration."""
        sample_texts = [
            "I have been feeling very sick lately with fever and headache",
            "The weather is really nice today in Nairobi",
            "My stomach hurts and I feel nauseous",
            "Going to watch a football match this evening",
            "Feeling anxious about my health condition",
            "Beautiful sunset over Lake Victoria",
            "Doctor recommended rest and medication",
            "Traffic is heavy on Uhuru Highway",
            "Having trouble sleeping due to stress",
            "Planning to visit Maasai Mara next month"
        ]
        
        # Generate labels (1 for health-related, 0 for not)
        labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        
        return pd.DataFrame({
            'text': sample_texts,
            'label': labels,
            'is_health_related': labels
        })

    def fit(self, data: pd.DataFrame) -> None:
        """Mock fit method."""
        self.is_fitted = True

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Mock transform method."""
        return self.preprocess_data(data)
