"""
Unit tests for model training and prediction module.

This test suite validates the HealthClassifier machine learning model that identifies
health-related content in social media posts. It covers model initialization, training,
prediction, performance evaluation, and persistence functionality.

Test Coverage:
- Model initialization with different algorithms
- Data preparation and preprocessing
- Training with various datasets
- Prediction accuracy and confidence scores
- Model persistence (save/load functionality)
- Cross-validation and performance metrics
- Error handling and edge cases
- Multi-language support validation
"""

import pytest
import pandas as pd
import numpy as np
import joblib
import tempfile
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys
import os

# Import the module to test
from model import HealthClassifier

class TestHealthClassifier:
    """
    Test cases for HealthClassifier class.
    
    This class contains comprehensive tests for the main ML model used in HealthScopeAI
    to classify health-related content from social media posts.
    """
    
    def test_init_default(self):
        """
        Test HealthClassifier initialization with default model.
        
        Verifies that the classifier initializes properly with default settings
        (logistic regression) and that all necessary components are created.
        """
        classifier = HealthClassifier()
        assert classifier.model_type == 'logistic_regression'
        assert classifier.model is not None
        assert classifier.preprocessor is not None
        assert isinstance(classifier.model_metrics, dict)
    
    def test_init_different_models(self):
        """
        Test initialization with different model types.
        
        Ensures that the classifier can be initialized with various ML algorithms
        and that each model type is properly configured.
        """
        models = ['logistic_regression', 'random_forest', 'svm', 'naive_bayes', 'gradient_boosting']
        
        for model_type in models:
            classifier = HealthClassifier(model_type=model_type)
            assert classifier.model_type == model_type
            assert classifier.model is not None
    
    def test_init_invalid_model(self):
        """
        Test initialization with invalid model type.
        
        Verifies that the classifier properly handles invalid model types
        by raising appropriate errors.
        """
        with pytest.raises(ValueError):
            HealthClassifier(model_type='invalid_model')
    
    def test_prepare_data(self, sample_health_data):
        """
        Test data preparation for training.
        
        Validates that the data preparation process correctly extracts features
        and targets from the input DataFrame for model training.
        """
        classifier = HealthClassifier()
        
        X, y = classifier.prepare_data(
            sample_health_data, 
            text_column='text', 
            target_column='is_health_related'
        )
        
        assert X.shape[0] == len(sample_health_data)
        assert len(y) == len(sample_health_data)
        assert X.shape[1] > 0  # Should have features
        assert all(label in [0, 1, True, False] for label in y)
    
    def test_train_model_basic(self, sample_health_data):
        """Test basic model training."""
        classifier = HealthClassifier()
        
        # Train on sample data
        metrics = classifier.train(
            sample_health_data,
            text_column='text',
            target_column='is_health_related',
            test_size=0.3
        )
        
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        
        # Model should be trained
        assert classifier.model is not None
        assert classifier.vectorizer is not None
    
    def test_train_insufficient_data(self):
        """Test training with insufficient data."""
        classifier = HealthClassifier()
        
        # Create minimal dataset
        small_data = pd.DataFrame({
            'text': ['health text', 'non health'],
            'is_health_related': [True, False]
        })
        
        # Should handle small datasets gracefully
        with pytest.warns(UserWarning):
            metrics = classifier.train(small_data, text_column='text', target_column='is_health_related')
        
        assert isinstance(metrics, dict)
    
    def test_predict_single_text(self, sample_health_data):
        """Test prediction on single text."""
        classifier = HealthClassifier()
        
        # Train first
        classifier.train(
            sample_health_data,
            text_column='text',
            target_column='is_health_related'
        )
        
        # Test predictions
        health_text = "I'm feeling anxious and stressed"
        result = classifier.predict(health_text)
        
        assert isinstance(result, dict)
        assert 'prediction' in result
        assert 'probability' in result
        assert 'confidence' in result
        assert isinstance(result['prediction'], (bool, int))
        assert 0 <= result['probability'] <= 1
    
    def test_predict_multiple_texts(self, sample_health_data):
        """Test prediction on multiple texts."""
        classifier = HealthClassifier()
        
        # Train first
        classifier.train(
            sample_health_data,
            text_column='text',
            target_column='is_health_related'
        )
        
        test_texts = [
            "I have a headache",
            "Beautiful weather today",
            "Feeling depressed"
        ]
        
        results = classifier.predict_batch(test_texts)
        
        assert len(results) == len(test_texts)
        assert all('prediction' in result for result in results)
        assert all('probability' in result for result in results)
    
    def test_predict_without_training(self):
        """Test prediction without training should raise error."""
        classifier = HealthClassifier()
        
        with pytest.raises((ValueError, AttributeError)):
            classifier.predict("some text")
    
    def test_evaluate_model(self, sample_health_data):
        """Test model evaluation."""
        classifier = HealthClassifier()
        
        # Prepare test data
        X, y = classifier.prepare_data(
            sample_health_data,
            text_column='text',
            target_column='is_health_related'
        )
        
        # Train model
        classifier.train(
            sample_health_data,
            text_column='text',
            target_column='is_health_related'
        )
        
        # Evaluate
        metrics = classifier.evaluate(X, y)
        
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert all(0 <= score <= 1 for score in metrics.values())
    
    def test_cross_validation(self, sample_health_data):
        """Test cross-validation."""
        classifier = HealthClassifier()
        
        cv_scores = classifier.cross_validate(
            sample_health_data,
            text_column='text',
            target_column='is_health_related',
            cv_folds=3
        )
        
        assert isinstance(cv_scores, dict)
        assert 'mean_accuracy' in cv_scores
        assert 'std_accuracy' in cv_scores
        assert 'scores' in cv_scores
        assert len(cv_scores['scores']) == 3
    
    def test_feature_importance(self, sample_health_data):
        """Test feature importance extraction."""
        # Test with tree-based model for feature importance
        classifier = HealthClassifier(model_type='random_forest')
        
        classifier.train(
            sample_health_data,
            text_column='text',
            target_column='is_health_related'
        )
        
        importance = classifier.get_feature_importance(top_n=10)
        
        assert isinstance(importance, dict)
        assert len(importance) <= 10
        assert all(isinstance(score, (int, float)) for score in importance.values())
    
    def test_save_and_load_model(self, sample_health_data, temp_directory):
        """Test model saving and loading."""
        classifier = HealthClassifier()
        
        # Train model
        classifier.train(
            sample_health_data,
            text_column='text',
            target_column='is_health_related'
        )
        
        # Save model
        model_path = temp_directory / "test_model.joblib"
        classifier.save_model(str(model_path))
        
        assert model_path.exists()
        
        # Load model
        new_classifier = HealthClassifier()
        new_classifier.load_model(str(model_path))
        
        # Test that loaded model works
        result = new_classifier.predict("I feel anxious")
        assert isinstance(result, dict)
        assert 'prediction' in result
    
    def test_hyperparameter_tuning(self, sample_health_data):
        """Test hyperparameter tuning."""
        classifier = HealthClassifier()
        
        # Define parameter grid for logistic regression
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2']
        }
        
        best_params = classifier.tune_hyperparameters(
            sample_health_data,
            text_column='text',
            target_column='is_health_related',
            param_grid=param_grid,
            cv_folds=3
        )
        
        assert isinstance(best_params, dict)
        assert 'C' in best_params
        assert 'penalty' in best_params
    
    def test_model_metrics_tracking(self, sample_health_data):
        """Test model metrics tracking."""
        classifier = HealthClassifier()
        
        metrics = classifier.train(
            sample_health_data,
            text_column='text',
            target_column='is_health_related'
        )
        
        # Check that metrics are stored
        assert len(classifier.model_metrics) > 0
        assert classifier.model_metrics['accuracy'] == metrics['accuracy']
    
    def test_prediction_confidence(self, sample_health_data):
        """Test prediction confidence scoring."""
        classifier = HealthClassifier()
        
        classifier.train(
            sample_health_data,
            text_column='text',
            target_column='is_health_related'
        )
        
        # Test high confidence prediction
        clear_health_text = "I have severe chest pain and need emergency care"
        result = classifier.predict(clear_health_text)
        
        assert result['confidence'] > 0.5  # Should be reasonably confident
        
        # Test low confidence prediction  
        ambiguous_text = "going to the place"
        result = classifier.predict(ambiguous_text)
        
        assert 0 <= result['confidence'] <= 1
    
    def test_multiclass_classification(self):
        """Test multiclass health category classification."""
        # Create multiclass data
        multiclass_data = pd.DataFrame({
            'text': [
                "I feel anxious and depressed",
                "Having fever and headache", 
                "Beautiful weather today",
                "Chest pain and breathing problems",
                "Stressed about work deadline",
                "Going for a walk"
            ],
            'category': [
                'mental_health', 'physical_health', 'non_health',
                'physical_health', 'mental_health', 'non_health'
            ]
        })
        
        classifier = HealthClassifier()
        
        metrics = classifier.train(
            multiclass_data,
            text_column='text',
            target_column='category'
        )
        
        assert isinstance(metrics, dict)
        # Should handle multiclass classification
        result = classifier.predict("I feel stressed")
        assert isinstance(result, dict)
    
    def test_text_preprocessing_integration(self, sample_health_data):
        """Test integration with preprocessing module."""
        classifier = HealthClassifier()
        
        # Train with raw text that needs preprocessing
        noisy_data = sample_health_data.copy()
        noisy_data['text'] = noisy_data['text'].apply(
            lambda x: f"@user {x} #health https://example.com 😷"
        )
        
        # Should handle noisy text through preprocessing
        metrics = classifier.train(
            noisy_data,
            text_column='text',
            target_column='is_health_related'
        )
        
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
    
    @pytest.mark.parametrize("model_type", [
        'logistic_regression', 'random_forest', 'svm', 'naive_bayes'
    ])
    def test_all_model_types(self, model_type, sample_health_data):
        """Parametrized test for all supported model types."""
        classifier = HealthClassifier(model_type=model_type)
        
        metrics = classifier.train(
            sample_health_data,
            text_column='text',
            target_column='is_health_related'
        )
        
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert metrics['accuracy'] >= 0  # Basic sanity check
    
    def test_error_handling(self):
        """Test error handling in model operations."""
        classifier = HealthClassifier()
        
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        with pytest.raises((ValueError, KeyError)):
            classifier.train(empty_df, text_column='text', target_column='target')
        
        # Test with missing columns
        bad_df = pd.DataFrame({'wrong_column': ['text']})
        with pytest.raises(KeyError):
            classifier.train(bad_df, text_column='text', target_column='target')
    
    def test_performance_benchmarks(self, sample_health_data):
        """Test training and prediction performance."""
        classifier = HealthClassifier()
        
        # Expand dataset for performance testing
        large_data = pd.concat([sample_health_data] * 20, ignore_index=True)
        
        import time
        
        # Test training time
        start_time = time.time()
        classifier.train(
            large_data,
            text_column='text',
            target_column='is_health_related'
        )
        training_time = time.time() - start_time
        
        assert training_time < 60  # Should train in under 1 minute
        
        # Test prediction time
        start_time = time.time()
        classifier.predict("I feel sick")
        prediction_time = time.time() - start_time
        
        assert prediction_time < 1  # Should predict in under 1 second
