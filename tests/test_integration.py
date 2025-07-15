"""
Integration tests for HealthScopeAI components.

This test suite validates the end-to-end integration of all HealthScopeAI components,
ensuring they work together seamlessly. It tests the complete pipeline from data
collection through preprocessing, model training, prediction, and visualization.

Test Coverage:
- Complete data pipeline (collection → preprocessing → model → visualization)
- Component integration and data flow validation
- End-to-end performance and accuracy testing
- Error propagation and handling across modules
- Data consistency throughout the pipeline
- Real-world workflow simulation
- System reliability and robustness
- Cross-module compatibility validation
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
import sys
import os
import joblib

# Import modules to test integration
from data_collection import DataCollector
from preprocessing import DataPreprocessor
from model import HealthClassifier
from geo_analysis import GeoAnalyzer

class TestIntegration:
    """
    Integration tests for the complete HealthScopeAI pipeline.
    
    These tests validate that all system components work together correctly
    to provide accurate health trend analysis and visualization capabilities.
    """
    
    def test_data_collection_to_preprocessing_pipeline(self):
        """
        Test integration between data collection and preprocessing.
        
        Validates that raw data collected from various sources can be properly
        processed and cleaned by the preprocessing module without data loss
        or corruption.
        """
        # Collect synthetic data to simulate real data collection
        collector = DataCollector()
        raw_data = collector.generate_synthetic_data(num_samples=20)
        
        # Preprocess the collected data
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.process_dataframe(raw_data, text_column='text')
        
        # Verify integration works correctly
        assert len(processed_data) == len(raw_data), "Data loss during preprocessing"
        assert 'cleaned_text' in processed_data.columns, "Cleaning step not applied"
        assert 'predicted_category' in processed_data.columns, "Categorization not performed"
        
        # Check data quality after processing
        assert not processed_data['cleaned_text'].isnull().any(), "Null values in cleaned text"
    
    def test_preprocessing_to_model_training_pipeline(self):
        """
        Test integration between preprocessing and model training.
        
        Ensures that preprocessed data can be successfully used to train
        the machine learning model with proper feature extraction and
        target variable preparation.
        """
        # Generate and preprocess data for model training
        collector = DataCollector()
        raw_data = collector.generate_synthetic_data(num_samples=50)
        
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.process_dataframe(raw_data, text_column='text')
        
        # Train model on processed data
        classifier = HealthClassifier()
        metrics = classifier.train(
            processed_data,
            text_column='cleaned_text',
            target_column='is_health_related'
        )
        
        # Verify integration
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert classifier.model is not None
        assert classifier.vectorizer is not None
        
        # Test prediction on new data
        test_text = "I feel sick and have a fever"
        result = classifier.predict(test_text)
        assert isinstance(result, dict)
        assert 'prediction' in result
    
    def test_model_to_geo_analysis_pipeline(self):
        """Test integration between model predictions and geo analysis."""
        # Prepare data with geographic information
        collector = DataCollector()
        data_with_geo = collector.generate_synthetic_data(num_samples=30)
        
        # Train model
        classifier = HealthClassifier()
        classifier.train(
            data_with_geo,
            text_column='text',
            target_column='is_health_related'
        )
        
        # Add model predictions
        predictions = []
        for text in data_with_geo['text']:
            pred = classifier.predict(text)
            predictions.append(pred['prediction'])
        
        data_with_geo['model_prediction'] = predictions
        
        # Perform geo analysis
        geo_analyzer = GeoAnalyzer()
        aggregated = geo_analyzer.aggregate_health_data(data_with_geo)
        
        # Verify integration
        assert isinstance(aggregated, dict)
        assert len(aggregated) > 0
        
        # Create map
        health_map = geo_analyzer.create_choropleth_map(aggregated)
        assert health_map is not None
    
    def test_complete_pipeline_end_to_end(self, temp_directory):
        """Test complete end-to-end pipeline."""
        # Step 1: Data Collection
        collector = DataCollector()
        raw_data = collector.generate_synthetic_data(num_samples=40)
        
        # Save raw data
        raw_file = temp_directory / "raw_data.csv"
        raw_data.to_csv(raw_file, index=False)
        
        # Step 2: Data Preprocessing
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.process_dataframe(raw_data, text_column='text')
        
        # Save processed data
        processed_file = temp_directory / "processed_data.csv"
        processed_data.to_csv(processed_file, index=False)
        
        # Step 3: Model Training
        classifier = HealthClassifier()
        training_metrics = classifier.train(
            processed_data,
            text_column='cleaned_text',
            target_column='is_health_related'
        )
        
        # Save model
        model_file = temp_directory / "trained_model.joblib"
        classifier.save_model(str(model_file))
        
        # Step 4: Model Evaluation
        test_texts = [
            "I have a headache and feel sick",
            "Beautiful weather in Nairobi today",
            "Feeling very anxious about work"
        ]
        
        predictions = []
        for text in test_texts:
            pred = classifier.predict(text)
            predictions.append(pred)
        
        # Step 5: Geo Analysis
        geo_analyzer = GeoAnalyzer()
        
        # Add predictions to processed data for geo analysis
        prediction_labels = [classifier.predict(text)['prediction'] for text in processed_data['text']]
        processed_data['final_prediction'] = prediction_labels
        
        # Aggregate by location
        geo_aggregated = geo_analyzer.aggregate_health_data(processed_data)
        
        # Generate report
        health_report = geo_analyzer.generate_health_report(processed_data)
        
        # Verify complete pipeline
        assert training_metrics['accuracy'] >= 0
        assert len(predictions) == 3
        assert all('prediction' in pred for pred in predictions)
        assert isinstance(geo_aggregated, dict)
        assert isinstance(health_report, dict)
        assert model_file.exists()
        assert processed_file.exists()
    
    def test_model_persistence_and_reload(self, temp_directory):
        """Test model saving and loading across sessions."""
        # Train and save model
        collector = DataCollector()
        training_data = collector.generate_synthetic_data(num_samples=30)
        
        classifier = HealthClassifier()
        original_metrics = classifier.train(
            training_data,
            text_column='text',
            target_column='is_health_related'
        )
        
        # Make prediction with original model
        test_text = "I feel depressed and anxious"
        original_prediction = classifier.predict(test_text)
        
        # Save model
        model_path = temp_directory / "persistent_model.joblib"
        classifier.save_model(str(model_path))
        
        # Create new classifier and load model
        new_classifier = HealthClassifier()
        new_classifier.load_model(str(model_path))
        
        # Make prediction with loaded model
        loaded_prediction = new_classifier.predict(test_text)
        
        # Verify consistency
        assert loaded_prediction['prediction'] == original_prediction['prediction']
        assert abs(loaded_prediction['probability'] - original_prediction['probability']) < 0.01
    
    def test_data_quality_through_pipeline(self):
        """Test data quality maintenance through the pipeline."""
        # Start with noisy data
        collector = DataCollector()
        noisy_data = pd.DataFrame({
            'text': [
                "I feel sick 😷 #health @doctor",  # Emojis, hashtags, mentions
                "",  # Empty text
                "FEELING VERY ANXIOUS!!!",  # All caps, punctuation
                "normal health text",
                "http://example.com/health check this out",  # URLs
                None,  # None value
                "a" * 200,  # Very long text
                "hii ni homa kubwa sana"  # Non-English (Swahili)
            ],
            'timestamp': pd.date_range('2025-01-01', periods=8),
            'location': ['Nairobi'] * 8,
            'source': ['twitter'] * 8,
            'is_health_related': [1, 0, 1, 1, 1, 0, 1, 1]
        })
        
        # Preprocess noisy data
        preprocessor = DataPreprocessor()
        clean_data = preprocessor.process_dataframe(noisy_data, text_column='text')
        
        # Remove rows with empty/invalid text
        clean_data = clean_data[clean_data['cleaned_text'].str.len() > 0]
        
        # Train model on clean data
        if len(clean_data) > 2:  # Need minimum data for training
            classifier = HealthClassifier()
            metrics = classifier.train(
                clean_data,
                text_column='cleaned_text',
                target_column='is_health_related'
            )
            
            # Verify data quality improvements
            assert len(clean_data) <= len(noisy_data)  # Should remove some bad data
            assert not clean_data['cleaned_text'].str.contains('@').any()  # No mentions
            assert not clean_data['cleaned_text'].str.contains('#').any()  # No hashtags
            assert not clean_data['cleaned_text'].str.contains('http').any()  # No URLs
    
    def test_multilingual_support_integration(self):
        """Test multilingual support across components."""
        # Create multilingual data
        multilingual_data = pd.DataFrame({
            'text': [
                "I feel sick and need help",  # English
                "Nina homa na maumivu ya kichwa",  # Swahili
                "Niko stressed sana na hii job",  # Sheng (mixed)
                "Feeling anxious about everything",  # English
                "Nimechoka na hii maisha",  # Swahili
            ],
            'location': ['Nairobi'] * 5,
            'is_health_related': [1, 1, 1, 1, 1]
        })
        
        # Process through pipeline
        preprocessor = DataPreprocessor()
        processed = preprocessor.process_dataframe(multilingual_data, text_column='text')
        
        # Train model
        classifier = HealthClassifier()
        metrics = classifier.train(
            processed,
            text_column='cleaned_text',
            target_column='is_health_related'
        )
        
        # Test predictions on multilingual text
        test_cases = [
            "I have a headache",
            "Nina maumivu ya kichwa",
            "Niko sick sana"
        ]
        
        predictions = [classifier.predict(text) for text in test_cases]
        
        # Verify multilingual handling
        assert all(isinstance(pred, dict) for pred in predictions)
        assert all('prediction' in pred for pred in predictions)
        # Model should handle different languages without crashing
    
    def test_real_time_prediction_simulation(self):
        """Test real-time prediction simulation."""
        # Prepare model
        collector = DataCollector()
        training_data = collector.generate_synthetic_data(num_samples=50)
        
        classifier = HealthClassifier()
        classifier.train(
            training_data,
            text_column='text',
            target_column='is_health_related'
        )
        
        # Simulate real-time data stream
        incoming_texts = [
            "Sudden chest pain, need help",
            "Beautiful morning in Mombasa",
            "Feeling depressed lately",
            "Traffic is terrible today",
            "Have fever and cough symptoms"
        ]
        
        # Process each text as it "arrives"
        real_time_results = []
        for text in incoming_texts:
            # Preprocess
            preprocessor = DataPreprocessor()
            processed_result = preprocessor.process_text(text)
            
            # Predict
            prediction = classifier.predict(text)
            
            # Combine results
            result = {
                'original_text': text,
                'cleaned_text': processed_result['cleaned_text'],
                'is_health_related': prediction['prediction'],
                'confidence': prediction['confidence'],
                'category': processed_result.get('category', 'unknown')
            }
            real_time_results.append(result)
        
        # Verify real-time processing
        assert len(real_time_results) == len(incoming_texts)
        health_count = sum(1 for r in real_time_results if r['is_health_related'])
        assert health_count > 0  # Should detect some health-related content
    
    def test_error_propagation_and_handling(self):
        """Test error handling across pipeline components."""
        # Test with problematic data
        problematic_data = pd.DataFrame({
            'text': [None, "", "normal text", 123, []],
            'location': ['Nairobi', None, 'InvalidCity', 'Mombasa', 'Kisumu'],
            'is_health_related': [1, 0, 1, None, 'invalid']
        })
        
        # Test preprocessing error handling
        preprocessor = DataPreprocessor()
        try:
            processed = preprocessor.process_dataframe(problematic_data, text_column='text')
            # Should handle errors gracefully
            assert isinstance(processed, pd.DataFrame)
        except Exception as e:
            # Should not crash the system
            assert False, f"Preprocessing failed with: {e}"
        
        # Test geo analysis error handling
        geo_analyzer = GeoAnalyzer()
        try:
            result = geo_analyzer.aggregate_health_data(problematic_data)
            assert isinstance(result, dict)
        except Exception as e:
            assert False, f"Geo analysis failed with: {e}"
    
    def test_performance_integration(self):
        """Test performance across integrated components."""
        import time
        
        # Generate larger dataset for performance testing
        collector = DataCollector()
        large_dataset = collector.generate_synthetic_data(num_samples=200)
        
        start_time = time.time()
        
        # Full pipeline performance test
        preprocessor = DataPreprocessor()
        processed = preprocessor.process_dataframe(large_dataset, text_column='text')
        
        classifier = HealthClassifier()
        metrics = classifier.train(
            processed,
            text_column='cleaned_text',
            target_column='is_health_related'
        )
        
        geo_analyzer = GeoAnalyzer()
        aggregated = geo_analyzer.aggregate_health_data(processed)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Performance assertions
        assert total_time < 120  # Should complete in under 2 minutes
        assert len(processed) == len(large_dataset)
        assert isinstance(metrics, dict)
        assert isinstance(aggregated, dict)
    
    def test_data_flow_consistency(self):
        """Test data consistency through the pipeline."""
        # Start with controlled data
        controlled_data = pd.DataFrame({
            'text': [
                'I have anxiety and depression',
                'Feeling sick with fever',
                'Great weather today',
                'Hospital visit needed',
                'Love this music'
            ],
            'location': ['Nairobi', 'Mombasa', 'Kisumu', 'Nairobi', 'Eldoret'],
            'is_health_related': [True, True, False, True, False]
        })
        
        # Track data through pipeline
        original_health_count = controlled_data['is_health_related'].sum()
        
        # Preprocessing
        preprocessor = DataPreprocessor()
        processed = preprocessor.process_dataframe(controlled_data, text_column='text')
        
        # Model training and prediction
        classifier = HealthClassifier()
        classifier.train(processed, text_column='cleaned_text', target_column='is_health_related')
        
        # Verify predictions maintain reasonable consistency
        predicted_labels = [classifier.predict(text)['prediction'] for text in controlled_data['text']]
        predicted_health_count = sum(predicted_labels)
        
        # Should have some correlation with original labels
        # (allowing for model uncertainty on borderline cases)
        consistency_ratio = min(predicted_health_count, original_health_count) / max(predicted_health_count, original_health_count)
        assert consistency_ratio > 0.3  # At least 30% consistency
    
    @pytest.mark.slow
    def test_stress_testing(self):
        """Stress test the integrated system."""
        # Create large dataset
        collector = DataCollector()
        stress_data = collector.generate_synthetic_data(num_samples=500)
        
        try:
            # Run full pipeline under stress
            preprocessor = DataPreprocessor()
            processed = preprocessor.process_dataframe(stress_data, text_column='text')
            
            classifier = HealthClassifier()
            metrics = classifier.train(processed, text_column='cleaned_text', target_column='is_health_related')
            
            # Batch predictions
            batch_texts = processed['text'][:100].tolist()
            batch_predictions = classifier.predict_batch(batch_texts)
            
            geo_analyzer = GeoAnalyzer()
            aggregated = geo_analyzer.aggregate_health_data(processed)
            
            # Verify system stability under load
            assert len(processed) == len(stress_data)
            assert len(batch_predictions) == 100
            assert isinstance(aggregated, dict)
            assert metrics['accuracy'] >= 0
            
        except Exception as e:
            assert False, f"System failed under stress: {e}"
