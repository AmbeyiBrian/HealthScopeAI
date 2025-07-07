"""
Data Collection Module for HealthScopeAI
Handles data collection from various sources including social media APIs and datasets.
"""

import pandas as pd
import numpy as np
import requests
import json
import time
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollector:
    """
    Main class for collecting health-related social media data from various sources.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the DataCollector.
        
        Args:
            config_path: Path to configuration file containing API keys
        """
        self.config = self._load_config(config_path)
        self.data_dir = Path("data/raw")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or environment variables."""
        config = {}
        
        # Try to load from file first
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        
        # Override with environment variables if available
        config.update({
            'twitter_bearer_token': os.getenv('TWITTER_BEARER_TOKEN'),
            'reddit_client_id': os.getenv('REDDIT_CLIENT_ID'),
            'reddit_client_secret': os.getenv('REDDIT_CLIENT_SECRET'),
            'reddit_user_agent': os.getenv('REDDIT_USER_AGENT', 'HealthScopeAI/1.0')
        })
        
        return config
    
    def collect_kaggle_datasets(self, dataset_urls: List[str]) -> pd.DataFrame:
        """
        Collect data from Kaggle datasets.
        
        Args:
            dataset_urls: List of Kaggle dataset URLs or identifiers
            
        Returns:
            Combined DataFrame from all datasets
        """
        logger.info("Collecting data from Kaggle datasets...")
        
        # For now, return sample data - replace with actual Kaggle API calls
        sample_data = self._generate_sample_data()
        
        # Save raw data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.data_dir / f"kaggle_data_{timestamp}.csv"
        sample_data.to_csv(output_path, index=False)
        logger.info(f"Saved Kaggle data to {output_path}")
        
        return sample_data
    
    def collect_twitter_data(self, queries: List[str], max_tweets: int = 1000) -> pd.DataFrame:
        """
        Collect tweets using Twitter API v2.
        
        Args:
            queries: List of search queries
            max_tweets: Maximum number of tweets to collect
            
        Returns:
            DataFrame with tweet data
        """
        logger.info("Collecting Twitter data...")
        
        if not self.config.get('twitter_bearer_token'):
            logger.warning("Twitter Bearer Token not found. Using sample data.")
            return self._generate_sample_twitter_data()
        
        # TODO: Implement actual Twitter API calls
        # For now, return sample data
        sample_data = self._generate_sample_twitter_data()
        
        # Save raw data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.data_dir / f"twitter_data_{timestamp}.csv"
        sample_data.to_csv(output_path, index=False)
        logger.info(f"Saved Twitter data to {output_path}")
        
        return sample_data
    
    def collect_reddit_data(self, subreddits: List[str], max_posts: int = 500) -> pd.DataFrame:
        """
        Collect Reddit posts from health-related subreddits.
        
        Args:
            subreddits: List of subreddit names
            max_posts: Maximum number of posts to collect
            
        Returns:
            DataFrame with Reddit post data
        """
        logger.info("Collecting Reddit data...")
        
        # TODO: Implement actual Reddit API calls using PRAW
        # For now, return sample data
        sample_data = self._generate_sample_reddit_data()
        
        # Save raw data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.data_dir / f"reddit_data_{timestamp}.csv"
        sample_data.to_csv(output_path, index=False)
        logger.info(f"Saved Reddit data to {output_path}")
        
        return sample_data
    
    def _generate_sample_data(self) -> pd.DataFrame:
        """Generate sample health-related social media data for testing."""
        
        # Health-related keywords and phrases
        health_texts = [
            "I've been feeling really anxious lately, can't sleep at night",
            "Got diagnosed with flu today, feeling terrible",
            "Mental health awareness is so important in our community",
            "Headache for three days straight, need to see a doctor",
            "Depression is real, seeking help should not be stigmatized",
            "Experiencing chest pain, should I go to the hospital?",
            "Feeling overwhelmed with work stress and pressure",
            "Stomach issues for the past week, very concerning",
            "Panic attacks have been getting worse recently",
            "Can't concentrate, feeling burnt out from everything",
            "Nairobi hospitals are overwhelmed with flu cases",
            "Mental health support groups in Kisumu are helpful",
            "Mombasa residents reporting high stress levels",
            "Nakuru county health officials urge caution",
            "Eldoret medical facilities seeing increase in anxiety cases"
        ]
        
        non_health_texts = [
            "Beautiful sunset today in Nairobi",
            "Traffic is terrible on Waiyaki Way",
            "Great football match last night",
            "Planning a trip to Maasai Mara",
            "New restaurant opened in Westlands",
            "Weather is perfect for outdoor activities",
            "Excited about the new movie release",
            "Working from home has its benefits",
            "Weekend plans include visiting family",
            "Love the new music from local artists"
        ]
        
        # Kenyan locations
        locations = [
            "Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret",
            "Thika", "Malindi", "Kitale", "Garissa", "Kakamega",
            "Nyeri", "Machakos", "Meru", "Embu", "Lamu"
        ]
        
        # Generate sample data
        np.random.seed(42)
        n_samples = 1000
        
        # Mix health and non-health texts
        texts = []
        labels = []
        
        for i in range(n_samples):
            if i < 600:  # 60% health-related
                text = np.random.choice(health_texts)
                label = 1
            else:  # 40% non-health
                text = np.random.choice(non_health_texts)
                label = 0
            
            texts.append(text)
            labels.append(label)
        
        # Create DataFrame
        data = pd.DataFrame({
            'text': texts,
            'label': labels,
            'location': np.random.choice(locations, n_samples),
            'timestamp': pd.date_range(
                start='2024-01-01', 
                periods=n_samples, 
                freq='H'
            ),
            'source': 'sample_data'
        })
        
        return data
    
    def _generate_sample_twitter_data(self) -> pd.DataFrame:
        """Generate sample Twitter data."""
        sample_data = self._generate_sample_data()
        sample_data['source'] = 'twitter'
        sample_data['platform'] = 'twitter'
        sample_data['username'] = [f"user_{i}" for i in range(len(sample_data))]
        return sample_data
    
    def _generate_sample_reddit_data(self) -> pd.DataFrame:
        """Generate sample Reddit data."""
        sample_data = self._generate_sample_data()
        sample_data['source'] = 'reddit'
        sample_data['platform'] = 'reddit'
        sample_data['subreddit'] = np.random.choice(
            ['mentalhealth', 'depression', 'anxiety', 'kenya', 'nairobi'], 
            len(sample_data)
        )
        return sample_data
    
    def combine_all_data(self) -> pd.DataFrame:
        """
        Combine data from all sources into a single DataFrame.
        
        Returns:
            Combined DataFrame with all collected data
        """
        logger.info("Combining data from all sources...")
        
        # Collect from all sources
        kaggle_data = self.collect_kaggle_datasets([])
        twitter_data = self.collect_twitter_data(["health Kenya", "mental health"])
        reddit_data = self.collect_reddit_data(["mentalhealth", "depression"])
        
        # Combine all data
        combined_data = pd.concat([
            kaggle_data, 
            twitter_data, 
            reddit_data
        ], ignore_index=True)
        
        # Remove duplicates
        combined_data = combined_data.drop_duplicates(subset=['text'])
        
        # Save combined data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.data_dir / f"combined_data_{timestamp}.csv"
        combined_data.to_csv(output_path, index=False)
        logger.info(f"Saved combined data to {output_path}")
        
        return combined_data

def main():
    """Main function to run data collection."""
    collector = DataCollector()
    
    # Collect data from all sources
    combined_data = collector.combine_all_data()
    
    print(f"Data collection complete!")
    print(f"Total samples collected: {len(combined_data)}")
    print(f"Health-related samples: {len(combined_data[combined_data['label'] == 1])}")
    print(f"Non-health samples: {len(combined_data[combined_data['label'] == 0])}")
    print(f"Unique locations: {combined_data['location'].nunique()}")
    
    # Display sample data
    print("\nSample data:")
    print(combined_data.head())

if __name__ == "__main__":
    main()
