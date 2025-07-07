"""
Data Preprocessing Module for HealthScopeAI
Handles text cleaning, NLP processing, and feature extraction.
"""

import pandas as pd
import numpy as np
import re
import string
from typing import List, Dict, Union, Optional
import logging
from pathlib import Path

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import spacy

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Main class for preprocessing health-related text data.
    """
    
    def __init__(self, language: str = 'en'):
        """
        Initialize the DataPreprocessor.
        
        Args:
            language: Language for preprocessing (default: 'en')
        """
        self.language = language
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Health-related keywords for filtering
        self.health_keywords = {
            'mental_health': [
                'anxiety', 'depression', 'stress', 'panic', 'mental', 'therapy',
                'counseling', 'psychiatrist', 'psychologist', 'mood', 'suicide',
                'bipolar', 'ptsd', 'trauma', 'grief', 'overwhelmed', 'burnout'
            ],
            'physical_health': [
                'pain', 'headache', 'fever', 'flu', 'cold', 'cough', 'sick',
                'hospital', 'doctor', 'medicine', 'treatment', 'symptoms',
                'chest pain', 'stomach', 'nausea', 'dizzy', 'fatigue'
            ],
            'locations': [
                'nairobi', 'mombasa', 'kisumu', 'nakuru', 'eldoret', 'thika',
                'kenya', 'malindi', 'kitale', 'garissa', 'kakamega', 'nyeri'
            ]
        }
        
        # Initialize spaCy model if available
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Using NLTK for preprocessing.")
            self.nlp = None
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text data.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        if self.nlp:
            doc = self.nlp(text)
            tokens = [token.text.lower() for token in doc if not token.is_punct]
        else:
            tokens = word_tokenize(text.lower())
        
        return tokens
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Filtered tokens
        """
        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens to their base form.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Lemmatized tokens
        """
        if self.nlp:
            doc = self.nlp(" ".join(tokens))
            return [token.lemma_ for token in doc]
        else:
            return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def extract_health_features(self, text: str) -> Dict[str, int]:
        """
        Extract health-related features from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of health features
        """
        text_lower = text.lower()
        features = {}
        
        # Count mental health keywords
        mental_count = sum(1 for keyword in self.health_keywords['mental_health'] 
                          if keyword in text_lower)
        features['mental_health_keywords'] = mental_count
        
        # Count physical health keywords
        physical_count = sum(1 for keyword in self.health_keywords['physical_health'] 
                           if keyword in text_lower)
        features['physical_health_keywords'] = physical_count
        
        # Check for location mentions
        location_count = sum(1 for location in self.health_keywords['locations'] 
                           if location in text_lower)
        features['location_keywords'] = location_count
        
        # Text length features
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        
        # Sentiment-related features (basic)
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'pain', 'hurt']
        positive_words = ['good', 'great', 'better', 'healing', 'recover']
        
        features['negative_sentiment'] = sum(1 for word in negative_words 
                                           if word in text_lower)
        features['positive_sentiment'] = sum(1 for word in positive_words 
                                           if word in text_lower)
        
        return features
    
    def process_text(self, text: str, return_features: bool = False) -> Union[str, Dict]:
        """
        Complete text preprocessing pipeline.
        
        Args:
            text: Raw text to process
            return_features: Whether to return extracted features
            
        Returns:
            Processed text or features dictionary
        """
        # Clean text
        cleaned_text = self.clean_text(text)
        
        if return_features:
            # Extract features
            features = self.extract_health_features(cleaned_text)
            return features
        
        # Tokenize
        tokens = self.tokenize_text(cleaned_text)
        
        # Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Lemmatize
        tokens = self.lemmatize_tokens(tokens)
        
        # Join back to text
        processed_text = " ".join(tokens)
        
        return processed_text
    
    def create_tfidf_features(self, texts: List[str], max_features: int = 5000) -> np.ndarray:
        """
        Create TF-IDF features from text data.
        
        Args:
            texts: List of texts to vectorize
            max_features: Maximum number of features
            
        Returns:
            TF-IDF feature matrix
        """
        logger.info(f"Creating TF-IDF features with max_features={max_features}")
        
        # Process texts
        processed_texts = [self.process_text(text) for text in texts]
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # Include bigrams
            min_df=2,  # Minimum document frequency
            max_df=0.8  # Maximum document frequency
        )
        
        # Fit and transform
        tfidf_matrix = vectorizer.fit_transform(processed_texts)
        
        # Store vectorizer for later use
        self.tfidf_vectorizer = vectorizer
        
        return tfidf_matrix.toarray()
    
    def process_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Process a pandas DataFrame containing text data.
        
        Args:
            df: DataFrame to process
            text_column: Name of the text column
            
        Returns:
            Processed DataFrame
        """
        logger.info(f"Processing DataFrame with {len(df)} rows")
        
        # Create a copy
        processed_df = df.copy()
        
        # Clean and process text
        processed_df['cleaned_text'] = processed_df[text_column].apply(self.clean_text)
        processed_df['processed_text'] = processed_df[text_column].apply(self.process_text)
        
        # Extract features
        features_list = []
        for text in processed_df[text_column]:
            features = self.extract_health_features(text)
            features_list.append(features)
        
        # Add features to DataFrame
        features_df = pd.DataFrame(features_list)
        processed_df = pd.concat([processed_df, features_df], axis=1)
        
        # Create binary health indicator
        processed_df['is_health_related'] = (
            (processed_df['mental_health_keywords'] > 0) | 
            (processed_df['physical_health_keywords'] > 0)
        ).astype(int)
        
        return processed_df
    
    def save_processed_data(self, df: pd.DataFrame, filename: str) -> None:
        """
        Save processed data to file.
        
        Args:
            df: DataFrame to save
            filename: Output filename
        """
        output_dir = Path("data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")

def main():
    """Main function to run data preprocessing."""
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load sample data
    data_dir = Path("data/raw")
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        logger.error("No CSV files found in data/raw directory")
        return
    
    # Load the most recent data file
    latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"Loading data from {latest_file}")
    
    df = pd.read_csv(latest_file)
    
    # Process the data
    processed_df = preprocessor.process_dataframe(df)
    
    # Save processed data
    output_filename = f"processed_{latest_file.stem}.csv"
    preprocessor.save_processed_data(processed_df, output_filename)
    
    # Display statistics
    print(f"Data preprocessing complete!")
    print(f"Total samples: {len(processed_df)}")
    print(f"Health-related samples: {processed_df['is_health_related'].sum()}")
    print(f"Average text length: {processed_df['text_length'].mean():.2f}")
    print(f"Average word count: {processed_df['word_count'].mean():.2f}")
    
    # Display sample processed data
    print("\nSample processed data:")
    print(processed_df[['text', 'processed_text', 'is_health_related']].head())

if __name__ == "__main__":
    main()
