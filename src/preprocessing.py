"""
Data preprocessing module for fake news detection.
Handles text cleaning, preprocessing, and feature extraction.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class TextPreprocessor:
    """Preprocesses text data for fake news detection."""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """
        Clean and preprocess text data.
        
        Args:
            text (str): Input text to clean
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize_and_stem(self, text):
        """
        Tokenize and stem text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Tokenized and stemmed text
        """
        # Tokenize
        tokens = text.split()
        
        # Remove stopwords and stem
        tokens = [
            self.stemmer.stem(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return ' '.join(tokens)
    
    def preprocess(self, text):
        """
        Complete preprocessing pipeline.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        text = self.clean_text(text)
        text = self.tokenize_and_stem(text)
        return text


class FeatureExtractor:
    """Extracts features from text using TF-IDF."""
    
    def __init__(self, max_features=5000):
        """
        Initialize feature extractor.
        
        Args:
            max_features (int): Maximum number of features to extract
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        self.fitted = False
    
    def fit_transform(self, texts):
        """
        Fit vectorizer and transform texts.
        
        Args:
            texts (list): List of text documents
            
        Returns:
            array: TF-IDF features
        """
        features = self.vectorizer.fit_transform(texts)
        self.fitted = True
        return features
    
    def transform(self, texts):
        """
        Transform texts using fitted vectorizer.
        
        Args:
            texts (list): List of text documents
            
        Returns:
            array: TF-IDF features
        """
        if not self.fitted:
            raise ValueError("Vectorizer must be fitted before transform")
        return self.vectorizer.transform(texts)
