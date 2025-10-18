"""
Fake News Detection System

A comprehensive machine learning project for detecting fake news.
"""

__version__ = '1.0.0'
__author__ = 'KMTanVeer'

from .preprocessing import TextPreprocessor, FeatureExtractor
from .models import FakeNewsDetector, compare_models
from .data_loader import DataLoader, create_sample_dataset

__all__ = [
    'TextPreprocessor',
    'FeatureExtractor',
    'FakeNewsDetector',
    'compare_models',
    'DataLoader',
    'create_sample_dataset'
]
