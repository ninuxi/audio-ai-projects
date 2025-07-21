# processors/__init__.py
"""
🎵 Audio Processing Modules Package
==================================

Individual specialized processors for different aspects of audio analysis.
Each processor handles a specific domain of audio feature extraction.
"""

from .feature_extractor import FeatureExtractor
from .quality_analyzer import QualityAnalyzer  
from .content_classifier import ContentClassifier
from .metadata_generator import MetadataGenerator

__all__ = [
    'FeatureExtractor',
    'QualityAnalyzer', 
    'ContentClassifier',
    'MetadataGenerator'
]

__version__ = "1.0.0"
__author__ = "Audio AI Projects"
