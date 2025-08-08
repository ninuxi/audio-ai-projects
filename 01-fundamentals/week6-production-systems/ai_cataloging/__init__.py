"""
Audio AI Projects - Advanced Audio Processing System
Copyright (c) 2025 Antonino Mainenti (ninuxi)
Licensed under MIT License - see LICENSE file
GitHub: https://github.com/ninuxi/audio-ai-projects
"""
"""
AI Cataloging Module
===================

Automated metadata generation and content analysis for cultural heritage audio.
Provides intelligent cataloging with cultural context and historical significance assessment.
"""

from .genre_classifier import GenreClassifier
from .metadata_generator import MetadataGenerator

__all__ = [
    'GenreClassifier',
    'MetadataGenerator'
]

__version__ = "1.0.0"
