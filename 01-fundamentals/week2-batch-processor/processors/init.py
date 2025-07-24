"""
Audio AI Projects - Advanced Audio Processing System
Copyright (c) 2025 Antonino Mainenti (ninuxi)
Licensed under MIT License - see LICENSE file
GitHub: https://github.com/ninuxi/audio-ai-projects
"""
"""
Audio processing modules for batch operations
"""
from .audio_normalizer import AudioNormalizer
from .noise_reducer import NoiseReducer
from .format_converter import FormatConverter
from .metadata_extractor import MetadataExtractor
from .batch_validator import BatchValidator

__all__ = [
    'AudioNormalizer',
    'NoiseReducer', 
    'FormatConverter',
    'MetadataExtractor',
    'BatchValidator'
]
