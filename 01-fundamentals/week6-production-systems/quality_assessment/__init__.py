"""
Quality Assessment Module
========================

Comprehensive audio quality analysis for cultural heritage preservation.
Provides technical assessment, degradation detection, and restoration recommendations.
"""

from .degradation_detector import DegradationDetector
from .quality_metrics_calculator import QualityMetrics

# AudioQualityAnalyzer commented out temporarily due to circular import
# from .audio_analyzer import AudioQualityAnalyzer

__all__ = [
    'DegradationDetector',
    'QualityMetrics'
    # 'AudioQualityAnalyzer'
]

__version__ = "1.0.0"