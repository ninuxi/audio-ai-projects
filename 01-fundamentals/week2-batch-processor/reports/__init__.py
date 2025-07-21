# reports/__init__.py
"""
📊 Audio Analysis Reports Package
=================================

Professional reporting modules for batch audio processing results.
Generates comprehensive analysis reports, visualizations, and summaries.
"""

from .analysis_reporter import AnalysisReporter
from .visualization_generator import VisualizationGenerator
from .export_manager import ExportManager

__all__ = [
    'AnalysisReporter',
    'VisualizationGenerator', 
    'ExportManager'
]

__version__ = "2.0.0"
__author__ = "Audio AI Projects - Week 2"
