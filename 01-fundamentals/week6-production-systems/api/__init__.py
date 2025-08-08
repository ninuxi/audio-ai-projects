"""
Audio AI Projects - Advanced Audio Processing System
Copyright (c) 2025 Antonino Mainenti (ninuxi)
Licensed under MIT License - see LICENSE file
GitHub: https://github.com/ninuxi/audio-ai-projects
"""
"""
API Module
==========

Institution integration endpoints for cultural heritage digitization system.
Provides RESTful APIs for institutional access and integration.
"""

from .api_endpoint_manager import APIManager

# Commented out temporarily - these modules need to be created
# from .auth_manager import AuthenticationManager
# from .batch_processor import BatchProcessor
# from .monitoring import APIMonitor
# from .institutional_adapter import InstitutionalAdapter

__all__ = [
    'APIManager'
    # 'AuthenticationManager',
    # 'BatchProcessor', 
    # 'APIMonitor',
    # 'InstitutionalAdapter'
]

__version__ = "1.0.0"
