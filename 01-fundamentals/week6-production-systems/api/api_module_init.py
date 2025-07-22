"""
API Module
==========

Institution integration endpoints for cultural heritage digitization system.
Provides RESTful APIs for institutional access and integration.
"""

from .endpoints import APIManager
from .auth_manager import AuthenticationManager
from .batch_processor import BatchProcessor
from .monitoring import APIMonitor
from .institutional_adapter import InstitutionalAdapter

__all__ = [
    'APIManager',
    'AuthenticationManager',
    'BatchProcessor', 
    'APIMonitor',
    'InstitutionalAdapter'
]

__version__ = "1.0.0"