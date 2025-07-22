"""
Database Module
==============

Cultural heritage data schemas and management system.
Provides database models, schemas, and utilities for managing cultural heritage audio metadata.
"""

from .schema_manager import DatabaseManager
from .cultural_schemas import CulturalHeritageSchema
from .metadata_models import AudioMetadataModel, CulturalContextModel
from .migration_utils import DatabaseMigrator
from .backup_manager import BackupManager

__all__ = [
    'DatabaseManager',
    'CulturalHeritageSchema',
    'AudioMetadataModel',
    'CulturalContextModel', 
    'DatabaseMigrator',
    'BackupManager'
]

__version__ = "1.0.0"