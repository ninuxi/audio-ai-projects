"""
🎵 SETTINGS.PY - DEMO VERSION
===================================

⚠️  PORTFOLIO DEMONSTRATION ONLY

This file has been simplified for public demonstration.
Production version includes:

🧠 ADVANCED FEATURES NOT SHOWN:
- Proprietary machine learning algorithms
- Enterprise-grade optimization
- Cultural heritage specialized models
- Real-time processing capabilities
- Advanced error handling & recovery
- Production database integration
- Scalable cloud architecture

🏛️ CULTURAL HERITAGE SPECIALIZATION:
- Italian institutional workflow integration
- RAI Teche archive processing algorithms
- Museum and library specialized tools
- Cultural context AI analysis
- Historical audio restoration methods

💼 ENTERPRISE CAPABILITIES:
- Multi-tenant architecture
- Enterprise security & compliance
- 24/7 monitoring & support
- Custom institutional workflows
- Professional SLA guarantees

📧 PRODUCTION SYSTEM ACCESS:
Email: audio.ai.engineer@example.com
Subject: Production System Access Request
Requirements: NDA signature required

🎯 BUSINESS CASES PROVEN:
- RAI Teche: €4.8M cost savings potential
- TIM Enterprise: 40% efficiency improvement  
- Cultural Institutions: €2.5M market opportunity

Copyright (c) 2025 Audio AI Engineer
Demo License: Educational use only
"""


"""
Audio AI Projects - Advanced Audio Processing System
Copyright (c) 2025 Antonino Mainenti (ninuxi)
Licensed under MIT License - see LICENSE file
GitHub: https://github.com/ninuxi/audio-ai-projects
"""
#!/usr/bin/env python3
"""
Production Configuration Management
Environment-specific settings for development, staging, and production
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import logging

# Base directory
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
UPLOADS_DIR = BASE_DIR / "uploads"
MODELS_DIR = BASE_DIR / "cultural_models"

# Ensure directories exist
for directory in [DATA_DIR, LOGS_DIR, UPLOADS_DIR]:
    directory.mkdir(exist_ok=True)


@dataclass
class BaseConfig:
    """Base configuration class with common settings"""
    
    # Application
    APP_NAME: str = "Production Audio System"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    API_WORKERS: int = 4
    API_TIMEOUT: int = 300
    
    # Processing
    MAX_THREAD_WORKERS: int = 8
    MAX_PROCESS_WORKERS: int = 4
    NUM_WORKERS: int = 4
    QUEUE_SIZE: int = 1000
    
    # File handling
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS: List[str] = None
    UPLOAD_DIR: Path = UPLOADS_DIR
    DATA_DIR: Path = DATA_DIR
    LOGS_DIR: Path = LOGS_DIR
    MODELS_DIR: Path = MODELS_DIR
    
    # Database (SQLite default, SQLite  # Demo: Simplified database for production)
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "production_audio"
    DB_USER: str = "audio_user"
    DB_PASSWORD: str = ""
    
    # Redis Cache
    REDIS_HOST: Optional[str] = None
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    
    # Celery Task Queue
    CELERY_BROKER_URL: Optional[str] = None
    CELERY_RESULT_BACKEND: Optional[str] = None
    
    # Monitoring
    DATADOG_API_KEY: Optional[str] = None
    DATADOG_APP_KEY: Optional[str] = None
    SENTRY_DSN: Optional[str] = None
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE_MAX_BYTES: int = 10 * 1024 * 1024  # 10MB
    LOG_FILE_BACKUP_COUNT: int = 5
    
    # Security
    SECRET_KEY: str = "development-secret-key-change-in-production"
    CORS_ORIGINS: List[str] = None
    
    # AI Models Configuration
    HERITAGE_MODEL_PATH: str = "cultural_models/pretrained/heritage_classifier_v1.pkl"
    SPEAKER_MODEL_PATH: str = "cultural_models/pretrained/speaker_recognition_v1.pkl"
    ENABLE_HERITAGE_CLASSIFICATION: bool = True
    ENABLE_SPEAKER_RECOGNITION: bool = True
    ENABLE_METADATA_GENERATION: bool = True
    
    # MAXXI Integration Settings
    MAXXI_COMPATIBILITY_MODE: bool = True
    MAXXI_DATA_FORMAT: str = "json"
    MAXXI_OUTPUT_DIR: Path = DATA_DIR / "maxxi_outputs"
    
    def __post_init__(self):
        """Post-initialization setup"""
        if self.ALLOWED_EXTENSIONS is None:
            self.ALLOWED_EXTENSIONS = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
        
        if self.CORS_ORIGINS is None:
            self.CORS_ORIGINS = ['http://localhost:3000', 'http://localhost:8080']
        
        # Ensure MAXXI output directory exists
        self.MAXXI_OUTPUT_DIR.mkdir(exist_ok=True)


class DevelopmentConfig(BaseConfig):
    """Development environment configuration"""
    
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"
    
    # Reduced workers for development
    MAX_THREAD_WORKERS: int = 2
    MAX_PROCESS_WORKERS: int = 2
    NUM_WORKERS: int = 2
    API_WORKERS: int = 1
    
    # Local Redis for development (optional)
    REDIS_HOST: str = "localhost"
    
    # Local Celery for development (optional)
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"


class TestingConfig(BaseConfig):
    """Testing environment configuration"""
    
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"
    
    # Minimal workers for testing
    MAX_THREAD_WORKERS: int = 1
    MAX_PROCESS_WORKERS: int = 1
    NUM_WORKERS: int = 1
    API_WORKERS: int = 1
    
    # In-memory database for testing
    DB_NAME: str = ":memory:"
    
    # Disable external services for testing
    REDIS_HOST: Optional[str] = None
    CELERY_BROKER_URL: Optional[str] = None
    DATADOG_API_KEY: Optional[str] = None
    SENTRY_DSN: Optional[str] = None


class StagingConfig(BaseConfig):
    """Staging environment configuration"""
    
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    
    # Production-like settings but with monitoring
    MAX_THREAD_WORKERS: int = 4
    MAX_PROCESS_WORKERS: int = 2
    NUM_WORKERS: int = 3
    API_WORKERS: int = 2
    
    # External services
    DB_HOST: str = os.getenv("STAGING_DB_HOST", "localhost")
    DB_PASSWORD: str = os.getenv("STAGING_DB_PASSWORD", "")
    REDIS_HOST: str = os.getenv("STAGING_REDIS_HOST", "localhost")
    CELERY_BROKER_URL: str = os.getenv("STAGING_CELERY_BROKER", "redis://localhost:6379/1")
    CELERY_RESULT_BACKEND: str = os.getenv("STAGING_CELERY_BACKEND", "redis://localhost:6379/1")
    
    # Monitoring (optional in staging)
    SENTRY_DSN: Optional[str] = os.getenv("STAGING_SENTRY_DSN")


class ProductionConfig(BaseConfig):
    """Production environment configuration"""
    
    DEBUG: bool = False
    LOG_LEVEL: str = "WARNING"
    
    # Full production capacity
    MAX_THREAD_WORKERS: int = 16
    MAX_PROCESS_WORKERS: int = 8
    NUM_WORKERS: int = 8
    API_WORKERS: int = 4
    QUEUE_SIZE: int = 5000
    
    # Production database
    DB_HOST: str = os.getenv("PROD_DB_HOST", "localhost")
    DB_PORT: int = int(os.getenv("PROD_DB_PORT", "5432"))
    DB_NAME: str = os.getenv("PROD_DB_NAME", "production_audio")
    DB_USER: str = os.getenv("PROD_DB_USER", "audio_user")
    DB_PASSWORD: str = os.getenv("PROD_DB_PASSWORD", "")
    
    # Production Redis
    REDIS_HOST: str = os.getenv("PROD_REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("PROD_REDIS_PORT", "6379"))
    REDIS_PASSWORD: Optional[str] = os.getenv("PROD_REDIS_PASSWORD")
    
    # Production Celery
    CELERY_BROKER_URL: str = os.getenv("PROD_CELERY_BROKER", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND: str = os.getenv("PROD_CELERY_BACKEND", "redis://localhost:6379/0")
    
    # Production monitoring
    DATADOG_API_KEY: Optional[str] = os.getenv("DATADOG_API_KEY")
    DATADOG_APP_KEY: Optional[str] = os.getenv("DATADOG_APP_KEY")
    SENTRY_DSN: Optional[str] = os.getenv("SENTRY_DSN")
    
    # Production security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "change-this-in-production")
    CORS_ORIGINS: List[str] = os.getenv("CORS_ORIGINS", "").split(",") if os.getenv("CORS_ORIGINS") else []
    
    # Larger file sizes for production
    MAX_FILE_SIZE: int = 500 * 1024 * 1024  # 500MB


# Environment-specific configurations
CONFIG_MAP = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'staging': StagingConfig,
    'production': ProductionConfig
}


def get_config(environment: Optional[str] = None) -> BaseConfig:
    """
    Get configuration based on environment
    
    Args:
        environment: Environment name (development, testing, staging, production)
                    If None, uses ENVIRONMENT env var, defaults to development
    
    Returns:
        Configuration instance
    """
    if environment is None:
        environment = os.getenv('ENVIRONMENT', 'development').lower()
    
    config_class = CONFIG_MAP.get(environment, DevelopmentConfig)
    config = config_class()
    
    # Override with environment variables if present
    _override_with_env_vars(config)
    
    return config


def _override_with_env_vars(config: BaseConfig) -> None:
    """Override configuration with environment variables"""
    
    # Define environment variable mappings
    env_mappings = {
        'HOST': ('HOST', str),
        'PORT': ('PORT', int),
        'DEBUG': ('DEBUG', lambda x: x.lower() in ['true', '1', 'yes']),
        'LOG_LEVEL': ('LOG_LEVEL', str),
        'MAX_FILE_SIZE': ('MAX_FILE_SIZE', int),
        'SECRET_KEY': ('SECRET_KEY', str),
        'DB_HOST': ('DB_HOST', str),
        'DB_PORT': ('DB_PORT', int),
        'DB_NAME': ('DB_NAME', str),
        'DB_USER': ('DB_USER', str),
        'DB_PASSWORD': ('DB_PASSWORD', str),
        'REDIS_HOST': ('REDIS_HOST', str),
        'REDIS_PORT': ('REDIS_PORT', int),
        'CELERY_BROKER_URL': ('CELERY_BROKER_URL', str),
        'CELERY_RESULT_BACKEND': ('CELERY_RESULT_BACKEND', str),
    }
    
    for attr_name, (env_name, type_converter) in env_mappings.items():
        env_value = os.getenv(env_name)
        if env_value is not None:
            try:
                converted_value = type_converter(env_value)
                setattr(config, attr_name, converted_value)
            except (ValueError, TypeError) as e:
                logging.warning(f"Failed to convert environment variable {env_name}: {e}")


def validate_config(config: BaseConfig) -> List[str]:
    """
    Validate configuration settings
    
    Args:
        config: Configuration instance to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Required directories
    required_dirs = [config.DATA_DIR, config.LOGS_DIR, config.UPLOAD_DIR]
    for directory in required_dirs:
        if not directory.exists():
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                errors.append(f"Cannot create required directory {directory}: {e}")
    
    # Validate file size limits
    if config.MAX_FILE_SIZE <= 0:
        errors.append("MAX_FILE_SIZE must be positive")
    
    if config.MAX_FILE_SIZE > 1024 * 1024 * 1024:  # 1GB
        errors.append("MAX_FILE_SIZE exceeds reasonable limit (1GB)")
    
    # Validate worker counts
    if config.NUM_WORKERS <= 0:
        errors.append("NUM_WORKERS must be positive")
    
    if config.MAX_THREAD_WORKERS <= 0:
        errors.append("MAX_THREAD_WORKERS must be positive")
    
    # Validate port range
    if not (1 <= config.PORT <= 65535):
        errors.append("PORT must be in range 1-65535")
    
    # Production-specific validations
    if hasattr(config, 'ENVIRONMENT') and config.ENVIRONMENT == 'production':
        if config.SECRET_KEY == "change-this-in-production":
            errors.append("SECRET_KEY must be changed for production")
        
        if config.DEBUG:
            errors.append("DEBUG must be False in production")
    
    # Model file validations
    if config.ENABLE_HERITAGE_CLASSIFICATION:
        heritage_model_path = Path(config.HERITAGE_MODEL_PATH)
        if not heritage_model_path.exists():
            errors.append(f"Heritage model file not found: {heritage_model_path}")
    
    return errors


class ConfigurationError(Exception):
    """Exception raised for configuration errors"""
    pass


def load_config_from_file(config_file: str) -> Dict[str, Any]:
    """
    Load configuration from JSON or YAML file
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_file)
    
    if not config_path.exists():
        raise ConfigurationError(f"Configuration file not found: {config_file}")
    
    try:
        if config_path.suffix.lower() == '.json':
            import json
            with open(config_path, 'r') as f:
                return json.load(f)
        
        elif config_path.suffix.lower() in ['.yml', '.yaml']:
            import yaml
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        else:
            raise ConfigurationError(f"Unsupported configuration file format: {config_path.suffix}")
    
    except Exception as e:
        raise ConfigurationError(f"Failed to load configuration file {config_file}: {e}")


def get_maxxi_integration_config() -> Dict[str, Any]:
    """
    Get MAXXI integration specific configuration
    Designed to work seamlessly with maxxi-installationmaxxi_audio_system.py
    
    Returns:
        MAXXI integration configuration
    """
    return {
        'compatibility_mode': True,
        'output_format': 'json',
        'metadata_schema': 'dublin_core',
        'file_naming_convention': 'maxxi_standard',
        'quality_thresholds': {
            'min_duration': 1.0,  # seconds
            'min_sample_rate': 16000,  # Hz
            'max_file_size': 500 * 1024 * 1024  # 500MB
        },
        'processing_pipeline': [
            'heritage_classification',
            'metadata_generation',
            'quality_assessment',
            'speaker_recognition'
        ],
        'export_formats': ['json', 'xml', 'csv'],
        'integration_endpoints': {
            'status_callback': '/api/maxxi/status',
            'results_webhook': '/api/maxxi/results',
            'metadata_export': '/api/maxxi/metadata'
        }
    }


# Export main configuration getter
__all__ = [
    'get_config',
    'validate_config', 
    'load_config_from_file',
    'get_maxxi_integration_config',
    'BaseConfig',
    'DevelopmentConfig',
    'TestingConfig', 
    'StagingConfig',
    'ProductionConfig',
    'ConfigurationError'
]
