#!/usr/bin/env python3
"""
Production Logging Configuration
Comprehensive logging setup for development, staging, and production environments
"""

import logging
import logging.config
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime

from .settings import get_config


def setup_production_logging(config=None, log_level: Optional[str] = None) -> None:
    """
    Setup production-ready logging configuration
    
    Args:
        config: Configuration object (if None, will load from environment)
        log_level: Override log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    if config is None:
        config = get_config()
    
    # Override log level if specified
    if log_level:
        config.LOG_LEVEL = log_level.upper()
    
    # Ensure logs directory exists
    config.LOGS_DIR.mkdir(exist_ok=True)
    
    # Create logging configuration
    logging_config = create_logging_config(config)
    
    # Apply configuration
    logging.config.dictConfig(logging_config)
    
    # Log startup information
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized - Level: {config.LOG_LEVEL}")
    logger.info(f"Log files location: {config.LOGS_DIR}")


def create_logging_config(config) -> Dict[str, Any]:
    """
    Create comprehensive logging configuration dictionary
    
    Args:
        config: Configuration object
        
    Returns:
        Logging configuration dictionary
    """
    
    # Define log file paths
    app_log_file = config.LOGS_DIR / "application.log"
    error_log_file = config.LOGS_DIR / "errors.log"
    access_log_file = config.LOGS_DIR / "access.log"
    performance_log_file = config.LOGS_DIR / "performance.log"
    
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        
        # Formatters
        'formatters': {
            'standard': {
                'format': config.LOG_FORMAT,
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'json': {
                '()': 'config.logging_config.JSONFormatter'
            },
            'performance': {
                'format': '%(asctime)s - PERF - %(name)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }
        },
        
        # Handlers
        'handlers': {
            'console': {
                'level': 'DEBUG' if config.DEBUG else 'INFO',
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
                'stream': 'ext://sys.stdout'
            },
            
            'file_app': {
                'level': config.LOG_LEVEL,
                'class': 'logging.handlers.RotatingFileHandler',
                'formatter': 'detailed',
                'filename': str(app_log_file),
                'maxBytes': config.LOG_FILE_MAX_BYTES,
                'backupCount': config.LOG_FILE_BACKUP_COUNT,
                'encoding': 'utf-8'
            },
            
            'file_error': {
                'level': 'ERROR',
                'class': 'logging.handlers.RotatingFileHandler', 
                'formatter': 'detailed',
                'filename': str(error_log_file),
                'maxBytes': config.LOG_FILE_MAX_BYTES,
                'backupCount': config.LOG_FILE_BACKUP_COUNT,
                'encoding': 'utf-8'
            },
            
            'file_access': {
                'level': 'INFO',
                'class': 'logging.handlers.RotatingFileHandler',
                'formatter': 'standard',
                'filename': str(access_log_file),
                'maxBytes': config.LOG_FILE_MAX_BYTES,
                'backupCount': config.LOG_FILE_BACKUP_COUNT,
                'encoding': 'utf-8'
            },
            
            'file_performance': {
                'level': 'INFO',
                'class': 'logging.handlers.RotatingFileHandler',
                'formatter': 'performance',
                'filename': str(performance_log_file),
                'maxBytes': config.LOG_FILE_MAX_BYTES,
                'backupCount': config.LOG_FILE_BACKUP_COUNT,
                'encoding': 'utf-8'
            }
        },
        
        # Loggers
        'loggers': {
            # Root logger
            '': {
                'level': config.LOG_LEVEL,
                'handlers': ['console', 'file_app', 'file_error'],
                'propagate': False
            },
            
            # Application loggers
            'production_audio_system': {
                'level': config.LOG_LEVEL,
                'handlers': ['console', 'file_app', 'file_error'],
                'propagate': False
            },
            
            'heritage_classifier': {
                'level': config.LOG_LEVEL,
                'handlers': ['file_app'],
                'propagate': True
            },
            
            'metadata_generator': {
                'level': config.LOG_LEVEL,
                'handlers': ['file_app'],
                'propagate': True
            },
            
            'speaker_recognition': {
                'level': config.LOG_LEVEL,
                'handlers': ['file_app'],
                'propagate': True
            },
            
            # Performance logger
            'performance': {
                'level': 'INFO',
                'handlers': ['file_performance'],
                'propagate': False
            },
            
            # Access logger (for API requests)
            'werkzeug': {
                'level': 'INFO',
                'handlers': ['file_access'],
                'propagate': False
            },
            
            'gunicorn.access': {
                'level': 'INFO',
                'handlers': ['file_access'],
                'propagate': False
            },
            
            'gunicorn.error': {
                'level': 'ERROR',
                'handlers': ['file_error'],
                'propagate': False
            },
            
            # Third-party library loggers
            'urllib3': {
                'level': 'WARNING',
                'propagate': True
            },
            
            'requests': {
                'level': 'WARNING', 
                'propagate': True
            },
            
            'celery': {
                'level': 'INFO',
                'handlers': ['file_app'],
                'propagate': False
            },
            
            'redis': {
                'level': 'WARNING',
                'propagate': True
            }
        }
    }
    
    # Add structured JSON logging for production
    if hasattr(config, 'ENVIRONMENT') and config.ENVIRONMENT == 'production':
        # Add JSON formatter for structured logging
        logging_config['handlers']['json_file'] = {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'formatter': 'json',
            'filename': str(config.LOGS_DIR / "structured.log"),
            'maxBytes': config.LOG_FILE_MAX_BYTES * 2,  # Larger for JSON logs
            'backupCount': config.LOG_FILE_BACKUP_COUNT,
            'encoding': 'utf-8'
        }
        
        # Add JSON handler to main loggers
        for logger_name in ['', 'production_audio_system']:
            logging_config['loggers'][logger_name]['handlers'].append('json_file')
    
    return logging_config


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        
        # Create base log entry
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add process/thread information
        log_entry['process_id'] = record.process
        log_entry['thread_id'] = record.thread
        
        # Add custom fields if present
        for key, value in record.__dict__.items():
            if key.startswith('custom_'):
                log_entry[key[7:]] = value  # Remove 'custom_' prefix
        
        return json.dumps(log_entry, ensure_ascii=False)


class PerformanceLogger:
    """
    Specialized logger for performance metrics and timing
    """
    
    def __init__(self, name: str = 'performance'):
        self.logger = logging.getLogger(name)
    
    def log_processing_time(self, operation: str, duration: float, 
                          metadata: Dict[str, Any] = None):
        """Log processing time for operations"""
        metadata = metadata or {}
        message = f"Operation: {operation}, Duration: {duration:.3f}s"
        
        if metadata:
            message += f", Metadata: {json.dumps(metadata)}"
        
        # Add custom fields for structured logging
        extra = {
            'custom_operation': operation,
            'custom_duration': duration,
            'custom_metadata': metadata
        }
        
        self.logger.info(message, extra=extra)
    
    def log_system_metrics(self, cpu_percent: float, memory_percent: float,
                         disk_percent: float, active_tasks: int):
        """Log system resource metrics"""
        message = f"System Metrics - CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%, Disk: {disk_percent:.1f}%, Tasks: {active_tasks}"
        
        extra = {
            'custom_cpu_percent': cpu_percent,
            'custom_memory_percent': memory_percent,
            'custom_disk_percent': disk_percent,
            'custom_active_tasks': active_tasks
        }
        
        self.logger.info(message, extra=extra)
    
    def log_api_request(self, endpoint: str, method: str, duration: float,
                       status_code: int, user_agent: str = None):
        """Log API request performance"""
        message = f"API Request - {method} {endpoint}, Duration: {duration:.3f}s, Status: {status_code}"
        
        extra = {
            'custom_endpoint': endpoint,
            'custom_method': method,
            'custom_duration': duration,
            'custom_status_code': status_code,
            'custom_user_agent': user_agent
        }
        
        self.logger.info(message, extra=extra)


class AuditLogger:
    """
    Audit logger for security and compliance events
    """
    
    def __init__(self, name: str = 'audit'):
        self.logger = logging.getLogger(name)
        # Set up separate audit log file
        self._setup_audit_handler()
    
    def _setup_audit_handler(self):
        """Setup separate handler for audit logs"""
        config = get_config()
        audit_log_file = config.LOGS_DIR / "audit.log"
        
        handler = logging.handlers.RotatingFileHandler(
            filename=str(audit_log_file),
            maxBytes=config.LOG_FILE_MAX_BYTES,
            backupCount=config.LOG_FILE_BACKUP_COUNT * 2,  # Keep more audit logs
            encoding='utf-8'
        )
        
        formatter = logging.Formatter(
            '%(asctime)s - AUDIT - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_file_access(self, user_id: str, file_path: str, operation: str):
        """Log file access events"""
        message = f"File Access - User: {user_id}, File: {file_path}, Operation: {operation}"
        self.logger.info(message)
    
    def log_authentication(self, user_id: str, success: bool, ip_address: str = None):
        """Log authentication attempts"""
        status = "SUCCESS" if success else "FAILURE"
        message = f"Authentication {status} - User: {user_id}"
        if ip_address:
            message += f", IP: {ip_address}"
        
        level = logging.INFO if success else logging.WARNING
        self.logger.log(level, message)
    
    def log_configuration_change(self, user_id: str, setting: str, 
                                old_value: str, new_value: str):
        """Log configuration changes"""
        message = f"Config Change - User: {user_id}, Setting: {setting}, Old: {old_value}, New: {new_value}"
        self.logger.warning(message)


def get_logger(name: str, level: str = None) -> logging.Logger:
    """
    Get configured logger instance
    
    Args:
        name: Logger name
        level: Optional log level override
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if level:
        logger.setLevel(getattr(logging, level.upper()))
    
    return logger


def log_performance_metrics(func):
    """
    Decorator to automatically log function performance metrics
    
    Usage:
        @log_performance_metrics
        def my_function():
            # function code
    """
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        perf_logger = PerformanceLogger()
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            perf_logger.log_processing_time(
                operation=f"{func.__module__}.{func.__name__}",
                duration=duration,
                metadata={'args_count': len(args), 'kwargs_count': len(kwargs)}
            )
            
            return result
        
        except Exception as e:
            duration = time.time() - start_time
            perf_logger.log_processing_time(
                operation=f"{func.__module__}.{func.__name__}",
                duration=duration,
                metadata={'error': str(e), 'args_count': len(args), 'kwargs_count': len(kwargs)}
            )
            raise
    
    return wrapper


def setup_maxxi_logging(maxxi_logs_dir: str = None):
    """
    Setup specialized logging for MAXXI integration
    Compatible with maxxi-installationmaxxi_audio_system.py
    
    Args:
        maxxi_logs_dir: Directory for MAXXI-specific logs
    """
    config = get_config()
    
    if maxxi_logs_dir:
        maxxi_dir = Path(maxxi_logs_dir)
    else:
        maxxi_dir = config.LOGS_DIR / "maxxi"
    
    maxxi_dir.mkdir(exist_ok=True)
    
    # Create MAXXI-specific logger
    maxxi_logger = logging.getLogger('maxxi_integration')
    maxxi_logger.setLevel(logging.INFO)
    
    # File handler for MAXXI logs
    maxxi_handler = logging.handlers.RotatingFileHandler(
        filename=str(maxxi_dir / "maxxi_integration.log"),
        maxBytes=config.LOG_FILE_MAX_BYTES,
        backupCount=config.LOG_FILE_BACKUP_COUNT,
        encoding='utf-8'
    )
    
    # Specialized formatter for MAXXI
    maxxi_formatter = logging.Formatter(
        '%(asctime)s - MAXXI - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    maxxi_handler.setFormatter(maxxi_formatter)
    maxxi_logger.addHandler(maxxi_handler)
    
    # Processing results logger
    results_logger = logging.getLogger('maxxi_results')
    results_handler = logging.handlers.RotatingFileHandler(
        filename=str(maxxi_dir / "processing_results.log"),
        maxBytes=config.LOG_FILE_MAX_BYTES,
        backupCount=config.LOG_FILE_BACKUP_COUNT,
        encoding='utf-8'
    )
    results_handler.setFormatter(maxxi_formatter)
    results_logger.addHandler(results_handler)
    results_logger.setLevel(logging.INFO)
    
    logging.getLogger(__name__).info(f"MAXXI logging initialized: {maxxi_dir}")


# Health check logging
def setup_health_monitoring_logs():
    """Setup logging for health monitoring and alerts"""
    config = get_config()
    
    health_logger = logging.getLogger('health_monitoring')
    health_logger.setLevel(logging.INFO)
    
    # Separate log file for health checks
    health_handler = logging.handlers.RotatingFileHandler(
        filename=str(config.LOGS_DIR / "health_monitoring.log"),
        maxBytes=config.LOG_FILE_MAX_BYTES,
        backupCount=config.LOG_FILE_BACKUP_COUNT,
        encoding='utf-8'
    )
    
    health_formatter = logging.Formatter(
        '%(asctime)s - HEALTH - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    health_handler.setFormatter(health_formatter)
    health_logger.addHandler(health_handler)


# Export main functions
__all__ = [
    'setup_production_logging',
    'get_logger',
    'PerformanceLogger', 
    'AuditLogger',
    'JSONFormatter',
    'log_performance_metrics',
    'setup_maxxi_logging',
    'setup_health_monitoring_logs'
]
