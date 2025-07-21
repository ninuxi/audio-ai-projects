#!/usr/bin/env python3
"""
Docker Configuration for Production Audio System
Container-specific settings and utilities
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .settings import BaseConfig, get_config


@dataclass 
class DockerConfig:
    """Docker-specific configuration settings"""
    
    # Container settings
    CONTAINER_NAME: str = "production-audio-system"
    IMAGE_NAME: str = "audio-ai/production-system"
    IMAGE_TAG: str = "latest"
    
    # Resource limits
    MEMORY_LIMIT: str = "4g"
    CPU_LIMIT: str = "2.0"
    CPU_RESERVATION: str = "1.0"
    MEMORY_RESERVATION: str = "2g"
    
    # Volume mounts
    DATA_VOLUME: str = "/app/data"
    LOGS_VOLUME: str = "/app/logs"
    MODELS_VOLUME: str = "/app/cultural_models"
    UPLOADS_VOLUME: str = "/app/uploads"
    
    # Network settings
    NETWORK_NAME: str = "audio-ai-network"
    EXPOSE_PORTS: List[int] = None
    
    # Health check
    HEALTH_CHECK_INTERVAL: str = "30s"
    HEALTH_CHECK_TIMEOUT: str = "10s"
    HEALTH_CHECK_RETRIES: int = 3
    
    # Environment-specific overrides
    ENVIRONMENT: str = "production"
    
    def __post_init__(self):
        if self.EXPOSE_PORTS is None:
            self.EXPOSE_PORTS = [8000, 9090]  # API + Metrics


def get_docker_compose_config(environment: str = "production") -> Dict[str, Any]:
    """
    Generate docker-compose configuration
    
    Args:
        environment: Deployment environment
        
    Returns:
        Docker compose configuration dictionary
    """
    
    config = get_config(environment)
    docker_config = DockerConfig()
    
    # Base service configuration
    service_config = {
        'version': '3.8',
        'services': {
            'production-audio-system': {
                'build': {
                    'context': '.',
                    'dockerfile': 'docker/Dockerfile',
                    'args': {
                        'ENVIRONMENT': environment,
                        'PYTHON_VERSION': '3.11'
                    }
                },
                'image': f"{docker_config.IMAGE_NAME}:{docker_config.IMAGE_TAG}",
                'container_name': docker_config.CONTAINER_NAME,
                'restart': 'unless-stopped',
                'ports': [
                    f"{config.PORT}:{config.PORT}",  # API port
                    "9090:9090"  # Metrics port
                ],
                'volumes': [
                    f"./data:{docker_config.DATA_VOLUME}",
                    f"./logs:{docker_config.LOGS_VOLUME}",
                    f"./cultural_models:{docker_config.MODELS_VOLUME}",
                    f"./uploads:{docker_config.UPLOADS_VOLUME}"
                ],
                'environment': _get_container_environment(config),
                'healthcheck': {
                    'test': ["CMD", "curl", "-f", f"http://localhost:{config.PORT}/health"],
                    'interval': docker_config.HEALTH_CHECK_INTERVAL,
                    'timeout': docker_config.HEALTH_CHECK_TIMEOUT,
                    'retries': docker_config.HEALTH_CHECK_RETRIES,
                    'start_period': '40s'
                },
                'deploy': {
                    'resources': {
                        'limits': {
                            'cpus': docker_config.CPU_LIMIT,
                            'memory': docker_config.MEMORY_LIMIT
                        },
                        'reservations': {
                            'cpus': docker_config.CPU_RESERVATION,
                            'memory': docker_config.MEMORY_RESERVATION
                        }
                    }
                },
                'depends_on': ['redis', 'postgres'] if environment == 'production' else [],
                'networks': [docker_config.NETWORK_NAME]
            }
        },
        'networks': {
            docker_config.NETWORK_NAME: {
                'driver': 'bridge'
            }
        },
        'volumes': {
            'audio_data': {},
            'audio_logs': {},
            'audio_models': {},
            'postgres_data': {} if environment == 'production' else None
        }
    }
    
    # Add Redis service for production
    if environment in ['production', 'staging']:
        service_config['services']['redis'] = {
            'image': 'redis:7-alpine',
            'container_name': 'audio-redis',
            'restart': 'unless-stopped',
            'ports': ['6379:6379'],
            'volumes': ['redis_data:/data'],
            'command': 'redis-server --appendonly yes',
            'networks': [docker_config.NETWORK_NAME],
            'healthcheck': {
                'test': ["CMD", "redis-cli", "ping"],
                'interval': '10s',
                'timeout': '5s',
                'retries': 5
            }
        }
        
        service_config['volumes']['redis_data'] = {}
    
    # Add PostgreSQL for production
    if environment == 'production':
        service_config['services']['postgres'] = {
            'image': 'postgres:15-alpine',
            'container_name': 'audio-postgres',
            'restart': 'unless-stopped',
            'ports': ['5432:5432'],
            'environment': {
                'POSTGRES_DB': config.DB_NAME,
                'POSTGRES_USER': config.DB_USER,
                'POSTGRES_PASSWORD': config.DB_PASSWORD,
                'PGDATA': '/var/lib/postgresql/data/pgdata'
            },
            'volumes': [
                'postgres_data:/var/lib/postgresql/data',
                './docker/postgres/init.sql:/docker-entrypoint-initdb.d/init.sql'
            ],
            'networks': [docker_config.NETWORK_NAME],
            'healthcheck': {
                'test': ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB}"],
                'interval': '10s',
                'timeout': '5s',
                'retries': 5
            }
        }
    
    # Add monitoring services for production
    if environment == 'production':
        # Prometheus
        service_config['services']['prometheus'] = {
            'image': 'prom/prometheus:latest',
            'container_name': 'audio-prometheus',
            'restart': 'unless-stopped',
            'ports': ['9091:9090'],
            'volumes': [
                './docker/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml',
                'prometheus_data:/prometheus'
            ],
            'command': [
                '--config.file=/etc/prometheus/prometheus.yml',
                '--storage.tsdb.path=/prometheus',
                '--web.console.libraries=/usr/share/prometheus/console_libraries',
                '--web.console.templates=/usr/share/prometheus/consoles',
                '--storage.tsdb.retention.time=200h',
                '--web.enable-lifecycle'
            ],
            'networks': [docker_config.NETWORK_NAME]
        }
        
        # Grafana
        service_config['services']['grafana'] = {
            'image': 'grafana/grafana:latest',
            'container_name': 'audio-grafana',
            'restart': 'unless-stopped',
            'ports': ['3000:3000'],
            'environment': {
                'GF_SECURITY_ADMIN_PASSWORD': 'admin'
            },
            'volumes': [
                'grafana_data:/var/lib/grafana',
                './docker/grafana/dashboards:/etc/grafana/provisioning/dashboards',
                './docker/grafana/datasources:/etc/grafana/provisioning/datasources'
            ],
            'networks': [docker_config.NETWORK_NAME]
        }
        
        service_config['volumes'].update({
            'prometheus_data': {},
            'grafana_data': {}
        })
    
    # Remove None values
    service_config['volumes'] = {k: v for k, v in service_config['volumes'].items() if v is not None}
    
    return service_config


def _get_container_environment(config: BaseConfig) -> Dict[str, str]:
    """Get environment variables for container"""
    
    return {
        'ENVIRONMENT': config.ENVIRONMENT if hasattr(config, 'ENVIRONMENT') else 'production',
        'HOST': '0.0.0.0',  # Bind to all interfaces in container
        'PORT': str(config.PORT),
        'LOG_LEVEL': config.LOG_LEVEL,
        'DEBUG': 'false',
        
        # Database
        'DB_HOST': config.DB_HOST,
        'DB_PORT': str(config.DB_PORT),
        'DB_NAME': config.DB_NAME,
        'DB_USER': config.DB_USER,
        'DB_PASSWORD': config.DB_PASSWORD,
        
        # Redis
        'REDIS_HOST': config.REDIS_HOST or 'redis',
        'REDIS_PORT': str(config.REDIS_PORT),
        
        # Celery
        'CELERY_BROKER_URL': config.CELERY_BROKER_URL or 'redis://redis:6379/0',
        'CELERY_RESULT_BACKEND': config.CELERY_RESULT_BACKEND or 'redis://redis:6379/0',
        
        # Monitoring
        'DATADOG_API_KEY': config.DATADOG_API_KEY or '',
        'SENTRY_DSN': config.SENTRY_DSN or '',
        
        # Security
        'SECRET_KEY': config.SECRET_KEY,
        
        # Processing
        'MAX_THREAD_WORKERS': str(config.MAX_THREAD_WORKERS),
        'MAX_PROCESS_WORKERS': str(config.MAX_PROCESS_WORKERS),
        'NUM_WORKERS': str(config.NUM_WORKERS),
        
        # File handling
        'MAX_FILE_SIZE': str(config.MAX_FILE_SIZE),
        'UPLOAD_DIR': '/app/uploads',
        'DATA_DIR': '/app/data',
        'LOGS_DIR': '/app/logs',
        'MODELS_DIR': '/app/cultural_models',
        
        # MAXXI Integration
        'MAXXI_COMPATIBILITY_MODE': 'true',
        'MAXXI_OUTPUT_DIR': '/app/data/maxxi_outputs'
    }


def get_kubernetes_manifests(environment: str = "production") -> Dict[str, Any]:
    """
    Generate Kubernetes deployment manifests
    
    Args:
        environment: Deployment environment
        
    Returns:
        Kubernetes manifests dictionary
    """
    
    config = get_config(environment)
    docker_config = DockerConfig()
    
    # Deployment manifest
    deployment = {
        'apiVersion': 'apps/v1',
        'kind': 'Deployment',
        'metadata': {
            'name': 'production-audio-system',
            'labels': {
                'app': 'production-audio-system',
                'version': 'v1'
            }
        },
        'spec': {
            'replicas': 3 if environment == 'production' else 1,
            'selector': {
                'matchLabels': {
                    'app': 'production-audio-system'
                }
            },
            'template': {
                'metadata': {
                    'labels': {
                        'app': 'production-audio-system'
                    }
                },
                'spec': {
                    'containers': [{
                        'name': 'audio-system',
                        'image': f"{docker_config.IMAGE_NAME}:{docker_config.IMAGE_TAG}",
                        'ports': [
                            {'containerPort': config.PORT, 'name': 'http'},
                            {'containerPort': 9090, 'name': 'metrics'}
                        ],
                        'env': [
                            {'name': k, 'value': v} 
                            for k, v in _get_container_environment(config).items()
                        ],
                        'resources': {
                            'requests': {
                                'memory': docker_config.MEMORY_RESERVATION,
                                'cpu': docker_config.CPU_RESERVATION
                            },
                            'limits': {
                                'memory': docker_config.MEMORY_LIMIT,
                                'cpu': docker_config.CPU_LIMIT
                            }
                        },
                        'volumeMounts': [
                            {'name': 'data-volume', 'mountPath': '/app/data'},
                            {'name': 'logs-volume', 'mountPath': '/app/logs'},
                            {'name': 'models-volume', 'mountPath': '/app/cultural_models'},
                            {'name': 'uploads-volume', 'mountPath': '/app/uploads'}
                        ],
                        'livenessProbe': {
                            'httpGet': {
                                'path': '/health',
                                'port': config.PORT
                            },
                            'initialDelaySeconds': 60,
                            'periodSeconds': 30,
                            'timeoutSeconds': 10
                        },
                        'readinessProbe': {
                            'httpGet': {
                                'path': '/health',
                                'port': config.PORT
                            },
                            'initialDelaySeconds': 30,
                            'periodSeconds': 10,
                            'timeoutSeconds': 5
                        }
                    }],
                    'volumes': [
                        {
                            'name': 'data-volume',
                            'persistentVolumeClaim': {'claimName': 'audio-data-pvc'}
                        },
                        {
                            'name': 'logs-volume',
                            'persistentVolumeClaim': {'claimName': 'audio-logs-pvc'}
                        },
                        {
                            'name': 'models-volume',
                            'persistentVolumeClaim': {'claimName': 'audio-models-pvc'}
                        },
                        {
                            'name': 'uploads-volume',
                            'persistentVolumeClaim': {'claimName': 'audio-uploads-pvc'}
                        }
                    ]
                }
            }
        }
    }
    
    # Service manifest
    service = {
        'apiVersion': 'v1',
        'kind': 'Service',
        'metadata': {
            'name': 'production-audio-system-service',
            'labels': {
                'app': 'production-audio-system'
            }
        },
        'spec': {
            'selector': {
                'app': 'production-audio-system'
            },
            'ports': [
                {
                    'name': 'http',
                    'port': 80,
                    'targetPort': config.PORT
                },
                {
                    'name': 'metrics',
                    'port': 9090,
                    'targetPort': 9090
                }
            ],
            'type': 'ClusterIP'
        }
    }
    
    # Ingress manifest
    ingress = {
        'apiVersion': 'networking.k8s.io/v1',
        'kind': 'Ingress',
        'metadata': {
            'name': 'production-audio-system-ingress',
            'annotations': {
                'kubernetes.io/ingress.class': 'nginx',
                'nginx.ingress.kubernetes.io/proxy-body-size': '500m',
                'nginx.ingress.kubernetes.io/proxy-read-timeout': '300',
                'nginx.ingress.kubernetes.io/proxy-send-timeout': '300'
            }
        },
        'spec': {
            'rules': [{
                'host': 'audio-api.example.com',
                'http': {
                    'paths': [{
                        'path': '/',
                        'pathType': 'Prefix',
                        'backend': {
                            'service': {
                                'name': 'production-audio-system-service',
                                'port': {'number': 80}
                            }
                        }
                    }]
                }
            }]
        }
    }
    
    return {
        'deployment': deployment,
        'service': service,
        'ingress': ingress
    }


def generate_dockerfile_content(environment: str = "production") -> str:
    """Generate Dockerfile content for the environment"""
    
    dockerfile_content = f"""# Production Audio System Dockerfile
# Multi-stage build for optimized production image

# Build stage
FROM python:3.11-slim as builder

# Set build arguments
ARG ENVIRONMENT={environment}
ARG PYTHON_VERSION=3.11

# Install system dependencies for building
RUN apt-get update && apt-get install -y \\
    build-essential \\
    libsndfile1-dev \\
    libffi-dev \\
    libssl-dev \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt requirements-prod.txt ./
RUN pip install --no-cache-dir --upgrade pip && \\
    pip install --no-cache-dir -r requirements-prod.txt

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1 \\
    ENVIRONMENT={environment} \\
    PATH="/app/.local/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \\
    libsndfile1 \\
    curl \\
    && rm -rf /var/lib/apt/lists/* \\
    && apt-get clean

# Create non-root user
RUN useradd --create-home --shell /bin/bash app

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=app:app . .

# Create required directories
RUN mkdir -p data logs uploads cultural_models && \\
    chown -R app:app data logs uploads cultural_models

# Switch to non-root user
USER app

# Expose ports
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["python", "production_audio_system.py"]
"""
    
    return dockerfile_content


# Export main functions
__all__ = [
    'DockerConfig',
    'get_docker_compose_config',
    'get_kubernetes_manifests', 
    'generate_dockerfile_content'
]
