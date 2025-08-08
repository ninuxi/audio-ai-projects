# Production Audio System Dockerfile
# Multi-stage build for optimized production deployment
# Compatible with maxxi-installationmaxxi_audio_system.py workflows

ARG PYTHON_VERSION=3.11
ARG ENVIRONMENT=production

# ==============================================================================
# Build Stage - Compile dependencies and prepare application
# ==============================================================================
FROM python:${PYTHON_VERSION}-slim as builder

# Set build arguments
ARG ENVIRONMENT
ARG DEBIAN_FRONTEND=noninteractive

# Metadata labels
LABEL maintainer="Audio AI Team"
LABEL description="Production Audio AI System"
LABEL version="1.0.0"

# Install system build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1-dev \
    libffi-dev \
    libssl-dev \
    pkg-config \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create requirements files if they don't exist
COPY requirements*.txt ./

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir --upgrade pip wheel setuptools && \
    pip install --no-cache-dir -r requirements.txt

# ==============================================================================
# Runtime Stage - Minimal production image
# ==============================================================================
FROM python:${PYTHON_VERSION}-slim as runtime

# Set build arguments
ARG ENVIRONMENT
ARG DEBIAN_FRONTEND=noninteractive

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    ENVIRONMENT=${ENVIRONMENT} \
    PATH="/app/.local/bin:$PATH"

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    libffi8 \
    libssl3 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r appgroup && \
    useradd -r -g appgroup -d /app -s /bin/bash -c "Application User" appuser

# Set working directory
WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python${PYTHON_VERSION}/site-packages/ /usr/local/lib/python${PYTHON_VERSION}/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy application code
COPY --chown=appuser:appgroup . .

# Create required directories and set permissions
RUN mkdir -p \
    data \
    logs \
    uploads \
    cultural_models \
    config \
    tests \
    docs \
    /tmp/audio-processing \
    && chown -R appuser:appgroup \
    data \
    logs \
    uploads \
    cultural_models \
    config \
    tests \
    docs \
    /tmp/audio-processing

# Set up logging directory with proper permissions
RUN mkdir -p /var/log/audio-ai && \
    chown appuser:appgroup /var/log/audio-ai && \
    ln -sf /app/logs /var/log/audio-ai

# Install additional MAXXI compatibility layer
COPY docker/maxxi-compatibility.py /app/maxxi-compatibility.py
RUN chown appuser:appgroup /app/maxxi-compatibility.py

# Copy health check script
COPY docker/healthcheck.sh /app/healthcheck.sh
RUN chmod +x /app/healthcheck.sh && \
    chown appuser:appgroup /app/healthcheck.sh

# Switch to non-root user
USER appuser

# Set up Python path and validate installation
RUN python -c "import sys; print(f'Python {sys.version}')" && \
    python -c "import librosa, numpy, sklearn; print('Audio AI dependencies OK')" && \
    python -c "from production_audio_system import ProductionAudioSystem; print('Application imports OK')"

# Expose ports
EXPOSE 8000 9090

# Add volume mount points
VOLUME ["/app/data", "/app/logs", "/app/uploads", "/app/cultural_models"]

# Health check configuration
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /app/healthcheck.sh

# Default command
CMD ["python", "production_audio_system.py"]

# ==============================================================================
# Development Stage - Extended image for development use
# ==============================================================================
FROM runtime as development

USER root

# Install development dependencies
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    strace \
    tcpdump \
    netcat-traditional \
    telnet \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    pytest-asyncio \
    ipython \
    jupyter \
    black \
    flake8 \
    mypy

# Set development environment
ENV ENVIRONMENT=development \
    DEBUG=true \
    LOG_LEVEL=DEBUG

USER appuser

# Development command with auto-reload
CMD ["python", "-u", "production_audio_system.py"]

# ==============================================================================
# Testing Stage - Optimized for CI/CD testing
# ==============================================================================
FROM development as testing

USER root

# Install testing-specific dependencies
RUN pip install --no-cache-dir \
    coverage \
    pytest-xdist \
    pytest-mock \
    factory-boy \
    faker

# Set testing environment
ENV ENVIRONMENT=testing \
    DEBUG=true \
    LOG_LEVEL=DEBUG \
    PYTHONPATH="/app:/app/tests"

USER appuser

# Create test data directories
RUN mkdir -p test_data test_results

# Testing command
CMD ["python", "-m", "pytest", "tests/", "-v", "--cov=production_audio_system", "--cov-report=html"]

# ==============================================================================
# Production Stage - Final optimized production image
# ==============================================================================
FROM runtime as production

# Production-specific optimizations
USER root

# Install production monitoring tools
RUN pip install --no-cache-dir \
    gunicorn \
    prometheus-client \
    sentry-sdk

# Set production environment variables
ENV ENVIRONMENT=production \
    DEBUG=false \
    LOG_LEVEL=WARNING \
    WORKERS=4 \
    TIMEOUT=300 \
    MAX_REQUESTS=1000 \
    MAX_REQUESTS_JITTER=50

USER appuser

# Production command with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--timeout", "300", "--access-logfile", "-", "--error-logfile", "-", "production_audio_system:app"]

# ==============================================================================
# MAXXI Integration Stage - Specialized for MAXXI workflows
# ==============================================================================
FROM production as maxxi-integration

USER root

# Install MAXXI-specific dependencies
RUN pip install --no-cache-dir \
    lxml \
    xmltodict \
    jsonschema

# Copy MAXXI integration files
COPY docker/maxxi-integration/ /app/maxxi-integration/
COPY maxxi-installationmaxxi_audio_system.py /app/
RUN chown -R appuser:appgroup /app/maxxi-integration/ /app/maxxi-installationmaxxi_audio_system.py

# Set MAXXI-specific environment variables
ENV MAXXI_COMPATIBILITY_MODE=true \
    MAXXI_OUTPUT_FORMAT=json \
    MAXXI_METADATA_SCHEMA=dublin_core \
    MAXXI_DATA_DIR=/app/data/maxxi_outputs

USER appuser

# Create MAXXI-specific directories
RUN mkdir -p \
    /app/data/maxxi_outputs \
    /app/data/maxxi_processed \
    /app/data/maxxi_archive

# MAXXI integration command
CMD ["python", "maxxi-compatibility.py"]
