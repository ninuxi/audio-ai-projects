# Production Audio System Documentation

## Overview

The Production Audio System is an enterprise-grade platform for AI-powered audio analysis and cultural heritage processing. It integrates advanced machine learning models for heritage classification, metadata generation, speaker recognition, and voice activity detection into a scalable, production-ready system.

### Key Features

- 🎯 **AI-Powered Analysis**: Heritage classification, speaker recognition, and metadata generation
- 🏭 **Production Ready**: Scalable architecture with load balancing and monitoring  
- 🐳 **Containerized Deployment**: Docker and Kubernetes support
- 🎭 **MAXXI Integration**: Compatible with existing MAXXI installation workflows
- 📊 **Comprehensive Monitoring**: Health checks, metrics, and performance tracking
- 🔒 **Enterprise Security**: Authentication, authorization, and audit logging

### System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Load Balancer │────│  API Gateway     │────│  Authentication │
│   (Nginx)       │    │  (Production     │    │  & Authorization│
└─────────────────┘    │   Audio System)  │    └─────────────────┘
                       └──────────────────┘
                               │
                    ┌──────────┼──────────┐
                    │          │          │
            ┌───────▼───┐ ┌────▼────┐ ┌───▼────┐
            │Processing │ │Heritage │ │Speaker │
            │  Workers  │ │Classifier│ │  AI    │
            └───────────┘ └─────────┘ └────────┘
                    │          │          │
            ┌───────▼──────────▼──────────▼────┐
            │     Message Queue (Redis)        │
            └──────────────────────────────────┘
                               │
            ┌──────────────────▼──────────────────┐
            │        Database (PostgreSQL)        │
            └─────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Docker & Docker Compose 20.10+
- Python 3.11+ (for development)
- 4GB+ RAM
- 10GB+ disk space

### Development Setup

1. **Clone and setup environment:**
   ```bash
   git clone <repository>
   cd week5-production
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   ```bash
   cp config/example.env .env
   # Edit .env with your configuration
   ```

3. **Run development server:**
   ```bash
   python production_audio_system.py
   ```

### Production Deployment

1. **Configure production environment:**
   ```bash
   # Set environment variables
   export ENVIRONMENT=production
   export DB_PASSWORD=your_secure_password
   export SECRET_KEY=your_secret_key
   ```

2. **Deploy with Docker Compose:**
   ```bash
   docker-compose up -d
   ```

3. **Verify deployment:**
   ```bash
   curl http://localhost:8000/health
   ```

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ENVIRONMENT` | Deployment environment (development/staging/production) | development | No |
| `HOST` | Server bind address | 0.0.0.0 | No |
| `PORT` | Server port | 8000 | No |
| `DEBUG` | Enable debug mode | false | No |
| `LOG_LEVEL` | Logging level (DEBUG/INFO/WARNING/ERROR) | INFO | No |
| `SECRET_KEY` | Application secret key | - | Yes (production) |
| `DB_HOST` | Database host | localhost | No |
| `DB_PORT` | Database port | 5432 | No |
| `DB_NAME` | Database name | production_audio | No |
| `DB_USER` | Database user | audio_user | No |
| `DB_PASSWORD` | Database password | - | Yes (production) |
| `REDIS_HOST` | Redis host | localhost | No |
| `REDIS_PORT` | Redis port | 6379 | No |
| `MAXXI_COMPATIBILITY_MODE` | Enable MAXXI integration | true | No |

### Configuration Files

- `config/settings.py` - Main configuration classes
- `config/logging_config.py` - Logging configuration
- `config/docker_config.py` - Container-specific settings

## API Reference

### Authentication

The API uses JWT-based authentication for production deployments:

```bash
# Get access token
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "password"}'

# Use token in requests
curl -H "Authorization: Bearer <token>" \
  http://localhost:8000/api/v1/process
```

### Core Endpoints

#### Health Check
```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "health_score": 0.95,
  "uptime_seconds": 3600,
  "system_metrics": {
    "cpu_percent": 25.5,
    "memory_percent": 45.2,
    "disk_percent": 60.0
  }
}
```

#### Process Audio
```bash
POST /api/v1/process
```

**Request:**
```bash
curl -X POST http://localhost:8000/api/v1/process \
  -F "audio_file=@example.wav" \
  -F "processing_type=comprehensive" \
  -F "priority=1"
```

**Response:**
```json
{
  "task_id": "uuid-task-id",
  "status": "queued",
  "message": "Audio processing task created"
}
```

#### Get Task Status
```bash
GET /api/v1/status/<task_id>
```

**Response:**
```json
{
  "task_id": "uuid-task-id",
  "status": "completed",
  "created_at": "2025-07-21T16:00:00Z",
  "completed_at": "2025-07-21T16:02:30Z",
  "processing_time": 150.5
}
```

#### Get Results
```bash
GET /api/v1/results/<task_id>
```

**Response:**
```json
{
  "task_id": "uuid-task-id",
  "audio_info": {
    "duration": 120.5,
    "sample_rate": 44100,
    "channels": 2
  },
  "results": {
    "heritage_classification": {
      "predicted_category": "folk_music",
      "confidence": 0.87
    },
    "speaker_recognition": {
      "speakers_detected": 2,
      "dominant_speaker": "Speaker_1"
    },
    "metadata_generation": {
      "dublin_core": {...},
      "cultural_tags": {...}
    }
  }
}
```

#### Metrics
```bash
GET /metrics
```

Returns Prometheus-formatted metrics for monitoring.

### MAXXI Integration Endpoints

#### MAXXI Status Callback
```bash
POST /api/maxxi/status
```

#### MAXXI Results Webhook  
```bash
POST /api/maxxi/results
```

#### MAXXI Metadata Export
```bash
GET /api/maxxi/metadata/<task_id>
```

## Processing Modules

### Heritage Classifier

Classifies audio content into cultural categories:

- **Folk Music**: Traditional songs and ballads
- **Classical Music**: Orchestral and chamber works  
- **Opera**: Operatic performances
- **Spoken Word**: Poetry and literary recordings
- **Historical Speeches**: Important historical addresses
- **Religious Chants**: Sacred music and ceremonies

**Usage:**
```python
from heritage_classifier import HeritageClassifier

classifier = HeritageClassifier()
classifier.load_model('cultural_models/pretrained/heritage_classifier_v1.pkl')
result = classifier.classify_audio('audio_file.wav')
```

### Metadata Generator

Generates comprehensive metadata following Dublin Core standards:

- **Technical Metadata**: Sample rate, bit depth, duration
- **Content Analysis**: Tempo, key, harmonic content
- **Cultural Tags**: Geographic, genre, period classifications
- **Dublin Core**: Standardized metadata schema

**Usage:**
```python
from metadata_generator import MetadataGenerator

generator = MetadataGenerator()
metadata = generator.generate_comprehensive_metadata('audio_file.wav')
```

### Speaker Recognition

Identifies and analyzes different speakers:

- **Speaker Diarization**: Who spoke when
- **Voice Characteristics**: Pitch, speech rate, energy
- **Speaker Statistics**: Speaking time, interruptions
- **Timeline Generation**: Detailed speaking timeline

**Usage:**
```python
from speaker_recognition import SpeakerRecognition

recognizer = SpeakerRecognition(n_speakers=2)
recognizer.train_speaker_clustering('training_audio.wav')
results = recognizer.identify_speakers('test_audio.wav')
```

## MAXXI Integration

### Overview

The system is designed to integrate seamlessly with the existing MAXXI installation workflow (`maxxi-installationmaxxi_audio_system.py`). This integration provides:

- **Workflow Compatibility**: Maintains existing MAXXI processing pipelines
- **Data Format Compatibility**: Outputs in MAXXI-compatible formats
- **Metadata Enhancement**: Enriches MAXXI metadata with AI-generated insights
- **Archive Integration**: Connects with MAXXI archival systems

### Integration Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   MAXXI         │    │  Production      │    │  Cultural       │
│   Installation  │◄──►│  Audio System    │◄──►│  AI Models      │
│   System        │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Archive       │    │  Processing      │    │  Heritage       │
│   Storage       │    │  Queue           │    │  Classification │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Configuration

Enable MAXXI integration in your configuration:

```python
# Environment variables
MAXXI_COMPATIBILITY_MODE=true
MAXXI_OUTPUT_FORMAT=json
MAXXI_METADATA_SCHEMA=dublin_core
MAXXI_OUTPUT_DIR=/app/data/maxxi_outputs

# Python configuration
from config.settings import get_maxxi_integration_config
maxxi_config = get_maxxi_integration_config()
```

### Data Flow

1. **Input Processing**: Audio files processed through standard pipeline
2. **AI Enhancement**: Heritage classification and metadata generation
3. **Format Conversion**: Results converted to MAXXI-compatible format
4. **Archive Integration**: Processed data stored in MAXXI archive structure
5. **Callback Notification**: MAXXI system notified of completion

### Example Integration

```python
# Initialize both systems
from production_audio_system import ProductionAudioSystem
from maxxi_integration import MAXXIConnector

production_system = ProductionAudioSystem()
maxxi_connector = MAXXIConnector()

# Process audio with MAXXI integration
def process_for_maxxi(audio_file):
    # Standard processing
    result = production_system.process_audio_comprehensive(audio_file)
    
    # MAXXI-specific formatting
    maxxi_data = maxxi_connector.format_for_maxxi(result)
    
    # Store in MAXXI archive
    maxxi_connector.store_in_archive(maxxi_data)
    
    return maxxi_data
```

## Deployment

### Docker Deployment

#### Single Container
```bash
# Build image
docker build -t audio-ai/production-system .

# Run container
docker run -d \
  --name production-audio-system \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  -e ENVIRONMENT=production \
  audio-ai/production-system
```

#### Docker Compose (Recommended)
```bash
# Production deployment
docker-compose -f docker-compose.yml up -d

# Development deployment
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# MAXXI integration
docker-compose --profile maxxi up -d
```

### Kubernetes Deployment

#### Basic Deployment
```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/namespace.yml
kubectl apply -f k8s/configmap.yml
kubectl apply -f k8s/secrets.yml
kubectl apply -f k8s/deployment.yml
kubectl apply -f k8s/service.yml
kubectl apply -f k8s/ingress.yml
```

#### Helm Chart
```bash
# Install with Helm
helm install production-audio-system ./helm/audio-ai-system \
  --namespace audio-ai \
  --create-namespace \
  --set environment=production \
  --set image.tag=latest
```

### Production Considerations

#### Resource Requirements

**Minimum Requirements:**
- CPU: 2 cores
- RAM: 4GB
- Disk: 20GB
- Network: 100 Mbps

**Recommended for Production:**
- CPU: 8 cores
- RAM: 16GB
- Disk: 100GB SSD
- Network: 1 Gbps

#### Scaling

**Horizontal Scaling:**
```bash
# Scale API servers
docker-compose up -d --scale production-audio-system=3

# Scale workers
docker-compose up -d --scale celery-worker=5
```

**Vertical Scaling:**
```yaml
# Docker Compose resource limits
deploy:
  resources:
    limits:
      cpus: '4.0'
      memory: 8G
    reservations:
      cpus: '2.0'
      memory: 4G
```

## Monitoring & Observability

### Health Monitoring

The system provides comprehensive health monitoring:

- **Application Health**: API responsiveness, worker status
- **System Health**:
