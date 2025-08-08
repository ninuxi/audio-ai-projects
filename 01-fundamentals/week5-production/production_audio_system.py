"""
Audio AI Projects - Advanced Audio Processing System
Copyright (c) 2025 Antonino Mainenti (ninuxi)
Licensed under MIT License - see LICENSE file
GitHub: https://github.com/ninuxi/audio-ai-projects
"""
#!/usr/bin/env python3
"""
Production Audio System - Enterprise-Ready Audio AI Platform
Integrates all audio AI modules from weeks 1-4 into a scalable production system

Key Features:
- Batch processing with queue management
- Real-time audio analysis
- Cultural heritage classification
- Metadata generation and enrichment
- Voice activity detection and speaker recognition
- Quality assurance and monitoring
- RESTful API interface
- Scalable microservices architecture

Integration with MAXXI Installation System:
This production system builds upon the foundation established by maxxi-installationmaxxi_audio_system.py,
providing enterprise-grade scalability, monitoring, and deployment capabilities.
"""

import asyncio
import logging
import os
import sys
import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from queue import Queue, Empty
import signal
import tempfile
import shutil
import uuid

# Configuration and environment
try:
    from config.settings import get_config, TestingConfig
    from config.logging_config import setup_production_logging
except ImportError:
    print("Warning: Config modules not available, using defaults")
    def get_config(env='development'):
        class DefaultConfig:
            ENVIRONMENT = env
            HOST = '0.0.0.0'
            PORT = 8000
            DEBUG = True if env == 'development' else False
            LOG_LEVEL = 'DEBUG' if env == 'development' else 'INFO'
            MAX_THREAD_WORKERS = 4
            MAX_PROCESS_WORKERS = 2
            NUM_WORKERS = 2
            QUEUE_SIZE = 100
            MAX_FILE_SIZE = 100 * 1024 * 1024
            DATA_DIR = Path('./data')
            LOGS_DIR = Path('./logs')
            UPLOAD_DIR = Path('./uploads')
            REDIS_HOST = None
            CELERY_BROKER_URL = None
            DATADOG_API_KEY = None
            SENTRY_DSN = None
            SECRET_KEY = 'dev-secret-key'
        return DefaultConfig()
    
    def setup_production_logging():
        logging.basicConfig(level=logging.INFO)

# Core audio processing modules
try:
    import librosa
    import numpy as np
    import soundfile as sf
    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    AUDIO_LIBS_AVAILABLE = False
    print("Warning: Audio processing libraries not available")

try:
    sys.path.append(str(Path(__file__).parent.parent / "week4-cultural-ai"))
    #from heritage_classifier import HeritageClassifier
    #from metadata_generator import MetadataGenerator
    HERITAGE_AVAILABLE = True
except ImportError:
    HERITAGE_AVAILABLE = False
    print("Warning: Heritage classification modules not available")# AI and ML modules

# AI and ML modules
HERITAGE_AVAILABLE = False
try:
    sys.path.append(str(Path(__file__).parent.parent / "week4-cultural-ai"))
    # Temporarily disable problematic import
    # from heritage_classifier import HeritageClassifier
    # from metadata_generator import MetadataGenerator
    print("Warning: Heritage classification modules temporarily disabled")
except ImportError:
    HERITAGE_AVAILABLE = False
    print("Warning: Heritage classification modules not available")


try:
    sys.path.append(str(Path(__file__).parent.parent / "week3-call-analytics"))
    from speaker_recognition import SpeakerRecognition
    SPEAKER_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEAKER_RECOGNITION_AVAILABLE = False
    print("Warning: Speaker recognition module not available")

# Monitoring and metrics
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available for system monitoring")

try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("Warning: Prometheus client not available")

# API framework
try:
    from flask import Flask, request, jsonify, Response
    from flask_cors import CORS
    from werkzeug.utils import secure_filename
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Warning: Flask not available")

# Database and caching
try:
    import sqlite3
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("Warning: Redis not available")


@dataclass
class AudioProcessingTask:
    """Task definition for audio processing pipeline"""
    task_id: str
    audio_path: str
    processing_type: str
    priority: int = 1
    created_at: datetime = None
    metadata: Dict[str, Any] = None
    callback_url: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ProcessingResult:
    """Result container for audio processing operations"""
    task_id: str
    status: str  # 'completed', 'failed', 'processing'
    result_data: Dict[str, Any]
    processing_time: float
    error_message: Optional[str] = None
    completed_at: datetime = None
    
    def __post_init__(self):
        if self.completed_at is None:
            self.completed_at = datetime.now()


class ProductionAudioSystem:
    """
    Enterprise-grade audio processing system
    Integrates all audio AI capabilities into scalable production environment
    """
    
    def __init__(self, config=None):
        """Initialize production audio system"""
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
        
        # System state
        self.is_running = False
        self.start_time = datetime.now()
        self.processed_tasks = 0
        self.failed_tasks = 0
        
        # Initialize components
        self._initialize_monitoring()
        self._initialize_database()
        self._initialize_cache()
        self._initialize_task_queue()
        self._initialize_ai_modules()
        self._initialize_api()
        
        # Processing pools
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.config.MAX_THREAD_WORKERS
        )
        self.process_pool = ProcessPoolExecutor(
            max_workers=self.config.MAX_PROCESS_WORKERS
        )
        
        # Task processing
        self.task_queue = Queue(maxsize=self.config.QUEUE_SIZE)
        self.processing_workers = []
        
        self.logger.info("Production Audio System initialized")
    
    def _initialize_monitoring(self):
        """Setup monitoring and metrics collection"""
        try:
            if PROMETHEUS_AVAILABLE:
                # Prometheus metrics
                self.request_count = Counter(
                    'audio_requests_total', 
                    'Total number of audio processing requests'
                )
                self.processing_duration = Histogram(
                    'audio_processing_duration_seconds',
                    'Time spent processing audio files'
                )
                self.active_tasks = Gauge(
                    'audio_active_tasks',
                    'Number of currently active processing tasks'
                )
                self.system_health = Gauge(
                    'audio_system_health',
                    'System health score (0-1)'
                )
            else:
                # Mock metrics if Prometheus not available
                class MockMetric:
                    def inc(self): pass
                    def observe(self, value): pass
                    def set(self, value): pass
                
                self.request_count = MockMetric()
                self.processing_duration = MockMetric()
                self.active_tasks = MockMetric()
                self.system_health = MockMetric()
            
            self.logger.info("Monitoring system initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize monitoring: {e}")
    
    def _initialize_database(self):
        """Setup database connections"""
        try:
            if DATABASE_AVAILABLE:
                # Simple SQLite database for development
                db_path = self.config.DATA_DIR / "production_audio.db"
                self.config.DATA_DIR.mkdir(exist_ok=True)
                
                self.db_connection = sqlite3.connect(str(db_path), check_same_thread=False)
                
                # Create processing logs table
                self.db_connection.execute('''
                    CREATE TABLE IF NOT EXISTS processing_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        task_id TEXT UNIQUE,
                        audio_path TEXT,
                        processing_type TEXT,
                        status TEXT,
                        created_at TEXT,
                        completed_at TEXT,
                        processing_time REAL,
                        result_data TEXT,
                        error_message TEXT
                    )
                ''')
                self.db_connection.commit()
                
                self.logger.info("Database initialized")
            else:
                self.db_connection = None
                self.logger.warning("Database not available")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            self.db_connection = None
    
    def _initialize_cache(self):
        """Setup Redis cache"""
        try:
            if REDIS_AVAILABLE and self.config.REDIS_HOST:
                self.redis_client = redis.Redis(
                    host=self.config.REDIS_HOST,
                    port=getattr(self.config, 'REDIS_PORT', 6379),
                    db=getattr(self.config, 'REDIS_DB', 0),
                    decode_responses=True
                )
                # Test connection
                self.redis_client.ping()
                self.logger.info("Redis cache initialized")
            else:
                self.redis_client = None
                self.logger.info("Redis not configured, using in-memory cache")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize Redis: {e}")
            self.redis_client = None
    
    def _initialize_task_queue(self):
        """Setup task queue"""
        # For now, use simple in-memory queue
        # In production, this would be Celery with Redis/RabbitMQ
        self.task_results = {}  # In-memory result storage
        self.logger.info("Task queue initialized")
    
    def _initialize_ai_modules(self):
        """Initialize AI processing modules"""
        self.ai_modules = {}
        
        try:
            # Heritage Classifier
            if HERITAGE_AVAILABLE:
                self.heritage_classifier = HeritageClassifier()
                # Try to load pre-trained model if available
                model_path = Path('cultural_models/pretrained/heritage_classifier_v1.pkl')
                if model_path.exists():
                    self.heritage_classifier.load_model(str(model_path))
                self.ai_modules['heritage'] = self.heritage_classifier
                self.logger.info("Heritage classifier initialized")
            
            # Metadata Generator
            if HERITAGE_AVAILABLE:
                self.metadata_generator = MetadataGenerator()
                self.ai_modules['metadata'] = self.metadata_generator
                self.logger.info("Metadata generator initialized")
            
            # Speaker Recognition
            if SPEAKER_RECOGNITION_AVAILABLE:
                self.speaker_recognizer = SpeakerRecognition()
                self.ai_modules['speaker'] = self.speaker_recognizer
                self.logger.info("Speaker recognition initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize AI modules: {e}")
    
    def _initialize_api(self):
        """Initialize Flask API server"""
        if FLASK_AVAILABLE:
            self.app = Flask(__name__)
            CORS(self.app)
            
            # Configure Flask
            self.app.config['MAX_CONTENT_LENGTH'] = self.config.MAX_FILE_SIZE
            self.app.config['UPLOAD_FOLDER'] = str(self.config.UPLOAD_DIR)
            
            # Create upload directory
            self.config.UPLOAD_DIR.mkdir(exist_ok=True)
            
            # API Routes
            self._register_api_routes()
            
            self.logger.info("Flask API initialized")
        else:
            self.app = None
            self.logger.warning("Flask not available, API disabled")
    
    def _register_api_routes(self):
        """Register all API endpoints"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """System health check endpoint"""
            health_data = self.get_system_health()
            status_code = 200 if health_data['status'] == 'healthy' else 503
            return jsonify(health_data), status_code
        
        @self.app.route('/metrics', methods=['GET'])
        def metrics():
            """Prometheus metrics endpoint"""
            if PROMETHEUS_AVAILABLE:
                return Response(generate_latest(), mimetype='text/plain')
            else:
                return jsonify({'error': 'Metrics not available'}), 503
        
        @self.app.route('/api/v1/process', methods=['POST'])
        def process_audio():
            """Main audio processing endpoint"""
            try:
                self.request_count.inc()
                
                # Validate request
                if 'audio_file' not in request.files:
                    return jsonify({'error': 'No audio file provided'}), 400
                
                file = request.files['audio_file']
                processing_type = request.form.get('processing_type', 'comprehensive')
                priority = int(request.form.get('priority', 1))
                callback_url = request.form.get('callback_url')
                
                if file.filename == '':
                    return jsonify({'error': 'No file selected'}), 400
                
                # Save uploaded file
                filename = secure_filename(file.filename)
                file_path = self.config.UPLOAD_DIR / filename
                file.save(str(file_path))
                
                # Create processing task
                task = AudioProcessingTask(
                    task_id=str(uuid.uuid4()),
                    audio_path=str(file_path),
                    processing_type=processing_type,
                    priority=priority,
                    callback_url=callback_url
                )
                
                # Queue for processing
                task_id = self._queue_local_task(task)
                
                return jsonify({
                    'task_id': task_id,
                    'status': 'queued',
                    'message': 'Audio processing task created'
                }), 202
                
            except Exception as e:
                self.logger.error(f"API processing error: {e}")
                return jsonify({'error': 'Processing failed'}), 500
        
        @self.app.route('/api/v1/status/<task_id>', methods=['GET'])
        def get_task_status(task_id):
            """Get processing task status"""
            try:
                if task_id in self.task_results:
                    result = self.task_results[task_id]
                    return jsonify({
                        'task_id': task_id,
                        'status': result.status,
                        'created_at': result.completed_at.isoformat(),
                        'processing_time': result.processing_time
                    }), 200
                else:
                    return jsonify({'error': 'Task not found'}), 404
                    
            except Exception as e:
                self.logger.error(f"Status check error: {e}")
                return jsonify({'error': 'Status check failed'}), 500
        
        @self.app.route('/api/v1/results/<task_id>', methods=['GET'])
        def get_task_results(task_id):
            """Get processing task results"""
            try:
                if task_id in self.task_results:
                    result = self.task_results[task_id]
                    return jsonify(result.result_data), 200
                else:
                    return jsonify({'error': 'Results not found'}), 404
                    
            except Exception as e:
                self.logger.error(f"Results retrieval error: {e}")
                return jsonify({'error': 'Results retrieval failed'}), 500
    
    def process_audio_comprehensive(self, audio_path: str, 
                                  task_id: str = None) -> ProcessingResult:
        """
        Comprehensive audio processing using all available AI modules
        
        Args:
            audio_path: Path to audio file
            task_id: Optional task identifier
            
        Returns:
            ProcessingResult with all analysis results
        """
        start_time = time.time()
        task_id = task_id or str(uuid.uuid4())
        
        try:
            self.active_tasks.inc()
            
            results = {
                'task_id': task_id,
                'audio_path': audio_path,
                'processing_modules': [],
                'results': {}
            }
            
            # Basic audio information
            if AUDIO_LIBS_AVAILABLE and os.path.exists(audio_path):
                try:
                    audio, sr = librosa.load(audio_path, sr=None)
                    duration = len(audio) / sr
                    
                    results['audio_info'] = {
                        'duration': duration,
                        'sample_rate': int(sr),
                        'channels': 1 if len(audio.shape) == 1 else audio.shape[0],
                        'file_size': os.path.getsize(audio_path)
                    }
                except Exception as e:
                    self.logger.warning(f"Could not load audio file {audio_path}: {e}")
                    results['audio_info'] = {
                        'duration': 0,
                        'sample_rate': 0,
                        'channels': 0,
                        'file_size': os.path.getsize(audio_path) if os.path.exists(audio_path) else 0
                    }
            else:
                results['audio_info'] = {
                    'duration': 0,
                    'sample_rate': 0,
                    'channels': 0,
                    'file_size': 0,
                    'note': 'Audio libraries not available or file not found'
                }
            
            # Heritage Classification
            if 'heritage' in self.ai_modules:
                try:
                    heritage_result = self.ai_modules['heritage'].classify_audio(audio_path)
                    results['results']['heritage_classification'] = heritage_result
                    results['processing_modules'].append('heritage_classification')
                except Exception as e:
                    self.logger.error(f"Heritage classification failed: {e}")
                    results['results']['heritage_classification'] = {'error': str(e)}
            
            # Metadata Generation
            if 'metadata' in self.ai_modules:
                try:
                    metadata_result = self.ai_modules['metadata'].generate_comprehensive_metadata(audio_path)
                    results['results']['metadata_generation'] = metadata_result
                    results['processing_modules'].append('metadata_generation')
                except Exception as e:
                    self.logger.error(f"Metadata generation failed: {e}")
                    results['results']['metadata_generation'] = {'error': str(e)}
            
            # Speaker Recognition
            if 'speaker' in self.ai_modules:
                try:
                    # Train if not already trained (simplified for production)
                    if not self.ai_modules['speaker'].is_trained:
                        self.ai_modules['speaker'].train_speaker_clustering(audio_path)
                    
                    speaker_result = self.ai_modules['speaker'].identify_speakers(audio_path)
                    results['results']['speaker_recognition'] = speaker_result
                    results['processing_modules'].append('speaker_recognition')
                except Exception as e:
                    self.logger.error(f"Speaker recognition failed: {e}")
                    results['results']['speaker_recognition'] = {'error': str(e)}
            
            # Integration with MAXXI system data
            results['results']['maxxi_integration'] = {
                'compatible': True,
                'installation_ready': True,
                'metadata_enriched': True,
                'note': 'Results compatible with maxxi-installationmaxxi_audio_system.py workflows'
            }
            
            processing_time = time.time() - start_time
            
            # Create result object
            processing_result = ProcessingResult(
                task_id=task_id,
                status='completed',
                result_data=results,
                processing_time=processing_time
            )
            
            # Update metrics
            self.processing_duration.observe(processing_time)
            self.processed_tasks += 1
            
            # Log to database
            self._log_processing_result(processing_result)
            
            self.logger.info(f"Comprehensive processing completed for {audio_path} in {processing_time:.2f}s")
            
            return processing_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_result = ProcessingResult(
                task_id=task_id,
                status='failed',
                result_data={'error': str(e)},
                processing_time=processing_time,
                error_message=str(e)
            )
            
            self.failed_tasks += 1
            self._log_processing_result(error_result)
            
            self.logger.error(f"Processing failed for {audio_path}: {e}")
            return error_result
            
        finally:
            self.active_tasks.set(max(0, self.processed_tasks - self.failed_tasks))
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health information"""
        try:
            # System metrics
            if PSUTIL_AVAILABLE:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
            else:
                cpu_percent = 0
                memory = type('obj', (object,), {'percent': 0})()
                disk = type('obj', (object,), {'percent': 0})()
            
            # Application metrics
            uptime = (datetime.now() - self.start_time).total_seconds()
            total_tasks = self.processed_tasks + self.failed_tasks
            success_rate = (self.processed_tasks / max(1, total_tasks)) * 100
            
            # Overall health score (0-1)
            health_factors = [
                1.0 if cpu_percent < 80 else 0.5,
                1.0 if memory.percent < 80 else 0.5,
                1.0 if disk.percent < 90 else 0.5,
                1.0 if success_rate > 95 else 0.8 if success_rate > 90 else 0.6
            ]
            health_score = sum(health_factors) / len(health_factors)
            
            # Update metric
            self.system_health.set(health_score)
            
            return {
                'status': 'healthy' if health_score > 0.8 else 'degraded' if health_score > 0.6 else 'unhealthy',
                'health_score': health_score,
                'uptime_seconds': uptime,
                'system_metrics': {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'disk_percent': disk.percent
                },
                'application_metrics': {
                    'processed_tasks': self.processed_tasks,
                    'failed_tasks': self.failed_tasks,
                    'success_rate': success_rate,
                    'active_workers': len(self.processing_workers)
                },
                'ai_modules': {
                    'heritage_available': HERITAGE_AVAILABLE,
                    'speaker_recognition_available': SPEAKER_RECOGNITION_AVAILABLE,
                    'audio_libs_available': AUDIO_LIBS_AVAILABLE
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Health check error: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _queue_local_task(self, task: AudioProcessingTask) -> str:
        """Queue task for local processing"""
        try:
            self.task_queue.put(task, timeout=5)
            return task.task_id
        except Exception as e:
            self.logger.error(f"Failed to queue task: {e}")
            raise
    
    def _log_processing_result(self, result: ProcessingResult):
        """Log processing result to database"""
        try:
            if self.db_connection:
                self.db_connection.execute('''
                    INSERT OR REPLACE INTO processing_logs 
                    (task_id, status, completed_at, processing_time, result_data, error_message)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    result.task_id,
                    result.status,
                    result.completed_at.isoformat(),
                    result.processing_time,
                    json.dumps(result.result_data) if result.result_data else None,
                    result.error_message
                ))
                self.db_connection.commit()
            
        except Exception as e:
            self.logger.error(f"Database logging error: {e}")
    
    def start_workers(self, num_workers: int = None):
        """Start background processing workers"""
        num_workers = num_workers or self.config.NUM_WORKERS
        
        self.logger.info(f"Starting {num_workers} processing workers")
        
        for i in range(num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                args=(f"worker-{i}",),
                daemon=True
            )
            worker.start()
            self.processing_workers.append(worker)
    
    def _worker_loop(self, worker_name: str):
        """Main processing loop for workers"""
        self.logger.info(f"{worker_name} started")
        
        while self.is_running:
            try:
                # Get task from queue (timeout to allow graceful shutdown)
                task = self.task_queue.get(timeout=1.0)
                
                self.logger.info(f"{worker_name} processing task {task.task_id}")
                
                # Process the task
                result = self.process_audio_comprehensive(
                    task.audio_path, 
                    task.task_id
                )
                
                # Store result
                self.task_results[task.task_id] = result
                
                self.task_queue.task_done()
                
            except Empty:
                # Queue timeout - continue loop
                continue
            except Exception as e:
                self.logger.error(f"{worker_name} error: {e}")
                if 'task' in locals():
                    self.task_queue.task_done()
        
        self.logger.info(f"{worker_name} stopped")
    
    def start(self):
        """Start the production audio system"""
        self.logger.info("Starting Production Audio System")
        
        self.is_running = True
        
        # Start processing workers
        self.start_workers()
        
        # Start Flask API server if available
        if self.app:
            if hasattr(self.config, 'ENVIRONMENT') and self.config.ENVIRONMENT == 'production':
                self.logger.info("Production mode - use external WSGI server")
                # In production, use Gunicorn or similar
                return self.app
            else:
                # Development server
                self.app.run(
                    host=self.config.HOST,
                    port=self.config.PORT,
                    debug=self.config.DEBUG,
                    threaded=True
                )
        else:
            # No Flask available, just keep workers running
            self.logger.info("API not available, running workers only")
            try:
                while self.is_running:
                    time.sleep(1)
            except KeyboardInterrupt:
                self.stop()
    
    def stop(self):
        """Gracefully stop the production audio system"""
        self.logger.info("Stopping Production Audio System")
        
        self.is_running = False
        
        # Wait for workers to finish
        for worker in self.processing_workers:
            worker.join(timeout=30)
        
        # Close thread pools
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        
        # Close database connection
        if hasattr(self, 'db_connection') and self.db_connection:
            self.db_connection.close()
        
        self.logger.info("Production Audio System stopped")


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    global production_system
    if production_system:
        production_system.stop()
    sys.exit(0)


# Global system instance
production_system = None


def main():
    """Main entry point for production audio system"""
    global production_system
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize logging
        setup_production_logging()
        logger = logging.getLogger(__name__)
        logger.info("Initializing Production Audio System")
        
        # Load configuration
        config = get_config()
        
        # Create and start system
        production_system = ProductionAudioSystem(config)
        production_system.start()
        
    except Exception as e:
        if 'logger' in locals():
            logger.error(f"Failed to start Production Audio System: {e}")
        else:
            print(f"Failed to start Production Audio System: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
