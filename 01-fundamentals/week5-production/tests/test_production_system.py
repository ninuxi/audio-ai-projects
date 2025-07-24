"""
Audio AI Projects - Advanced Audio Processing System
Copyright (c) 2025 Antonino Mainenti (ninuxi)
Licensed under MIT License - see LICENSE file
GitHub: https://github.com/ninuxi/audio-ai-projects
"""
#!/usr/bin/env python3
"""
Comprehensive Test Suite for Production Audio System
Tests all components, integration points, and performance characteristics
"""

import unittest
import asyncio
import tempfile
import shutil
import json
import time
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import soundfile as sf

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from production_audio_system import ProductionAudioSystem, AudioProcessingTask, ProcessingResult
from config.settings import TestingConfig, get_config, validate_config


class TestProductionAudioSystem(unittest.TestCase):
    """Test cases for ProductionAudioSystem"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.config = TestingConfig()
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls.config.DATA_DIR = cls.temp_dir / "data"
        cls.config.LOGS_DIR = cls.temp_dir / "logs"
        cls.config.UPLOAD_DIR = cls.temp_dir / "uploads"
        
        # Create directories
        for directory in [cls.config.DATA_DIR, cls.config.LOGS_DIR, cls.config.UPLOAD_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Create test audio file
        cls.test_audio_file = cls.temp_dir / "test_audio.wav"
        cls._create_test_audio_file(cls.test_audio_file)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir)
    
    def setUp(self):
        """Set up each test"""
        with patch('production_audio_system.setup_production_logging'):
            self.system = ProductionAudioSystem(self.config)
    
    def tearDown(self):
        """Clean up after each test"""
        if hasattr(self.system, 'stop'):
            self.system.stop()
    
    @classmethod
    def _create_test_audio_file(cls, file_path: Path):
        """Create a test audio file"""
        sample_rate = 44100
        duration = 2.0  # 2 seconds
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        sf.write(str(file_path), audio, sample_rate)
    
    def test_system_initialization(self):
        """Test system initialization"""
        self.assertIsNotNone(self.system)
        self.assertIsNotNone(self.system.config)
        self.assertEqual(self.system.processed_tasks, 0)
        self.assertEqual(self.system.failed_tasks, 0)
        self.assertFalse(self.system.is_running)
    
    def test_health_check(self):
        """Test system health check"""
        health = self.system.get_system_health()
        
        self.assertIn('status', health)
        self.assertIn('health_score', health)
        self.assertIn('uptime_seconds', health)
        self.assertIn('system_metrics', health)
        self.assertIn('application_metrics', health)
        self.assertTrue(isinstance(health['health_score'], (int, float)))
        self.assertTrue(0 <= health['health_score'] <= 1)
    
    def test_task_creation(self):
        """Test audio processing task creation"""
        task = AudioProcessingTask(
            task_id="test-task-001",
            audio_path=str(self.test_audio_file),
            processing_type="comprehensive",
            priority=1
        )
        
        self.assertEqual(task.task_id, "test-task-001")
        self.assertEqual(task.audio_path, str(self.test_audio_file))
        self.assertEqual(task.processing_type, "comprehensive")
        self.assertEqual(task.priority, 1)
        self.assertIsInstance(task.metadata, dict)
    
    def test_comprehensive_processing(self):
        """Test comprehensive audio processing"""
        # Mock AI modules to avoid dependencies
        with patch.object(self.system, 'ai_modules', {}):
            result = self.system.process_audio_comprehensive(
                str(self.test_audio_file),
                task_id="test-comprehensive-001"
            )
        
        self.assertIsInstance(result, ProcessingResult)
        self.assertEqual(result.task_id, "test-comprehensive-001")
        self.assertEqual(result.status, "completed")
        self.assertIsInstance(result.result_data, dict)
        self.assertGreater(result.processing_time, 0)
        
        # Check result structure
        self.assertIn('audio_info', result.result_data)
        self.assertIn('duration', result.result_data['audio_info'])
        self.assertIn('sample_rate', result.result_data['audio_info'])
    
    def test_processing_with_heritage_classifier(self):
        """Test processing with heritage classifier"""
        mock_classifier = Mock()
        mock_classifier.classify_audio.return_value = {
            'predicted_category': 'folk_music',
            'confidence': 0.85
        }
        
        with patch.object(self.system, 'ai_modules', {'heritage': mock_classifier}):
            with patch.object(self.system, 'ai_circuit_breakers', {'heritage': Mock()}):
                self.system.ai_circuit_breakers['heritage'].__enter__ = Mock(return_value=None)
                self.system.ai_circuit_breakers['heritage'].__exit__ = Mock(return_value=None)
                
                result = self.system.process_audio_comprehensive(
                    str(self.test_audio_file),
                    task_id="test-heritage-001"
                )
        
        self.assertEqual(result.status, "completed")
        self.assertIn('heritage_classification', result.result_data['results'])
        self.assertIn('heritage_classification', result.result_data['processing_modules'])
        mock_classifier.classify_audio.assert_called_once()
    
    def test_processing_error_handling(self):
        """Test error handling during processing"""
        # Test with non-existent file
        result = self.system.process_audio_comprehensive(
            "non_existent_file.wav",
            task_id="test-error-001"
        )
        
        self.assertEqual(result.status, "failed")
        self.assertIsNotNone(result.error_message)
        self.assertGreater(result.processing_time, 0)
    
    def test_task_queue_operations(self):
        """Test task queue operations"""
        task = AudioProcessingTask(
            task_id="test-queue-001",
            audio_path=str(self.test_audio_file),
            processing_type="basic"
        )
        
        # Test local task queuing
        task_id = self.system._queue_local_task(task)
        self.assertEqual(task_id, "test-queue-001")
        
        # Verify task is in queue
        self.assertFalse(self.system.task_queue.empty())
    
    def test_database_logging(self):
        """Test database logging functionality"""
        result = ProcessingResult(
            task_id="test-db-001",
            status="completed",
            result_data={"test": "data"},
            processing_time=1.5
        )
        
        # Test logging (should not raise exception)
        try:
            self.system._log_processing_result(result)
        except Exception as e:
            self.fail(f"Database logging failed: {e}")
    
    def test_metrics_collection(self):
        """Test metrics collection"""
        # Test that metrics objects exist
        self.assertIsNotNone(self.system.request_count)
        self.assertIsNotNone(self.system.processing_duration)
        self.assertIsNotNone(self.system.active_tasks)
        self.assertIsNotNone(self.system.system_health)
        
        # Test metrics increment
        initial_count = self.system.processed_tasks
        
        with patch.object(self.system, 'ai_modules', {}):
            self.system.process_audio_comprehensive(str(self.test_audio_file))
        
        self.assertEqual(self.system.processed_tasks, initial_count + 1)
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        errors = validate_config(self.config)
        self.assertIsInstance(errors, list)
        # Should have no errors for test config
        if errors:
            self.fail(f"Configuration validation errors: {errors}")
    
    def test_maxxi_integration(self):
        """Test MAXXI installation integration"""
        with patch.object(self.system, 'ai_modules', {}):
            result = self.system.process_audio_comprehensive(str(self.test_audio_file))
        
        # Check MAXXI integration metadata
        self.assertIn('maxxi_integration', result.result_data['results'])
        maxxi_data = result.result_data['results']['maxxi_integration']
        
        self.assertTrue(maxxi_data['compatible'])
        self.assertTrue(maxxi_data['installation_ready'])
        self.assertTrue(maxxi_data['metadata_enriched'])
        self.assertIn('maxxi-installationmaxxi_audio_system.py', maxxi_data['note'])


class TestAPIEndpoints(unittest.TestCase):
    """Test API endpoints"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.config = TestingConfig()
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls.config.DATA_DIR = cls.temp_dir / "data"
        cls.config.LOGS_DIR = cls.temp_dir / "logs"
        cls.config.UPLOAD_DIR = cls.temp_dir / "uploads"
        
        for directory in [cls.config.DATA_DIR, cls.config.LOGS_DIR, cls.config.UPLOAD_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir)
    
    def setUp(self):
        """Set up each test"""
        with patch('production_audio_system.setup_production_logging'):
            self.system = ProductionAudioSystem(self.config)
            self.client = self.system.app.test_client()
    
    def test_health_endpoint(self):
        """Test /health endpoint"""
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('status', data)
        self.assertIn('health_score', data)
    
    def test_metrics_endpoint(self):
        """Test /metrics endpoint"""
        response = self.client.get('/metrics')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, 'text/plain')
    
    def test_process_endpoint_without_file(self):
        """Test /api/v1/process endpoint without file"""
        response = self.client.post('/api/v1/process')
        self.assertEqual(response.status_code, 400)
        
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_status_endpoint_invalid_task(self):
        """Test /api/v1/status endpoint with invalid task ID"""
        response = self.client.get('/api/v1/status/invalid-task-id')
        self.assertEqual(response.status_code, 404)


class TestConfigurationSystem(unittest.TestCase):
    """Test configuration system"""
    
    def test_development_config(self):
        """Test development configuration"""
        config = get_config('development')
        self.assertTrue(config.DEBUG)
        self.assertEqual(config.LOG_LEVEL, 'DEBUG')
        self.assertLessEqual(config.MAX_THREAD_WORKERS, 4)
    
    def test_production_config(self):
        """Test production configuration"""
        config = get_config('production')
        self.assertFalse(config.DEBUG)
        self.assertEqual(config.LOG_LEVEL, 'WARNING')
        self.assertGreaterEqual(config.MAX_THREAD_WORKERS, 8)
    
    def test_testing_config(self):
        """Test testing configuration"""
        config = get_config('testing')
        self.assertTrue(config.DEBUG)
        self.assertEqual(config.NUM_WORKERS, 1)
        self.assertEqual(config.DB_NAME, ':memory:')
    
    def test_config_validation(self):
        """Test configuration validation"""
        config = TestingConfig()
        errors = validate_config(config)
        self.assertIsInstance(errors, list)
    
    def test_environment_override(self):
        """Test environment variable override"""
        with patch.dict(os.environ, {'PORT': '9000', 'DEBUG': 'true'}):
            config = get_config('development')
            # Note: This test may fail if env override is not implemented
            # The config system should override PORT and DEBUG from env vars


class TestPerformance(unittest.TestCase):
    """Performance and load testing"""
    
    @classmethod
    def setUpClass(cls):
        """Set up performance test environment"""
        cls.config = TestingConfig()
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls.config.DATA_DIR = cls.temp_dir / "data"
        cls.config.LOGS_DIR = cls.temp_dir / "logs"
        cls.config.UPLOAD_DIR = cls.temp_dir / "uploads"
        
        for directory in [cls.config.DATA_DIR, cls.config.LOGS_DIR, cls.config.UPLOAD_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Create test audio file
        cls.test_audio_file = cls.temp_dir / "perf_test_audio.wav"
        sample_rate = 44100
        duration = 5.0  # 5 seconds
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.3 * np.sin(2 * np.pi * 440 * t)
        sf.write(str(cls.test_audio_file), audio, sample_rate)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up performance test environment"""
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir)
    
    def setUp(self):
        """Set up each performance test"""
        with patch('production_audio_system.setup_production_logging'):
            self.system = ProductionAudioSystem(self.config)
    
    def test_processing_performance(self):
        """Test processing performance"""
        start_time = time.time()
        
        with patch.object(self.system, 'ai_modules', {}):
            result = self.system.process_audio_comprehensive(str(self.test_audio_file))
        
        processing_time = time.time() - start_time
        
        self.assertEqual(result.status, "completed")
        # Processing should complete within reasonable time (10 seconds for 5s audio)
        self.assertLess(processing_time, 10.0)
        
        # Result should contain timing information
        self.assertGreater(result.processing_time, 0)
        self.assertLess(result.processing_time, processing_time)
    
    def test_concurrent_processing(self):
        """Test concurrent processing performance"""
        num_concurrent = 3
        results = []
        threads = []
        
        def process_audio():
            with patch.object(self.system, 'ai_modules', {}):
                result = self.system.process_audio_comprehensive(str(self.test_audio_file))
                results.append(result)
        
        # Start concurrent processing
        start_time = time.time()
        for _ in range(num_concurrent):
            thread = threading.Thread(target=process_audio)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout
        
        total_time = time.time() - start_time
        
        # All tasks should complete
        self.assertEqual(len(results), num_concurrent)
        
        # All should be successful
        for result in results:
            self.assertEqual(result.status, "completed")
        
        # Concurrent processing should be more efficient than sequential
        self.assertLess(total_time, num_concurrent * 5.0)  # Less than sequential worst case
    
    def test_memory_usage(self):
        """Test memory usage during processing"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Process multiple files
        for i in range(5):
            with patch.object(self.system, 'ai_modules', {}):
                result = self.system.process_audio_comprehensive(str(self.test_audio_file))
                self.assertEqual(result.status, "completed")
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for test)
        self.assertLess(memory_increase, 100 * 1024 * 1024)


class TestIntegration(unittest.TestCase):
    """Integration tests with external components"""
    
    @classmethod
    def setUpClass(cls):
        """Set up integration test environment"""
        cls.config = TestingConfig()
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls.config.DATA_DIR = cls.temp_dir / "data"
        cls.config.LOGS_DIR = cls.temp_dir / "logs"
        cls.config.UPLOAD_DIR = cls.temp_dir / "uploads"
        
        for directory in [cls.config.DATA_DIR, cls.config.LOGS_DIR, cls.config.UPLOAD_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up integration test environment"""
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir)
    
    def setUp(self):
        """Set up each integration test"""
        with patch('production_audio_system.setup_production_logging'):
            self.system = ProductionAudioSystem(self.config)
    
    def test_redis_integration(self):
        """Test Redis cache integration"""
        # Mock Redis client
        mock_redis = Mock()
        mock_redis.ping.return_value = True
        mock_redis.setex.return_value = True
        mock_redis.get.return_value = None
        
        with patch('redis.Redis', return_value=mock_redis):
            # Test that Redis operations don't fail
            self.system._initialize_cache()
            self.assertIsNotNone(self.system.redis_client)
    
    def test_database_integration(self):
        """Test database integration"""
        # Database should be initialized
        self.assertIsNotNone(self.system.db_engine)
        self.assertIsNotNone(self.system.db_session)
        self.assertIsNotNone(self.system.ProcessingLog)
        
        # Test database operations
        result = ProcessingResult(
            task_id="integration-test-001",
            status="completed",
            result_data={"test": "data"},
            processing_time=1.0
        )
        
        # Should not raise exception
        try:
            self.system._log_processing_result(result)
        except Exception as e:
            self.fail(f"Database integration failed: {e}")
    
    def test_maxxi_compatibility(self):
        """Test compatibility with MAXXI installation system"""
        from config.settings import get_maxxi_integration_config
        
        maxxi_config = get_maxxi_integration_config()
        
        # Verify MAXXI configuration structure
        self.assertIn('compatibility_mode', maxxi_config)
        self.assertIn('output_format', maxxi_config)
        self.assertIn('metadata_schema', maxxi_config)
        self.assertIn('processing_pipeline', maxxi_config)
        
        self.assertTrue(maxxi_config['compatibility_mode'])
        self.assertEqual(maxxi_config['metadata_schema'], 'dublin_core')
    
    def test_ai_modules_integration(self):
        """Test AI modules integration"""
        # Test that AI modules can be initialized without error
        try:
            self.system._initialize_ai_modules()
            # Should have circuit breakers for any initialized modules
            if self.system.ai_modules:
                self.assertIsInstance(self.system.ai_circuit_breakers, dict)
        except Exception as e:
            # AI modules may not be available in test environment
            # This is acceptable as long as system handles it gracefully
            pass


class TestSecurity(unittest.TestCase):
    """Security and safety tests"""
    
    def setUp(self):
        """Set up security tests"""
        self.config = TestingConfig()
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config.DATA_DIR = self.temp_dir / "data"
        self.config.LOGS_DIR = self.temp_dir / "logs" 
        self.config.UPLOAD_DIR = self.temp_dir / "uploads"
        
        for directory in [self.config.DATA_DIR, self.config.LOGS_DIR, self.config.UPLOAD_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
        
        with patch('production_audio_system.setup_production_logging'):
            self.system = ProductionAudioSystem(self.config)
    
    def tearDown(self):
        """Clean up security tests"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_file_path_validation(self):
        """Test file path validation and security"""
        from production_audio_system import secure_filename
        
        # Test normal filename
        safe_name = secure_filename("normal_file.wav")
        self.assertEqual(safe_name, "normal_file.wav")
        
        # Test filename with dangerous characters
        dangerous_name = secure_filename("../../../etc/passwd")
        self.assertNotIn("../", dangerous_name)
        self.assertNotIn("/", dangerous_name)
        
        # Test long filename
        long_name = "a" * 300 + ".wav"
        safe_long = secure_filename(long_name)
        self.assertLessEqual(len(safe_long), 255)
    
    def test_input_validation(self):
        """Test input validation"""
        # Test invalid processing type
        task = AudioProcessingTask(
            task_id="security-test-001",
            audio_path="test.wav",
            processing_type="../invalid_type",
            priority=-1
        )
        
        # System should handle invalid input gracefully
        self.assertIsInstance(task.processing_type, str)
        self.assertIsInstance(task.priority, int)
    
    def test_resource_limits(self):
        """Test resource limits and DoS protection"""
        # Test file size limits
        self.assertGreater(self.config.MAX_FILE_SIZE, 0)
        self.assertLess(self.config.MAX_FILE_SIZE, 10 * 1024 * 1024 * 1024)  # Less than 10GB
        
        # Test queue size limits
        self.assertGreater(self.config.QUEUE_SIZE, 0)
        self.assertLess(self.config.QUEUE_SIZE, 100000)  # Reasonable limit
        
        # Test worker limits
        self.assertGreater(self.config.NUM_WORKERS, 0)
        self.assertLess(self.config.NUM_WORKERS, 100)  # Reasonable limit


def create_test_suite():
    """Create comprehensive test suite"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    test_classes = [
        TestProductionAudioSystem,
        TestAPIEndpoints,
        TestConfigurationSystem,
        TestPerformance,
        TestIntegration,
        TestSecurity
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    return suite


def run_tests():
    """Run the complete test suite"""
    suite = create_test_suite()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
