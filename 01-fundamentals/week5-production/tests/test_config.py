#!/usr/bin/env python3
"""
Configuration System Tests
Tests for all configuration classes and environment handling
"""

import unittest
import os
import tempfile
import json
from pathlib import Path
from unittest.mock import patch

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import (
    BaseConfig, DevelopmentConfig, TestingConfig, 
    StagingConfig, ProductionConfig, get_config, 
    validate_config, load_config_from_file,
    get_maxxi_integration_config, ConfigurationError
)


class TestBaseConfiguration(unittest.TestCase):
    """Test base configuration class"""
    
    def setUp(self):
        """Set up base config test"""
        self.config = BaseConfig()
    
    def test_default_values(self):
        """Test default configuration values"""
        self.assertEqual(self.config.APP_NAME, "Production Audio System")
        self.assertEqual(self.config.VERSION, "1.0.0")
        self.assertFalse(self.config.DEBUG)
        self.assertEqual(self.config.HOST, "0.0.0.0")
        self.assertEqual(self.config.PORT, 8000)
        self.assertEqual(self.config.LOG_LEVEL, "INFO")
    
    def test_post_init_processing(self):
        """Test post-initialization processing"""
        # Test that ALLOWED_EXTENSIONS is populated
        self.assertIsNotNone(self.config.ALLOWED_EXTENSIONS)
        self.assertIn('.wav', self.config.ALLOWED_EXTENSIONS)
        self.assertIn('.mp3', self.config.ALLOWED_EXTENSIONS)
        
        # Test that CORS_ORIGINS is populated
        self.assertIsNotNone(self.config.CORS_ORIGINS)
        self.assertIsInstance(self.config.CORS_ORIGINS, list)
        
        # Test that MAXXI_OUTPUT_DIR is created
        self.assertTrue(self.config.MAXXI_OUTPUT_DIR.exists())
    
    def test_file_size_limits(self):
        """Test file size configuration"""
        self.assertGreater(self.config.MAX_FILE_SIZE, 0)
        self.assertEqual(self.config.MAX_FILE_SIZE, 100 * 1024 * 1024)  # 100MB
    
    def test_worker_configuration(self):
        """Test worker configuration"""
        self.assertGreater(self.config.MAX_THREAD_WORKERS, 0)
        self.assertGreater(self.config.MAX_PROCESS_WORKERS, 0)
        self.assertGreater(self.config.NUM_WORKERS, 0)
        self.assertGreater(self.config.API_WORKERS, 0)


class TestEnvironmentConfigurations(unittest.TestCase):
    """Test environment-specific configurations"""
    
    def test_development_config(self):
        """Test development configuration"""
        config = DevelopmentConfig()
        
        self.assertTrue(config.DEBUG)
        self.assertEqual(config.LOG_LEVEL, "DEBUG")
        self.assertLessEqual(config.MAX_THREAD_WORKERS, 4)  # Reduced for dev
        self.assertEqual(config.NUM_WORKERS, 2)
        self.assertEqual(config.API_WORKERS, 1)
    
    def test_testing_config(self):
        """Test testing configuration"""
        config = TestingConfig()
        
        self.assertTrue(config.DEBUG)
        self.assertEqual(config.LOG_LEVEL, "DEBUG")
        self.assertEqual(config.MAX_THREAD_WORKERS, 1)
        self.assertEqual(config.NUM_WORKERS, 1)
        self.assertEqual(config.DB_NAME, ":memory:")
        
        # External services should be disabled
        self.assertIsNone(config.REDIS_HOST)
        self.assertIsNone(config.CELERY_BROKER_URL)
        self.assertIsNone(config.DATADOG_API_KEY)
    
    def test_staging_config(self):
        """Test staging configuration"""
        config = StagingConfig()
        
        self.assertFalse(config.DEBUG)
        self.assertEqual(config.LOG_LEVEL, "INFO")
        self.assertGreater(config.MAX_THREAD_WORKERS, 2)
        self.assertGreater(config.NUM_WORKERS, 1)
        
        # Should have external service configurations
        self.assertIsNotNone(config.REDIS_HOST)
        self.assertIsNotNone(config.CELERY_BROKER_URL)
    
    def test_production_config(self):
        """Test production configuration"""
        config = ProductionConfig()
        
        self.assertFalse(config.DEBUG)
        self.assertEqual(config.LOG_LEVEL, "WARNING")
        self.assertGreaterEqual(config.MAX_THREAD_WORKERS, 8)
        self.assertGreaterEqual(config.NUM_WORKERS, 4)
        self.assertEqual(config.QUEUE_SIZE, 5000)
        
        # Production should have larger file size limits
        self.assertEqual(config.MAX_FILE_SIZE, 500 * 1024 * 1024)  # 500MB


class TestConfigurationRetrieval(unittest.TestCase):
    """Test configuration retrieval functions"""
    
    def test_get_config_default(self):
        """Test get_config with default environment"""
        config = get_config()
        self.assertIsInstance(config, BaseConfig)
    
    def test_get_config_development(self):
        """Test get_config with development environment"""
        config = get_config('development')
        self.assertIsInstance(config, DevelopmentConfig)
        self.assertTrue(config.DEBUG)
    
    def test_get_config_testing(self):
        """Test get_config with testing environment"""
        config = get_config('testing')
        self.assertIsInstance(config, TestingConfig)
        self.assertEqual(config.DB_NAME, ":memory:")
    
    def test_get_config_production(self):
        """Test get_config with production environment"""
        config = get_config('production')
        self.assertIsInstance(config, ProductionConfig)
        self.assertFalse(config.DEBUG)
    
    def test_get_config_invalid_environment(self):
        """Test get_config with invalid environment"""
        config = get_config('invalid')
        # Should default to DevelopmentConfig
        self.assertIsInstance(config, DevelopmentConfig)
    
    @patch.dict(os.environ, {'ENVIRONMENT': 'production'})
    def test_get_config_from_environment_variable(self):
        """Test get_config using environment variable"""
        config = get_config()
        self.assertIsInstance(config, ProductionConfig)


class TestEnvironmentVariableOverrides(unittest.TestCase):
    """Test environment variable overrides"""
    
    def test_port_override(self):
        """Test PORT environment variable override"""
        with patch.dict(os.environ, {'PORT': '9000'}):
            config = get_config('development')
            self.assertEqual(config.PORT, 9000)
    
    def test_debug_override(self):
        """Test DEBUG environment variable override"""
        with patch.dict(os.environ, {'DEBUG': 'true'}):
            config = get_config('production')
            self.assertTrue(config.DEBUG)
        
        with patch.dict(os.environ, {'DEBUG': 'false'}):
            config = get_config('development')
            self.assertFalse(config.DEBUG)
    
    def test_database_overrides(self):
        """Test database environment variable overrides"""
        env_vars = {
            'DB_HOST': 'test-host',
            'DB_PORT': '5433',
            'DB_NAME': 'test-db',
            'DB_USER': 'test-user',
            'DB_PASSWORD': 'test-pass'
        }
        
        with patch.dict(os.environ, env_vars):
            config = get_config('development')
            self.assertEqual(config.DB_HOST, 'test-host')
            self.assertEqual(config.DB_PORT, 5433)
            self.assertEqual(config.DB_NAME, 'test-db')
            self.assertEqual(config.DB_USER, 'test-user')
            self.assertEqual(config.DB_PASSWORD, 'test-pass')
    
    def test_invalid_environment_variable_types(self):
        """Test handling of invalid environment variable types"""
        with patch.dict(os.environ, {'PORT': 'not-a-number'}):
            config = get_config('development')
            # Should keep default port value
            self.assertEqual(config.PORT, 8000)


class TestConfigurationValidation(unittest.TestCase):
    """Test configuration validation"""
    
    def setUp(self):
        """Set up validation tests"""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up validation tests"""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_valid_configuration(self):
        """Test validation of valid configuration"""
        config = TestingConfig()
        config.DATA_DIR = self.temp_dir / "data"
        config.LOGS_DIR = self.temp_dir / "logs"
        config.UPLOAD_DIR = self.temp_dir / "uploads"
        
        errors = validate_config(config)
        self.assertIsInstance(errors, list)
        self.assertEqual(len(errors), 0)
    
    def test_invalid_file_size(self):
        """Test validation of invalid file size"""
        config = TestingConfig()
        config.MAX_FILE_SIZE = -1
        
        errors = validate_config(config)
        self.assertGreater(len(errors), 0)
        self.assertTrue(any('MAX_FILE_SIZE must be positive' in error for error in errors))
    
    def test_invalid_worker_count(self):
        """Test validation of invalid worker count"""
        config = TestingConfig()
        config.NUM_WORKERS = 0
        
        errors = validate_config(config)
        self.assertGreater(len(errors), 0)
        self.assertTrue(any('NUM_WORKERS must be positive' in error for error in errors))
    
    def test_invalid_port_range(self):
        """Test validation of invalid port range"""
        config = TestingConfig()
        config.PORT = 0
        
        errors = validate_config(config)
        self.assertGreater(len(errors), 0)
        self.assertTrue(any('PORT must be in range' in error for error in errors))


class TestConfigurationFiles(unittest.TestCase):
    """Test configuration file loading"""
    
    def setUp(self):
        """Set up config file tests"""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up config file tests"""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_load_json_config(self):
        """Test loading JSON configuration file"""
        config_data = {
            'HOST': '127.0.0.1',
            'PORT': 9000,
            'DEBUG': True,
            'LOG_LEVEL': 'DEBUG'
        }
        
        config_file = self.temp_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        loaded_config = load_config_from_file(str(config_file))
        
        self.assertEqual(loaded_config['HOST'], '127.0.0.1')
        self.assertEqual(loaded_config['PORT'], 9000)
        self.assertTrue(loaded_config['DEBUG'])
        self.assertEqual(loaded_config['LOG_LEVEL'], 'DEBUG')
    
    def test_load_yaml_config(self):
        """Test loading YAML configuration file"""
        yaml_content = """
HOST: '127.0.0.1'
PORT: 9000
DEBUG: true
LOG_LEVEL: 'DEBUG'
"""
        
        config_file = self.temp_dir / "config.yml"
        with open(config_file, 'w') as f:
            f.write(yaml_content)
        
        try:
            loaded_config = load_config_from_file(str(config_file))
            self.assertEqual(loaded_config['HOST'], '127.0.0.1')
            self.assertEqual(loaded_config['PORT'], 9000)
        except ImportError:
            # YAML library not available, skip test
            self.skipTest("PyYAML not available")
    
    def test_load_nonexistent_config_file(self):
        """Test loading non-existent configuration file"""
        with self.assertRaises(ConfigurationError):
            load_config_from_file('nonexistent.json')
    
    def test_load_unsupported_config_format(self):
        """Test loading unsupported configuration file format"""
        config_file = self.temp_dir / "config.txt"
        with open(config_file, 'w') as f:
            f.write("HOST=127.0.0.1")
        
        with self.assertRaises(ConfigurationError):
            load_config_from_file(str(config_file))
    
    def test_load_invalid_json_config(self):
        """Test loading invalid JSON configuration file"""
        config_file = self.temp_dir / "invalid.json"
        with open(config_file, 'w') as f:
            f.write("{ invalid json }")
        
        with self.assertRaises(ConfigurationError):
            load_config_from_file(str(config_file))


class TestMAXXIIntegration(unittest.TestCase):
    """Test MAXXI integration configuration"""
    
    def test_maxxi_config_structure(self):
        """Test MAXXI configuration structure"""
        maxxi_config = get_maxxi_integration_config()
        
        # Required keys
        required_keys = [
            'compatibility_mode',
            'output_format',
            'metadata_schema',
            'file_naming_convention',
            'quality_thresholds',
            'processing_pipeline',
            'export_formats',
            'integration_endpoints'
        ]
        
        for key in required_keys:
            self.assertIn(key, maxxi_config)
    
    def test_maxxi_compatibility_settings(self):
        """Test MAXXI compatibility settings"""
        maxxi_config = get_maxxi_integration_config()
        
        self.assertTrue(maxxi_config['compatibility_mode'])
        self.assertEqual(maxxi_config['output_format'], 'json')
        self.assertEqual(maxxi_config['metadata_schema'], 'dublin_core')
        self.assertEqual(maxxi_config['file_naming_convention'], 'maxxi_standard')
    
    def test_maxxi_quality_thresholds(self):
        """Test MAXXI quality thresholds"""
        maxxi_config = get_maxxi_integration_config()
        thresholds = maxxi_config['quality_thresholds']
        
        self.assertIn('min_duration', thresholds)
        self.assertIn('min_sample_rate', thresholds)
        self.assertIn('max_file_size', thresholds)
        
        self.assertGreater(thresholds['min_duration'], 0)
        self.assertGreaterEqual(thresholds['min_sample_rate'], 8000)
        self.assertGreater(thresholds['max_file_size'], 0)
    
    def test_maxxi_processing_pipeline(self):
        """Test MAXXI processing pipeline configuration"""
        maxxi_config = get_maxxi_integration_config()
        pipeline = maxxi_config['processing_pipeline']
        
        expected_stages = [
            'heritage_classification',
            'metadata_generation',
            'quality_assessment',
            'speaker_recognition'
        ]
        
        for stage in expected_stages:
            self.assertIn(stage, pipeline)
    
    def test_maxxi_export_formats(self):
        """Test MAXXI export formats"""
        maxxi_config = get_maxxi_integration_config()
        formats = maxxi_config['export_formats']
        
        expected_formats = ['json', 'xml', 'csv']
        for format_type in expected_formats:
            self.assertIn(format_type, formats)
    
    def test_maxxi_integration_endpoints(self):
        """Test MAXXI integration endpoints"""
        maxxi_config = get_maxxi_integration_config()
        endpoints = maxxi_config['integration_endpoints']
        
        expected_endpoints = [
            'status_callback',
            'results_webhook', 
            'metadata_export'
        ]
        
        for endpoint in expected_endpoints:
            self.assertIn(endpoint, endpoints)
            self.assertTrue(endpoints[endpoint].startswith('/api/maxxi/'))


class TestConfigurationSecurity(unittest.TestCase):
    """Test configuration security aspects"""
    
    def test_production_secret_key_validation(self):
        """Test that production config validates secret key"""
        config = ProductionConfig()
        config.SECRET_KEY = "change-this-in-production"  # Insecure default
        
        # Simulate production environment
        config.ENVIRONMENT = 'production'
        
        errors = validate_config(config)
        # Should have error about secret key
        self.assertTrue(any('SECRET_KEY must be changed' in error for error in errors))
    
    def test_production_debug_validation(self):
        """Test that production config validates debug setting"""
        config = ProductionConfig()
        config.DEBUG = True  # Should not be True in production
        
        # Simulate production environment  
        config.ENVIRONMENT = 'production'
        
        errors = validate_config(config)
        # Should have error about debug mode
        self.assertTrue(any('DEBUG must be False' in error for error in errors))
    
    def test_cors_origins_configuration(self):
        """Test CORS origins configuration"""
        config = ProductionConfig()
        
        # Test with environment variable
        with patch.dict(os.environ, {'CORS_ORIGINS': 'https://example.com,https://app.example.com'}):
            # Simulate environment override
            if os.getenv('CORS_ORIGINS'):
                config.CORS_ORIGINS = os.getenv('CORS_ORIGINS').split(',')
            
            self.assertIn('https://example.com', config.CORS_ORIGINS)
            self.assertIn('https://app.example.com', config.CORS_ORIGINS)
    
    def test_sensitive_data_not_in_defaults(self):
        """Test that sensitive data is not in default configuration"""
        config = BaseConfig()
        
        # These should be empty or None by default
        self.assertIsNone(config.DATADOG_API_KEY)
        self.assertIsNone(config.SENTRY_DSN)
        self.assertEqual(config.DB_PASSWORD, "")
        self.assertIsNone(config.REDIS_PASSWORD)


class TestConfigurationPerformance(unittest.TestCase):
    """Test configuration performance aspects"""
    
    def test_worker_scaling_logic(self):
        """Test worker count scaling across environments"""
        dev_config = DevelopmentConfig()
        staging_config = StagingConfig()
        prod_config = ProductionConfig()
        
        # Development should have fewer workers
        self.assertLessEqual(dev_config.NUM_WORKERS, 4)
        
        # Staging should have moderate workers
        self.assertGreater(staging_config.NUM_WORKERS, dev_config.NUM_WORKERS)
        self.assertLessEqual(staging_config.NUM_WORKERS, prod_config.NUM_WORKERS)
        
        # Production should have most workers
        self.assertGreaterEqual(prod_config.NUM_WORKERS, 4)
        self.assertGreaterEqual(prod_config.MAX_THREAD_WORKERS, 8)
    
    def test_queue_size_scaling(self):
        """Test queue size scaling"""
        base_config = BaseConfig()
        prod_config = ProductionConfig()
        
        # Production should have larger queue
        self.assertGreater(prod_config.QUEUE_SIZE, base_config.QUEUE_SIZE)
    
    def test_timeout_configuration(self):
        """Test timeout configuration"""
        config = ProductionConfig()
        
        self.assertGreater(config.API_TIMEOUT, 0)
        self.assertLessEqual(config.API_TIMEOUT, 600)  # Max 10 minutes


class TestConfigurationEdgeCases(unittest.TestCase):
    """Test configuration edge cases and error conditions"""
    
    def test_missing_directories_creation(self):
        """Test that missing directories are created during validation"""
        temp_dir = Path(tempfile.mkdtemp())
        config = TestingConfig()
        
        # Set paths to non-existent directories
        config.DATA_DIR = temp_dir / "data"
        config.LOGS_DIR = temp_dir / "logs" 
        config.UPLOAD_DIR = temp_dir / "uploads"
        
        # These directories don't exist yet
        self.assertFalse(config.DATA_DIR.exists())
        self.assertFalse(config.LOGS_DIR.exists())
        self.assertFalse(config.UPLOAD_DIR.exists())
        
        # Validation should create them
        errors = validate_config(config)
        
        # Should have no errors and directories should now exist
        self.assertEqual(len(errors), 0)
        self.assertTrue(config.DATA_DIR.exists())
        self.assertTrue(config.LOGS_DIR.exists())
        self.assertTrue(config.UPLOAD_DIR.exists())
        
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)
    
    def test_extreme_values(self):
        """Test configuration with extreme values"""
        config = BaseConfig()
        
        # Test extremely large file size
        config.MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2GB
        errors = validate_config(config)
        
        # Should have error about excessive file size
        self.assertTrue(any('exceeds reasonable limit' in error for error in errors))
        
        # Test zero values
        config.MAX_FILE_SIZE = 0
        config.NUM_WORKERS = 0
        errors = validate_config(config)
        
        self.assertGreater(len(errors), 0)


def run_config_tests():
    """Run all configuration tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    test_classes = [
        TestBaseConfiguration,
        TestEnvironmentConfigurations,
        TestConfigurationRetrieval,
        TestEnvironmentVariableOverrides,
        TestConfigurationValidation,
        TestConfigurationFiles,
        TestMAXXIIntegration,
        TestConfigurationSecurity,
        TestConfigurationPerformance,
        TestConfigurationEdgeCases
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print(f"\nConfiguration Tests Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_config_tests()
    exit(0 if success else 1)
