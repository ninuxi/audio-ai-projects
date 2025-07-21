# processors/metadata_generator.py
"""
📋 Metadata Generator Module
===========================

Specialized module for generating comprehensive metadata from audio analysis.
Creates structured metadata for database storage and ML pipeline integration.
"""

import numpy as np
import os
import hashlib
from datetime import datetime
from typing import Dict, List, Any
import json

class MetadataGenerator:
    """
    Professional metadata generation for batch audio processing
    Combines technical analysis, content classification, and file information
    """
    
    def __init__(self, config: Dict):
        """Initialize with metadata configuration"""
        self.config = config
        self.sr = config['audio']['sample_rate']
        
        # Output configuration
        self.output_config = config.get('output', {})
        self.include_metadata = self.output_config.get('dataset', {}).get('include_metadata', True)
        
        print(f"📋 MetadataGenerator initialized")
    
    def generate_metadata(self, audio_file_path: str, audio: np.ndarray, 
                         features: Dict, quality_report: Dict, 
                         classification: Dict) -> Dict:
        """Generate comprehensive metadata from all analysis results"""
        
        metadata = {}
        
        # File metadata
        metadata.update(self._generate_file_metadata(audio_file_path, audio))
        
        # Technical metadata
        metadata.update(self._generate_technical_metadata(audio, features))
        
        # Quality metadata
        metadata.update(self._generate_quality_metadata(quality_report))
        
        # Content metadata  
        metadata.update(self._generate_content_metadata(classification))
        
        # Processing metadata
        metadata.update(self._generate_processing_metadata())
        
        # ML-ready features
        if self.include_metadata:
            metadata.update(self._generate_ml_features(features))
        
        return metadata
    
    def _generate_file_metadata(self, file_path: str, audio: np.ndarray) -> Dict:
        """Generate file-related metadata"""
        
        file_metadata = {
            'file_info': {
                'filename': os.path.basename(file_path),
                'file_path': file_path,
                'file_size_bytes': os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                'file_extension': os.path.splitext(file_path)[1].lower(),
                'file_hash': self._calculate_file_hash(file_path),
                'creation_date': self._get_file_creation_date(file_path),
                'last_modified': self._get_file_modification_date(file_path)
            },
            
            'audio_info': {
                'duration_seconds': float(len(audio) / self.sr),
                'duration_samples': len(audio),
                'sample_rate': self.sr,
                'channels': 1,  # Assuming mono after librosa loading
                'bit_depth': 32,  # Float32 after librosa processing
                'format': 'processed_float32'
            }
        }
        
        return file_metadata
    
    def _generate_technical_metadata(self, audio: np.ndarray, features: Dict) -> Dict:
        """Generate technical audio metadata"""
        
        technical_metadata = {
            'technical_analysis': {
                # Signal characteristics
                'peak_amplitude': float(np.max(np.abs(audio))),
                'rms_level': float(np.sqrt(np.mean(audio**2))),
                'dynamic_range_db': self._calculate_dynamic_range_db(audio),
                'crest_factor': self._calculate_crest_factor(audio),
                
                # Frequency domain
                'spectral_centroid_hz': features.get('spectral_centroid_mean', 0),
                'spectral_bandwidth_hz': features.get('spectral_bandwidth_mean', 0),
                'spectral_rolloff_hz': features.get('spectral_rolloff_mean', 0),
                'spectral_flatness': features.get('spectral_flatness_mean', 0),
                
                # Temporal characteristics
                'zero_crossing_rate': features.get('zcr_mean', 0),
                'silence_ratio': features.get('silence_ratio', 0),
                
                # Rhythm analysis
                'tempo_bpm': features.get('tempo', 0),
                'rhythm_regularity': features.get('rhythm_regularity', 0),
                'beats_detected': features.get('beats_count', 0),
                'onsets_detected': features.get('onsets_count', 0)
            }
        }
        
        return technical_metadata
    
    def _generate_quality_metadata(self, quality_report: Dict) -> Dict:
        """Generate quality assessment metadata"""
        
        quality_metadata = {
            'quality_assessment': {
                'overall_quality': quality_report.get('overall_quality', 'unknown'),
                'quality_score': quality_report.get('quality_score', 0),
                'snr_db': quality_report.get('snr_db', 0),
                'dynamic_range_quality': quality_report.get('dynamic_range_quality', 'unknown'),
                'frequency_balance_score': quality_report.get('frequency_balance_score', 0),
                'consistency_score': quality_report.get('consistency_score', 1.0),
                'clipping_detected': quality_report.get('clipping_ratio', 0) > 0.01,
                'issues_detected': len(quality_report.get('issues', [])),
                'recommendations_count': len(quality_report.get('recommendations', []))
            },
            
            'quality_flags': {
                'is_high_quality': quality_report.get('quality_score', 0) > 80,
                'needs_processing': quality_report.get('quality_score', 0) < 60,
                'has_technical_issues': len(quality_report.get('issues', [])) > 0,
                'suitable_for_ml': quality_report.get('quality_score', 0) > 50
            }
        }
        
        return quality_metadata
    
    def _generate_content_metadata(self, classification: Dict) -> Dict:
        """Generate content classification metadata"""
        
        content_metadata = {
            'content_classification': {
                'content_type': classification.get('content_type', 'unknown'),
                'content_confidence': classification.get('content_confidence', 0),
                'genre': classification.get('genre', 'unknown'),
                'genre_confidence': classification.get('genre_confidence', 0),
                'classification_method': classification.get('classification_method', 'unknown')
            },
            
            'content_characteristics': classification.get('characteristics', {}),
            
            'content_flags': {
                'is_music': classification.get('content_type') == 'music',
                'is_speech': classification.get('content_type') == 'speech',
                'is_instrumental': classification.get('characteristics', {}).get('vocal_content') == 'instrumental',
                'has_rhythm': classification.get('characteristics', {}).get('rhythm_type') in ['regular', 'moderate'],
                'high_confidence_classification': classification.get('content_confidence', 0) > 0.7
            }
        }
        
        return content_metadata
    
    def _generate_processing_metadata(self) -> Dict:
        """Generate processing and analysis metadata"""
        
        processing_metadata = {
            'processing_info': {
                'analysis_timestamp': datetime.now().isoformat(),
                'processing_version': '2.0.0',
                'sample_rate_used': self.sr,
                'analysis_duration_seconds': 0,  # Would be filled by calling code
                'features_extracted': True,
                'quality_assessed': True,
                'content_classified': True
            },
            
            'analysis_parameters': {
                'mfcc_coefficients': self.config.get('features', {}).get('mfcc', {}).get('n_mfcc', 13),
                'fft_window_size': self.config.get('features', {}).get('mfcc', {}).get('n_fft', 2048),
                'hop_length': self.config.get('features', {}).get('mfcc', {}).get('hop_length', 512),
                'normalization_applied': self.config.get('audio', {}).get('normalize_audio', True)
            }
        }
        
        return processing_metadata
    
    def _generate_ml_features(self, features: Dict) -> Dict:
        """Generate ML-ready feature vectors"""
        
        # Extract numeric features for ML
        ml_features = {}
        
        # Collect all numeric features
        for key, value in features.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                ml_features[f'feature_{key}'] = float(value)
        
        # Create feature vectors for common ML tasks
        feature_vectors = {
            'ml_features': {
                # Basic feature vector (most important features)
                'basic_vector': [
                    features.get('rms_mean', 0),
                    features.get('spectral_centroid_mean', 0),
                    features.get('spectral_bandwidth_mean', 0),
                    features.get('zcr_mean', 0),
                    features.get('tempo', 0)
                ],
                
                # MFCC vector
                'mfcc_vector': [features.get(f'mfcc_{i}_mean', 0) for i in range(13)],
                
                # Spectral feature vector
                'spectral_vector': [
                    features.get('spectral_centroid_mean', 0),
                    features.get('spectral_bandwidth_mean', 0),
                    features.get('spectral_rolloff_mean', 0),
                    features.get('spectral_contrast_mean', 0),
                    features.get('spectral_flatness_mean', 0)
                ],
                
                # Temporal feature vector
                'temporal_vector': [
                    features.get('zcr_mean', 0),
                    features.get('rms_mean', 0),
                    features.get('tempo', 0),
                    features.get('rhythm_regularity', 0)
                ]
            }
        }
        
        return feature_vectors
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of audio file"""
        if not os.path.exists(file_path):
            return ""
        
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except:
            return ""
    
    def _get_file_creation_date(self, file_path: str) -> str:
        """Get file creation date"""
        if not os.path.exists(file_path):
            return ""
        
        try:
            timestamp = os.path.getctime(file_path)
            return datetime.fromtimestamp(timestamp).isoformat()
        except:
            return ""
    
    def _get_file_modification_date(self, file_path: str) -> str:
        """Get file modification date"""
        if not os.path.exists(file_path):
            return ""
        
        try:
            timestamp = os.path.getmtime(file_path)
            return datetime.fromtimestamp(timestamp).isoformat()
        except:
            return ""
    
    def _calculate_dynamic_range_db(self, audio: np.ndarray) -> float:
        """Calculate dynamic range in dB"""
        if len(audio) == 0:
            return 0
        
        peak = np.max(np.abs(audio))
        rms = np.sqrt(np.mean(audio**2))
        
        if rms > 0:
            return float(20 * np.log10(peak / rms))
        else:
            return 0
    
    def _calculate_crest_factor(self, audio: np.ndarray) -> float:
        """Calculate crest factor (peak-to-RMS ratio)"""
        if len(audio) == 0:
            return 0
        
        peak = np.max(np.abs(audio))
        rms = np.sqrt(np.mean(audio**2))
        
        return float(peak / rms) if rms > 0 else 0
    
    def export_metadata_json(self, metadata: Dict, output_path: str) -> bool:
        """Export metadata to JSON file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"❌ Error exporting metadata to {output_path}: {e}")
            return False
    
    def create_dataset_entry(self, metadata: Dict) -> Dict:
        """Create a flattened dataset entry suitable for CSV/DataFrame"""
        
        entry = {}
        
        # Flatten nested metadata structure
        def flatten_dict(d: Dict, prefix: str = '') -> Dict:
            items = []
            for k, v in d.items():
                new_key = f"{prefix}_{k}" if prefix else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key).items())
                elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], (int, float)):
                    # Handle feature vectors
                    for i, val in enumerate(v):
                        items.append((f"{new_key}_{i}", val))
                elif isinstance(v, (str, int, float, bool)):
                    items.append((new_key, v))
            return dict(items)
        
        # Flatten all metadata
        entry = flatten_dict(metadata)
        
        # Ensure consistent data types
        for key, value in entry.items():
            if isinstance(value, np.floating):
                entry[key] = float(value)
            elif isinstance(value, np.integer):
                entry[key] = int(value)
        
        return entry
    
    def batch_generate_metadata(self, file_paths: List[str], audio_list: List[np.ndarray],
                               features_list: List[Dict], quality_reports: List[Dict],
                               classifications: List[Dict]) -> List[Dict]:
        """Generate metadata for multiple audio files"""
        
        print(f"📋 Generating metadata for {len(file_paths)} files...")
        
        metadata_list = []
        
        for i, (file_path, audio, features, quality, classification) in enumerate(
            zip(file_paths, audio_list, features_list, quality_reports, classifications)):
            
            try:
                metadata = self.generate_metadata(file_path, audio, features, quality, classification)
                metadata_list.append(metadata)
                
                if (i + 1) % 50 == 0:
                    print(f"   Generated metadata for {i + 1}/{len(file_paths)} files...")
                    
            except Exception as e:
                print(f"❌ Error generating metadata for {file_path}: {e}")
                # Create minimal metadata entry for failed files
                minimal_metadata = {
                    'file_info': {'filename': os.path.basename(file_path), 'error': str(e)},
                    'processing_info': {'analysis_timestamp': datetime.now().isoformat()}
                }
                metadata_list.append(minimal_metadata)
        
        print(f"✅ Metadata generation completed!")
        return metadata_list
    
    def create_summary_report(self, metadata_list: List[Dict]) -> Dict:
        """Create summary report from metadata collection"""
        
        if not metadata_list:
            return {}
        
        # Aggregate statistics
        total_files = len(metadata_list)
        total_duration = sum(m.get('audio_info', {}).get('duration_seconds', 0) for m in metadata_list)
        
        # Quality distribution
        quality_distribution = {}
        for metadata in metadata_list:
            quality = metadata.get('quality_assessment', {}).get('overall_quality', 'unknown')
            quality_distribution[quality] = quality_distribution.get(quality, 0) + 1
        
        # Content type distribution
        content_distribution = {}
        for metadata in metadata_list:
            content_type = metadata.get('content_classification', {}).get('content_type', 'unknown')
            content_distribution[content_type] = content_distribution.get(content_type, 0) + 1
        
        # File format distribution
        format_distribution = {}
        for metadata in metadata_list:
            file_ext = metadata.get('file_info', {}).get('file_extension', 'unknown')
            format_distribution[file_ext] = format_distribution.get(file_ext, 0) + 1
        
        summary = {
            'dataset_summary': {
                'total_files': total_files,
                'total_duration_seconds': total_duration,
                'total_duration_hours': total_duration / 3600,
                'average_file_duration': total_duration / total_files if total_files > 0 else 0,
                'analysis_date': datetime.now().isoformat()
            },
            
            'quality_distribution': quality_distribution,
            'content_distribution': content_distribution,
            'format_distribution': format_distribution,
            
            'dataset_statistics': {
                'high_quality_files': sum(1 for m in metadata_list 
                                        if m.get('quality_flags', {}).get('is_high_quality', False)),
                'ml_suitable_files': sum(1 for m in metadata_list 
                                       if m.get('quality_flags', {}).get('suitable_for_ml', False)),
                'music_files': sum(1 for m in metadata_list 
                                 if m.get('content_flags', {}).get('is_music', False)),
                'speech_files': sum(1 for m in metadata_list 
                                  if m.get('content_flags', {}).get('is_speech', False))
            }
        }
        
        return summary

# Demo usage
if __name__ == "__main__":
    # Test configuration
    test_config = {
        'audio': {'sample_rate': 22050, 'normalize_audio': True},
        'features': {'mfcc': {'n_mfcc': 13, 'n_fft': 2048, 'hop_length': 512}},
        'output': {'dataset': {'include_metadata': True}}
    }
    
    # Initialize generator
    generator = MetadataGenerator(test_config)
    
    # Create test data
    test_audio = np.random.randn(22050 * 3)  # 3 seconds of noise
    test_features = {
        'rms_mean': 0.1,
        'spectral_centroid_mean': 2000,
        'zcr_mean': 0.08,
        'tempo': 120,
        'mfcc_0_mean': -10.5
    }
    test_quality = {
        'overall_quality': 'good',
        'quality_score': 75,
        'snr_db': 20
    }
    test_classification = {
        'content_type': 'music',
        'content_confidence': 0.8,
        'genre': 'pop'
    }
    
    # Generate metadata
    print("📋 METADATA GENERATION TEST")
    print("=" * 40)
    
    metadata = generator.generate_metadata(
        'test_audio.wav', test_audio, test_features, test_quality, test_classification
    )
    
    print(f"✅ Generated metadata with {len(metadata)} main sections:")
    for section in metadata.keys():
        print(f"   • {section}")
    
    # Test dataset entry creation
    dataset_entry = generator.create_dataset_entry(metadata)
    print(f"\n📊 Dataset entry has {len(dataset_entry)} flattened features")
