# processors/feature_extractor.py
"""
🎵 Feature Extractor Module
===========================

Specialized module for extracting comprehensive audio features
optimized for batch processing and machine learning pipelines.
"""

import numpy as np
import librosa
from typing import Dict, Optional, List
import warnings
warnings.filterwarnings('ignore')

class FeatureExtractor:
    """
    Professional audio feature extraction for batch processing
    Optimized for consistency and ML pipeline integration
    """
    
    def __init__(self, config: Dict):
        """Initialize with configuration from config.yaml"""
        self.config = config
        self.sr = config['audio']['sample_rate']
        
        # Feature extraction parameters
        self.mfcc_config = config['features']['mfcc']
        self.spectral_config = config['features']['spectral']
        self.chroma_config = config['features']['chroma']
        
        print(f"🎵 FeatureExtractor initialized (sr={self.sr})")
    
    def extract_all_features(self, audio: np.ndarray) -> Dict:
        """Extract comprehensive feature set from audio"""
        
        features = {}
        
        # Basic audio properties
        features.update(self._extract_basic_features(audio))
        
        # Temporal features
        features.update(self._extract_temporal_features(audio))
        
        # Spectral features  
        features.update(self._extract_spectral_features(audio))
        
        # Perceptual features
        features.update(self._extract_perceptual_features(audio))
        
        # Rhythm features
        features.update(self._extract_rhythm_features(audio))
        
        return features
    
    def _extract_basic_features(self, audio: np.ndarray) -> Dict:
        """Extract basic audio properties"""
        
        return {
            'duration': len(audio) / self.sr,
            'sample_count': len(audio),
            'rms_energy': float(np.sqrt(np.mean(audio**2))),
            'peak_amplitude': float(np.max(np.abs(audio))),
            'dynamic_range': float(np.max(audio) - np.min(audio)),
            'silence_ratio': self._calculate_silence_ratio(audio)
        }
    
    def _extract_temporal_features(self, audio: np.ndarray) -> Dict:
        """Extract time-domain features"""
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(
            audio, 
            frame_length=self.spectral_config['n_fft'],
            hop_length=self.spectral_config['hop_length']
        )[0]
        
        return {
            'zcr_mean': float(np.mean(zcr)),
            'zcr_std': float(np.std(zcr)),
            'zcr_max': float(np.max(zcr)),
            'zcr_min': float(np.min(zcr))
        }
    
    def _extract_spectral_features(self, audio: np.ndarray) -> Dict:
        """Extract frequency-domain features"""
        
        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio, 
            sr=self.sr,
            hop_length=self.spectral_config['hop_length']
        )[0]
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio, 
            sr=self.sr,
            hop_length=self.spectral_config['hop_length']
        )[0]
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio, 
            sr=self.sr,
            hop_length=self.spectral_config['hop_length']
        )[0]
        
        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(
            y=audio, 
            sr=self.sr,
            hop_length=self.spectral_config['hop_length']
        )
        
        # Spectral flatness
        spectral_flatness = librosa.feature.spectral_flatness(
            y=audio,
            hop_length=self.spectral_config['hop_length']
        )[0]
        
        return {
            # Spectral Centroid
            'spectral_centroid_mean': float(np.mean(spectral_centroid)),
            'spectral_centroid_std': float(np.std(spectral_centroid)),
            
            # Spectral Rolloff
            'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
            'spectral_rolloff_std': float(np.std(spectral_rolloff)),
            
            # Spectral Bandwidth
            'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
            'spectral_bandwidth_std': float(np.std(spectral_bandwidth)),
            
            # Spectral Contrast (summarize 7 bands)
            'spectral_contrast_mean': float(np.mean(spectral_contrast)),
            'spectral_contrast_std': float(np.std(spectral_contrast)),
            
            # Spectral Flatness
            'spectral_flatness_mean': float(np.mean(spectral_flatness)),
            'spectral_flatness_std': float(np.std(spectral_flatness))
        }
    
    def _extract_perceptual_features(self, audio: np.ndarray) -> Dict:
        """Extract perceptually relevant features"""
        
        # MFCC
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sr,
            n_mfcc=self.mfcc_config['n_mfcc'],
            n_fft=self.mfcc_config['n_fft'],
            hop_length=self.mfcc_config['hop_length']
        )
        
        # Chroma
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=self.sr,
            n_chroma=self.chroma_config['n_chroma'],
            hop_length=self.spectral_config['hop_length']
        )
        
        features = {}
        
        # MFCC features (mean and std for each coefficient)
        for i in range(self.mfcc_config['n_mfcc']):
            features[f'mfcc_{i}_mean'] = float(np.mean(mfcc[i]))
            features[f'mfcc_{i}_std'] = float(np.std(mfcc[i]))
        
        # Chroma features
        for i in range(self.chroma_config['n_chroma']):
            features[f'chroma_{i}_mean'] = float(np.mean(chroma[i]))
        
        return features
    
    def _extract_rhythm_features(self, audio: np.ndarray) -> Dict:
        """Extract rhythm and temporal structure features"""
        
        if not self.config['features']['rhythm']['tempo_estimation']:
            return {}
        
        try:
            # Tempo and beat tracking
            tempo, beats = librosa.beat.beat_track(y=audio, sr=self.sr)
            
            # Onset detection
            if self.config['features']['rhythm']['onset_detection']:
                onsets = librosa.onset.onset_detect(y=audio, sr=self.sr, units='time')
            else:
                onsets = []
            
            # Rhythm regularity
            rhythm_regularity = self._calculate_rhythm_regularity(beats) if len(beats) > 2 else 0
            
            return {
                'tempo': float(tempo),
                'beats_count': len(beats),
                'onsets_count': len(onsets),
                'rhythm_regularity': rhythm_regularity,
                'beats_per_second': len(beats) / (len(audio) / self.sr) if len(audio) > 0 else 0
            }
            
        except Exception as e:
            print(f"⚠️ Rhythm extraction failed: {e}")
            return {
                'tempo': 0.0,
                'beats_count': 0,
                'onsets_count': 0,
                'rhythm_regularity': 0.0,
                'beats_per_second': 0.0
            }
    
    def _calculate_silence_ratio(self, audio: np.ndarray) -> float:
        """Calculate ratio of silent samples"""
        silence_threshold = 0.01  # Configurable
        silent_samples = np.sum(np.abs(audio) < silence_threshold)
        return silent_samples / len(audio)
    
    def _calculate_rhythm_regularity(self, beats: np.ndarray) -> float:
        """Calculate rhythm regularity from beat positions"""
        if len(beats) < 3:
            return 0.0
        
        # Convert beat frames to time
        beat_times = librosa.frames_to_time(beats, sr=self.sr)
        
        # Calculate inter-beat intervals
        intervals = np.diff(beat_times)
        
        # Regularity = 1 - coefficient of variation
        if np.mean(intervals) > 0:
            regularity = 1 - (np.std(intervals) / np.mean(intervals))
            return max(0, min(1, regularity))
        
        return 0.0
    
    def get_feature_names(self) -> List[str]:
        """Get list of all extracted feature names"""
        
        # This would be generated dynamically based on config
        feature_names = [
            # Basic features
            'duration', 'sample_count', 'rms_energy', 'peak_amplitude', 
            'dynamic_range', 'silence_ratio',
            
            # Temporal features
            'zcr_mean', 'zcr_std', 'zcr_max', 'zcr_min',
            
            # Spectral features
            'spectral_centroid_mean', 'spectral_centroid_std',
            'spectral_rolloff_mean', 'spectral_rolloff_std',
            'spectral_bandwidth_mean', 'spectral_bandwidth_std',
            'spectral_contrast_mean', 'spectral_contrast_std',
            'spectral_flatness_mean', 'spectral_flatness_std'
        ]
        
        # Add MFCC features
        for i in range(self.mfcc_config['n_mfcc']):
            feature_names.extend([f'mfcc_{i}_mean', f'mfcc_{i}_std'])
        
        # Add Chroma features  
        for i in range(self.chroma_config['n_chroma']):
            feature_names.append(f'chroma_{i}_mean')
        
        # Add rhythm features if enabled
        if self.config['features']['rhythm']['tempo_estimation']:
            feature_names.extend([
                'tempo', 'beats_count', 'onsets_count', 
                'rhythm_regularity', 'beats_per_second'
            ])
        
        return feature_names
    
    def extract_features_batch(self, audio_list: List[np.ndarray]) -> List[Dict]:
        """Extract features from multiple audio arrays"""
        
        print(f"🔄 Extracting features from {len(audio_list)} audio files...")
        
        features_list = []
        
        for i, audio in enumerate(audio_list):
            try:
                features = self.extract_all_features(audio)
                features_list.append(features)
                
                if (i + 1) % 10 == 0:
                    print(f"   Processed {i + 1}/{len(audio_list)} files...")
                    
            except Exception as e:
                print(f"❌ Error processing audio {i}: {e}")
                features_list.append({})  # Empty features for failed processing
        
        print(f"✅ Feature extraction completed!")
        return features_list

# Demo usage
if __name__ == "__main__":
    # Test configuration
    test_config = {
        'audio': {'sample_rate': 22050},
        'features': {
            'mfcc': {'n_mfcc': 13, 'n_fft': 2048, 'hop_length': 512},
            'spectral': {'n_fft': 2048, 'hop_length': 512},
            'chroma': {'n_chroma': 12},
            'rhythm': {'tempo_estimation': True, 'onset_detection': True}
        }
    }
    
    # Initialize extractor
    extractor = FeatureExtractor(test_config)
    
    # Generate test audio
    sr = 22050
    duration = 3
    t = np.linspace(0, duration, sr * duration)
    test_audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # A4 note
    
    # Extract features
    features = extractor.extract_all_features(test_audio)
    
    print(f"\n🎵 FEATURE EXTRACTION TEST")
    print(f"Features extracted: {len(features)}")
    print(f"Sample features:")
    for key, value in list(features.items())[:10]:
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
