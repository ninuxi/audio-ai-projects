"""
Audio AI Projects - Advanced Audio Processing System
Copyright (c) 2025 Antonino Mainenti (ninuxi)
Licensed under MIT License - see LICENSE file
GitHub: https://github.com/ninuxi/audio-ai-projects
"""
#!/usr/bin/env python3
"""
Speaker Recognition System for Call Analytics
Identifies and classifies different speakers in call center recordings
"""

import librosa
import numpy as np
import soundfile as sf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class SpeakerRecognition:
    """
    Speaker identification and diarization system for call center analytics
    """
    
    def __init__(self, n_speakers: int = 2, sample_rate: int = 16000):
        """
        Initialize speaker recognition system
        
        Args:
            n_speakers: Expected number of speakers (default: 2 for agent + customer)
            sample_rate: Target sample rate for processing
        """
        self.n_speakers = n_speakers
        self.sample_rate = sample_rate
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=min(20, 5))  # Dinamico in base ai dati
        self.kmeans = KMeans(n_clusters=n_speakers, random_state=42, n_init=10)
        
        # Speaker profiles
        self.speaker_profiles = {}
        self.is_trained = False
        
    def extract_speaker_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract MFCC features for speaker identification
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Feature vector for speaker identification
        """
        try:
            # Resample if necessary
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            
            # Extract MFCCs (mel-frequency cepstral coefficients)
            mfccs = librosa.feature.mfcc(
                y=audio, 
                sr=self.sample_rate, 
                n_mfcc=13,
                n_fft=2048,
                hop_length=512,
                n_mels=40
            )
            
            # Extract spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)
            
            # Extract pitch features
            pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate)
            pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
            
            # Combine features
            feature_vector = np.concatenate([
                np.mean(mfccs, axis=1),  # Mean MFCCs
                np.std(mfccs, axis=1),   # Standard deviation MFCCs
                [np.mean(spectral_centroids)],
                [np.std(spectral_centroids)],
                [np.mean(spectral_rolloff)],
                [np.std(spectral_rolloff)],
                [np.mean(zero_crossing_rate)],
                [np.std(zero_crossing_rate)],
                [np.mean(spectral_bandwidth)],
                [np.std(spectral_bandwidth)],
                [pitch_mean]
            ])
            
            return feature_vector
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return np.zeros(31)  # Return zero vector on error
    
    def segment_audio(self, audio: np.ndarray, sr: int, 
                     segment_duration: float = 2.0) -> List[Tuple[np.ndarray, float, float]]:
        """
        Segment audio into chunks for speaker analysis
        
        Args:
            audio: Audio signal
            sr: Sample rate
            segment_duration: Duration of each segment in seconds
            
        Returns:
            List of (audio_segment, start_time, end_time) tuples
        """
        segment_samples = int(segment_duration * sr)
        segments = []
        
        for i in range(0, len(audio), segment_samples):
            end_idx = min(i + segment_samples, len(audio))
            segment = audio[i:end_idx]
            
            # Skip very short or silent segments
            if len(segment) < segment_samples * 0.5:
                continue
                
            # Check if segment has sufficient energy
            energy = np.mean(segment ** 2)
            if energy > 0.001:  # Threshold for non-silent segments
                start_time = i / sr
                end_time = end_idx / sr
                segments.append((segment, start_time, end_time))
        
        return segments
    
    def train_speaker_clustering(self, audio_file: str) -> Dict[str, Any]:
        """
        Train speaker clustering model on audio file
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Training results and statistics
        """
        try:
            print(f"Training speaker recognition on: {audio_file}")
            
            # Load audio
            audio, sr = librosa.load(audio_file, sr=self.sample_rate)
            
            # Segment audio
            segments = self.segment_audio(audio, sr)
            print(f"Created {len(segments)} audio segments")
            
            if len(segments) < self.n_speakers:
                raise ValueError(f"Not enough segments ({len(segments)}) for {self.n_speakers} speakers")
            
            # Extract features for all segments
            features = []
            segment_info = []
            
            for segment, start_time, end_time in segments:
                feature_vector = self.extract_speaker_features(segment, sr)
                if not np.all(feature_vector == 0):  # Skip zero vectors
                    features.append(feature_vector)
                    segment_info.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': end_time - start_time
                    })
            
            features = np.array(features)
            print(f"Extracted features from {len(features)} segments")
            
            if len(features) < self.n_speakers:
                raise ValueError(f"Not enough valid features ({len(features)}) for clustering")
            
            # Normalize features
            features_normalized = self.scaler.fit_transform(features)
            
            # Apply PCA for dimensionality reduction
            features_pca = self.pca.fit_transform(features_normalized)
            
            # Cluster speakers
            speaker_labels = self.kmeans.fit_predict(features_pca)
            
            # Create speaker profiles
            self.speaker_profiles = {}
            for speaker_id in range(self.n_speakers):
                speaker_segments = np.where(speaker_labels == speaker_id)[0]
                speaker_features = features[speaker_segments]
                
                self.speaker_profiles[f'Speaker_{speaker_id}'] = {
                    'mean_features': np.mean(speaker_features, axis=0),
                    'std_features': np.std(speaker_features, axis=0),
                    'segment_count': len(speaker_segments),
                    'total_duration': sum(segment_info[i]['duration'] for i in speaker_segments)
                }
            
            self.is_trained = True
            
            # Calculate training statistics
            training_results = {
                'total_segments': len(segments),
                'valid_segments': len(features),
                'speakers_identified': self.n_speakers,
                'feature_dimension': features.shape[1],
                'pca_components': self.pca.n_components_,
                'speaker_profiles': self.speaker_profiles,
                'training_file': audio_file
            }
            
            print(f"✅ Speaker clustering trained successfully!")
            print(f"   Speakers identified: {self.n_speakers}")
            print(f"   Segments processed: {len(features)}")
            
            return training_results
            
        except Exception as e:
            print(f"❌ Error training speaker recognition: {e}")
            return {'error': str(e)}
    
    def identify_speakers(self, audio_file: str) -> Dict[str, Any]:
        """
        Identify speakers in audio file using trained model
        
        Args:
            audio_file: Path to audio file to analyze
            
        Returns:
            Speaker identification results
        """
        if not self.is_trained:
            return {'error': 'Model not trained. Call train_speaker_clustering first.'}
        
        try:
            print(f"Analyzing speakers in: {audio_file}")
            
            # Load audio
            audio, sr = librosa.load(audio_file, sr=self.sample_rate)
            
            # Segment audio
            segments = self.segment_audio(audio, sr)
            
            # Extract features and predict speakers
            speaker_timeline = []
            speaker_stats = {f'Speaker_{i}': {'duration': 0, 'segments': 0} 
                           for i in range(self.n_speakers)}
            
            for segment, start_time, end_time in segments:
                feature_vector = self.extract_speaker_features(segment, sr)
                
                if not np.all(feature_vector == 0):
                    # Normalize and apply PCA
                    feature_normalized = self.scaler.transform([feature_vector])
                    feature_pca = self.pca.transform(feature_normalized)
                    
                    # Predict speaker
                    speaker_id = self.kmeans.predict(feature_pca)[0]
                    speaker_name = f'Speaker_{speaker_id}'
                    
                    speaker_timeline.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'speaker': speaker_name,
                        'confidence': self._calculate_confidence(feature_vector, speaker_id)
                    })
                    
                    # Update statistics
                    speaker_stats[speaker_name]['duration'] += (end_time - start_time)
                    speaker_stats[speaker_name]['segments'] += 1
            
            # Calculate speaking percentages
            total_duration = sum(stats['duration'] for stats in speaker_stats.values())
            for speaker in speaker_stats:
                if total_duration > 0:
                    speaker_stats[speaker]['percentage'] = (
                        speaker_stats[speaker]['duration'] / total_duration * 100
                    )
                else:
                    speaker_stats[speaker]['percentage'] = 0
            
            results = {
                'audio_file': audio_file,
                'total_duration': len(audio) / sr,
                'analyzed_duration': total_duration,
                'speaker_timeline': speaker_timeline,
                'speaker_statistics': speaker_stats,
                'dominant_speaker': max(speaker_stats.keys(), 
                                      key=lambda x: speaker_stats[x]['duration'])
            }
            
            print(f"✅ Speaker analysis completed!")
            print(f"   Analyzed duration: {total_duration:.1f}s")
            print(f"   Segments identified: {len(speaker_timeline)}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error in speaker identification: {e}")
            return {'error': str(e)}
    
    def _calculate_confidence(self, features: np.ndarray, predicted_speaker: int) -> float:
        """Calculate confidence score for speaker prediction"""
        try:
            if f'Speaker_{predicted_speaker}' in self.speaker_profiles:
                profile = self.speaker_profiles[f'Speaker_{predicted_speaker}']
                mean_features = profile['mean_features']
                std_features = profile['std_features']
                
                # Calculate normalized distance
                distance = np.linalg.norm((features - mean_features) / (std_features + 1e-6))
                confidence = max(0, 1 - distance / 10)  # Normalize to 0-1 range
                return confidence
            else:
                return 0.5
        except:
            return 0.5
    
    def generate_speaker_report(self, results: Dict[str, Any], 
                              output_file: str = None) -> str:
        """
        Generate detailed speaker analysis report
        
        Args:
            results: Speaker identification results
            output_file: Optional output file path
            
        Returns:
            Report text or file path
        """
        try:
            report_lines = []
            report_lines.append("# Speaker Recognition Report")
            report_lines.append("=" * 50)
            report_lines.append(f"Audio File: {results.get('audio_file', 'N/A')}")
            report_lines.append(f"Total Duration: {results.get('total_duration', 0):.2f} seconds")
            report_lines.append(f"Analyzed Duration: {results.get('analyzed_duration', 0):.2f} seconds")
            report_lines.append(f"Dominant Speaker: {results.get('dominant_speaker', 'N/A')}")
            report_lines.append("")
            
            # Speaker Statistics
            report_lines.append("## Speaker Statistics")
            report_lines.append("-" * 30)
            
            if 'speaker_statistics' in results:
                for speaker, stats in results['speaker_statistics'].items():
                    report_lines.append(f"{speaker}:")
                    report_lines.append(f"  Duration: {stats['duration']:.2f}s")
                    report_lines.append(f"  Percentage: {stats['percentage']:.1f}%")
                    report_lines.append(f"  Segments: {stats['segments']}")
                    report_lines.append("")
            
            # Timeline Summary
            if 'speaker_timeline' in results and len(results['speaker_timeline']) > 0:
                report_lines.append("## Timeline Summary (First 10 segments)")
                report_lines.append("-" * 40)
                
                for i, segment in enumerate(results['speaker_timeline'][:10]):
                    report_lines.append(
                        f"{segment['start_time']:.1f}s - {segment['end_time']:.1f}s: "
                        f"{segment['speaker']} (conf: {segment['confidence']:.2f})"
                    )
                
                if len(results['speaker_timeline']) > 10:
                    report_lines.append(f"... and {len(results['speaker_timeline']) - 10} more segments")
            
            report_text = "\n".join(report_lines)
            
            if output_file:
                with open(output_file, 'w') as f:
                    f.write(report_text)
                print(f"📄 Speaker report saved to: {output_file}")
                return output_file
            else:
                return report_text
                
        except Exception as e:
            error_msg = f"Error generating speaker report: {e}"
            print(f"❌ {error_msg}")
            return error_msg
    
    def visualize_speaker_timeline(self, results: Dict[str, Any], 
                                 output_file: str = None) -> str:
        """
        Create visualization of speaker timeline
        
        Args:
            results: Speaker identification results
            output_file: Optional output file for plot
            
        Returns:
            Path to saved plot or error message
        """
        try:
            if 'speaker_timeline' not in results:
                return "No timeline data available"
            
            timeline = results['speaker_timeline']
            if not timeline:
                return "Empty timeline"
            
            # Create plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Timeline plot
            colors = ['blue', 'red', 'green', 'orange', 'purple']
            speaker_colors = {}
            
            for i, segment in enumerate(timeline):
                speaker = segment['speaker']
                if speaker not in speaker_colors:
                    speaker_colors[speaker] = colors[len(speaker_colors) % len(colors)]
                
            ax1.barh(0, segment['end_time'] - segment['start_time'], 
                        left=segment['start_time'], height=0.5,
                        color=speaker_colors[speaker], alpha=0.7,
                        label=speaker)
            
            ax1.set_xlabel('Time (seconds)')
            ax1.set_ylabel('Speaker Activity')
            ax1.set_title('Speaker Timeline')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Speaking time pie chart
            if 'speaker_statistics' in results:
                stats = results['speaker_statistics']
                speakers = list(stats.keys())
                durations = [stats[speaker]['duration'] for speaker in speakers]
                colors_pie = [speaker_colors.get(speaker, 'gray') for speaker in speakers]
                
                ax2.pie(durations, labels=speakers, colors=colors_pie, autopct='%1.1f%%')
                ax2.set_title('Speaking Time Distribution')
            
            plt.tight_layout()
            
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"📊 Speaker timeline plot saved to: {output_file}")
                return output_file
            else:
                plt.show()
                return "Plot displayed"
                
        except Exception as e:
            error_msg = f"Error creating speaker visualization: {e}"
            print(f"❌ {error_msg}")
            return error_msg
    
    def save_model(self, model_file: str) -> bool:
        """Save trained model to file"""
        try:
            if not self.is_trained:
                print("❌ No trained model to save")
                return False
            
            model_data = {
                'n_speakers': self.n_speakers,
                'sample_rate': self.sample_rate,
                'speaker_profiles': self.speaker_profiles,
                'scaler_mean': self.scaler.mean_.tolist(),
                'scaler_scale': self.scaler.scale_.tolist(),
                'pca_components': self.pca.components_.tolist(),
                'pca_mean': self.pca.mean_.tolist(),
                'kmeans_centers': self.kmeans.cluster_centers_.tolist()
            }
            
            with open(model_file, 'w') as f:
                json.dump(model_data, f, indent=2)
            
            print(f"💾 Model saved to: {model_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False
    
    def load_model(self, model_file: str) -> bool:
        """Load trained model from file"""
        try:
            with open(model_file, 'r') as f:
                model_data = json.load(f)
            
            self.n_speakers = model_data['n_speakers']
            self.sample_rate = model_data['sample_rate']
            self.speaker_profiles = model_data['speaker_profiles']
            
            # Restore scaler
            self.scaler.mean_ = np.array(model_data['scaler_mean'])
            self.scaler.scale_ = np.array(model_data['scaler_scale'])
            
            # Restore PCA
            self.pca.components_ = np.array(model_data['pca_components'])
            self.pca.mean_ = np.array(model_data['pca_mean'])
            
            # Restore KMeans
            self.kmeans.cluster_centers_ = np.array(model_data['kmeans_centers'])
            
            self.is_trained = True
            print(f"📥 Model loaded from: {model_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False


def demo_speaker_recognition():
    """Demonstration of speaker recognition capabilities"""
    print("🎯 Speaker Recognition Demo")
    print("=" * 40)
    
    # Create sample audio data for demo
    sr = 16000
    duration = 10  # 10 seconds
    t = np.linspace(0, duration, sr * duration)
    
    # Create synthetic multi-speaker audio
    # Speaker 1: Lower frequency voice
    speaker1_audio = 0.5 * np.sin(2 * np.pi * 150 * t) * (1 + 0.1 * np.sin(2 * np.pi * 5 * t))
    # Speaker 2: Higher frequency voice  
    speaker2_audio = 0.3 * np.sin(2 * np.pi * 250 * t) * (1 + 0.1 * np.sin(2 * np.pi * 3 * t))
    
    # Create alternating pattern (simulating conversation)
    combined_audio = np.zeros_like(t)
    segment_length = sr * 2  # 2 second segments
    
    for i in range(0, len(t), segment_length):
        end_idx = min(i + segment_length, len(t))
        if (i // segment_length) % 2 == 0:
            combined_audio[i:end_idx] = speaker1_audio[i:end_idx]
        else:
            combined_audio[i:end_idx] = speaker2_audio[i:end_idx]
    
    # Add some noise
    noise = 0.05 * np.random.normal(0, 1, len(combined_audio))
    combined_audio += noise
    
    # Save demo audio
    demo_file = "demo_conversation.wav"
    sf.write(demo_file, combined_audio, sr)
    
    print(f"📁 Created demo audio file: {demo_file}")
    
    # Initialize speaker recognition
    recognizer = SpeakerRecognition(n_speakers=2, sample_rate=sr)
    
    # Train on demo audio
    training_results = recognizer.train_speaker_clustering(demo_file)
    
    if 'error' not in training_results:
        # Analyze the same file (in real use, this would be different files)
        analysis_results = recognizer.identify_speakers(demo_file)
        
        if 'error' not in analysis_results:
            # Generate report
            report = recognizer.generate_speaker_report(analysis_results)
            print("\n" + report)
            
            # Create visualization
            plot_file = "speaker_timeline_demo.png"
            recognizer.visualize_speaker_timeline(analysis_results, plot_file)
            
            # Save model
            model_file = "speaker_model_demo.json"
            recognizer.save_model(model_file)
            
            print(f"\n✅ Demo completed successfully!")
            print(f"📄 Files created: {demo_file}, {plot_file}, {model_file}")
        else:
            print(f"❌ Analysis failed: {analysis_results['error']}")
    else:
        print(f"❌ Training failed: {training_results['error']}")


if __name__ == "__main__":
    demo_speaker_recognition()
