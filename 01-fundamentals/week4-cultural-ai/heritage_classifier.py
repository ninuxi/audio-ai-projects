"""
Audio AI Projects - Advanced Audio Processing System
Copyright (c) 2025 Antonino Mainenti (ninuxi)
Licensed under MIT License - see LICENSE file
GitHub: https://github.com/ninuxi/audio-ai-projects
"""
#!/usr/bin/env python3
"""
Heritage Classifier - Cultural Content Classification System
Classifies audio content into cultural categories using AI techniques
"""

import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class HeritageClassifier:
    """
    AI system for classifying cultural and heritage audio content
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize heritage classification system
        
        Args:
            model_type: Type of ML model ('random_forest', 'svm', 'neural_network')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
        # Cultural categories for classification
        self.cultural_categories = {
            'folk_music': 'Traditional folk songs and ballads',
            'classical_music': 'Classical compositions and orchestral works',
            'opera': 'Operatic performances and arias',
            'spoken_word': 'Poetry, speeches, and literary recordings',
            'historical_speeches': 'Important historical addresses and declarations',
            'traditional_stories': 'Folklore, legends, and oral traditions',
            'religious_chants': 'Sacred music and religious ceremonies',
            'regional_dialects': 'Local languages and dialect preservation',
            'instrumental_traditional': 'Traditional instrumental music',
            'contemporary_cultural': 'Modern cultural expressions'
        }
        
        # Initialize model based on type
        self._initialize_model()
        
        # Feature extraction parameters
        self.sample_rate = 22050
        self.n_mfcc = 13
        self.n_mels = 40
        
    def _initialize_model(self):
        """Initialize the machine learning model"""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        elif self.model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            )
        else:
            # Default to random forest
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    def extract_cultural_features(self, audio_path: str) -> np.ndarray:
        """
        Extract comprehensive features for cultural classification
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Feature vector for classification
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Basic audio features
            features = []
            
            # 1. MFCCs (Mel-frequency cepstral coefficients)
            mfccs = librosa.feature.mfcc(
                y=audio, sr=sr, n_mfcc=self.n_mfcc, n_fft=2048, hop_length=512
            )
            features.extend([
                np.mean(mfccs, axis=1),
                np.std(mfccs, axis=1),
                np.max(mfccs, axis=1),
                np.min(mfccs, axis=1)
            ])
            
            # 2. Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
            
            features.extend([
                [np.mean(spectral_centroids), np.std(spectral_centroids)],
                [np.mean(spectral_rolloff), np.std(spectral_rolloff)],
                [np.mean(spectral_bandwidth), np.std(spectral_bandwidth)],
                [np.mean(zero_crossing_rate), np.std(zero_crossing_rate)]
            ])
            
            # 3. Harmonic and percussive components
            harmonic, percussive = librosa.effects.hpss(audio)
            harmonic_energy = np.mean(harmonic ** 2)
            percussive_energy = np.mean(percussive ** 2)
            features.append([harmonic_energy, percussive_energy])
            
            # 4. Chroma features (for musical content)
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            features.extend([
                np.mean(chroma, axis=1),
                np.std(chroma, axis=1)
            ])
            
            # 5. Tempo and rhythm features
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
            features.append([tempo, len(beats) / (len(audio) / sr)])  # beats per second
            
            # 6. Tonal features
            tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
            features.extend([
                np.mean(tonnetz, axis=1),
                np.std(tonnetz, axis=1)
            ])
            
            # 7. Cultural-specific features
            # Pitch variation (important for traditional music)
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
            pitch_values = pitches[pitches > 0]
            if len(pitch_values) > 0:
                pitch_variation = np.std(pitch_values)
                pitch_mean = np.mean(pitch_values)
            else:
                pitch_variation = 0
                pitch_mean = 0
            
            features.append([pitch_mean, pitch_variation])
            
            # 8. Energy distribution across frequency bands
            stft = librosa.stft(audio)
            magnitude = np.abs(stft)
            
            # Low, mid, high frequency energy
            freq_bins = magnitude.shape[0]
            low_freq_energy = np.mean(magnitude[:freq_bins//3, :])
            mid_freq_energy = np.mean(magnitude[freq_bins//3:2*freq_bins//3, :])
            high_freq_energy = np.mean(magnitude[2*freq_bins//3:, :])
            
            features.append([low_freq_energy, mid_freq_energy, high_freq_energy])
            
            # Flatten all features
            feature_vector = np.concatenate([np.array(f).flatten() for f in features])
            
            return feature_vector
            
        except Exception as e:
            print(f"Error extracting features from {audio_path}: {e}")
            # Return zero vector on error
            return np.zeros(100)  # Approximate expected feature length
    
    def create_training_dataset(self, data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create training dataset from organized audio files
        
        Args:
            data_dir: Directory with subdirectories for each cultural category
            
        Returns:
            Features array and labels array
        """
        features_list = []
        labels_list = []
        
        data_path = Path(data_dir)
        
        if not data_path.exists():
            print(f"Data directory {data_dir} does not exist")
            return np.array([]), np.array([])
        
        print(f"Creating training dataset from: {data_dir}")
        
        # Process each category directory
        for category in self.cultural_categories.keys():
            category_path = data_path / category
            
            if category_path.exists():
                print(f"Processing category: {category}")
                audio_files = list(category_path.glob("*.wav")) + list(category_path.glob("*.mp3"))
                
                for audio_file in audio_files:
                    features = self.extract_cultural_features(str(audio_file))
                    
                    if not np.all(features == 0):  # Skip failed extractions
                        features_list.append(features)
                        labels_list.append(category)
                        
                print(f"  Processed {len(audio_files)} files")
            else:
                print(f"Category directory not found: {category_path}")
        
        if len(features_list) == 0:
            print("No valid audio files found for training")
            return np.array([]), np.array([])
        
        # Convert to arrays and ensure consistent feature length
        features_array = np.array(features_list)
        labels_array = np.array(labels_list)
        
        print(f"Dataset created: {len(features_array)} samples, {features_array.shape[1]} features")
        
        return features_array, labels_array
    
    def train_classifier(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """
        Train the cultural heritage classifier
        
        Args:
            features: Feature matrix
            labels: Labels array
            
        Returns:
            Training results and metrics
        """
        if len(features) == 0:
            return {'error': 'No training data provided'}
        
        try:
            print(f"Training {self.model_type} classifier...")
            print(f"Dataset: {len(features)} samples, {features.shape[1]} features")
            
            # Encode labels
            labels_encoded = self.label_encoder.fit_transform(labels)
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features_scaled, labels_encoded, 
                test_size=0.2, 
                random_state=42, 
                stratify=labels_encoded
            )
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            # Cross-validation
            cv_scores = cross_val_score(self.model, features_scaled, labels_encoded, cv=5)
            
            # Predictions for detailed metrics
            y_pred = self.model.predict(X_test)
            
            # Classification report
            class_names = self.label_encoder.classes_
            class_report = classification_report(
                y_test, y_pred, 
                target_names=class_names, 
                output_dict=True
            )
            
            self.is_trained = True
            
            training_results = {
                'model_type': self.model_type,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'features_count': features.shape[1],
                'categories': list(class_names),
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'cv_mean_accuracy': np.mean(cv_scores),
                'cv_std_accuracy': np.std(cv_scores),
                'classification_report': class_report,
                'training_completed': datetime.now().isoformat()
            }
            
            print(f"✅ Training completed!")
            print(f"   Train accuracy: {train_score:.3f}")
            print(f"   Test accuracy: {test_score:.3f}")
            print(f"   CV accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
            
            return training_results
            
        except Exception as e:
            error_msg = f"Training error: {e}"
            print(f"❌ {error_msg}")
            return {'error': error_msg}
    
    def classify_audio(self, audio_path: str, return_probabilities: bool = True) -> Dict[str, Any]:
        """
        Classify cultural content of audio file
        
        Args:
            audio_path: Path to audio file
            return_probabilities: Whether to return probability scores
            
        Returns:
            Classification results
        """
        if not self.is_trained:
            return {'error': 'Model not trained. Call train_classifier first.'}
        
        try:
            print(f"Classifying: {audio_path}")
            
            # Extract features
            features = self.extract_cultural_features(audio_path)
            
            if np.all(features == 0):
                return {'error': 'Failed to extract features from audio'}
            
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Predict
            prediction = self.model.predict(features_scaled)[0]
            predicted_category = self.label_encoder.inverse_transform([prediction])[0]
            
            result = {
                'audio_file': audio_path,
                'predicted_category': predicted_category,
                'category_description': self.cultural_categories.get(predicted_category, 'Unknown'),
                'classification_time': datetime.now().isoformat()
            }
            
            # Add probabilities if requested and supported
            if return_probabilities and hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features_scaled)[0]
                categories = self.label_encoder.classes_
                
                prob_dict = {}
                for i, category in enumerate(categories):
                    prob_dict[category] = float(probabilities[i])
                
                result['probabilities'] = prob_dict
                result['confidence'] = float(max(probabilities))
            
            print(f"✅ Classification: {predicted_category}")
            if 'confidence' in result:
                print(f"   Confidence: {result['confidence']:.3f}")
            
            return result
            
        except Exception as e:
            error_msg = f"Classification error: {e}"
            print(f"❌ {error_msg}")
            return {'error': error_msg}
    
    def batch_classify(self, audio_directory: str, output_file: str = None) -> Dict[str, Any]:
        """
        Classify all audio files in a directory
        
        Args:
            audio_directory: Directory containing audio files
            output_file: Optional output file for results
            
        Returns:
            Batch classification results
        """
        if not self.is_trained:
            return {'error': 'Model not trained'}
        
        try:
            audio_dir = Path(audio_directory)
            if not audio_dir.exists():
                return {'error': f'Directory not found: {audio_directory}'}
            
            print(f"Batch classifying files in: {audio_directory}")
            
            # Find audio files
            audio_files = []
            for ext in ['*.wav', '*.mp3', '*.flac', '*.ogg']:
                audio_files.extend(audio_dir.glob(ext))
            
            if not audio_files:
                return {'error': 'No audio files found'}
            
            results = []
            category_counts = {}
            
            for audio_file in audio_files:
                classification = self.classify_audio(str(audio_file))
                
                if 'error' not in classification:
                    results.append(classification)
                    
                    # Count categories
                    category = classification['predicted_category']
                    category_counts[category] = category_counts.get(category, 0) + 1
            
            batch_results = {
                'directory': audio_directory,
                'total_files': len(audio_files),
                'classified_files': len(results),
                'failed_files': len(audio_files) - len(results),
                'category_distribution': category_counts,
                'detailed_results': results,
                'batch_completed': datetime.now().isoformat()
            }
            
            # Save results if output file specified
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(batch_results, f, indent=2)
                print(f"📄 Results saved to: {output_file}")
            
            print(f"✅ Batch classification completed:")
            print(f"   Processed: {len(results)}/{len(audio_files)} files")
            for category, count in category_counts.items():
                print(f"   {category}: {count} files")
            
            return batch_results
            
        except Exception as e:
            error_msg = f"Batch classification error: {e}"
            print(f"❌ {error_msg}")
            return {'error': error_msg}
    
    def generate_classification_report(self, results: Dict[str, Any], 
                                     report_file: str = None) -> str:
        """
        Generate detailed classification report
        
        Args:
            results: Classification results
            report_file: Optional output file
            
        Returns:
            Report text or file path
        """
        try:
            report_lines = []
            report_lines.append("# Cultural Heritage Classification Report")
            report_lines.append("=" * 50)
            report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append("")
            
            if 'directory' in results:
                # Batch classification report
                report_lines.append("## Batch Classification Summary")
                report_lines.append(f"Directory: {results['directory']}")
                report_lines.append(f"Total Files: {results['total_files']}")
                report_lines.append(f"Successfully Classified: {results['classified_files']}")
                report_lines.append(f"Failed: {results['failed_files']}")
                report_lines.append("")
                
                # Category distribution
                report_lines.append("## Cultural Category Distribution")
                report_lines.append("-" * 30)
                
                if 'category_distribution' in results:
                    total_classified = sum(results['category_distribution'].values())
                    for category, count in results['category_distribution'].items():
                        percentage = (count / total_classified) * 100 if total_classified > 0 else 0
                        description = self.cultural_categories.get(category, 'Unknown')
                        report_lines.append(f"{category}: {count} files ({percentage:.1f}%)")
                        report_lines.append(f"  Description: {description}")
                        report_lines.append("")
                
                # Detailed results sample
                if 'detailed_results' in results and results['detailed_results']:
                    report_lines.append("## Sample Classifications")
                    report_lines.append("-" * 30)
                    
                    sample_size = min(10, len(results['detailed_results']))
                    for i, result in enumerate(results['detailed_results'][:sample_size]):
                        filename = Path(result['audio_file']).name
                        category = result['predicted_category']
                        confidence = result.get('confidence', 'N/A')
                        
                        report_lines.append(f"{i+1}. {filename}")
                        report_lines.append(f"   Category: {category}")
                        if confidence != 'N/A':
                            report_lines.append(f"   Confidence: {confidence:.3f}")
                        report_lines.append("")
                    
                    if len(results['detailed_results']) > 10:
                        remaining = len(results['detailed_results']) - 10
                        report_lines.append(f"... and {remaining} more files")
            
            else:
                # Single file classification report
                report_lines.append("## Single File Classification")
                report_lines.append(f"File: {results.get('audio_file', 'N/A')}")
                report_lines.append(f"Category: {results.get('predicted_category', 'N/A')}")
                report_lines.append(f"Description: {results.get('category_description', 'N/A')}")
                if 'confidence' in results:
                    report_lines.append(f"Confidence: {results['confidence']:.3f}")
                
                if 'probabilities' in results:
                    report_lines.append("")
                    report_lines.append("## All Category Probabilities")
                    report_lines.append("-" * 30)
                    
                    sorted_probs = sorted(results['probabilities'].items(), 
                                        key=lambda x: x[1], reverse=True)
                    
                    for category, prob in sorted_probs:
                        report_lines.append(f"{category}: {prob:.3f}")
            
            # Cultural categories reference
            report_lines.append("")
            report_lines.append("## Cultural Categories Reference")
            report_lines.append("-" * 40)
            for category, description in self.cultural_categories.items():
                report_lines.append(f"**{category}**: {description}")
            
            report_text = "\n".join(report_lines)
            
            if report_file:
                with open(report_file, 'w') as f:
                    f.write(report_text)
                print(f"📄 Classification report saved to: {report_file}")
                return report_file
            else:
                return report_text
                
        except Exception as e:
            error_msg = f"Error generating report: {e}"
            print(f"❌ {error_msg}")
            return error_msg
    
    def visualize_classification_results(self, results: Dict[str, Any], 
                                       output_file: str = None) -> str:
        """
        Create visualization of classification results
        
        Args:
            results: Classification results
            output_file: Optional output file for plot
            
        Returns:
            Path to saved plot or error message
        """
        try:
            if 'category_distribution' not in results:
                return "No category distribution data available"
            
            # Create subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Pie chart of category distribution
            categories = list(results['category_distribution'].keys())
            counts = list(results['category_distribution'].values())
            
            ax1.pie(counts, labels=categories, autopct='%1.1f%%', startangle=90)
            ax1.set_title('Cultural Category Distribution')
            
            # Bar chart of category counts
            ax2.bar(categories, counts, color='skyblue', alpha=0.7)
            ax2.set_title('File Count by Category')
            ax2.set_ylabel('Number of Files')
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"📊 Classification visualization saved to: {output_file}")
                return output_file
            else:
                plt.show()
                return "Plot displayed"
                
        except Exception as e:
            error_msg = f"Error creating visualization: {e}"
            print(f"❌ {error_msg}")
            return error_msg
    
    def save_model(self, model_file: str) -> bool:
        """Save trained model to file"""
        try:
            if not self.is_trained:
                print("❌ No trained model to save")
                return False
            
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'model_type': self.model_type,
                'cultural_categories': self.cultural_categories,
                'is_trained': self.is_trained
            }
            
            joblib.dump(model_data, model_file)
            print(f"💾 Model saved to: {model_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False
    
    def load_model(self, model_file: str) -> bool:
        """Load trained model from file"""
        try:
            model_data = joblib.load(model_file)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            self.model_type = model_data['model_type']
            self.cultural_categories = model_data['cultural_categories']
            self.is_trained = model_data['is_trained']
            
            print(f"📥 Model loaded from: {model_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False


def demo_heritage_classifier():
    """Demonstration of heritage classification capabilities"""
    print("🏛️ Heritage Classifier Demo")
    print("=" * 40)
    
    # Initialize classifier
    classifier = HeritageClassifier(model_type='random_forest')
    
    # Create synthetic training data for demo
    print("📊 Creating synthetic training data...")
    
    # Generate synthetic audio features for different categories
    np.random.seed(42)
    
    features_list = []
    labels_list = []
    
    categories = ['folk_music', 'classical_music', 'spoken_word', 'religious_chants']
    
    for category in categories:
        print(f"Generating samples for: {category}")
        
        for _ in range(20):  # 20 samples per category
            # Create category-specific synthetic features
            base_features = np.random.random(100)
            
            if category == 'folk_music':
                # Folk music: moderate tempo, traditional harmonies
                base_features[:20] += np.random.normal(0.5, 0.1, 20)
            elif category == 'classical_music':
                # Classical: complex harmonies, wide dynamic range
                base_features[20:40] += np.random.normal(0.7, 0.15, 20)
            elif category == 'spoken_word':
                # Spoken: lower spectral content, speech patterns
                base_features[40:60] += np.random.normal(0.3, 0.1, 20)
            elif category == 'religious_chants':
                # Chants: repetitive, modal harmonies
                base_features[60:80] += np.random.normal(0.6, 0.12, 20)
            
            features_list.append(base_features)
            labels_list.append(category)
    
    features_array = np.array(features_list)
    labels_array = np.array(labels_list)
    
    print(f"✅ Created synthetic dataset: {len(features_array)} samples")
    
    # Train classifier
    training_results = classifier.train_classifier(features_array, labels_array)
    
    if 'error' not in training_results:
        print(f"🎯 Training Results:")
        print(f"   Accuracy: {training_results['test_accuracy']:.3f}")
        print(f"   Categories: {len(training_results['categories'])}")
        
        # Test classification with synthetic data
        print("\n🧪 Testing classification...")
        
        # Create test sample (folk music characteristics)
        test_features = np.random.random(100)
        test_features[:20] += np.random.normal(0.5, 0.1, 20)
        
        # Save as temporary "audio file" for demonstration

        # Since we can't create actual audio, we'll simulate the classification
        print("📁 Simulating audio classification...")
                
        # Since we can't create actual audio, we'll simulate the classification
        print("


        # Mock classification result
        simulated_result = {
            'audio_file': 'demo_folk_song.wav',
            'predicted_category': 'folk_music',
            'category_description': classifier.cultural_categories['folk_music'],
            'confidence': 0.85,
            'probabilities': {
                'folk_music': 0.85,
                'classical_music': 0.08,
                'spoken_word': 0.04,
                'religious_chants': 0.03
            }
        }
        
        print("✅ Classification Results:")
        print(f"   Predicted: {simulated_result['predicted_category']}")
        print(f"   Confidence: {simulated_result['confidence']:.3f}")
        print(f"   Description: {simulated_result['category_description']}")
        
        # Generate report
        report = classifier.generate_classification_report(simulated_result)
        print(f"\n📄 Generated classification report")
        
        # Save model
        model_file = "heritage_classifier_demo.pkl"
        if classifier.save_model(model_file):
            print(f"💾 Model saved to: {model_file}")
        
        print(f"\n🎉 Heritage Classifier demo completed successfully!")
        
    else:
        print(f"❌ Training failed: {training_results['error']}")


if __name__ == "__main__":
    demo_heritage_classifier()
