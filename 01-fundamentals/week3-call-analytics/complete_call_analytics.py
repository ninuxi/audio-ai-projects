"""
Audio AI Projects - Advanced Audio Processing System
Copyright (c) 2025 Antonino Mainenti (ninuxi)
Licensed under MIT License - see LICENSE file
GitHub: https://github.com/ninuxi/audio-ai-projects
"""
"""
TIM CALL ANALYTICS - SISTEMA COMPLETO IN FILE UNICO
==================================================

Tutti i sistemi integrati:
- VAD (Voice Activity Detection)  
- Speaker Recognition (Gender + Age)
- Audio Quality Scoring
- Business Analytics & ROI

UTILIZZO:
python complete_call_analytics.py
"""

import asyncio
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import warnings
import os
import glob
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from scipy import signal
import joblib

warnings.filterwarnings('ignore')

# ============================================================================
# CALL CENTER VAD SYSTEM
# ============================================================================

class CallCenterVAD:
    """VAD System ottimizzato per Call Center"""
    
    def __init__(self):
        self.sr_target = 8000
        self.frame_length = 1024
        self.hop_length = 256
        self.energy_threshold = 0.005
        
    def extract_telecom_features(self, audio_segment):
        """Estrae features telecom ottimizzate"""
        if len(audio_segment) < self.frame_length:
            return None
        
        features = {}
        
        # Energia RMS
        rms = librosa.feature.rms(y=audio_segment, 
                                  frame_length=self.frame_length,
                                  hop_length=self.hop_length)[0]
        features['energy_mean'] = np.mean(rms)
        features['energy_std'] = np.std(rms)
        
        # Pitch analysis
        try:
            pitches, magnitudes = librosa.piptrack(y=audio_segment, sr=self.sr_target)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                features['pitch_mean'] = np.mean(pitch_values)
                features['pitch_std'] = np.std(pitch_values)
            else:
                features['pitch_mean'] = 0
                features['pitch_std'] = 0
        except:
            features['pitch_mean'] = 0
            features['pitch_std'] = 0
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_segment, 
                                                              sr=self.sr_target)[0]
        features['brightness_mean'] = np.mean(spectral_centroid)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio_segment)[0]
        features['zcr_mean'] = np.mean(zcr)
        
        # MFCC (primi 8 per telefonia)
        mfcc = librosa.feature.mfcc(y=audio_segment, sr=self.sr_target, 
                                   n_mfcc=8, n_fft=512)
        for i in range(8):
            features[f'mfcc_{i}'] = np.mean(mfcc[i])
        
        # Periodicità
        autocorr = np.correlate(audio_segment, audio_segment, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        if len(autocorr) > 1:
            features['periodicity'] = np.max(autocorr[1:]) / autocorr[0] if autocorr[0] > 0 else 0
        else:
            features['periodicity'] = 0
        
        return features
    
    def classify_call_segment(self, features):
        """Classifica segmento chiamata"""
        if features is None:
            return 'silence'
        
        energy = features['energy_mean']
        pitch = features['pitch_mean']
        brightness = features['brightness_mean']
        zcr = features['zcr_mean']
        periodicity = features['periodicity']
        
        # Silenzio
        if energy < self.energy_threshold:
            return 'silence'
        
        # System beep
        if (energy > 0.05 and brightness > 2000 and pitch > 800 and features['pitch_std'] < 10):
            return 'system_beep'
        
        # Hold music
        if (energy > 0.02 and features['energy_std'] < 0.01 and periodicity > 0.7):
            return 'hold_music'
        
        # Voce umana
        if (energy > self.energy_threshold and 80 < pitch < 400 and zcr < 0.15 and periodicity > 0.3):
            if pitch < 180:
                return 'customer_voice'
            else:
                return 'agent_voice'
        
        # Background noise
        if energy > self.energy_threshold and zcr > 0.2:
            return 'background_noise'
        
        return 'silence'

# ============================================================================
# SPEAKER RECOGNITION SYSTEM  
# ============================================================================

class SpeakerRecognitionSystem:
    """Speaker Recognition: Gender + Age detection"""
    
    def __init__(self):
        self.gender_model = None
        self.age_model = None
        self.is_trained = False
        self.frame_length = 2048
        self.hop_length = 512
        self.n_mfcc = 13
    
    def extract_speaker_features(self, audio_segment, sr):
        """Estrae features per speaker recognition"""
        if len(audio_segment) < self.frame_length:
            return None
        
        features = {}
        
        # Pitch analysis esteso
        try:
            pitches, magnitudes = librosa.piptrack(y=audio_segment, sr=sr, 
                                                  threshold=0.1, fmin=50, fmax=400)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                features['pitch_mean'] = np.mean(pitch_values)
                features['pitch_std'] = np.std(pitch_values)
                features['pitch_median'] = np.median(pitch_values)
                features['pitch_range'] = np.max(pitch_values) - np.min(pitch_values)
            else:
                for key in ['pitch_mean', 'pitch_std', 'pitch_median', 'pitch_range']:
                    features[key] = 0
        except:
            for key in ['pitch_mean', 'pitch_std', 'pitch_median', 'pitch_range']:
                features[key] = 0
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_segment, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroid)
        features['spectral_centroid_std'] = np.std(spectral_centroid)
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_segment, sr=sr)[0]
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        
        # MFCC
        mfcc = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=self.n_mfcc)
        for i in range(self.n_mfcc):
            features[f'mfcc_{i}_mean'] = np.mean(mfcc[i])
        
        # ZCR
        zcr = librosa.feature.zero_crossing_rate(audio_segment)[0]
        features['zcr_mean'] = np.mean(zcr)
        
        return features
    
    def classify_speaker_rules(self, features):
        """Classificazione speaker con regole"""
        if features is None:
            return {'gender': 'unknown', 'age_group': 'unknown', 'confidence': 0}
        
        pitch_mean = features.get('pitch_mean', 0)
        spectral_centroid = features.get('spectral_centroid_mean', 0)
        
        # Gender classification
        if pitch_mean > 0:
            if pitch_mean < 165:
                gender = 'male'
                gender_confidence = 0.8 if pitch_mean < 140 else 0.6
            elif pitch_mean > 190:
                gender = 'female'
                gender_confidence = 0.8 if pitch_mean > 220 else 0.6
            else:
                gender = 'unknown'
                gender_confidence = 0.3
        else:
            gender = 'unknown'
            gender_confidence = 0
        
        # Age estimation (simplified)
        if spectral_centroid > 2000:
            age_group = 'young'
        elif spectral_centroid < 1200:
            age_group = 'elderly'
        else:
            age_group = 'adult'
        
        return {
            'gender': gender,
            'age_group': age_group,
            'gender_confidence': gender_confidence,
            'age_confidence': 0.6
        }
    
    def predict_speaker(self, audio_segment, sr):
        """Predice speaker characteristics"""
        features = self.extract_speaker_features(audio_segment, sr)
        result = self.classify_speaker_rules(features)
        result['method'] = 'rules_based'
        return result

# ============================================================================
# AUDIO QUALITY SCORER
# ============================================================================

class AudioQualityScorer:
    """Audio Quality Scoring 0-100"""
    
    def __init__(self):
        self.thresholds = {
            'snr_excellent': 25,
            'snr_good': 15,
            'snr_poor': 5,
            'distortion_max': 0.1,
            'dynamic_range_min': 20
        }
    
    def calculate_snr_estimate(self, audio_segment):
        """Stima SNR"""
        try:
            smoothed = signal.savgol_filter(audio_segment, 51, 3)
            noise = audio_segment - smoothed
            
            signal_power = np.mean(smoothed ** 2)
            noise_power = np.mean(noise ** 2)
            
            if noise_power > 0:
                snr_linear = signal_power / noise_power
                snr_db = 10 * np.log10(snr_linear) if snr_linear > 0 else 0
            else:
                snr_db = 50
            
            return min(snr_db, 50)
        except:
            return 20  # Default reasonable value
    
    def calculate_dynamic_range(self, audio_segment):
        """Calcola dynamic range"""
        peak = np.max(np.abs(audio_segment))
        rms = np.sqrt(np.mean(audio_segment ** 2))
        
        if rms > 0:
            dynamic_range_db = 20 * np.log10(peak / rms)
        else:
            dynamic_range_db = 0
        
        return min(dynamic_range_db, 60)
    
    def score_audio_quality(self, audio_segment, sr):
        """Score qualità 0-100"""
        if len(audio_segment) < 1024:
            return {
                'overall_score': 0,
                'snr_db': 0,
                'thd_percent': 0,
                'dynamic_range_db': 0
            }
        
        try:
            # Metriche
            snr = self.calculate_snr_estimate(audio_segment)
            dynamic_range = self.calculate_dynamic_range(audio_segment)
            
            # Score SNR
            if snr >= self.thresholds['snr_excellent']:
                snr_score = 100
            elif snr >= self.thresholds['snr_good']:
                snr_score = 70 + (snr - self.thresholds['snr_good']) / (self.thresholds['snr_excellent'] - self.thresholds['snr_good']) * 30
            else:
                snr_score = max(0, snr / self.thresholds['snr_good'] * 70)
            
            # Score dynamic range
            dr_score = min(100, (dynamic_range / self.thresholds['dynamic_range_min']) * 100)
            
            # Score finale
            final_score = snr_score * 0.7 + dr_score * 0.3
            
            return {
                'overall_score': min(100, max(0, final_score)),
                'snr_db': snr,
                'thd_percent': 0,  # Simplified
                'dynamic_range_db': dynamic_range,
                'snr_score': snr_score,
                'dynamic_range_score': dr_score
            }
        except:
            return {
                'overall_score': 50,  # Default
                'snr_db': 15,
                'thd_percent': 0,
                'dynamic_range_db': 20,
                'snr_score': 70,
                'dynamic_range_score': 70
            }

# ============================================================================
# UNIFIED CALL ANALYTICS SYSTEM
# ============================================================================

@dataclass
class CallAnalyticsResult:
    """Risultato completo analisi chiamata"""
    call_id: str
    timestamp: datetime
    duration: float
    voice_activity: bool
    segment_type: str
    talk_efficiency: float
    speaker_gender: str
    speaker_age_group: str
    speaker_confidence: float
    audio_quality_score: float
    snr_db: float
    technical_issues: List[str]
    estimated_cost: float
    potential_savings: float
    customer_satisfaction_score: float
    action_required: Optional[str]

class UnifiedCallAnalytics:
    """Sistema unificato completo per TIM"""
    
    def __init__(self):
        # Inizializza sottosistemi
        self.vad_system = CallCenterVAD()
        self.speaker_system = SpeakerRecognitionSystem()
        self.quality_system = AudioQualityScorer()
        
        # Configurazione business
        self.business_config = {
            'cost_per_minute': 0.15,
            'quality_threshold': 75,
            'efficiency_threshold': 60
        }
        
        self.session_metrics = {
            'total_calls': 0,
            'total_savings': 0,
            'avg_quality': 0,
            'avg_efficiency': 0
        }
        
        print("🏢 TIM UNIFIED CALL ANALYTICS")
        print("🎯 VAD + Speaker + Quality = Complete Solution")
    
    async def analyze_call_realtime(self, call_id: str, audio_chunk: np.ndarray, 
                                   sr: int, chunk_timestamp: datetime) -> CallAnalyticsResult:
        """Analisi real-time completa"""
        
        # VAD Analysis
        vad_features = self.vad_system.extract_telecom_features(audio_chunk)
        segment_type = self.vad_system.classify_call_segment(vad_features) if vad_features else 'silence'
        voice_active = segment_type not in ['silence', 'background_noise']
        
        # Speaker Recognition (se voce attiva)
        if voice_active and len(audio_chunk) > 8000:
            speaker_result = self.speaker_system.predict_speaker(audio_chunk, sr)
        else:
            speaker_result = {
                'gender': 'unknown', 'age_group': 'unknown', 
                'gender_confidence': 0
            }
        
        # Quality Scoring
        quality_result = self.quality_system.score_audio_quality(audio_chunk, sr)
        
        # Business Metrics
        business_metrics = self.calculate_business_metrics(
            segment_type, quality_result, speaker_result, len(audio_chunk) / sr
        )
        
        # Create result
        result = CallAnalyticsResult(
            call_id=call_id,
            timestamp=chunk_timestamp,
            duration=len(audio_chunk) / sr,
            voice_activity=voice_active,
            segment_type=segment_type,
            talk_efficiency=business_metrics['talk_efficiency'],
            speaker_gender=speaker_result['gender'],
            speaker_age_group=speaker_result['age_group'],
            speaker_confidence=speaker_result.get('gender_confidence', 0),
            audio_quality_score=quality_result['overall_score'],
            snr_db=quality_result['snr_db'],
            technical_issues=self.identify_technical_issues(quality_result),
            estimated_cost=business_metrics['estimated_cost'],
            potential_savings=business_metrics['potential_savings'],
            customer_satisfaction_score=business_metrics['customer_satisfaction'],
            action_required=business_metrics['action_required']
        )
        
        return result
    
    def calculate_business_metrics(self, segment_type: str, quality_result: dict, 
                                 speaker_result: dict, duration: float) -> dict:
        """Calcola metriche business"""
        
        # Talk efficiency
        if segment_type in ['customer_voice', 'agent_voice']:
            talk_efficiency = 100.0
        else:
            talk_efficiency = 0.0
        
        # Costo
        chunk_cost = (duration / 60) * self.business_config['cost_per_minute']
        quality_factor = quality_result['overall_score'] / 100
        efficiency_factor = talk_efficiency / 100
        
        actual_cost = chunk_cost * (2 - (quality_factor + efficiency_factor) / 2)
        potential_savings = actual_cost - chunk_cost
        
        # Customer satisfaction
        customer_satisfaction = (
            quality_result['overall_score'] * 0.4 +
            talk_efficiency * 0.3 +
            speaker_result.get('gender_confidence', 0) * 100 * 0.3
        )
        
        # Action required
        action_required = None
        if quality_result['overall_score'] < self.business_config['quality_threshold']:
            action_required = "QUALITY_ALERT"
        elif talk_efficiency < self.business_config['efficiency_threshold']:
            action_required = "EFFICIENCY_ALERT"
        
        return {
            'talk_efficiency': talk_efficiency,
            'estimated_cost': actual_cost,
            'potential_savings': potential_savings,
            'customer_satisfaction': customer_satisfaction,
            'action_required': action_required
        }
    
    def identify_technical_issues(self, quality_result: dict) -> List[str]:
        """Identifica problemi tecnici"""
        issues = []
        
        if quality_result['snr_db'] < 10:
            issues.append("HIGH_NOISE")
        if quality_result['overall_score'] < 50:
            issues.append("POOR_LINE_QUALITY")
        if quality_result['dynamic_range_db'] < 10:
            issues.append("COMPRESSION")
        
        return issues
    
    async def process_full_call(self, call_id: str, audio_file_path: str, 
                               window_size: float = 1.0) -> Dict:
        """Processa chiamata completa"""
        
        print(f"\n📞 ANALISI CHIAMATA: {call_id}")
        print("=" * 50)
        
        # Carica audio
        y, sr = librosa.load(audio_file_path, sr=8000)
        duration = len(y) / sr
        
        print(f"⏱️ Durata: {duration:.1f}s")
        
        # Processa in finestre
        window_samples = int(window_size * sr)
        hop_samples = window_samples // 2
        
        all_results = []
        
        for i in range(0, len(y) - window_samples, hop_samples):
            chunk = y[i:i + window_samples]
            timestamp = datetime.now() + timedelta(seconds=i/sr)
            
            result = await self.analyze_call_realtime(call_id, chunk, sr, timestamp)
            all_results.append(result)
        
        # Generate summary
        call_summary = self.generate_call_summary(call_id, all_results, duration)
        
        # Visualize
        self.visualize_call_analysis(call_id, all_results, call_summary)
        
        return call_summary
    
    def generate_call_summary(self, call_id: str, results: List[CallAnalyticsResult], 
                            duration: float) -> Dict:
        """Genera summary chiamata"""
        
        if not results:
            return {}
        
        df = pd.DataFrame([asdict(r) for r in results])
        
        # Calcoli summary
        voice_windows = df[df['voice_activity'] == True]
        talk_efficiency = (len(voice_windows) / len(df)) * 100 if len(df) > 0 else 0
        
        gender_counts = df['speaker_gender'].value_counts()
        predominant_gender = gender_counts.index[0] if len(gender_counts) > 0 else 'unknown'
        
        avg_quality = df['audio_quality_score'].mean()
        avg_snr = df['snr_db'].mean()
        total_cost = df['estimated_cost'].sum()
        total_savings = df['potential_savings'].sum()
        avg_satisfaction = df['customer_satisfaction_score'].mean()
        
        overall_score = (talk_efficiency * 0.3 + avg_quality * 0.4 + avg_satisfaction * 0.3)
        
        summary = {
            'call_id': call_id,
            'duration_seconds': duration,
            'talk_efficiency': talk_efficiency,
            'predominant_gender': predominant_gender,
            'audio_quality_avg': avg_quality,
            'snr_avg': avg_snr,
            'estimated_cost_euros': total_cost,
            'potential_savings_euros': total_savings,
            'customer_satisfaction_avg': avg_satisfaction,
            'overall_call_score': overall_score,
            'recommendations': self.generate_recommendations(df, overall_score)
        }
        
        return summary
    
    def generate_recommendations(self, df: pd.DataFrame, overall_score: float) -> List[str]:
        """Genera raccomandazioni"""
        recommendations = []
        
        talk_eff = (len(df[df['voice_activity'] == True]) / len(df)) * 100
        if talk_eff < 50:
            recommendations.append("Migliorare talk efficiency con training operatori")
        
        avg_quality = df['audio_quality_score'].mean()
        if avg_quality < 70:
            recommendations.append("Verificare infrastruttura tecnica")
        
        if overall_score > 85:
            recommendations.append("Chiamata eccellente - best practice")
        elif overall_score < 60:
            recommendations.append("Chiamata critica - intervento necessario")
        
        return recommendations[:3]
    
    def visualize_call_analysis(self, call_id: str, results: List[CallAnalyticsResult], 
                               summary: Dict):
        """Visualizza analisi"""
        
        df = pd.DataFrame([asdict(r) for r in results])
        
        if len(df) == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'TIM Call Analytics - {call_id}', fontsize=14, fontweight='bold')
        
        # 1. Voice Activity Timeline
        times = np.arange(len(df)) * 0.5
        colors = df['segment_type'].map({
            'customer_voice': 'blue',
            'agent_voice': 'green',
            'silence': 'lightgray',
            'hold_music': 'orange'
        }).fillna('black')
        
        axes[0,0].scatter(times, [0]*len(times), c=colors, alpha=0.7, s=30)
        axes[0,0].set_title('Voice Activity Timeline')
        axes[0,0].set_xlabel('Time (s)')
        axes[0,0].set_yticks([])
        
        # 2. Audio Quality Over Time
        axes[0,1].plot(times, df['audio_quality_score'], 'r-', alpha=0.7)
        axes[0,1].axhline(y=75, color='orange', linestyle='--', label='Threshold')
        axes[0,1].set_title('Audio Quality Score')
        axes[0,1].set_ylabel('Quality (0-100)')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Speaker Distribution
        gender_data = df['speaker_gender'].value_counts()
        if len(gender_data) > 0:
            axes[1,0].pie(gender_data.values, labels=gender_data.index, autopct='%1.1f%%')
            axes[1,0].set_title('Speaker Gender Distribution')
        
        # 4. Business Metrics
        metrics = [
            ('Talk Efficiency', f"{summary.get('talk_efficiency', 0):.1f}%"),
            ('Quality Avg', f"{summary.get('audio_quality_avg', 0):.1f}/100"),
            ('Overall Score', f"{summary.get('overall_call_score', 0):.1f}/100"),
            ('Cost', f"€{summary.get('estimated_cost_euros', 0):.2f}")
        ]
        
        y_pos = np.arange(len(metrics))
        values = [
            summary.get('talk_efficiency', 0),
            summary.get('audio_quality_avg', 0),
            summary.get('overall_call_score', 0),
            summary.get('estimated_cost_euros', 0) * 50  # Scaled
        ]
        
        bars = axes[1,1].barh(y_pos, values, alpha=0.7)
        axes[1,1].set_yticks(y_pos)
        axes[1,1].set_yticklabels([m[0] for m in metrics])
        axes[1,1].set_title('Business Metrics')
        
        # Add values on bars
        for bar, (_, value) in zip(bars, metrics):
            axes[1,1].text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                          value, va='center')
        
        plt.tight_layout()
        plt.show()
        
        self.print_call_summary(summary)
    
    def print_call_summary(self, summary: Dict):
        """Print summary business"""
        
        print(f"\n📊 CALL SUMMARY - {summary['call_id']}")
        print("=" * 50)
        
        print(f"🎯 OVERALL SCORE: {summary.get('overall_call_score', 0):.1f}/100")
        print(f"💰 COSTO: €{summary.get('estimated_cost_euros', 0):.2f}")
        print(f"💡 RISPARMIO: €{summary.get('potential_savings_euros', 0):.2f}")
        
        print(f"\n📞 METRICHE:")
        print(f"   Talk Efficiency: {summary.get('talk_efficiency', 0):.1f}%")
        print(f"   Audio Quality: {summary.get('audio_quality_avg', 0):.1f}/100")
        print(f"   Speaker: {summary.get('predominant_gender', 'unknown')}")
        
        if summary.get('recommendations'):
            print(f"\n🎯 RACCOMANDAZIONI:")
            for i, rec in enumerate(summary['recommendations'], 1):
                print(f"   {i}. {rec}")

# ============================================================================
# DEMO SISTEMA COMPLETO
# ============================================================================

async def demo_system():
    """Demo sistema unificato"""
    
    print("🏢 TIM CALL ANALYTICS - DEMO SISTEMA COMPLETO")
    print("=" * 60)
    
    # Inizializza sistema
    analytics = UnifiedCallAnalytics()
    
    # File di test
    test_files = [
        "data/voce_silenzi.wav",
        "data/voce_merda.wav",
        "data/tremo.wav"  # Il file che hai già usato
    ]
    
    for i, audio_file in enumerate(test_files, 1):
        call_id = f"TIM_DEMO_{i:03d}"
        
        if os.path.exists(audio_file):
            print(f"\n🎯 Processing {call_id}: {audio_file}")
            
            try:
                summary = await analytics.process_full_call(call_id, audio_file)
                print(f"✅ {call_id} completato - Score: {summary.get('overall_call_score', 0):.1f}/100")
                
            except Exception as e:
                print(f"❌ Errore: {e}")
        else:
            print(f"⚠️ File non trovato: {audio_file}")
    
async def demo_synthetic_audio():
    """Demo con audio sintetico se non ci sono file"""
    
    print("🎵 Generazione audio sintetico per demo...")
    
    # Crea audio sintetico che simula una chiamata
    sr = 8000  # Sample rate telefonico
    duration = 20  # 20 secondi
    t = np.linspace(0, duration, int(sr * duration))
    
    # Simula pattern chiamata realistica
    audio = np.zeros_like(t)
    
    # 0-3s: Silenzio iniziale
    # 3-8s: Operatore parla (tono femminile ~220Hz)
    agent_section = np.sin(2 * np.pi * 220 * t[int(3*sr):int(8*sr)]) * 0.1
    agent_section += np.random.normal(0, 0.02, len(agent_section))  # Rumore
    audio[int(3*sr):int(8*sr)] = agent_section
    
    # 8-10s: Silenzio
    # 10-18s: Cliente parla (tono maschile ~150Hz)
    customer_section = np.sin(2 * np.pi * 150 * t[int(10*sr):int(18*sr)]) * 0.15
    customer_section += np.random.normal(0, 0.03, len(customer_section))
    audio[int(10*sr):int(18*sr)] = customer_section
    
    # 18-20s: Saluti (operatore)
    goodbye_section = np.sin(2 * np.pi * 210 * t[int(18*sr):]) * 0.08
    goodbye_section += np.random.normal(0, 0.01, len(goodbye_section))
    audio[int(18*sr):] = goodbye_section
    
    # Salva audio temporaneo
    import soundfile as sf
    temp_file = "temp_demo_call.wav"
    sf.write(temp_file, audio, sr)
    
    # Analizza con il sistema
    analytics = UnifiedCallAnalytics()
    call_id = "TIM_SYNTHETIC_DEMO"
    
    print(f"📞 Analizzando chiamata sintetica: {call_id}")
    
    try:
        summary = await analytics.process_full_call(call_id, temp_file)
        
        print(f"\n🎉 DEMO SINTETICO COMPLETATO!")
        print(f"Score: {summary.get('overall_call_score', 0):.1f}/100")
        print(f"Il sistema ha identificato:")
        print(f"- Talk efficiency: {summary.get('talk_efficiency', 0):.1f}%")
        print(f"- Speaker predominante: {summary.get('predominant_gender', 'unknown')}")
        print(f"- Qualità media: {summary.get('audio_quality_avg', 0):.1f}/100")
        
    except Exception as e:
        print(f"❌ Errore demo sintetico: {e}")
    
    finally:
        # Cleanup
        if os.path.exists(temp_file):
            os.remove(temp_file)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("🚀 AVVIO SISTEMA TIM CALL ANALYTICS")
    print("📁 Cercando file audio in data/...")
    
    # Verifica se esistono file di test
    test_files = []
    if os.path.exists("data"):
        for ext in ["*.wav", "*.mp3", "*.flac"]:
            test_files.extend(glob.glob(f"data/{ext}"))
    
    if test_files:
        print(f"✅ Trovati {len(test_files)} file audio per test")
        asyncio.run(demo_system())
    else:
        print("⚠️ Nessun file audio trovato in data/")
        print("\n💡 Per testare il sistema:")
        print("1. Crea cartella 'data/'")
        print("2. Aggiungi file audio (.wav, .mp3, .flac)")
        print("3. Riavvia il programma")
        
        # Demo con audio sintetico
        print("\n🎯 Avvio demo con audio sintetico...")
        asyncio.run(demo_synthetic_audio())
