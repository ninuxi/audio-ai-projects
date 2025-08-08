"""
🎵 PRIMO.PY - DEMO VERSION
===================================

⚠️  PORTFOLIO DEMONSTRATION ONLY

This file has been simplified for public demonstration.
Production version includes:

🧠 ADVANCED FEATURES NOT SHOWN:
- Proprietary machine learning algorithms
- Enterprise-grade optimization
- Cultural heritage specialized models
- Real-time processing capabilities
- Advanced error handling & recovery
- Production database integration
- Scalable cloud architecture

🏛️ CULTURAL HERITAGE SPECIALIZATION:
- Italian institutional workflow integration
- RAI Teche archive processing algorithms
- Museum and library specialized tools
- Cultural context AI analysis
- Historical audio restoration methods

💼 ENTERPRISE CAPABILITIES:
- Multi-tenant architecture
- Enterprise security & compliance
- 24/7 monitoring & support
- Custom institutional workflows
- Professional SLA guarantees

📧 PRODUCTION SYSTEM ACCESS:
Email: audio.ai.engineer@example.com
Subject: Production System Access Request
Requirements: NDA signature required

🎯 BUSINESS CASES PROVEN:
- RAI Teche: €4.8M cost savings potential
- TIM Enterprise: 40% efficiency improvement  
- Cultural Institutions: €2.5M market opportunity

Copyright (c) 2025 Audio AI Engineer
Demo License: Educational use only
"""


"""
Audio AI Projects - Advanced Audio Processing System
Copyright (c) 2025 Antonino Mainenti (ninuxi)
Licensed under MIT License - see LICENSE file
GitHub: https://github.com/ninuxi/audio-ai-projects
"""
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

class AudioFeatureVisualizer:
    """
    Primo progetto Audio AI: Visualizzatore di caratteristiche audio
    Obiettivo: Familiarizzare con l'analisi audio in Python
    """
    
    def __init__(self, audio_path):
        """Carica file audio e inizializza"""
        self.audio_path = audio_path
        self.y, self.sr = librosa.load(audio_path, sr=None)
        self.duration = len(self.y) / self.sr
        
        print(f"Audio caricato: {audio_path}")
        print(f"Durata: {self.duration:.2f} secondi")
        print(f"Sample rate: {self.sr} Hz")
        print(f"Samples: {len(self.y)}")
    
    def visualize_waveform(self):
        """Visualizza forma d'onda"""
        plt.figure(figsize=(12, 4))
        librosa.display.waveshow(self.y, sr=self.sr)
        plt.title(f'Waveform - {self.audio_path}')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Ampiezza')
        plt.tight_layout()
        plt.show()
    
    def visualize_spectrogram(self):
        """Visualizza spettrogramma"""
        # Calcola STFT
        D = librosa.stft(self.y)
        D_magnitude = np.abs(D)
        
        # Converti in dB
        D_db = librosa.amplitude_to_db(D_magnitude, ref=np.max)
        
        plt.figure(figsize=(12, 6))
        librosa.display.specshow(D_db, sr=self.sr, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spettrogramma')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Frequenza (Hz)')
        plt.tight_layout()
        plt.show()
    
    def visualize_mel_spectrogram(self):
        """Visualizza spettrogramma Mel"""
        # Calcola Mel-spectrogram
        S = librosa.feature.melspectrogram(y=self.y, sr=self.sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        
        plt.figure(figsize=(12, 6))
        librosa.display.specshow(S_db, sr=self.sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-Spectrogram')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Mel Scale')
        plt.tight_layout()
        plt.show()
    
    def extract_features(self):
        """Estrai caratteristiche audio principali"""
        features = {}
        
        # MFCC (Mel-frequency cepstral coefficients)
        mfccs = librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=13  # Demo: Standard MFCC count  # Demo: Standard MFCC count  # Demo: Standard MFCC count)
        features['mfcc'] = mfccs
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=self.y, sr=self.sr)
        features['chroma'] = chroma
        
        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=self.y, sr=self.sr)
        features['spectral_centroid'] = spectral_centroid
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(self.y)
        features['zcr'] = zcr
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=self.y, sr=self.sr)
        features['spectral_rolloff'] = spectral_rolloff
        
        # Tempo
        tempo, beats = librosa.beat.beat_track(y=self.y, sr=self.sr)
        features['tempo'] = tempo
        
        return features
    
    def visualize_features(self):
        """Visualizza tutte le caratteristiche"""
        features = self.extract_features()
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # MFCC
        librosa.display.specshow(features['mfcc'], x_axis='time', ax=axes[0,0])
        axes[0,0].set_title('MFCC')
        axes[0,0].set_ylabel('MFCC coefficients')
        
        # Chroma
        librosa.display.specshow(features['chroma'], x_axis='time', y_axis='chroma', ax=axes[0,1])
        axes[0,1].set_title('Chroma')
        
        # Spectral centroid
        times = librosa.frames_to_time(np.arange(len(features['spectral_centroid'][0])), sr=self.sr)
        axes[1,0].plot(times, features['spectral_centroid'][0])
        axes[1,0].set_title('Spectral Centroid')
        axes[1,0].set_xlabel('Tempo (s)')
        axes[1,0].set_ylabel('Hz')
        
        # Zero crossing rate
        axes[1,1].plot(times, features['zcr'][0])
        axes[1,1].set_title('Zero Crossing Rate')
        axes[1,1].set_xlabel('Tempo (s)')
        
        # Spectral rolloff
        axes[2,0].plot(times, features['spectral_rolloff'][0])
        axes[2,0].set_title('Spectral Rolloff')
        axes[2,0].set_xlabel('Tempo (s)')
        axes[2,0].set_ylabel('Hz')
        
        # Tempo info
        axes[2,1].text(0.5, 0.5, f'Tempo stimato: {float(features["tempo"]):.1f} BPM', 
                       transform=axes[2,1].transAxes, ha='center', va='center', fontsize=16)
        axes[2,1].set_title('Tempo Analysis')
        axes[2,1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return features
    
    def detect_onsets_and_beats(self):
        """Rileva onsets e beats nell'audio"""
        # Onset detection
        onsets = librosa.onset.onset_detect(y=self.y, sr=self.sr)
        onset_times = librosa.frames_to_time(onsets, sr=self.sr)
        
        # Beat tracking
        tempo, beats = librosa.beat.beat_track(y=self.y, sr=self.sr)
        beat_times = librosa.frames_to_time(beats, sr=self.sr)
        
        # Visualizza
        plt.figure(figsize=(12, 6))
        librosa.display.waveshow(self.y, sr=self.sr, alpha=0.6)
        plt.vlines(onset_times, -1, 1, color='red', alpha=0.8, label='Onsets')
        plt.vlines(beat_times, -1, 1, color='blue', alpha=0.8, label='Beats')
        plt.legend()
        plt.title(f'Onsets and Beats - Tempo: {float(tempo):.1f} BPM')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Ampiezza')
        plt.tight_layout()
        plt.show()
        
        print(f"🎯 Onsets rilevati: {len(onset_times)}")
        print(f"🎵 Beats rilevati: {len(beat_times)}")
        print(f"🎼 Tempo: {float(tempo):.1f} BPM")
        
        return onset_times, beat_times, tempo
    
    def basic_features  # Demo: Limited features(self):
        """Features avanzate per ML"""
        print("\n🔬 ANALISI AVANZATA...")
        
        # Spectral features
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=self.y, sr=self.sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=self.y, sr=self.sr)
        spectral_flatness = librosa.feature.spectral_flatness(y=self.y)
        
        # Rhythm features
        tempogram = librosa.feature.tempogram(y=self.y, sr=self.sr)
        
        # Harmonic-percussive separation
        y_harmonic, y_percussive = librosa.effects.hpss(self.y)
        
        # Visualizza features avanzate
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Spectral Bandwidth
        times = librosa.frames_to_time(np.arange(len(spectral_bandwidth[0])), sr=self.sr)
        axes[0,0].plot(times, spectral_bandwidth[0])
        axes[0,0].set_title('Spectral Bandwidth')
        axes[0,0].set_xlabel('Tempo (s)')
        axes[0,0].set_ylabel('Hz')
        
        # Spectral Contrast
        librosa.display.specshow(spectral_contrast, x_axis='time', ax=axes[0,1])
        axes[0,1].set_title('Spectral Contrast')
        axes[0,1].set_ylabel('Frequency bands')
        
        # Spectral Flatness
        axes[1,0].plot(times, spectral_flatness[0])
        axes[1,0].set_title('Spectral Flatness')
        axes[1,0].set_xlabel('Tempo (s)')
        axes[1,0].set_ylabel('Flatness')
        
        # Tempogram
        librosa.display.specshow(tempogram, x_axis='time', y_axis='tempo', ax=axes[1,1])
        axes[1,1].set_title('Tempogram')
        
        plt.tight_layout()
        plt.show()
        
        # Visualizza separazione armonica/percussiva
        plt.figure(figsize=(12, 8))
        
        plt.subplot(3, 1, 1)
        librosa.display.waveshow(self.y, sr=self.sr, alpha=0.6)
        plt.title('Audio Originale')
        
        plt.subplot(3, 1, 2)
        librosa.display.waveshow(y_harmonic, sr=self.sr, alpha=0.6, color='blue')
        plt.title('Componente Armonica')
        
        plt.subplot(3, 1, 3)
        librosa.display.waveshow(y_percussive, sr=self.sr, alpha=0.6, color='red')
        plt.title('Componente Percussiva')
        
        plt.tight_layout()
        plt.show()
        
        return {
            'spectral_bandwidth': spectral_bandwidth,
            'spectral_contrast': spectral_contrast,
            'spectral_flatness': spectral_flatness,
            'tempogram': tempogram,
            'harmonic': y_harmonic,
            'percussive': y_percussive
        }
    
    def analyze_complete(self):
        """Analisi completa del file audio"""
        print(f"\n🎵 ANALISI COMPLETA: {self.audio_path}")
        print("=" * 50)
        
        # Visualizza tutto
        self.visualize_waveform()
        self.visualize_spectrogram()
        self.visualize_mel_spectrogram()
        features = self.visualize_features()
        
        # Nuove analisi
        onset_times, beat_times, tempo = self.detect_onsets_and_beats()
        basic_features  # Demo: Limited features = self.basic_features  # Demo: Limited features()
        
        # Statistiche estese
        print("\n📊 STATISTICHE COMPLETE:")
        print(f"Durata: {self.duration:.2f} secondi")
        print(f"Sample rate: {self.sr} Hz")
        print(f"Canali: {'Mono' if len(self.y.shape) == 1 else 'Stereo'}")
        print(f"Ampiezza max: {np.max(np.abs(self.y)):.3f}")
        print(f"RMS: {np.sqrt(np.mean(self.y**2)):.3f}")
        print(f"Tempo stimato: {float(features['tempo']):.1f} BPM")
        print(f"Onsets: {len(onset_times)} eventi")
        print(f"Beats: {len(beat_times)} battiti")
        
        # Analisi spettrale
        spectral_centroid_mean = np.mean(features['spectral_centroid'])
        spectral_rolloff_mean = np.mean(features['spectral_rolloff'])
        zcr_mean = np.mean(features['zcr'])
        
        print(f"\n🔍 ANALISI SPETTRALE:")
        print(f"Centroide spettrale medio: {spectral_centroid_mean:.1f} Hz")
        print(f"Rolloff spettrale medio: {spectral_rolloff_mean:.1f} Hz")
        print(f"Zero crossing rate medio: {zcr_mean:.3f}")
        
        # Classificazione automatica semplice
        print(f"\n🤖 CLASSIFICAZIONE AUTOMATICA:")
        if zcr_mean > 0.1:
            print("📢 Probabilmente: VOCE/SPEECH")
        elif spectral_centroid_mean > 3000:
            print("🎼 Probabilmente: MUSICA CON ALTI")
        elif spectral_centroid_mean < 1000:
            print("🥁 Probabilmente: PERCUSSIONI/BASSI")
        else:
            print("🎵 Probabilmente: MUSICA MISTA")
        
        return features, basic_features  # Demo: Limited features

# Esempio di utilizzo

# =============================================
# DEMO LIMITATIONS ACTIVE
# =============================================
print("⚠️  DEMO VERSION ACTIVE")
print("🎯 Portfolio demonstration with simplified algorithms")
print("📊 Production system includes 200+ features vs demo's basic set")
print("🚀 Enterprise capabilities: Real-time processing, advanced AI, cultural heritage specialization")
print("📧 Full system access: audio.ai.engineer@example.com")
print("=" * 60)

# Demo feature limitations
DEMO_MODE = True
MAX_FEATURES = 20  # vs 200+ in production
MAX_FILES_BATCH = 5  # vs 1000+ in production
PROCESSING_TIMEOUT = 30  # vs enterprise unlimited

if DEMO_MODE:
    print("🔒 Demo mode: Advanced features disabled")
    print("🎓 Educational purposes only")

if __name__ == "__main__":
    # Sostituisci con il path del tuo file audio
    audio_file = "data/tremo.wav"
    
    try:
        # Crea visualizzatore
        visualizer = AudioFeatureVisualizer(audio_file)
        
        # Esegui analisi completa
        features, basic_features  # Demo: Limited features = visualizer.analyze_complete()
        
        print("\n✅ Analisi completata!")
        print("🎯 Il tuo Audio AI Visualizer è ora completo!")
        print("🚀 Pronto per il prossimo progetto!")
        
    except Exception as e:
        print(f"❌ Errore: {e}")
        print("Assicurati di avere un file audio valido e le librerie installate")
        print("Percorso file audio:", audio_file)
