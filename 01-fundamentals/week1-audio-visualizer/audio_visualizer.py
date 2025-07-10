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
        mfccs = librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=13)
        features['mfcc'] = mfccs
        
        # Chroma features
        chroma = librosa.feature.chroma(y=self.y, sr=self.sr)
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
        axes[2,1].text(0.5, 0.5, f'Tempo stimato: {features["tempo"]:.1f} BPM', 
                       transform=axes[2,1].transAxes, ha='center', va='center', fontsize=16)
        axes[2,1].set_title('Tempo Analysis')
        axes[2,1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return features
    
    def analyze_complete(self):
        """Analisi completa del file audio"""
        print(f"\n🎵 ANALISI COMPLETA: {self.audio_path}")
        print("=" * 50)
        
        # Visualizza tutto
        self.visualize_waveform()
        self.visualize_spectrogram()
        self.visualize_mel_spectrogram()
        features = self.visualize_features()
        
        # Statistiche
        print("\n📊 STATISTICHE:")
        print(f"Durata: {self.duration:.2f} secondi")
        print(f"Sample rate: {self.sr} Hz")
        print(f"Canali: {'Mono' if len(self.y.shape) == 1 else 'Stereo'}")
        print(f"Ampiezza max: {np.max(np.abs(self.y)):.3f}")
        print(f"RMS: {np.sqrt(np.mean(self.y**2)):.3f}")
        print(f"Tempo stimato: {features['tempo']:.1f} BPM")
        
        return features

# Esempio di utilizzo
if __name__ == "__main__":
    # Sostituisci con il path del tuo file audio
    audio_file = "data/tremo.wav"
    
    try:
        # Crea visualizzatore
        visualizer = AudioFeatureVisualizer(audio_file)
        
        # Esegui analisi completa
        features = visualizer.analyze_complete()
        
        print("\n✅ Analisi completata!")
        print("Questo è il tuo primo progetto Audio AI!")
        
    except Exception as e:
        print(f"❌ Errore: {e}")
        print("Assicurati di avere un file audio valido e le librerie installate")
