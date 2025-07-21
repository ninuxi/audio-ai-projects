# examples/sample_analysis.py
"""
🎵 Sample Audio Analysis Examples
================================

Demonstrate audio feature extraction and visualization
with different types of audio content.
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

def generate_sample_audio():
    """Generate synthetic audio samples for demonstration"""
    
    sr = 22050
    duration = 5  # seconds
    t = np.linspace(0, duration, sr * duration)
    
    samples = {}
    
    # 1. Pure tone (musical note A4)
    samples['pure_tone'] = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    # 2. Complex musical chord (C major)
    chord_freqs = [261.63, 329.63, 392.00]  # C, E, G
    chord = sum(0.3 * np.sin(2 * np.pi * freq * t) for freq in chord_freqs)
    samples['musical_chord'] = chord
    
    # 3. Speech-like signal (formant synthesis)
    f1, f2 = 730, 1090  # Vowel /a/ formants
    speech = (0.3 * np.sin(2 * np.pi * f1 * t) + 
              0.2 * np.sin(2 * np.pi * f2 * t) +
              0.1 * np.random.randn(len(t)))
    samples['speech_like'] = speech
    
    # 4. Noise (white noise)
    samples['white_noise'] = 0.1 * np.random.randn(len(t))
    
    return samples, sr

def analyze_sample(audio, sr, title):
    """Analyze single audio sample"""
    
    print(f"\n🔍 Analyzing: {title}")
    print("-" * 30)
    
    # Basic features
    duration = len(audio) / sr
    rms = np.sqrt(np.mean(audio**2))
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio)[0])
    
    # Spectral features
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    
    # MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=5)
    
    print(f"Duration: {duration:.2f}s")
    print(f"RMS Energy: {rms:.4f}")
    print(f"Zero Crossing Rate: {zcr:.4f}")
    print(f"Spectral Centroid: {spectral_centroid:.1f} Hz")
    print(f"MFCC 1-5 (mean): {np.mean(mfcc, axis=1)}")
    
    return {
        'duration': duration,
        'rms': rms,
        'zcr': zcr,
        'spectral_centroid': spectral_centroid,
        'mfcc': mfcc
    }

def create_comparison_visualization():
    """Create visualization comparing different audio types"""
    
    samples, sr = generate_sample_audio()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Audio Feature Analysis Comparison', fontsize=16)
    
    sample_names = list(samples.keys())
    
    for i, (name, audio) in enumerate(samples.items()):
        if i >= 4:  # Only plot first 4
            break
            
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        # Waveform
        time = np.linspace(0, len(audio)/sr, len(audio))
        ax.plot(time[:1000], audio[:1000])  # Plot first second
        ax.set_title(f'{name.replace("_", " ").title()}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        
        # Analyze and print results
        analyze_sample(audio, sr, name)
    
    plt.tight_layout()
    plt.savefig('examples/audio_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n💾 Visualization saved: examples/audio_comparison.png")
    
    return samples, sr

if __name__ == "__main__":
    print("🎵 AUDIO ANALYSIS EXAMPLES")
    print("=" * 40)
    
    # Generate and analyze samples
    samples, sr = create_comparison_visualization()
    
    print(f"\n🎯 COMPARISON SUMMARY:")
    print("- Pure tone: Clean sinusoid, low ZCR")
    print("- Musical chord: Complex harmonics, mid ZCR") 
    print("- Speech-like: Formant structure, high ZCR")
    print("- White noise: Random, very high ZCR")
    
    print(f"\n📊 Use these examples to understand how different")
    print("audio types produce different feature patterns!")
