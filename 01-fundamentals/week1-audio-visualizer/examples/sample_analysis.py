"""
Audio AI Projects - Advanced Audio Processing System
Copyright (c) 2025 Antonino Mainenti (ninuxi)
Licensed under MIT License - see LICENSE file
GitHub: https://github.com/ninuxi/audio-ai-projects
"""
# examples/sample_analysis.py
"""
🎵 Sample Audio Analysis Examples - FIXED VERSION
================================================

Demonstrate audio feature extraction and visualization
with different types of audio content.
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os

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
    
    print("🎵 Generating synthetic audio samples...")
    samples, sr = generate_sample_audio()
    
    print("📊 Creating visualization...")
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
    
    # CORREZIONE: Assicurati che la directory examples esista
    print("📁 Creating examples directory...")
    os.makedirs('examples', exist_ok=True)
    
    # Salva il grafico
    output_path = 'examples/audio_comparison.png'
    print(f"💾 Saving visualization to: {output_path}")
    
    try:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualization saved successfully!")
    except Exception as e:
        print(f"❌ Error saving visualization: {e}")
        print("💡 Trying alternative path...")
        # Fallback: salva nella directory corrente
        fallback_path = 'audio_comparison.png'
        plt.savefig(fallback_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved to current directory: {fallback_path}")
    
    # Mostra il grafico
    print("🖼️ Displaying visualization...")
    plt.show()
    
    return samples, sr

def save_sample_audio_files():
    """Save sample audio as WAV files for testing"""
    
    print("\n🎵 Generating sample audio files...")
    samples, sr = generate_sample_audio()
    
    # Assicurati che la directory esista
    os.makedirs('examples', exist_ok=True)
    
    try:
        import soundfile as sf
        
        for name, audio in samples.items():
            filename = f'examples/sample_{name}.wav'
            sf.write(filename, audio, sr)
            print(f"✅ Saved: {filename}")
            
    except ImportError:
        print("⚠️ soundfile not available - using scipy.io.wavfile")
        try:
            from scipy.io import wavfile
            
            for name, audio in samples.items():
                filename = f'examples/sample_{name}.wav'
                # Normalize to 16-bit range
                audio_int16 = (audio * 32767).astype(np.int16)
                wavfile.write(filename, sr, audio_int16)
                print(f"✅ Saved: {filename}")
                
        except ImportError:
            print("❌ Cannot save audio files - install soundfile or scipy")

def run_complete_demo():
    """Run complete demonstration"""
    
    print("🎵 AUDIO ANALYSIS EXAMPLES - COMPLETE DEMO")
    print("=" * 50)
    
    # Check current directory
    print(f"📍 Current directory: {os.getcwd()}")
    print(f"📁 Directory contents: {os.listdir('.')}")
    
    try:
        # Generate and analyze samples
        samples, sr = create_comparison_visualization()
        
        # Save sample audio files
        save_sample_audio_files()
        
        print(f"\n🎯 COMPARISON SUMMARY:")
        print("- Pure tone: Clean sinusoid, low ZCR")
        print("- Musical chord: Complex harmonics, mid ZCR") 
        print("- Speech-like: Formant structure, high ZCR")
        print("- White noise: Random, very high ZCR")
        
        print(f"\n📊 Use these examples to understand how different")
        print("audio types produce different feature patterns!")
        
        print(f"\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("💡 Make sure you have librosa and matplotlib installed:")
        print("   pip install librosa matplotlib")

if __name__ == "__main__":
    run_complete_demo()
