# 📚 Week 1: Audio Feature Visualizer - Technical Documentation

## 🎯 Overview

The Audio Feature Visualizer is the foundation project that introduces core audio processing concepts through hands-on visualization. This documentation covers the technical implementation, feature extraction methods, and interpretation guidelines.

## 🏗️ Architecture

```
Audio Input → LibROSA Processing → Feature Extraction → Visualization → Analysis
```

### Core Components

1. **Audio Loader** - Handles multiple audio formats
2. **Feature Extractor** - Computes audio characteristics
3. **Visualizer** - Creates professional plots
4. **Classifier** - Basic content type detection

## 🔬 Feature Extraction Methods

### 1. Temporal Features

#### RMS Energy
```python
rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)
```
- **Purpose**: Measures signal power over time
- **Range**: 0.0 to 1.0 (normalized)
- **Interpretation**: Higher values = louder audio

#### Zero Crossing Rate (ZCR)
```python
zcr = librosa.feature.zero_crossing_rate(audio, frame_length=2048, hop_length=512)
```
- **Purpose**: Counts zero crossings per frame
- **Range**: 0.0 to 1.0
- **Interpretation**: 
  - High ZCR (>0.1) → Speech, noise
  - Low ZCR (<0.05) → Tonal music, sustained sounds

### 2. Spectral Features

#### Spectral Centroid
```python
centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
```
- **Purpose**: "Brightness" of sound (weighted frequency average)
- **Range**: 0 to Nyquist frequency (sr/2)
- **Interpretation**:
  - High centroid → Bright, high-frequency content
  - Low centroid → Dark, bass-heavy content

#### Spectral Rolloff
```python
rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85)
```
- **Purpose**: Frequency below which 85% of energy lies
- **Applications**: Music/speech discrimination

#### Spectral Bandwidth
```python
bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
```
- **Purpose**: Width of frequency distribution
- **Interpretation**: Narrow bandwidth = tonal, Wide bandwidth = noisy

### 3. Perceptual Features

#### MFCC (Mel-Frequency Cepstral Coefficients)
```python
mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=2048)
```
- **Purpose**: Compact representation of spectral envelope
- **Parameters**:
  - `n_mfcc=13`: Standard for speech recognition
  - `n_fft=2048`: Window size for frequency analysis
- **Coefficients**:
  - MFCC 0: Overall energy (often excluded)
  - MFCC 1-2: Formant information (crucial for speech)
  - MFCC 3-12: Fine spectral details

#### Chroma Features
```python
chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_chroma=12)
```
- **Purpose**: Pitch class profiles (C, C#, D, D#, E, F, F#, G, G#, A, A#, B)
- **Applications**: Harmony analysis, key detection

### 4. Rhythm Features

#### Tempo Detection
```python
tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
```
- **Output**: 
  - `tempo`: BPM (beats per minute)
  - `beats`: Frame indices of detected beats

#### Onset Detection
```python
onsets = librosa.onset.onset_detect(y=audio, sr=sr, units='time')
```
- **Purpose**: Detect note/event beginnings
- **Applications**: Rhythm analysis, segmentation

## 📊 Visualization Techniques

### 1. Waveform Plot
```python
librosa.display.waveshow(audio, sr=sr, alpha=0.6)
```
- **Shows**: Amplitude variations over time
- **Best for**: Overview of signal structure, clipping detection

### 2. Spectrogram
```python
D = librosa.stft(audio)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(D)), 
                        sr=sr, x_axis='time', y_axis='hz')
```
- **Shows**: Frequency content evolution
- **Color**: Intensity represents magnitude

### 3. Mel-Spectrogram
```python
S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
librosa.display.specshow(librosa.power_to_db(S), 
                        sr=sr, x_axis='time', y_axis='mel')
```
- **Shows**: Perceptually-relevant frequency analysis
- **Advantages**: Better for human auditory perception

### 4. Feature Evolution Plots
```python
times = librosa.frames_to_time(range(len(feature)), sr=sr, hop_length=512)
plt.plot(times, feature)
```
- **Shows**: How features change over time
- **Applications**: Pattern recognition, anomaly detection

## 🎯 Classification Rules

### Content Type Detection

#### Speech Detection
```python
def is_speech(zcr_mean, spectral_centroid_mean, energy):
    return (zcr_mean > 0.1 and 
            2000 < spectral_centroid_mean < 6000 and 
            energy > 0.01)
```

#### Music Detection
```python
def is_music(spectral_centroid_mean, harmonic_content, rhythm_regularity):
    return (spectral_centroid_mean < 3000 and 
            harmonic_content > 0.3 and 
            rhythm_regularity > 0.5)
```

#### Silence Detection
```python
def is_silence(energy, zcr_mean):
    return energy < 0.001 or zcr_mean < 0.01
```

## ⚡ Performance Optimization

### Memory Management
```python
# Load only necessary duration
y, sr = librosa.load(audio_file, duration=30)  # First 30 seconds

# Use lower sample rate for faster processing
y, sr = librosa.load(audio_file, sr=16000)  # Instead of 22050
```

### Computational Efficiency
```python
# Larger hop length = faster processing
hop_length = 1024  # Instead of 512

# Fewer MFCC coefficients
n_mfcc = 8  # Instead of 13
```

## 🔧 Troubleshooting

### Common Issues

1. **File Loading Errors**
   ```python
   # Solution: Check file format and path
   try:
       y, sr = librosa.load(file_path)
   except Exception as e:
       print(f"Error loading {file_path}: {e}")
   ```

2. **Memory Issues**
   ```python
   # Solution: Process in chunks
   chunk_duration = 30  # seconds
   y, sr = librosa.load(file_path, duration=chunk_duration)
   ```

3. **Empty/Silent Audio**
   ```python
   # Solution: Check for minimum energy
   if np.max(np.abs(y)) < 0.001:
       print("Warning: Very quiet or silent audio")
   ```

## 📐 Parameter Guidelines

### Frame Analysis Parameters
- **frame_length**: 2048 (good balance), 4096 (high resolution), 1024 (fast)
- **hop_length**: frame_length/4 (detailed), frame_length/2 (standard)
- **window**: 'hann' (default, smooth), 'hamming' (sharper)

### Feature-Specific Settings
- **MFCC**: n_mfcc=13 (speech), n_mfcc=20 (music analysis)
- **Mel-spectrogram**: n_mels=128 (standard), n_mels=64 (faster)
- **Chroma**: n_chroma=12 (standard), n_chroma=24 (microtonal)

## 🎓 Educational Notes

### For Audio Engineering Students
- Focus on understanding spectral vs temporal features
- Practice interpreting spectrograms visually
- Learn correlation between features and perceptual qualities

### For AI/ML Students  
- MFCC are standard input features for audio ML models
- Feature normalization is crucial for ML applications
- Consider feature delta (derivatives) for dynamic information

### For Music Technology Students
- Chroma features connect to music theory concepts
- Tempo detection enables rhythm-based applications
- Harmonic/percussive separation useful for remixing

## 🔗 Further Reading

- [LibROSA Documentation](https://librosa.org/doc/latest/)
- [Audio Signal Processing Fundamentals](https://ccrma.stanford.edu/~jos/filters/)
- [Music Information Retrieval](https://musicinformationretrieval.com/)

---
**📝 This documentation covers the technical foundation needed to understand and extend the audio feature visualization system.**
