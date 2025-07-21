import librosa
import soundfile as sf
import numpy as np
from scipy.signal import wiener

class NoiseReducer:
    """Reduce background noise from audio"""
    
    def __init__(self):
        self.noise_threshold = 0.1
    
    def reduce_noise(self, audio_path: str, output_path: str) -> bool:
        """
        Apply noise reduction to audio file
        
        Args:
            audio_path: Input audio file path
            output_path: Output cleaned audio path
            
        Returns:
            bool: Success status
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=None)
            
            # Estimate noise from first 0.5 seconds
            noise_sample = audio[:int(0.5 * sr)]
            noise_level = np.std(noise_sample)
            
            # Apply Wiener filter for noise reduction
            if noise_level > self.noise_threshold:
                filtered_audio = wiener(audio, mysize=5)
            else:
                filtered_audio = audio
            
            # Save cleaned audio
            sf.write(output_path, filtered_audio, sr)
            return True
            
        except Exception as e:
            print(f"Noise reduction error for {audio_path}: {e}")
            return False
