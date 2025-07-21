import librosa
import soundfile as sf
import numpy as np
from typing import Tuple, Optional

class AudioNormalizer:
    """Normalize audio levels for consistent processing"""
    
    def __init__(self, target_lufs: float = -23.0):
        self.target_lufs = target_lufs
    
    def normalize_audio(self, audio_path: str, output_path: str) -> bool:
        """
        Normalize audio to target LUFS level
        
        Args:
            audio_path: Input audio file path
            output_path: Output normalized audio path
            
        Returns:
            bool: Success status
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=None)
            
            # Calculate RMS and normalize
            rms = np.sqrt(np.mean(audio**2))
            if rms > 0:
                target_rms = 10**(self.target_lufs/20)
                normalized_audio = audio * (target_rms / rms)
            else:
                normalized_audio = audio
            
            # Prevent clipping
            if np.max(np.abs(normalized_audio)) > 1.0:
                normalized_audio = normalized_audio / np.max(np.abs(normalized_audio))
            
            # Save normalized audio
            sf.write(output_path, normalized_audio, sr)
            return True
            
        except Exception as e:
            print(f"Normalization error for {audio_path}: {e}")
            return False
