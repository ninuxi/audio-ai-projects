"""
Audio AI Projects - Advanced Audio Processing System
Copyright (c) 2025 Antonino Mainenti (ninuxi)
Licensed under MIT License - see LICENSE file
GitHub: https://github.com/ninuxi/audio-ai-projects
"""
import librosa
import soundfile as sf
from pathlib import Path
import subprocess

class FormatConverter:
    """Convert audio between different formats"""
    
    def __init__(self):
        self.supported_formats = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    
    def convert_format(self, input_path: str, output_path: str, 
                      target_format: str = 'wav') -> bool:
        """
        Convert audio file to target format
        
        Args:
            input_path: Input audio file path
            output_path: Output audio file path
            target_format: Target audio format
            
        Returns:
            bool: Success status
        """
        try:
            # Load and resave in target format
            audio, sr = librosa.load(input_path, sr=None)
            
            # Set output path with correct extension
            output_file = Path(output_path).with_suffix(f'.{target_format}')
            
            # Save in target format
            sf.write(str(output_file), audio, sr, format=target_format.upper())
            return True
            
        except Exception as e:
            print(f"Format conversion error: {e}")
            return False
    
    def batch_convert(self, input_dir: str, output_dir: str, 
                     target_format: str = 'wav') -> int:
        """
        Convert all audio files in directory
        
        Returns:
            int: Number of successfully converted files
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        converted_count = 0
        
        for audio_file in input_path.rglob('*'):
            if audio_file.suffix.lower() in self.supported_formats:
                output_file = output_path / f"{audio_file.stem}.{target_format}"
                
                if self.convert_format(str(audio_file), str(output_file), target_format):
                    converted_count += 1
        
        return converted_count
