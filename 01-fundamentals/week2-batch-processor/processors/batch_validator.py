import librosa
from pathlib import Path
import json
from typing import Dict, List, Any
import hashlib

class BatchValidator:
    """Validate audio files and batch processing results"""
    
    def __init__(self):
        self.min_duration = 0.1  # Minimum 0.1 seconds
        self.max_duration = 3600  # Maximum 1 hour
        self.supported_formats = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    
    def validate_audio_file(self, audio_path: str) -> Dict[str, Any]:
        """
        Validate individual audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dict with validation results
        """
        validation = {
            'file_path': audio_path,
            'is_valid': False,
            'errors': [],
            'warnings': [],
            'file_hash': None
        }
        
        try:
            file_path = Path(audio_path)
            
            # Check if file exists
            if not file_path.exists():
                validation['errors'].append('File does not exist')
                return validation
            
            # Check file extension
            if file_path.suffix.lower() not in self.supported_formats:
                validation['errors'].append(f'Unsupported format: {file_path.suffix}')
            
            # Try to load audio
            try:
                audio, sr = librosa.load(audio_path, sr=None)
                duration = len(audio) / sr
                
                # Duration validation
                if duration < self.min_duration:
                    validation['errors'].append(f'Duration too short: {duration:.2f}s')
                elif duration > self.max_duration:
                    validation['warnings'].append(f'Duration very long: {duration:.2f}s')
                
                # Audio quality checks
                if len(audio) == 0:
                    validation['errors'].append('Empty audio data')
                
                # Calculate file hash for integrity
                with open(audio_path, 'rb') as f:
                    validation['file_hash'] = hashlib.md5(f.read()).hexdigest()
                
                # If no errors, mark as valid
                if not validation['errors']:
                    validation['is_valid'] = True
                
            except Exception as e:
                validation['errors'].append(f'Audio loading error: {str(e)}')
        
        except Exception as e:
            validation['errors'].append(f'Validation error: {str(e)}')
        
        return validation
    
    def validate_batch(self, input_dir: str, report_file: str) -> Dict[str, Any]:
        """
        Validate all audio files in directory
        
        Args:
            input_dir: Directory containing audio files
            report_file: Path to save validation report
            
        Returns:
            Dict with batch validation summary
        """
        input_path = Path(input_dir)
        validation_results = []
        summary = {
            'total_files': 0,
            'valid_files': 0,
            'invalid_files': 0,
            'warnings': 0,
            'errors': []
        }
        
        # Validate each audio file
        for audio_file in input_path.rglob('*'):
            if audio_file.suffix.lower() in self.supported_formats:
                result = self.validate_audio_file(str(audio_file))
                validation_results.append(result)
                summary['total_files'] += 1
                
                if result['is_valid']:
                    summary['valid_files'] += 1
                else:
                    summary['invalid_files'] += 1
                    summary['errors'].extend(result['errors'])
                
                summary['warnings'] += len(result['warnings'])
        
        # Generate report
        report = {
            'summary': summary,
            'detailed_results': validation_results,
            'timestamp': str(Path().absolute())
        }
        
        # Save report
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return summary
