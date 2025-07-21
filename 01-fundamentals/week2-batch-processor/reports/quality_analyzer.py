import librosa
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Any

class QualityAnalyzer:
    """Analyze audio quality metrics across batch processing"""
    
    def __init__(self):
        self.quality_thresholds = {
            'min_snr': 10,  # dB
            'max_clipping': 0.01,  # 1% clipping threshold
            'min_dynamic_range': 20,  # dB
            'max_silence_ratio': 0.1  # 10% silence
        }
    
    def analyze_audio_quality(self, audio_path: str) -> Dict[str, Any]:
        """
        Analyze quality metrics for single audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dict with quality metrics
        """
        quality_metrics = {
            'file_path': audio_path,
            'snr': 0,
            'dynamic_range': 0,
            'clipping_ratio': 0,
            'silence_ratio': 0,
            'rms_level': 0,
            'peak_level': 0,
            'quality_score': 0,
            'issues': []
        }
        
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=None)
            
            # Signal-to-Noise Ratio estimation
            # Use quieter segments as noise estimation
            frame_length = int(0.1 * sr)  # 100ms frames
            frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=frame_length//2)
            frame_energy = np.sum(frames**2, axis=0)
            
            # Estimate noise from quietest 10% of frames
            noise_threshold = np.percentile(frame_energy, 10)
            signal_power = np.mean(frame_energy)
            
            if noise_threshold > 0:
                quality_metrics['snr'] = 10 * np.log10(signal_power / noise_threshold)
            
            # Dynamic Range
            rms_values = []
            for i in range(0, len(audio), frame_length):
                frame = audio[i:i+frame_length]
                if len(frame) > 0:
                    rms_values.append(np.sqrt(np.mean(frame**2)))
            
            if rms_values:
                max_rms = max(rms_values)
                min_rms = min([r for r in rms_values if r > 0]) if any(r > 0 for r in rms_values) else max_rms
                
                if min_rms > 0:
                    quality_metrics['dynamic_range'] = 20 * np.log10(max_rms / min_rms)
            
            # Clipping Detection
            peak_threshold = 0.99
            clipped_samples = np.sum(np.abs(audio) >= peak_threshold)
            quality_metrics['clipping_ratio'] = clipped_samples / len(audio)
            
            # Silence Ratio
            silence_threshold = 0.01
            silent_samples = np.sum(np.abs(audio) < silence_threshold)
            quality_metrics['silence_ratio'] = silent_samples / len(audio)
            
            # RMS and Peak Levels
            quality_metrics['rms_level'] = np.sqrt(np.mean(audio**2))
            quality_metrics['peak_level'] = np.max(np.abs(audio))
            
            # Quality Assessment
            issues = []
            if quality_metrics['snr'] < self.quality_thresholds['min_snr']:
                issues.append(f"Low SNR: {quality_metrics['snr']:.1f} dB")
            
            if quality_metrics['clipping_ratio'] > self.quality_thresholds['max_clipping']:
                issues.append(f"Clipping detected: {quality_metrics['clipping_ratio']*100:.1f}%")
            
            if quality_metrics['dynamic_range'] < self.quality_thresholds['min_dynamic_range']:
                issues.append(f"Low dynamic range: {quality_metrics['dynamic_range']:.1f} dB")
            
            if quality_metrics['silence_ratio'] > self.quality_thresholds['max_silence_ratio']:
                issues.append(f"Excessive silence: {quality_metrics['silence_ratio']*100:.1f}%")
            
            quality_metrics['issues'] = issues
            
            # Overall Quality Score (0-100)
            score = 100
            score -= max(0, (self.quality_thresholds['min_snr'] - quality_metrics['snr']) * 2)
            score -= quality_metrics['clipping_ratio'] * 100
            score -= max(0, (self.quality_thresholds['min_dynamic_range'] - quality_metrics['dynamic_range']))
            score -= quality_metrics['silence_ratio'] * 50
            
            quality_metrics['quality_score'] = max(0, score)
            
        except Exception as e:
            quality_metrics['error'] = str(e)
        
        return quality_metrics
    
    def batch_quality_analysis(self, input_dir: str, output_report: str) -> Dict[str, Any]:
        """
        Analyze quality for all audio files in directory
        
        Returns:
            Dict with batch quality analysis
        """
        input_path = Path(input_dir)
        quality_results = []
        
        # Analyze each file
        for audio_file in input_path.rglob('*'):
            if audio_file.suffix.lower() in ['.wav', '.mp3', '.flac', '.ogg', '.m4a']:
                quality_metrics = self.analyze_audio_quality(str(audio_file))
                quality_results.append(quality_metrics)
        
        # Calculate batch statistics
        valid_results = [r for r in quality_results if 'error' not in r]
        
        if valid_results:
            batch_stats = {
                'total_files': len(quality_results),
                'analyzed_files': len(valid_results),
                'average_snr': np.mean([r['snr'] for r in valid_results]),
                'average_dynamic_range': np.mean([r['dynamic_range'] for r in valid_results]),
                'average_quality_score': np.mean([r['quality_score'] for r in valid_results]),
                'files_with_issues': len([r for r in valid_results if r['issues']]),
                'common_issues': {}
            }
            
            # Count common issues
            all_issues = []
            for result in valid_results:
                all_issues.extend(result['issues'])
            
            for issue in set(all_issues):
                batch_stats['common_issues'][issue] = all_issues.count(issue)
            
        else:
            batch_stats = {'total_files': len(quality_results), 'analyzed_files': 0}
        
        # Generate report
        report = {
            'batch_statistics': batch_stats,
            'individual_results': quality_results,
            'recommendations': self._generate_recommendations(batch_stats)
        }
        
        # Save report
        with open(output_report, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def _generate_recommendations(self, batch_stats: Dict[str, Any]) -> List[str]:
        """Generate quality improvement recommendations"""
        recommendations = []
        
        if 'average_snr' in batch_stats:
            if batch_stats['average_snr'] < 15:
                recommendations.append("Consider applying noise reduction to improve SNR")
            
            if batch_stats['average_dynamic_range'] < 25:
                recommendations.append("Audio may benefit from dynamic range expansion")
            
            if batch_stats['files_with_issues'] > batch_stats['analyzed_files'] * 0.3:
                recommendations.append("High percentage of files with quality issues - review source material")
        
        if not recommendations:
            recommendations.append("Audio quality appears acceptable for batch processing")
        
        return recommendations
