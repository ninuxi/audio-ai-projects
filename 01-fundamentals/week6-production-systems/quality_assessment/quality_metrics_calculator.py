"""
🎵 QUALITY_METRICS_CALCULATOR.PY - DEMO VERSION
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
"""
Quality Metrics Calculator
=========================

Calculates comprehensive quality metrics for cultural heritage audio.
Provides standardized measurements and scoring systems.
"""

import asyncio
import numpy as np
import librosa
from scipy import signal
from typing import Dict, Any
import logging

class QualityMetrics:
    """
    Calculates standardized quality metrics for cultural heritage audio.
    
    Provides measurements for:
    - Signal-to-noise ratio (SNR)
    - Total harmonic distortion (THD)
    - Dynamic range
    - Frequency response
    - Temporal consistency
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize quality metrics calculator"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Metric calculation parameters
        self.reference_level = self.config.get('reference_level', -20.0)  # dBFS
        self.noise_gate_threshold = self.config.get('noise_gate_threshold', -60.0)  # dBFS
        
    async def calculate_metrics(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Calculate comprehensive quality metrics"""
        
        try:
            self.logger.debug(f"Calculating quality metrics for audio: {len(audio_data)} samples at {sample_rate}Hz")
            
            # Calculate all metrics concurrently
            metric_tasks = [
                self._calculate_snr(audio_data, sample_rate),
                self._calculate_thd(audio_data, sample_rate),
                self._calculate_dynamic_range(audio_data),
                self._calculate_frequency_metrics(audio_data, sample_rate),
                self._calculate_temporal_metrics(audio_data, sample_rate),
                self._calculate_psychoacoustic_metrics(audio_data, sample_rate)
            ]
            
            results = await asyncio.gather(*metric_tasks)
            
            # Compile comprehensive metrics
            quality_metrics = {
                'snr_metrics': results[0],
                'distortion_metrics': results[1],
                'dynamic_metrics': results[2],
                'frequency_metrics': results[3],
                'temporal_metrics': results[4],
                'psychoacoustic_metrics': results[5],
                'overall_score': 0.0
            }
            
            # Calculate overall quality score
            quality_metrics['overall_score'] = self._calculate_overall_score(quality_metrics)
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"Quality metrics calculation failed: {e}")
            return {'error': str(e), 'overall_score': 0.0}
    
    async def _calculate_snr(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Calculate signal-to-noise ratio metrics"""
        
        # Method 1: Statistical approach
        # Assume first 10% is noise, rest is signal + noise
        noise_samples = int(0.1 * len(audio_data))
        noise_segment = audio_data[:noise_samples]
        
        noise_power = np.mean(noise_segment**2)
        signal_power = np.mean(audio_data**2)
        
        snr_statistical = 10 * np.log10((signal_power - noise_power) / (noise_power + 1e-10))
        
        # Method 2: Spectral approach
        frequencies, psd = signal.welch(audio_data, sample_rate, nperseg=2048)
        
        # Estimate noise floor as minimum PSD
        noise_floor = np.percentile(psd, 10)
        signal_level = np.mean(psd)
        
        snr_spectral = 10 * np.log10(signal_level / (noise_floor + 1e-10))
        
        # Method 3: Dynamic approach (signal vs quiet segments)
        snr_dynamic = await self._calculate_dynamic_snr(audio_data, sample_rate)
        
        return {
            'snr_statistical_db': float(snr_statistical),
            'snr_spectral_db': float(snr_spectral),
            'snr_dynamic_db': float(snr_dynamic),
            'snr_average_db': float(np.mean([snr_statistical, snr_spectral, snr_dynamic])),
            'noise_floor_db': float(10 * np.log10(noise_power + 1e-10))
        }
    
    async def _calculate_dynamic_snr(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """Calculate SNR using dynamic range analysis"""
        
        # Divide into short segments
        segment_length = int(0.1 * sample_rate)  # 100ms segments
        segment_powers = []
        
        for i in range(0, len(audio_data) - segment_length, segment_length):
            segment = audio_data[i:i + segment_length]
            power = np.mean(segment**2)
            segment_powers.append(power)
        
        segment_powers = np.array(segment_powers)
        
        # Separate signal and noise segments based on power
        power_threshold = np.percentile(segment_powers, 30)  # Bottom 30% as noise
        
        noise_segments = segment_powers[segment_powers <= power_threshold]
        signal_segments = segment_powers[segment_powers > power_threshold]
        
        if len(noise_segments) > 0 and len(signal_segments) > 0:
            noise_power = np.mean(noise_segments)
            signal_power = np.mean(signal_segments)
            
            return 10 * np.log10((signal_power - noise_power) / (noise_power + 1e-10))
        
        return 20.0  # Default if can't separate
    
    async def _calculate_thd(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Calculate total harmonic distortion metrics"""
        
        # Use a segment of audio for THD calculation
        segment_length = min(len(audio_data), 8192)
        segment = audio_data[:segment_length]
        
        # FFT analysis
        fft_data = np.fft.fft(segment)
        freqs = np.fft.fftfreq(len(segment), 1/sample_rate)
        magnitude = np.abs(fft_data[:len(fft_data)//2])
        positive_freqs = freqs[:len(freqs)//2]
        
        # Find fundamental frequency
        # Skip DC and very low frequencies
        start_bin = int(50 * len(segment) / sample_rate)  # Skip below 50Hz
        
        if start_bin >= len(magnitude):
            return {'thd_percent': 5.0, 'thd_db': -26.0, 'fundamental_freq': 440.0}
        
        fundamental_bin = np.argmax(magnitude[start_bin:]) + start_bin
        fundamental_freq = positive_freqs[fundamental_bin]
        fundamental_magnitude = magnitude[fundamental_bin]
        
        # Calculate harmonic content
        harmonic_power = 0.0
        harmonic_count = 0
        
        for harmonic in range(2, 6):  # 2nd to 5th harmonics
            harmonic_freq = fundamental_freq * harmonic
            
            if harmonic_freq > sample_rate / 2:
                break
            
            # Find closest bin to harmonic frequency
            harmonic_bin = int(harmonic_freq * len(segment) / sample_rate)
            
            if harmonic_bin < len(magnitude):
                harmonic_power += magnitude[harmonic_bin]**2
                harmonic_count += 1
        
        # Calculate THD
        if fundamental_magnitude > 0:
            fundamental_power = fundamental_magnitude**2
            thd_ratio = np.sqrt(harmonic_power) / fundamental_magnitude
            thd_percent = thd_ratio * 100
            thd_db = 20 * np.log10(thd_ratio + 1e-10)
        else:
            thd_percent = 50.0
            thd_db = -6.0
        
        return {
            'thd_percent': float(min(100.0, max(0.0, thd_percent))),
            'thd_db': float(max(-100.0, thd_db)),
            'fundamental_frequency_hz': float(fundamental_freq),
            'harmonics_detected': harmonic_count,
            'fundamental_level_db': float(20 * np.log10(fundamental_magnitude + 1e-10))
        }
    
    async def _calculate_dynamic_range(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Calculate dynamic range metrics"""
        
        # Basic dynamic range
        peak_level = np.max(np.abs(audio_data))
        rms_level = np.sqrt(np.mean(audio_data**2))
        
        if rms_level > 0:
            crest_factor = peak_level / rms_level
            crest_factor_db = 20 * np.log10(crest_factor)
        else:
            crest_factor_db = 0.0
        
        # Percentile-based dynamic range
        percentile_95 = np.percentile(np.abs(audio_data), 95)
        percentile_10 = np.percentile(np.abs(audio_data), 10)
        
        percentile_range_db = 20 * np.log10((percentile_95 + 1e-10) / (percentile_10 + 1e-10))
        
        # EBU R128-style loudness range (simplified)
        loudness_range = await self._calculate_loudness_range(audio_data)
        
        return {
            'peak_level_dbfs': float(20 * np.log10(peak_level + 1e-10)),
            'rms_level_dbfs': float(20 * np.log10(rms_level + 1e-10)),
            'crest_factor_db': float(crest_factor_db),
            'percentile_range_db': float(percentile_range_db),
            'loudness_range_lu': float(loudness_range),
            'dynamic_range_score': float(min(1.0, max(0.0, (crest_factor_db - 3) / 15)))
        }
    
    async def _calculate_loudness_range(self, audio_data: np.ndarray) -> float:
        """Calculate loudness range (simplified EBU R128)"""
        
        # Simplified loudness calculation
        # In real implementation, would use proper K-weighting
        
        # Divide into 400ms blocks
        block_length = int(0.4 * 44100)  # Assume 44.1kHz for simplicity
        
        if len(audio_data) < block_length:
            return 10.0  # Default range
        
        block_loudnesses = []
        
        for i in range(0, len(audio_data) - block_length, block_length):
            block = audio_data[i:i + block_length]
            # Simplified loudness calculation
            mean_square = np.mean(block**2)
            if mean_square > 0:
                loudness = -0.691 + 10 * np.log10(mean_square)  # Simplified LUFS
                block_loudnesses.append(loudness)
        
        if len(block_loudnesses) < 2:
            return 10.0
        
        # Calculate loudness range as difference between 95th and 10th percentile
        loudness_array = np.array(block_loudnesses)
        lra = np.percentile(loudness_array, 95) - np.percentile(loudness_array, 10)
        
        return max(0.0, min(50.0, lra))  # Clamp to reasonable range
    
    async def _calculate_frequency_metrics(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Calculate frequency domain metrics"""
        
        # Power spectral density
        frequencies, psd = signal.welch(audio_data, sample_rate, nperseg=2048)
        
        # Frequency band analysis
        bands = {
            'sub_bass': (20, 60),
            'bass': (60, 250),
            'low_mid': (250, 500),
            'mid': (500, 2000),
            'high_mid': (2000, 4000),
            'presence': (4000, 8000),
            'brilliance': (8000, 20000)
        }
        
        band_energies = {}
        for band_name, (low_freq, high_freq) in bands.items():
            band_mask = (frequencies >= low_freq) & (frequencies <= high_freq)
            if np.any(band_mask):
                band_energy = np.sum(psd[band_mask])
                band_energies[f'{band_name}_energy_db'] = float(10 * np.log10(band_energy + 1e-10))
            else:
                band_energies[f'{band_name}_energy_db'] = -60.0
        
        # Spectral characteristics
        spectral_centroid = float(np.sum(frequencies * psd) / (np.sum(psd) + 1e-10))
        spectral_spread = float(np.sqrt(np.sum(((frequencies - spectral_centroid) ** 2) * psd) / (np.sum(psd) + 1e-10)))
        spectral_skewness = await self._calculate_spectral_skewness(frequencies, psd, spectral_centroid, spectral_spread)
        
        # Frequency response flatness
        flatness = np.exp(np.mean(np.log(psd + 1e-10))) / (np.mean(psd) + 1e-10)
        
        # High frequency rolloff
        hf_mask = frequencies > 8000
        if np.any(hf_mask):
            hf_energy = np.sum(psd[hf_mask])
            total_energy = np.sum(psd)
            hf_ratio = hf_energy / (total_energy + 1e-10)
        else:
            hf_ratio = 0.0
        
        return {
            **band_energies,
            'spectral_centroid_hz': spectral_centroid,
            'spectral_spread_hz': spectral_spread,
            'spectral_skewness': spectral_skewness,
            'spectral_flatness': float(flatness),
            'high_frequency_ratio': float(hf_ratio),
            'bandwidth_hz': float(frequencies[-1] - frequencies[0]),
            'frequency_balance_score': await self._calculate_frequency_balance_score(band_energies)
        }
    
    async def _calculate_spectral_skewness(self, frequencies: np.ndarray, psd: np.ndarray, 
                                         centroid: float, spread: float) -> float:
        """Calculate spectral skewness"""
        
        if spread == 0:
            return 0.0
        
        skewness = np.sum(((frequencies - centroid) ** 3) * psd) / ((spread ** 3) * (np.sum(psd) + 1e-10))
        return float(skewness)
    
    async def _calculate_frequency_balance_score(self, band_energies: Dict[str, float]) -> float:
        """Calculate frequency balance score"""
        
        # Extract energy values
        energies = [energy for key, energy in band_energies.items() if 'energy_db' in key]
        
        if len(energies) < 3:
            return 0.5
        
        # Calculate standard deviation (lower is more balanced)
        std_dev = np.std(energies)
        
        # Convert to score (0-1, where 1 is perfectly balanced)
        balance_score = max(0.0, min(1.0, 1.0 - std_dev / 20.0))  # Normalize by 20dB range
        
        return float(balance_score)
    
    async def _calculate_temporal_metrics(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Calculate temporal domain metrics"""
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio_data, frame_length=2048, hop_length=512)[0]
        
        # RMS energy over time
        rms = librosa.feature.rms(y=audio_data, frame_length=2048, hop_length=512)[0]
        
        # Temporal stability metrics
        zcr_stability = 1.0 - (np.std(zcr) / (np.mean(zcr) + 1e-10))
        rms_stability = 1.0 - (np.std(rms) / (np.mean(rms) + 1e-10))
        
        # Onset detection
        onset_frames = librosa.onset.onset_detect(y=audio_data, sr=sample_rate, units='frames')
        onset_rate = len(onset_frames) / (len(audio_data) / sample_rate)
        
        # Tempo estimation (if applicable)
        try:
            tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
            tempo_confidence = float(tempo > 60 and tempo < 200)  # Reasonable tempo range
        except:
            tempo = 0.0
            tempo_confidence = 0.0
        
        return {
            'zero_crossing_rate_mean': float(np.mean(zcr)),
            'zero_crossing_rate_std': float(np.std(zcr)),
            'zcr_stability': float(max(0.0, min(1.0, zcr_stability))),
            'rms_energy_mean': float(np.mean(rms)),
            'rms_energy_std': float(np.std(rms)),
            'rms_stability': float(max(0.0, min(1.0, rms_stability))),
            'onset_rate_per_second': float(onset_rate),
            'onset_count': len(onset_frames),
            'tempo_bpm': float(tempo),
            'tempo_confidence': tempo_confidence,
            'temporal_consistency_score': float((zcr_stability + rms_stability) / 2)
        }
    
    async def _calculate_psychoacoustic_metrics(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Calculate psychoacoustic quality metrics"""
        
        # Bark scale analysis
        bark_spectrum = await self._calculate_bark_spectrum(audio_data, sample_rate)
        
        # Masking threshold estimation
        masking_threshold = await self._estimate_masking_threshold(audio_data, sample_rate)
        
        # Perceived loudness (simplified)
        perceived_loudness = await self._calculate_perceived_loudness(audio_data, sample_rate)
        
        # Sharpness calculation
        sharpness = await self._calculate_sharpness(bark_spectrum)
        
        # Roughness estimation
        roughness = await self._calculate_roughness(audio_data, sample_rate)
        
        return {
            'bark_spectrum_energy': bark_spectrum,
            'masking_threshold_db': masking_threshold,
            'perceived_loudness_sone': perceived_loudness,
            'sharpness_acum': sharpness,
            'roughness_asper': roughness,
            'psychoacoustic_quality_score': await self._calculate_psychoacoustic_score(
                perceived_loudness, sharpness, roughness
            )
        }
    
    async def _calculate_bark_spectrum(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Calculate energy in Bark frequency bands"""
        
        # Bark scale frequency bands (approximate)
        bark_bands = [
            (0, 100), (100, 200), (200, 300), (300, 400), (400, 510),
            (510, 630), (630, 770), (770, 920), (920, 1080), (1080, 1270),
            (1270, 1480), (1480, 1720), (1720, 2000), (2000, 2320), (2320, 2700),
            (2700, 3150), (3150, 3700), (3700, 4400), (4400, 5300), (5300, 6400),
            (6400, 7700), (7700, 9500), (9500, 12000), (12000, 15500)
        ]
        
        # Calculate PSD
        frequencies, psd = signal.welch(audio_data, sample_rate, nperseg=2048)
        
        bark_energies = {}
        for i, (low_freq, high_freq) in enumerate(bark_bands):
            if high_freq > sample_rate / 2:
                break
                
            band_mask = (frequencies >= low_freq) & (frequencies <= high_freq)
            if np.any(band_mask):
                band_energy = np.sum(psd[band_mask])
                bark_energies[f'bark_band_{i}_energy'] = float(10 * np.log10(band_energy + 1e-10))
        
        return bark_energies
    
    async def _estimate_masking_threshold(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """Estimate psychoacoustic masking threshold"""
        
        # Simplified masking threshold calculation
        # In real implementation, would use detailed psychoacoustic model
        
        # Calculate spectral peaks
        frequencies, psd = signal.welch(audio_data, sample_rate, nperseg=2048)
        
        # Find spectral peaks
        peaks, _ = signal.find_peaks(psd, height=np.max(psd) * 0.1)
        
        if len(peaks) > 0:
            # Use strongest peak as masker
            strongest_peak_idx = peaks[np.argmax(psd[peaks])]
            masker_level = 10 * np.log10(psd[strongest_peak_idx] + 1e-10)
            
            # Simplified masking threshold (actual calculation is much more complex)
            masking_threshold = masker_level - 20  # Approximate 20dB masking
        else:
            masking_threshold = -40.0  # Default threshold
        
        return float(masking_threshold)
    
    async def _calculate_perceived_loudness(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """Calculate perceived loudness in sones (simplified)"""
        
        # Simplified loudness calculation
        # Real implementation would use Stevens' power law and frequency weighting
        
        rms_level = np.sqrt(np.mean(audio_data**2))
        
        if rms_level > 0:
            # Convert to approximate sone scale
            db_spl = 20 * np.log10(rms_level) + 94  # Rough conversion to SPL
            
            # Stevens' power law approximation
            if db_spl > 40:
                sones = 2 ** ((db_spl - 40) / 10)
            else:
                sones = 0.1
        else:
            sones = 0.0
        
        return float(min(100.0, max(0.0, sones)))
    
    async def _calculate_sharpness(self, bark_spectrum: Dict[str, float]) -> float:
        """Calculate sharpness in acum units"""
        
        # Simplified sharpness calculation
        # Real implementation would use proper Bark scale weighting
        
        bark_energies = list(bark_spectrum.values())
        
        if len(bark_energies) == 0:
            return 1.0
        
        # Weight higher frequencies more heavily
        weights = np.linspace(1, 4, len(bark_energies))  # Linear weighting for simplicity
        weighted_energy = np.sum(np.array(bark_energies) * weights)
        total_energy = np.sum(bark_energies)
        
        if total_energy > 0:
            sharpness = weighted_energy / total_energy / 2  # Normalize to approximate acum
        else:
            sharpness = 1.0
        
        return float(max(0.5, min(5.0, sharpness)))
    
    async def _calculate_roughness(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """Calculate roughness in asper units"""
        
        # Simplified roughness calculation based on modulation depth
        # Real implementation would analyze beating patterns
        
        # Calculate short-term energy variations
        frame_length = int(0.02 * sample_rate)  # 20ms frames
        hop_length = int(0.01 * sample_rate)    # 10ms hop
        
        if len(audio_data) < frame_length * 2:
            return 0.0
        
        frame_energies = []
        for i in range(0, len(audio_data) - frame_length, hop_length):
            frame = audio_data[i:i + frame_length]
            energy = np.mean(frame**2)
            frame_energies.append(energy)
        
        if len(frame_energies) < 2:
            return 0.0
        
        # Calculate energy modulation
        energy_array = np.array(frame_energies)
        energy_variation = np.std(energy_array) / (np.mean(energy_array) + 1e-10)
        
        # Convert to approximate asper scale
        roughness = min(2.0, energy_variation * 5)  # Scale factor for asper approximation
        
        return float(roughness)
    
    async def _calculate_psychoacoustic_score(self, loudness: float, sharpness: float, 
                                           roughness: float) -> float:
        """Calculate overall psychoacoustic quality score"""
        
        # Normalize and weight psychoacoustic parameters
        
        # Loudness score (optimal around 4-16 sones for music)
        if 4 <= loudness <= 16:
            loudness_score = 1.0
        elif loudness < 4:
            loudness_score = loudness / 4
        else:
            loudness_score = max(0.2, 16 / loudness)
        
        # Sharpness score (optimal around 1-2 acum)
        if 1 <= sharpness <= 2:
            sharpness_score = 1.0
        elif sharpness < 1:
            sharpness_score = sharpness
        else:
            sharpness_score = max(0.2, 2 / sharpness)
        
        # Roughness score (lower is better)
        roughness_score = max(0.0, 1.0 - roughness / 2)
        
        # Weighted combination
        psychoacoustic_score = (
            0.4 * loudness_score +
            0.3 * sharpness_score +
            0.3 * roughness_score
        )
        
        return float(max(0.0, min(1.0, psychoacoustic_score)))
    
    def _calculate_overall_score(self, quality_metrics: Dict[str, Any]) -> float:
        """Calculate overall quality score from all metrics"""
        
        scores = []
        weights = []
        
        # SNR contribution (30% weight)
        if 'snr_metrics' in quality_metrics:
            snr_avg = quality_metrics['snr_metrics'].get('snr_average_db', 20)
            snr_score = max(0.0, min(1.0, (snr_avg + 10) / 40))  # -10 to 30 dB range
            scores.append(snr_score)
            weights.append(0.3)
        
        # Distortion contribution (20% weight)
        if 'distortion_metrics' in quality_metrics:
            thd_percent = quality_metrics['distortion_metrics'].get('thd_percent', 10)
            thd_score = max(0.0, 1.0 - thd_percent / 10)  # 10% THD = 0 score
            scores.append(thd_score)
            weights.append(0.2)
        
        # Dynamic range contribution (20% weight)
        if 'dynamic_metrics' in quality_metrics:
            dr_score = quality_metrics['dynamic_metrics'].get('dynamic_range_score', 0.5)
            scores.append(dr_score)
            weights.append(0.2)
        
        # Frequency balance contribution (15% weight)
        if 'frequency_metrics' in quality_metrics:
            freq_balance = quality_metrics['frequency_metrics'].get('frequency_balance_score', 0.5)
            scores.append(freq_balance)
            weights.append(0.15)
        
        # Temporal consistency contribution (10% weight)
        if 'temporal_metrics' in quality_metrics:
            temporal_score = quality_metrics['temporal_metrics'].get('temporal_consistency_score', 0.5)
            scores.append(temporal_score)
            weights.append(0.1)
        
        # Psychoacoustic contribution (5% weight)
        if 'psychoacoustic_metrics' in quality_metrics:
            psycho_score = quality_metrics['psychoacoustic_metrics'].get('psychoacoustic_quality_score', 0.5)
            scores.append(psycho_score)
            weights.append(0.05)
        
        # Calculate weighted average
        if len(scores) > 0:
            total_weight = sum(weights)
            weighted_score = sum(s * w for s, w in zip(scores, weights)) / (total_weight + 1e-10)
            return float(max(0.0, min(1.0, weighted_score)))
        
        return 0.5  # Default score
