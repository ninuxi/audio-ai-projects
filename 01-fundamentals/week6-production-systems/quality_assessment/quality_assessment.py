"""
Restoration Advisor
==================

Provides recommendations for audio restoration based on quality analysis.
Suggests appropriate restoration techniques and parameters.
"""

import asyncio
import logging
import numpy as np
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

@dataclass
class RestorationRecommendation:
    """Data class for restoration recommendations"""
    technique: str
    priority: int  # 1-10, higher is more urgent
    parameters: Dict[str, Any]
    expected_improvement: Dict[str, float]
    complexity: str  # 'simple', 'moderate', 'complex'
    estimated_time_hours: float
    estimated_cost_eur: float

class RestorationAdvisor:
    """
    Analyzes audio quality issues and provides targeted restoration recommendations.
    
    Provides advice for:
    - Noise reduction techniques
    - Frequency response correction
    - Dynamic range restoration
    - Artifact removal
    - Degradation-specific treatments
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize restoration advisor"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Restoration technique database
        self.restoration_techniques = self._load_restoration_techniques()
        
        # Cost and time estimates
        self.base_hourly_rate = self.config.get('hourly_rate_eur', 50.0)
        self.complexity_multipliers = {
            'simple': 1.0,
            'moderate': 2.0,
            'complex': 4.0,
            'very_complex': 8.0
        }
    
    def _load_restoration_techniques(self) -> Dict[str, Dict[str, Any]]:
        """Load restoration technique database"""
        
        return {
            'spectral_noise_reduction': {
                'description': 'Remove broadband noise using spectral subtraction',
                'targets': ['hiss', 'broadband_noise', 'surface_noise'],
                'complexity': 'moderate',
                'expected_snr_improvement': 8.0,
                'base_time_hours': 2.0,
                'parameters': {
                    'noise_reduction_db': 12.0,
                    'preservation_ratio': 0.3,
                    'frequency_smoothing': 3
                }
            },
            
            'click_pop_removal': {
                'description': 'Remove impulsive noise (clicks, pops)',
                'targets': ['clicks', 'pops', 'crackle'],
                'complexity': 'simple',
                'expected_snr_improvement': 5.0,
                'base_time_hours': 1.0,
                'parameters': {
                    'detection_threshold': 3.0,
                    'interpolation_method': 'autoregressive',
                    'max_gap_ms': 10
                }
            },
            
            'hum_removal': {
                'description': 'Remove power line hum and harmonics',
                'targets': ['power_line_hum', 'electrical_interference'],
                'complexity': 'simple',
                'expected_snr_improvement': 10.0,
                'base_time_hours': 0.5,
                'parameters': {
                    'fundamental_freq': 50.0,
                    'harmonics_count': 5,
                    'notch_width_hz': 2.0
                }
            },
            
            'frequency_response_correction': {
                'description': 'Correct frequency response imbalances',
                'targets': ['high_freq_loss', 'frequency_imbalance'],
                'complexity': 'moderate',
                'expected_snr_improvement': 3.0,
                'base_time_hours': 1.5,
                'parameters': {
                    'eq_type': 'parametric',
                    'bands_count': 7,
                    'max_boost_db': 6.0
                }
            },
            
            'dynamic_range_restoration': {
                'description': 'Restore dynamic range and reduce compression',
                'targets': ['compression', 'limited_dynamics'],
                'complexity': 'complex',
                'expected_snr_improvement': 2.0,
                'base_time_hours': 3.0,
                'parameters': {
                    'expansion_ratio': 1.5,
                    'threshold_db': -30.0,
                    'attack_ms': 1.0,
                    'release_ms': 100.0
                }
            },
            
            'wow_flutter_correction': {
                'description': 'Correct pitch instability (wow and flutter)',
                'targets': ['wow', 'flutter', 'pitch_instability'],
                'complexity': 'very_complex',
                'expected_snr_improvement': 1.0,
                'base_time_hours': 8.0,
                'parameters': {
                    'analysis_window_ms': 50,
                    'correction_strength': 0.8,
                    'frequency_range_hz': [50, 8000]
                }
            },
            
            'dropout_repair': {
                'description': 'Repair dropouts and missing audio segments',
                'targets': ['dropouts', 'missing_segments'],
                'complexity': 'moderate',
                'expected_snr_improvement': 4.0,
                'base_time_hours': 2.5,
                'parameters': {
                    'interpolation_method': 'spectral',
                    'max_gap_ms': 100,
                    'cross_fade_ms': 5
                }
            },
            
            'clipping_restoration': {
                'description': 'Restore clipped/distorted audio',
                'targets': ['clipping', 'distortion'],
                'complexity': 'complex',
                'expected_snr_improvement': 6.0,
                'base_time_hours': 4.0,
                'parameters': {
                    'threshold_detection': 0.95,
                    'restoration_method': 'cubic_spline',
                    'harmonics_restoration': True
                }
            },
            
            'azimuth_correction': {
                'description': 'Correct azimuth errors in tape recordings',
                'targets': ['azimuth_errors', 'phase_issues'],
                'complexity': 'complex',
                'expected_snr_improvement': 3.0,
                'base_time_hours': 3.5,
                'parameters': {
                    'phase_correction': True,
                    'frequency_dependent': True,
                    'stereo_processing': True
                }
            },
            
            'speed_correction': {
                'description': 'Correct playback speed variations',
                'targets': ['speed_variations', 'tempo_drift'],
                'complexity': 'moderate',
                'expected_snr_improvement': 1.0,
                'base_time_hours': 2.0,
                'parameters': {
                    'reference_pitch': 440.0,
                    'correction_method': 'phase_vocoder',
                    'preserve_formants': True
                }
            }
        }
    
    async def generate_recommendations(self, audio_data: np.ndarray, sample_rate: int, 
                                     quality_score: float) -> List[RestorationRecommendation]:
        """Generate restoration recommendations based on audio analysis"""
        
        try:
            self.logger.info("Generating restoration recommendations")
            
            # Analyze audio issues
            audio_issues = await self._analyze_audio_issues(audio_data, sample_rate, quality_score)
            
            # Generate recommendations based on issues
            recommendations = []
            
            for issue_type, issue_data in audio_issues.items():
                if issue_data.get('detected', False):
                    technique_recommendations = await self._get_techniques_for_issue(
                        issue_type, issue_data
                    )
                    recommendations.extend(technique_recommendations)
            
            # Sort by priority (highest first)
            recommendations.sort(key=lambda x: x.priority, reverse=True)
            
            # Add workflow recommendations
            workflow_recommendations = await self._generate_workflow_recommendations(recommendations)
            
            self.logger.info(f"Generated {len(recommendations)} restoration recommendations")
            
            return {
                'individual_recommendations': recommendations,
                'workflow_recommendations': workflow_recommendations,
                'total_estimated_time_hours': sum(r.estimated_time_hours for r in recommendations),
                'total_estimated_cost_eur': sum(r.estimated_cost_eur for r in recommendations),
                'overall_complexity': self._assess_overall_complexity(recommendations)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {e}")
            return {
                'individual_recommendations': [],
                'workflow_recommendations': [],
                'error': str(e)
            }
    
    async def _analyze_audio_issues(self, audio_data: np.ndarray, sample_rate: int, 
                                  quality_score: float) -> Dict[str, Any]:
        """Analyze specific audio issues that need restoration"""
        
        issues = {}
        
        # Noise analysis
        noise_analysis = await self._analyze_noise_issues(audio_data, sample_rate)
        issues.update(noise_analysis)
        
        # Frequency response issues
        frequency_issues = await self._analyze_frequency_issues(audio_data, sample_rate)
        issues.update(frequency_issues)
        
        # Dynamic range issues
        dynamic_issues = await self._analyze_dynamic_issues(audio_data, sample_rate)
        issues.update(dynamic_issues)
        
        # Distortion and artifacts
        artifact_issues = await self._analyze_artifact_issues(audio_data, sample_rate)
        issues.update(artifact_issues)
        
        # Temporal issues
        temporal_issues = await self._analyze_temporal_issues(audio_data, sample_rate)
        issues.update(temporal_issues)
        
        return issues
    
    async def _analyze_noise_issues(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Analyze noise-related issues"""
        
        # Calculate noise characteristics
        noise_floor = np.percentile(np.abs(audio_data), 10)
        signal_level = np.sqrt(np.mean(audio_data**2))
        
        snr = 20 * np.log10((signal_level + 1e-10) / (noise_floor + 1e-10))
        
        # Spectral analysis for noise type identification
        from scipy import signal as scipy_signal
        frequencies, psd = scipy_signal.welch(audio_data, sample_rate, nperseg=2048)
        
        # Detect different noise types
        issues = {}
        
        # Broadband noise (hiss)
        high_freq_noise = np.mean(psd[frequencies > 4000])
        total_noise = np.mean(psd)
        hiss_ratio = high_freq_noise / (total_noise + 1e-10)
        
        issues['hiss'] = {
            'detected': hiss_ratio > 1.5 and snr < 25,
            'severity': min(1.0, hiss_ratio / 3.0),
            'snr_db': snr,
            'hiss_ratio': hiss_ratio
        }
        
        # Power line hum
        hum_detected = False
        hum_strength = 0.0
        
        for hum_freq in [50, 60]:
            freq_idx = np.argmin(np.abs(frequencies - hum_freq))
            local_psd = psd[max(0, freq_idx-2):min(len(psd), freq_idx+3)]
            avg_psd = np.mean(psd)
            
            if len(local_psd) > 0:
                peak_strength = np.max(local_psd) / (avg_psd + 1e-10)
                if peak_strength > 3.0:
                    hum_detected = True
                    hum_strength = max(hum_strength, peak_strength)
        
        issues['power_line_hum'] = {
            'detected': hum_detected,
            'severity': min(1.0, hum_strength / 10.0),
            'strength_ratio': hum_strength
        }
        
        # Surface noise/crackle
        # Analyze short-term energy variations
        frame_size = int(0.01 * sample_rate)  # 10ms frames
        energy_variations = []
        
        for i in range(0, len(audio_data) - frame_size, frame_size):
            frame = audio_data[i:i + frame_size]
            energy = np.mean(frame**2)
            energy_variations.append(energy)
        
        if len(energy_variations) > 1:
            energy_cv = np.std(energy_variations) / (np.mean(energy_variations) + 1e-10)
            
            issues['surface_noise'] = {
                'detected': energy_cv > 2.0,
                'severity': min(1.0, energy_cv / 5.0),
                'energy_variation': energy_cv
            }
        
        return issues
    
    async def _analyze_frequency_issues(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Analyze frequency response issues"""
        
        from scipy import signal as scipy_signal
        frequencies, psd = scipy_signal.welch(audio_data, sample_rate, nperseg=2048)
        
        issues = {}
        
        # High frequency loss
        hf_mask = frequencies > 8000
        lf_mask = frequencies < 1000
        
        if np.any(hf_mask) and np.any(lf_mask):
            hf_energy = np.mean(psd[hf_mask])
            lf_energy = np.mean(psd[lf_mask])
            hf_lf_ratio = hf_energy / (lf_energy + 1e-10)
            
            issues['high_freq_loss'] = {
                'detected': hf_lf_ratio < 0.1,
                'severity': min(1.0, (0.5 - hf_lf_ratio) / 0.4),
                'hf_lf_ratio': hf_lf_ratio
            }
        
        # Frequency response imbalance
        # Divide spectrum into bands and check for imbalances
        bands = [(20, 200), (200, 2000), (2000, 8000), (8000, 20000)]
        band_energies = []
        
        for low, high in bands:
            band_mask = (frequencies >= low) & (frequencies <= high)
            if np.any(band_mask):
                band_energy = np.mean(psd[band_mask])
                band_energies.append(band_energy)
        
        if len(band_energies) > 2:
            energy_std = np.std(band_energies)
            energy_mean = np.mean(band_energies)
            imbalance_ratio = energy_std / (energy_mean + 1e-10)
            
            issues['frequency_imbalance'] = {
                'detected': imbalance_ratio > 5.0,
                'severity': min(1.0, imbalance_ratio / 10.0),
                'imbalance_ratio': imbalance_ratio
            }
        
        return issues
    
    async def _analyze_dynamic_issues(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Analyze dynamic range issues"""
        
        issues = {}
        
        # Calculate dynamic range metrics
        peak_level = np.max(np.abs(audio_data))
        rms_level = np.sqrt(np.mean(audio_data**2))
        
        if rms_level > 0:
            crest_factor = peak_level / rms_level
            crest_factor_db = 20 * np.log10(crest_factor)
        else:
            crest_factor_db = 0
        
        # Detect compression/limiting
        issues['compression'] = {
            'detected': crest_factor_db < 6.0,  # Very low crest factor indicates compression
            'severity': min(1.0, (10.0 - crest_factor_db) / 8.0),
            'crest_factor_db': crest_factor_db
        }
        
        # Analyze RMS consistency (over-compression indicator)
        window_size = int(0.1 * sample_rate)  # 100ms windows
        rms_values = []
        
        for i in range(0, len(audio_data) - window_size, window_size):
            window = audio_data[i:i + window_size]
            window_rms = np.sqrt(np.mean(window**2))
            rms_values.append(window_rms)
        
        if len(rms_values) > 1:
            rms_variation = np.std(rms_values) / (np.mean(rms_values) + 1e-10)
            
            issues['limited_dynamics'] = {
                'detected': rms_variation < 0.2,  # Very consistent RMS indicates limiting
                'severity': min(1.0, (0.5 - rms_variation) / 0.3),
                'rms_variation': rms_variation
            }
        
        return issues
    
    async def _analyze_artifact_issues(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Analyze artifacts and distortion issues"""
        
        issues = {}
        
        # Click and pop detection
        # Use median filter to detect outliers
        from scipy import signal as scipy_signal
        median_filtered = scipy_signal.medfilt(audio_data, kernel_size=7)
        difference = np.abs(audio_data - median_filtered)
        
        click_threshold = np.std(difference) * 3.0
        clicks = np.sum(difference > click_threshold)
        click_rate = clicks / (len(audio_data) / sample_rate)
        
        issues['clicks'] = {
            'detected': click_rate > 0.5,  # More than 0.5 clicks per second
            'severity': min(1.0, click_rate / 5.0),
            'click_rate_per_second': click_rate,
            'total_clicks': clicks
        }
        
        # Clipping detection
        clipping_threshold = 0.98
        clipped_samples = np.sum(np.abs(audio_data) >= clipping_threshold)
        clipping_percentage = clipped_samples / len(audio_data) * 100
        
        issues['clipping'] = {
            'detected': clipping_percentage > 0.01,  # More than 0.01% clipped
            'severity': min(1.0, clipping_percentage / 1.0),
            'clipping_percentage': clipping_percentage,
            'clipped_samples': clipped_samples
        }
        
        # Dropout detection
        # Look for segments with very low energy
        segment_size = int(0.05 * sample_rate)  # 50ms segments
        dropouts = 0
        
        for i in range(0, len(audio_data) - segment_size, segment_size):
            segment = audio_data[i:i + segment_size]
            segment_energy = np.mean(segment**2)
            
            if segment_energy < 1e-6:  # Very low energy indicates dropout
                dropouts += 1
        
        dropout_percentage = dropouts / (len(audio_data) / segment_size) * 100
        
        issues['dropouts'] = {
            'detected': dropout_percentage > 1.0,  # More than 1% dropouts
            'severity': min(1.0, dropout_percentage / 10.0),
            'dropout_percentage': dropout_percentage,
            'dropout_count': dropouts
        }
        
        return issues
    
    async def _analyze_temporal_issues(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Analyze temporal stability issues"""
        
        issues = {}
        
        # Wow and flutter detection (simplified)
        # Analyze short-term frequency variations
        if len(audio_data) > sample_rate:  # At least 1 second of audio
            
            # Use spectral centroid as a proxy for pitch variations
            import librosa
            spectral_centroid = librosa.feature.spectral_centroid(
                y=audio_data, sr=sample_rate, hop_length=512
            )[0]
            
            if len(spectral_centroid) > 10:
                # Remove outliers and smooth
                from scipy import signal as scipy_signal
                smoothed_centroid = scipy_signal.medfilt(spectral_centroid, kernel_size=5)
                
                # Calculate variations
                centroid_variations = np.abs(spectral_centroid - smoothed_centroid)
                relative_variations = centroid_variations / (smoothed_centroid + 1e-10)
                
                # Analyze wow (slow variations)
                wow_magnitude = np.std(relative_variations)
                
                issues['wow'] = {
                    'detected': wow_magnitude > 0.02,  # 2% variation
                    'severity': min(1.0, wow_magnitude / 0.1),
                    'magnitude_percent': wow_magnitude * 100
                }
                
                # Analyze flutter (fast variations)
                # Look for rapid changes in centroid
                centroid_diff = np.diff(spectral_centroid)
                flutter_magnitude = np.std(centroid_diff)
                
                issues['flutter'] = {
                    'detected': flutter_magnitude > 100,  # Arbitrary threshold
                    'severity': min(1.0, flutter_magnitude / 500),
                    'magnitude': flutter_magnitude
                }
        
        # Speed/pitch consistency
        # This would require more sophisticated analysis in real implementation
        issues['speed_variations'] = {
            'detected': False,  # Placeholder
            'severity': 0.0
        }
        
        return issues
    
    async def _get_techniques_for_issue(self, issue_type: str, issue_data: Dict[str, Any]) -> List[RestorationRecommendation]:
        """Get restoration techniques for a specific issue"""
        
        recommendations = []
        severity = issue_data.get('severity', 0.5)
        
        # Find techniques that target this issue
        for technique_name, technique_data in self.restoration_techniques.items():
            if issue_type in technique_data['targets']:
                
                # Calculate priority based on severity and technique effectiveness
                base_priority = int(severity * 8) + 2  # 2-10 range
                
                # Adjust parameters based on severity
                parameters = technique_data['parameters'].copy()
                parameters = self._adjust_parameters_for_severity(parameters, severity, issue_type)
                
                # Calculate time and cost estimates
                base_time = technique_data['base_time_hours']
                complexity = technique_data['complexity']
                
                time_multiplier = self.complexity_multipliers.get(complexity, 2.0)
                severity_multiplier = 0.5 + severity  # 0.5 to 1.5 range
                
                estimated_time = base_time * time_multiplier * severity_multiplier
                estimated_cost = estimated_time * self.base_hourly_rate
                
                # Expected improvement
                expected_improvement = {
                    'snr_improvement_db': technique_data['expected_snr_improvement'] * severity,
                    'quality_score_improvement': 0.1 + (severity * 0.2)
                }
                
                recommendation = RestorationRecommendation(
                    technique=technique_name,
                    priority=base_priority,
                    parameters=parameters,
                    expected_improvement=expected_improvement,
                    complexity=complexity,
                    estimated_time_hours=estimated_time,
                    estimated_cost_eur=estimated_cost
                )
                
                recommendations.append(recommendation)
        
        return recommendations
    
    def _adjust_parameters_for_severity(self, parameters: Dict[str, Any], 
                                      severity: float, issue_type: str) -> Dict[str, Any]:
        """Adjust technique parameters based on issue severity"""
        
        adjusted = parameters.copy()
        
        # Adjust parameters based on severity
        if issue_type == 'hiss' and 'noise_reduction_db' in adjusted:
            # More aggressive noise reduction for severe hiss
            adjusted['noise_reduction_db'] = adjusted['noise_reduction_db'] * (0.7 + severity * 0.6)
        
        elif issue_type == 'clicks' and 'detection_threshold' in adjusted:
            # Lower threshold for severe click problems
            adjusted['detection_threshold'] = adjusted['detection_threshold'] * (1.2 - severity * 0.4)
        
        elif issue_type == 'power_line_hum' and 'notch_width_hz' in adjusted:
            # Wider notch for severe hum
            adjusted['notch_width_hz'] = adjusted['notch_width_hz'] * (0.8 + severity * 0.4)
        
        elif issue_type == 'high_freq_loss' and 'max_boost_db' in adjusted:
            # More boost for severe high frequency loss
            adjusted['max_boost_db'] = adjusted['max_boost_db'] * (0.5 + severity * 0.8)
        
        elif issue_type == 'compression' and 'expansion_ratio' in adjusted:
            # More expansion for severe compression
            adjusted['expansion_ratio'] = adjusted['expansion_ratio'] * (0.8 + severity * 0.4)
        
        return adjusted
    
    async def _generate_workflow_recommendations(self, recommendations: List[RestorationRecommendation]) -> Dict[str, Any]:
        """Generate workflow recommendations for restoration process"""
        
        if not recommendations:
            return {
                'processing_order': [],
                'total_phases': 0,
                'estimated_total_time': 0,
                'workflow_complexity': 'none'
            }
        
        # Define processing order based on technique types
        technique_order = {
            'speed_correction': 1,           # First: correct timing issues
            'click_pop_removal': 2,          # Second: remove impulse noise
            'hum_removal': 3,                # Third: remove tonal interference
            'spectral_noise_reduction': 4,   # Fourth: remove broadband noise
            'dropout_repair': 5,             # Fifth: repair missing segments
            'clipping_restoration': 6,       # Sixth: restore distorted segments
            'frequency_response_correction': 7, # Seventh: correct frequency balance
            'dynamic_range_restoration': 8,  # Eighth: restore dynamics
            'wow_flutter_correction': 9,     # Ninth: correct pitch variations
            'azimuth_correction': 10         # Last: final corrections
        }
        
        # Sort recommendations by processing order
        ordered_recommendations = sorted(
            recommendations, 
            key=lambda x: technique_order.get(x.technique, 99)
        )
        
        # Group into phases
        phases = []
        current_phase = []
        last_order = 0
        
        for rec in ordered_recommendations:
            order = technique_order.get(rec.technique, 99)
            
            if order != last_order and current_phase:
                phases.append(current_phase)
                current_phase = []
            
            current_phase.append(rec)
            last_order = order
        
        if current_phase:
            phases.append(current_phase)
        
        # Calculate workflow metrics
        total_time = sum(rec.estimated_time_hours for rec in recommendations)
        
        workflow_complexity = self._assess_workflow_complexity(recommendations, phases)
        
        # Generate processing instructions
        processing_order = []
        for i, phase in enumerate(phases):
            phase_info = {
                'phase_number': i + 1,
                'phase_name': self._get_phase_name(phase),
                'techniques': [rec.technique for rec in phase],
                'estimated_time_hours': sum(rec.estimated_time_hours for rec in phase),
                'can_run_parallel': len(phase) > 1 and self._can_run_parallel(phase),
                'dependencies': self._get_phase_dependencies(i, phases)
            }
            processing_order.append(phase_info)
        
        return {
            'processing_order': processing_order,
            'total_phases': len(phases),
            'estimated_total_time_hours': total_time,
            'workflow_complexity': workflow_complexity,
            'parallel_processing_possible': any(p['can_run_parallel'] for p in processing_order),
            'critical_path_hours': self._calculate_critical_path(phases),
            'quality_gates': self._recommend_quality_gates(phases)
        }
    
    def _get_phase_name(self, phase: List[RestorationRecommendation]) -> str:
        """Get descriptive name for processing phase"""
        
        technique_names = [rec.technique for rec in phase]
        
        if any('noise' in name for name in technique_names):
            return "Noise Reduction Phase"
        elif any('click' in name or 'pop' in name for name in technique_names):
            return "Artifact Removal Phase"
        elif any('frequency' in name for name in technique_names):
            return "Frequency Correction Phase"
        elif any('dynamic' in name for name in technique_names):
            return "Dynamic Range Restoration Phase"
        elif any('wow' in name or 'flutter' in name for name in technique_names):
            return "Pitch Correction Phase"
        else:
            return f"Restoration Phase ({len(phase)} techniques)"
    
    def _can_run_parallel(self, phase: List[RestorationRecommendation]) -> bool:
        """Check if techniques in phase can run in parallel"""
        
        # Simple rule: techniques targeting different issues can run parallel
        targeted_issues = set()
        
        for rec in phase:
            technique_data = self.restoration_techniques.get(rec.technique, {})
            targets = technique_data.get('targets', [])
            
            # Check for conflicts
            for target in targets:
                if target in targeted_issues:
                    return False  # Conflict found
                targeted_issues.add(target)
        
        return len(phase) > 1
    
    def _get_phase_dependencies(self, phase_index: int, phases: List[List[RestorationRecommendation]]) -> List[int]:
        """Get dependencies for a processing phase"""
        
        # Simple dependency model: each phase depends on previous phases
        if phase_index == 0:
            return []
        else:
            return list(range(phase_index))
    
    def _calculate_critical_path(self, phases: List[List[RestorationRecommendation]]) -> float:
        """Calculate critical path time for workflow"""
        
        # Simple calculation: assume sequential processing of phases
        # In real implementation, would consider parallel processing
        
        total_time = 0
        for phase in phases:
            if len(phase) == 1:
                total_time += phase[0].estimated_time_hours
            else:
                # If multiple techniques, assume longest one determines phase time
                phase_time = max(rec.estimated_time_hours for rec in phase)
                total_time += phase_time
        
        return total_time
    
    def _recommend_quality_gates(self, phases: List[List[RestorationRecommendation]]) -> List[Dict[str, Any]]:
        """Recommend quality control checkpoints"""
        
        quality_gates = []
        
        for i, phase in enumerate(phases):
            gate = {
                'after_phase': i + 1,
                'checkpoint_name': f"Quality Check {i + 1}",
                'recommended_tests': self._get_quality_tests_for_phase(phase),
                'abort_criteria': self._get_abort_criteria_for_phase(phase),
                'estimated_check_time_minutes': 15 + len(phase) * 5
            }
            quality_gates.append(gate)
        
        return quality_gates
    
    def _get_quality_tests_for_phase(self, phase: List[RestorationRecommendation]) -> List[str]:
        """Get recommended quality tests for a phase"""
        
        tests = ['A/B comparison with original', 'Spectral analysis']
        
        phase_techniques = [rec.technique for rec in phase]
        
        if any('noise' in tech for tech in phase_techniques):
            tests.extend(['SNR measurement', 'Noise floor analysis'])
        
        if any('click' in tech or 'pop' in tech for tech in phase_techniques):
            tests.extend(['Artifact detection scan', 'Listening test for clicks'])
        
        if any('frequency' in tech for tech in phase_techniques):
            tests.extend(['Frequency response measurement', 'EQ curve validation'])
        
        if any('dynamic' in tech for tech in phase_techniques):
            tests.extend(['Dynamic range measurement', 'Compression detection'])
        
        return tests
    
    def _get_abort_criteria_for_phase(self, phase: List[RestorationRecommendation]) -> List[str]:
        """Get criteria for aborting restoration if quality degrades"""
        
        criteria = [
            'Overall quality score decreases by more than 10%',
            'Introduction of new artifacts',
            'Unnatural sound characteristics'
        ]
        
        phase_techniques = [rec.technique for rec in phase]
        
        if any('noise' in tech for tech in phase_techniques):
            criteria.append('SNR improvement less than 3dB')
        
        if any('frequency' in tech for tech in phase_techniques):
            criteria.append('Frequency response becomes more unbalanced')
        
        return criteria
    
    def _assess_overall_complexity(self, recommendations: List[RestorationRecommendation]) -> str:
        """Assess overall complexity of restoration project"""
        
        if not recommendations:
            return 'none'
        
        complexity_scores = {
            'simple': 1,
            'moderate': 2,
            'complex': 3,
            'very_complex': 4
        }
        
        total_complexity = sum(complexity_scores.get(rec.complexity, 2) for rec in recommendations)
        avg_complexity = total_complexity / len(recommendations)
        
        if avg_complexity < 1.5:
            return 'simple'
        elif avg_complexity < 2.5:
            return 'moderate'
        elif avg_complexity < 3.5:
            return 'complex'
        else:
            return 'very_complex'
    
    def _assess_workflow_complexity(self, recommendations: List[RestorationRecommendation], 
                                  phases: List[List[RestorationRecommendation]]) -> str:
        """Assess workflow complexity"""
        
        technique_count = len(recommendations)
        phase_count = len(phases)
        
        if technique_count <= 2 and phase_count <= 2:
            return 'simple'
        elif technique_count <= 4 and phase_count <= 3:
            return 'moderate'
        elif technique_count <= 7 and phase_count <= 5:
            return 'complex'
        else:
            return 'very_complex'