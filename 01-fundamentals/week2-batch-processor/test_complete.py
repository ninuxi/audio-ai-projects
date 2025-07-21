#!/usr/bin/env python3
"""
TEST SUITE COMPLETO WEEK 2 - Batch Processor
Testa tutto il sistema con file audio reali e genera report dettagliati
"""

import os
import sys
import tempfile
import shutil
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
import json
import time
import matplotlib.pyplot as plt

# Importa tutti i moduli
from processors.audio_normalizer import AudioNormalizer
from processors.noise_reducer import NoiseReducer
from processors.format_converter import FormatConverter
from processors.metadata_extractor import MetadataExtractor
from processors.batch_validator import BatchValidator
from reports.batch_report_generator import BatchReportGenerator
from reports.quality_analyzer import QualityAnalyzer
from reports.processing_stats import ProcessingStats

class CompleteTestSuite:
    """Test Suite Completo per Week 2"""
    
    def __init__(self):
        self.test_dir = Path("test_results")
        self.test_audio_dir = self.test_dir / "audio_files"
        self.output_dir = self.test_dir / "processed"
        self.reports_dir = self.test_dir / "reports"
        
        # Risultati dei test
        self.results = {
            'setup': False,
            'audio_generation': False,
            'processors': {},
            'reports': {},
            'integration': False,
            'performance': {},
            'overall_score': 0
        }
        
        self.test_files = []
        
    def setup_test_environment(self):
        """Prepara l'ambiente di test"""
        print("🔧 SETUP AMBIENTE DI TEST")
        print("=" * 50)
        
        try:
            # Crea directories
            for directory in [self.test_dir, self.test_audio_dir, self.output_dir, self.reports_dir]:
                directory.mkdir(parents=True, exist_ok=True)
            
            print(f"✅ Directories create: {self.test_dir}")
            self.results['setup'] = True
            
        except Exception as e:
            print(f"❌ Errore setup: {e}")
            self.results['setup'] = False
    
    def generate_test_audio_files(self):
        """Genera diversi tipi di file audio per il testing"""
        print("\n🎵 GENERAZIONE FILE AUDIO DI TEST")
        print("=" * 50)
        
        try:
            sample_rate = 44100
            duration = 3  # 3 secondi
            t = np.linspace(0, duration, int(sample_rate * duration))
            
            # 1. Audio normale - tono puro 440Hz
            print("📁 Creando audio normale...")
            audio_normal = 0.5 * np.sin(2 * np.pi * 440 * t)
            normal_file = self.test_audio_dir / "01_normal_tone.wav"
            sf.write(str(normal_file), audio_normal, sample_rate)
            self.test_files.append(str(normal_file))
            
            # 2. Audio con rumore
            print("📁 Creando audio con rumore...")
            noise = np.random.normal(0, 0.15, len(audio_normal))
            audio_noisy = audio_normal + noise
            noisy_file = self.test_audio_dir / "02_noisy_audio.wav"
            sf.write(str(noisy_file), audio_noisy, sample_rate)
            self.test_files.append(str(noisy_file))
            
            # 3. Audio con clipping
            print("📁 Creando audio con clipping...")
            audio_clipped = np.clip(audio_normal * 2.5, -1, 1)
            clipped_file = self.test_audio_dir / "03_clipped_audio.wav"
            sf.write(str(clipped_file), audio_clipped, sample_rate)
            self.test_files.append(str(clipped_file))
            
            # 4. Audio molto silenzioso
            print("📁 Creando audio silenzioso...")
            audio_quiet = audio_normal * 0.02
            quiet_file = self.test_audio_dir / "04_quiet_audio.wav"
            sf.write(str(quiet_file), audio_quiet, sample_rate)
            self.test_files.append(str(quiet_file))
            
            # 5. Audio complesso (mix di frequenze)
            print("📁 Creando audio complesso...")
            freq1 = 440 * np.sin(2 * np.pi * 440 * t)
            freq2 = 0.5 * 880 * np.sin(2 * np.pi * 880 * t)
            freq3 = 0.3 * 220 * np.sin(2 * np.pi * 220 * t)
            audio_complex = 0.3 * (freq1 + freq2 + freq3)
            complex_file = self.test_audio_dir / "05_complex_audio.wav"
            sf.write(str(complex_file), audio_complex, sample_rate)
            self.test_files.append(str(complex_file))
            
            # 6. Audio con pause di silenzio
            print("📁 Creando audio con pause...")
            silence = np.zeros(int(sample_rate * 0.5))
            audio_with_pauses = np.concatenate([
                silence, audio_normal[:sample_rate], silence, 
                audio_normal[sample_rate:2*sample_rate], silence
            ])
            pauses_file = self.test_audio_dir / "06_audio_with_pauses.wav"
            sf.write(str(pauses_file), audio_with_pauses, sample_rate)
            self.test_files.append(str(pauses_file))
            
            print(f"✅ Creati {len(self.test_files)} file audio di test")
            self.results['audio_generation'] = True
            
        except Exception as e:
            print(f"❌ Errore generazione audio: {e}")
            self.results['audio_generation'] = False
    
    def test_processors(self):
        """Test completo di tutti i processors"""
        print("\n🧪 TEST PROCESSORS")
        print("=" * 50)
        
        if not self.test_files:
            print("❌ Nessun file audio disponibile")
            return
        
        # Test AudioNormalizer
        print("\n📊 Test AudioNormalizer...")
        try:
            normalizer = AudioNormalizer()
            successes = 0
            
            for test_file in self.test_files:
                output_file = self.output_dir / f"normalized_{Path(test_file).name}"
                if normalizer.normalize_audio(test_file, str(output_file)):
                    successes += 1
            
            success_rate = (successes / len(self.test_files)) * 100
            self.results['processors']['AudioNormalizer'] = {
                'success_rate': success_rate,
                'status': '✅ PASS' if success_rate >= 80 else '⚠️ PARTIAL' if success_rate >= 50 else '❌ FAIL'
            }
            print(f"   Tasso successo: {success_rate:.0f}%")
            
        except Exception as e:
            self.results['processors']['AudioNormalizer'] = {'status': f'❌ ERROR: {e}', 'success_rate': 0}
        
        # Test NoiseReducer
        print("\n🔇 Test NoiseReducer...")
        try:
            noise_reducer = NoiseReducer()
            successes = 0
            
            for test_file in self.test_files[:3]:  # Test sui primi 3 file
                output_file = self.output_dir / f"denoised_{Path(test_file).name}"
                if noise_reducer.reduce_noise(test_file, str(output_file)):
                    successes += 1
            
            success_rate = (successes / 3) * 100
            self.results['processors']['NoiseReducer'] = {
                'success_rate': success_rate,
                'status': '✅ PASS' if success_rate >= 80 else '⚠️ PARTIAL' if success_rate >= 50 else '❌ FAIL'
            }
            print(f"   Tasso successo: {success_rate:.0f}%")
            
        except Exception as e:
            self.results['processors']['NoiseReducer'] = {'status': f'❌ ERROR: {e}', 'success_rate': 0}
        
        # Test FormatConverter
        print("\n🔄 Test FormatConverter...")
        try:
            converter = FormatConverter()
            successes = 0
            
            for i, test_file in enumerate(self.test_files[:2]):
                formats = ['flac', 'ogg']
                target_format = formats[i % len(formats)]
                output_file = self.output_dir / f"converted_{Path(test_file).stem}.{target_format}"
                
                if converter.convert_format(test_file, str(output_file), target_format):
                    successes += 1
            
            success_rate = (successes / 2) * 100
            self.results['processors']['FormatConverter'] = {
                'success_rate': success_rate,
                'status': '✅ PASS' if success_rate >= 80 else '⚠️ PARTIAL' if success_rate >= 50 else '❌ FAIL'
            }
            print(f"   Tasso successo: {success_rate:.0f}%")
            
        except Exception as e:
            self.results['processors']['FormatConverter'] = {'status': f'❌ ERROR: {e}', 'success_rate': 0}
        
        # Test MetadataExtractor
        print("\n📋 Test MetadataExtractor...")
        try:
            extractor = MetadataExtractor()
            successes = 0
            
            for test_file in self.test_files:
                # Prova diversi nomi di metodi possibili
                metadata = None
                methods_to_try = ['extract_audio_metadata', 'extract_metadata', 'get_metadata']
                
                for method_name in methods_to_try:
                    if hasattr(extractor, method_name):
                        method = getattr(extractor, method_name)
                        try:
                            metadata = method(test_file)
                            break
                        except:
                            continue
                
                if metadata and isinstance(metadata, dict) and len(metadata) > 0:
                    successes += 1
            
            success_rate = (successes / len(self.test_files)) * 100
            self.results['processors']['MetadataExtractor'] = {
                'success_rate': success_rate,
                'status': '✅ PASS' if success_rate >= 80 else '⚠️ PARTIAL' if success_rate >= 50 else '❌ FAIL'
            }
            print(f"   Tasso successo: {success_rate:.0f}%")
            
        except Exception as e:
            self.results['processors']['MetadataExtractor'] = {'status': f'❌ ERROR: {e}', 'success_rate': 0}
        
        # Test BatchValidator
        print("\n✔️ Test BatchValidator...")
        try:
            validator = BatchValidator()
            successes = 0
            
            for test_file in self.test_files:
                validation = validator.validate_audio_file(test_file)
                if validation and 'is_valid' in validation:
                    successes += 1
            
            success_rate = (successes / len(self.test_files)) * 100
            self.results['processors']['BatchValidator'] = {
                'success_rate': success_rate,
                'status': '✅ PASS' if success_rate >= 80 else '⚠️ PARTIAL' if success_rate >= 50 else '❌ FAIL'
            }
            print(f"   Tasso successo: {success_rate:.0f}%")
            
        except Exception as e:
            self.results['processors']['BatchValidator'] = {'status': f'❌ ERROR: {e}', 'success_rate': 0}
    
    def test_reports_modules(self):
        """Test dei moduli di reporting"""
        print("\n📊 TEST REPORTS")
        print("=" * 50)
        
        # Test QualityAnalyzer
        print("\n📈 Test QualityAnalyzer...")
        try:
            analyzer = QualityAnalyzer()
            successes = 0
            
            for test_file in self.test_files[:3]:
                quality = analyzer.analyze_audio_quality(test_file)
                if quality and 'quality_score' in quality and 'snr' in quality:
                    successes += 1
            
            success_rate = (successes / 3) * 100
            self.results['reports']['QualityAnalyzer'] = {
                'success_rate': success_rate,
                'status': '✅ PASS' if success_rate >= 80 else '⚠️ PARTIAL' if success_rate >= 50 else '❌ FAIL'
            }
            print(f"   Tasso successo: {success_rate:.0f}%")
            
        except Exception as e:
            self.results['reports']['QualityAnalyzer'] = {'status': f'❌ ERROR: {e}', 'success_rate': 0}
        
        # Test ProcessingStats
        print("\n📊 Test ProcessingStats...")
        try:
            stats = ProcessingStats()
            
            # Simula batch processing
            stats.start_batch("test_batch", len(self.test_files))
            
            for i, test_file in enumerate(self.test_files):
                processing_time = 0.1 + (i * 0.02)
                success = i < len(self.test_files) - 1  # Simula un fallimento
                error = None if success else "Simulated error"
                
                stats.record_file_processing(test_file, success, processing_time, error)
            
            batch_results = stats.finish_batch()
            
            if batch_results and 'success_rate' in batch_results and 'throughput' in batch_results:
                self.results['reports']['ProcessingStats'] = {
                    'success_rate': 100,
                    'status': '✅ PASS'
                }
            else:
                self.results['reports']['ProcessingStats'] = {
                    'success_rate': 0,
                    'status': '❌ FAIL'
                }
            print(f"   Batch results generati correttamente")
            
        except Exception as e:
            self.results['reports']['ProcessingStats'] = {'status': f'❌ ERROR: {e}', 'success_rate': 0}
        
        # Test BatchReportGenerator
        print("\n📝 Test BatchReportGenerator...")
        try:
            generator = BatchReportGenerator(str(self.reports_dir))
            
            # Crea dati di test
            test_results = {
                'summary': {
                    'total_files': len(self.test_files),
                    'successful': len(self.test_files) - 1,
                    'failed': 1,
                    'success_rate': ((len(self.test_files) - 1) / len(self.test_files)) * 100,
                    'total_time': 5.2
                },
                'detailed_results': [
                    {
                        'file': test_file,
                        'status': 'success' if i < len(self.test_files) - 1 else 'failed',
                        'time': 0.5 + (i * 0.1),
                        'errors': [] if i < len(self.test_files) - 1 else ['Test error']
                    }
                    for i, test_file in enumerate(self.test_files)
                ]
            }
            
            report_path = generator.generate_processing_report(test_results)
            
            if Path(report_path).exists():
                self.results['reports']['BatchReportGenerator'] = {
                    'success_rate': 100,
                    'status': '✅ PASS'
                }
                print(f"   Report generato: {report_path}")
            else:
                self.results['reports']['BatchReportGenerator'] = {
                    'success_rate': 0,
                    'status': '❌ FAIL'
                }
                
        except Exception as e:
            self.results['reports']['BatchReportGenerator'] = {'status': f'❌ ERROR: {e}', 'success_rate': 0}
    
    def test_integration_workflow(self):
        """Test del workflow completo integrato"""
        print("\n🔗 TEST INTEGRAZIONE WORKFLOW")
        print("=" * 50)
        
        try:
            # 1. Validazione batch
            print("1️⃣ Validazione batch...")
            validator = BatchValidator()
            validation_report = self.reports_dir / "validation_batch.json"
            
            validation_summary = validator.validate_batch(
                str(self.test_audio_dir), 
                str(validation_report)
            )
            
            # 2. Analisi qualità batch  
            print("2️⃣ Analisi qualità batch...")
            analyzer = QualityAnalyzer()
            quality_report = self.reports_dir / "quality_batch.json"
            
            quality_summary = analyzer.batch_quality_analysis(
                str(self.test_audio_dir),
                str(quality_report)
            )
            
            # 3. Processing con statistiche
            print("3️⃣ Processing con statistiche...")
            stats = ProcessingStats()
            stats.start_batch("integration_test", len(self.test_files))
            
            normalizer = AudioNormalizer()
            processed_files = 0
            
            for test_file in self.test_files:
                start_time = time.time()
                output_file = self.output_dir / f"integration_{Path(test_file).name}"
                
                success = normalizer.normalize_audio(test_file, str(output_file))
                processing_time = time.time() - start_time
                
                stats.record_file_processing(test_file, success, processing_time)
                if success:
                    processed_files += 1
            
            batch_results = stats.finish_batch()
            
            # 4. Report finale
            print("4️⃣ Generazione report finale...")
            generator = BatchReportGenerator(str(self.reports_dir))
            final_report = generator.generate_processing_report(batch_results)
            
            # Verifica risultati
            reports_exist = [
                validation_report.exists(),
                quality_report.exists(),
                Path(final_report).exists()
            ]
            
            integration_score = (
                (processed_files / len(self.test_files)) * 0.4 +  # 40% processing success
                (sum(reports_exist) / len(reports_exist)) * 0.6     # 60% reports generation
            ) * 100
            
            self.results['integration'] = integration_score >= 70
            self.results['performance']['integration_score'] = integration_score
            
            print(f"✅ Workflow integrato completato (Score: {integration_score:.0f}%)")
            
        except Exception as e:
            print(f"❌ Errore integrazione: {e}")
            self.results['integration'] = False
            self.results['performance']['integration_score'] = 0
    
    def generate_final_report(self):
        """Genera il report finale completo"""
        print("\n📋 GENERAZIONE REPORT FINALE")
        print("=" * 50)
        
        # Calcola score complessivo
        all_scores = []
        
        # Score processors
        for result in self.results['processors'].values():
            if isinstance(result, dict) and 'success_rate' in result:
                all_scores.append(result['success_rate'])
        
        # Score reports
        for result in self.results['reports'].values():
            if isinstance(result, dict) and 'success_rate' in result:
                all_scores.append(result['success_rate'])
        
        # Score integrazione
        if 'integration_score' in self.results['performance']:
            all_scores.append(self.results['performance']['integration_score'])
        
        overall_score = sum(all_scores) / len(all_scores) if all_scores else 0
        self.results['overall_score'] = overall_score
        
        # Salva risultati dettagliati
        results_file = self.reports_dir / "complete_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"📊 Risultati dettagliati salvati in: {results_file}")
        
        return overall_score
    
    def print_final_summary(self):
        """Stampa il sommario finale"""
        print("\n" + "=" * 60)
        print("🎯 RISULTATO FINALE TEST SUITE WEEK 2")
        print("=" * 60)
        
        print(f"\n📁 Directory test: {self.test_dir}")
        print(f"🎵 File audio creati: {len(self.test_files)}")
        
        print(f"\n🧪 PROCESSORS:")
        for name, result in self.results['processors'].items():
            if isinstance(result, dict):
                print(f"   {name}: {result['status']}")
        
        print(f"\n📊 REPORTS:")
        for name, result in self.results['reports'].items():
            if isinstance(result, dict):
                print(f"   {name}: {result['status']}")
        
        integration_status = "✅ PASS" if self.results['integration'] else "❌ FAIL"
        print(f"\n🔗 INTEGRAZIONE: {integration_status}")
        
        overall_score = self.results['overall_score']
        print(f"\n📈 SCORE COMPLESSIVO: {overall_score:.0f}%")
        
        if overall_score >= 90:
            print("🏆 ECCELLENTE! Sistema completamente funzionale")
        elif overall_score >= 75:
            print("✅ BUONO! Sistema funzionale con piccoli problemi")
        elif overall_score >= 60:
            print("⚠️ DISCRETO! Sistema parzialmente funzionale")
        else:
            print("❌ PROBLEMI! Sistema richiede correzioni")
        
        print(f"\n💡 Controlla i file in {self.test_dir} per dettagli aggiuntivi")
    
    def run_complete_test(self):
        """Esegue il test suite completo"""
        print("🚀 AVVIO TEST SUITE COMPLETO WEEK 2")
        print("=" * 60)
        
        # Esegui tutti i test
        self.setup_test_environment()
        
        if self.results['setup']:
            self.generate_test_audio_files()
            
            if self.results['audio_generation']:
                self.test_processors()
                self.test_reports_modules()
                self.test_integration_workflow()
                
                # Report finale
                overall_score = self.generate_final_report()
                self.print_final_summary()
                
                return overall_score >= 60
        
        return False


if __name__ == "__main__":
    test_suite = CompleteTestSuite()
    success = test_suite.run_complete_test()
    
    if success:
        print("\n🎉 TEST SUITE COMPLETATO CON SUCCESSO!")
    else:
        print("\n⚠️ TEST SUITE COMPLETATO CON PROBLEMI!")
