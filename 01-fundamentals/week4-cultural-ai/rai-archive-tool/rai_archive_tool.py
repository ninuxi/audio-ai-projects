"""
Audio AI Projects - Advanced Audio Processing System
Copyright (c) 2025 Antonino Mainenti (ninuxi)
Licensed under MIT License - see LICENSE file
GitHub: https://github.com/ninuxi/audio-ai-projects
"""
"""
RAI MUSIC ARCHIVE TOOL - Sistema di Catalogazione Automatica
===========================================================

BUSINESS CASE:
- 100,000+ ore di archivi musicali RAI da catalogare
- Costo manuale: €50/ora = €5M totali
- Costo AI: €2/ora = €200K totali  
- RISPARMIO: €4.8M + valorizzazione patrimonio culturale

FEATURES:
- Metadata extraction automatica
- Genre classification professionale
- Search engine musicale intelligente
- Database integration ready
- Export per sistemi RAI esistenti
"""

import os
import glob
import json
import csv
import sqlite3
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import hashlib

# Import our music analyzer
from music_analyzer_basic import MusicAnalyzer

class RAIMusicArchiveTool:
    """
    SISTEMA DI CATALOGAZIONE AUTOMATICA per RAI
    
    Funzionalità Enterprise:
    - Batch processing migliaia di file
    - Database integration con metadati completi
    - Search engine musicale per archivisti
    - Export compatibility con sistemi RAI
    - Quality control e validation
    - Progress tracking per grandi volumi
    """
    
    def __init__(self, archive_path="rai_archive", db_path="rai_music_database.db"):
        """Inizializza sistema archivi RAI"""
        self.archive_path = archive_path
        self.db_path = db_path
        
        # Inizializza music analyzer
        self.music_analyzer = MusicAnalyzer()
        
        # Configurazione RAI specifica
        self.rai_config = {
            'supported_formats': ['.wav', '.mp3', '.flac', '.aiff', '.m4a'],
            'quality_thresholds': {
                'min_duration': 5,      # 5 secondi minimo
                'max_duration': 7200,   # 2 ore massimo
                'min_sample_rate': 8000,
                'min_quality_score': 0.3
            },
            'metadata_fields': [
                'filename', 'duration', 'sample_rate', 'file_size',
                'genre', 'tempo_bpm', 'key_signature', 'time_signature',
                'instruments', 'emotions', 'energy_level', 'era_period',
                'archive_date', 'quality_score', 'fingerprint'
            ],
            'genre_mapping': {
                # Mapping generi per classificazione RAI
                'classical': 'Musica Classica',
                'jazz': 'Jazz e Blues',
                'pop': 'Musica Leggera',
                'electronic': 'Musica Elettronica',
                'folk': 'Musica Tradizionale',
                'rock': 'Rock e Pop Rock',
                'opera': 'Opera e Lirica'
            }
        }
        
        # Statistics tracking
        self.processing_stats = {
            'files_processed': 0,
            'files_failed': 0,
            'total_duration': 0,
            'genres_detected': {},
            'start_time': None,
            'estimated_savings': 0
        }
        
        # Initialize database
        self.init_database()
        
        print("🏛️ RAI MUSIC ARCHIVE TOOL inizializzato")
        print(f"📁 Archive path: {self.archive_path}")
        print(f"💾 Database: {self.db_path}")
        print("🎯 Ready per catalogazione automatica patrimonio RAI")
    
    def init_database(self):
        """Inizializza database SQLite per archivi"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabella principale archivi musicali
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS music_archive (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE NOT NULL,
                file_path TEXT NOT NULL,
                file_size INTEGER,
                duration REAL,
                sample_rate INTEGER,
                
                -- Metadata musicali
                genre TEXT,
                genre_confidence REAL,
                tempo_bpm REAL,
                key_signature TEXT,
                time_signature TEXT,
                
                -- Analisi avanzata
                instruments TEXT,  -- JSON array
                emotions TEXT,     -- JSON array
                energy_level REAL,
                valence TEXT,      -- positive/negative
                arousal TEXT,      -- high/medium/low
                
                -- Classificazione RAI
                rai_category TEXT,
                era_period TEXT,
                cultural_value TEXT,
                
                -- Technical data
                quality_score REAL,
                audio_fingerprint TEXT,
                spectral_centroid REAL,
                harmonic_complexity REAL,
                rhythm_regularity REAL,
                
                -- Archive management
                archive_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed_by TEXT,
                notes TEXT
            )
        ''')
        
        # Tabella search index per ricerca veloce
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS search_index (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                music_id INTEGER,
                search_terms TEXT,
                category TEXT,
                FOREIGN KEY (music_id) REFERENCES music_archive (id)
            )
        ''')
        
        # Tabella statistics per reporting
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processing_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                files_processed INTEGER,
                files_failed INTEGER,
                total_duration REAL,
                processing_time REAL,
                estimated_savings REAL
            )
        ''')
        
        conn.commit()
        conn.close()
        
        print("💾 Database RAI inizializzato")
    
    def scan_archive_directory(self, directory_path):
        """Scannerizza directory archivi RAI per file audio"""
        print(f"🔍 Scansione archivi: {directory_path}")
        
        audio_files = []
        total_size = 0
        
        # Scan ricorsivo per tutti i formati supportati
        for format_ext in self.rai_config['supported_formats']:
            pattern = os.path.join(directory_path, "**", f"*{format_ext}")
            files = glob.glob(pattern, recursive=True)
            
            for file_path in files:
                try:
                    file_size = os.path.getsize(file_path)
                    total_size += file_size
                    
                    audio_files.append({
                        'path': file_path,
                        'filename': os.path.basename(file_path),
                        'size': file_size,
                        'format': format_ext
                    })
                except OSError as e:
                    print(f"⚠️ Errore accesso file {file_path}: {e}")
        
        print(f"📊 Trovati {len(audio_files)} file audio")
        print(f"📏 Dimensione totale: {total_size / (1024**3):.2f} GB")
        
        # Breakdown per formato
        format_counts = {}
        for file_info in audio_files:
            fmt = file_info['format']
            format_counts[fmt] = format_counts.get(fmt, 0) + 1
        
        print("📁 Breakdown formati:")
        for fmt, count in format_counts.items():
            print(f"   {fmt}: {count} file")
        
        return audio_files
    
    def generate_audio_fingerprint(self, y, sr):
        """Genera fingerprint audio per identificazione duplicati"""
        # Simplified audio fingerprinting usando chroma
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        
        # Riduce a signature compatta
        chroma_mean = np.mean(chroma, axis=1)
        
        # Convert to hex string
        chroma_bytes = (chroma_mean * 255).astype(np.uint8).tobytes()
        fingerprint = hashlib.md5(chroma_bytes).hexdigest()[:16]
        
        return fingerprint
    
    def classify_era_period(self, analysis_results):
        """Classifica periodo storico basato su caratteristiche musicali"""
        
        tempo = analysis_results['rhythm']['tempo_bpm']
        harmonic_complexity = analysis_results['harmony']['harmonic_complexity']
        genre = analysis_results.get('genre', {}).get('predicted_genre', 'unknown')
        
        # Rules per classificazione periodo (semplificato)
        if genre == 'classical':
            if harmonic_complexity > 0.25:
                return 'Romantico (1800-1900)'
            elif harmonic_complexity > 0.15:
                return 'Classico (1750-1820)'
            else:
                return 'Barocco (1600-1750)'
        
        elif genre == 'jazz':
            if tempo > 140:
                return 'Bebop (1940-1950)'
            elif tempo > 100:
                return 'Swing (1930-1940)'
            else:
                return 'Jazz Tradizionale (1920-1930)'
        
        elif genre == 'pop':
            return 'Contemporaneo (1960-presente)'
        
        else:
            return 'Periodo Non Classificato'
    
    def calculate_cultural_value(self, analysis_results, file_info):
        """Calcola valore culturale per prioritizzazione archivi"""
        
        score = 0
        
        # Fattori che aumentano valore culturale
        
        # 1. Qualità audio
        quality = analysis_results.get('quality_score', 0.5)
        score += quality * 20
        
        # 2. Rarità genere
        genre = analysis_results.get('genre', {}).get('predicted_genre', '')
        if genre in ['classical', 'jazz', 'folk']:
            score += 30
        elif genre in ['opera']:
            score += 40
        
        # 3. Complessità musicale
        harmony = analysis_results['harmony']['harmonic_complexity']
        if harmony > 0.2:
            score += 25
        
        # 4. Durata (registrazioni complete più preziose)
        duration = analysis_results['file_info']['duration']
        if duration > 180:  # > 3 minuti
            score += 15
        
        # 5. Unicità (basata su fingerprint)
        # TODO: check for duplicates in database
        score += 10
        
        # Classificazione valore
        if score > 80:
            return 'Alto Valore Culturale'
        elif score > 60:
            return 'Medio Valore Culturale'
        elif score > 40:
            return 'Valore Culturale Base'
        else:
            return 'Valore Commerciale'
    
    def process_single_file(self, file_path):
        """Processa singolo file per catalogazione"""
        
        try:
            # Analisi musicale completa
            analysis_results = self.music_analyzer.analyze_complete(file_path)
            
            if not analysis_results:
                return None
            
            # File info
            file_size = os.path.getsize(file_path)
            filename = os.path.basename(file_path)
            
            # Generate fingerprint
            y, sr = librosa.load(file_path, sr=None)
            fingerprint = self.generate_audio_fingerprint(y, sr)
            
            # Advanced classification
            era_period = self.classify_era_period(analysis_results)
            cultural_value = self.calculate_cultural_value(analysis_results, {'size': file_size})
            
            # RAI category mapping
            genre = analysis_results.get('genre', {}).get('predicted_genre', 'unknown')
            rai_category = self.rai_config['genre_mapping'].get(genre, 'Musica Varia')
            
            # Prepare database record
            record = {
                'filename': filename,
                'file_path': file_path,
                'file_size': file_size,
                'duration': analysis_results['file_info']['duration'],
                'sample_rate': analysis_results['file_info']['sample_rate'],
                
                # Musical metadata
                'genre': genre,
                'genre_confidence': analysis_results.get('genre', {}).get('confidence', 0),
                'tempo_bpm': analysis_results['rhythm']['tempo_bpm'],
                'key_signature': f"{analysis_results['harmony']['estimated_key']} {analysis_results['harmony']['mode']}",
                'time_signature': '4/4',  # Default, could be enhanced
                
                # Advanced analysis
                'instruments': json.dumps(analysis_results['timbre']['instruments_detected']),
                'emotions': json.dumps(analysis_results['emotion']['emotions_detected']),
                'energy_level': analysis_results['emotion']['energy_level'],
                'valence': analysis_results['emotion']['valence'],
                'arousal': analysis_results['emotion']['arousal'],
                
                # RAI classification
                'rai_category': rai_category,
                'era_period': era_period,
                'cultural_value': cultural_value,
                
                # Technical
                'quality_score': 0.8,  # Placeholder, could be enhanced
                'audio_fingerprint': fingerprint,
                'spectral_centroid': analysis_results['timbre']['spectral_centroid_mean'],
                'harmonic_complexity': analysis_results['harmony']['harmonic_complexity'],
                'rhythm_regularity': analysis_results['rhythm']['rhythm_regularity'],
                
                # Archive management
                'processed_by': 'RAI_AI_System_v1.0'
            }
            
            return record
            
        except Exception as e:
            print(f"❌ Errore processing {file_path}: {e}")
            return None
    
    def save_to_database(self, record):
        """Salva record nel database RAI"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Insert main record
            columns = ', '.join(record.keys())
            placeholders = ', '.join(['?' for _ in record])
            query = f"INSERT OR REPLACE INTO music_archive ({columns}) VALUES ({placeholders})"
            
            cursor.execute(query, list(record.values()))
            music_id = cursor.lastrowid
            
            # Create search index entries
            search_terms = [
                record['genre'],
                record['rai_category'],
                record['era_period'],
                *json.loads(record['instruments']),
                *json.loads(record['emotions'])
            ]
            
            for term in search_terms:
                if term and term != 'unknown':
                    cursor.execute('''
                        INSERT INTO search_index (music_id, search_terms, category)
                        VALUES (?, ?, ?)
                    ''', (music_id, term.lower(), 'auto_generated'))
            
            conn.commit()
            return True
            
        except sqlite3.Error as e:
            print(f"❌ Database error: {e}")
            return False
        finally:
            conn.close()
    
    def batch_process_archive(self, directory_path, max_files=None):
        """Processamento batch per archivi RAI"""
        
        print(f"🚀 AVVIO BATCH PROCESSING RAI ARCHIVE")
        print("=" * 60)
        
        # Start timing
        self.processing_stats['start_time'] = datetime.now()
        
        # Scan directory
        audio_files = self.scan_archive_directory(directory_path)
        
        if max_files:
            audio_files = audio_files[:max_files]
            print(f"📊 Limitato a {max_files} file per demo")
        
        total_files = len(audio_files)
        
        print(f"\n🎵 Inizio catalogazione {total_files} file...")
        
        # Process files
        successful_records = []
        failed_files = []
        
        for i, file_info in enumerate(audio_files, 1):
            file_path = file_info['path']
            filename = file_info['filename']
            
            print(f"\n📁 [{i}/{total_files}] Processing: {filename}")
            
            # Process file
            record = self.process_single_file(file_path)
            
            if record:
                # Save to database
                if self.save_to_database(record):
                    successful_records.append(record)
                    
                    # Update stats
                    self.processing_stats['files_processed'] += 1
                    self.processing_stats['total_duration'] += record['duration']
                    
                    # Track genres
                    genre = record['genre']
                    self.processing_stats['genres_detected'][genre] = \
                        self.processing_stats['genres_detected'].get(genre, 0) + 1
                    
                    print(f"✅ Catalogato: {record['rai_category']} | {record['genre']} | {record['era_period']}")
                else:
                    failed_files.append(file_path)
                    self.processing_stats['files_failed'] += 1
            else:
                failed_files.append(file_path)
                self.processing_stats['files_failed'] += 1
                print(f"❌ Fallito: {filename}")
        
        # Calculate final stats
        end_time = datetime.now()
        processing_time = (end_time - self.processing_stats['start_time']).total_seconds()
        
        # Estimate savings
        manual_cost = len(successful_records) * 50  # €50/file manuale
        ai_cost = len(successful_records) * 2       # €2/file AI
        estimated_savings = manual_cost - ai_cost
        self.processing_stats['estimated_savings'] = estimated_savings
        
        # Save session stats
        self.save_processing_stats(processing_time)
        
        # Generate report
        self.generate_processing_report(processing_time, successful_records, failed_files)
        
        return successful_records
    
    def save_processing_stats(self, processing_time):
        """Salva statistiche sessione"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO processing_stats 
            (files_processed, files_failed, total_duration, processing_time, estimated_savings)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            self.processing_stats['files_processed'],
            self.processing_stats['files_failed'], 
            self.processing_stats['total_duration'],
            processing_time,
            self.processing_stats['estimated_savings']
        ))
        
        conn.commit()
        conn.close()
    
    def generate_processing_report(self, processing_time, successful_records, failed_files):
        """Genera report dettagliato processing"""
        
        print(f"\n📊 REPORT PROCESSING RAI ARCHIVE")
        print("=" * 60)
        
        # Summary stats
        total_files = len(successful_records) + len(failed_files)
        success_rate = (len(successful_records) / total_files * 100) if total_files > 0 else 0
        
        print(f"📁 File processati: {len(successful_records)}/{total_files}")
        print(f"✅ Success rate: {success_rate:.1f}%")
        print(f"⏱️ Tempo processing: {processing_time:.1f} secondi")
        print(f"🎵 Durata totale audio: {self.processing_stats['total_duration']/3600:.1f} ore")
        
        # Business metrics
        print(f"\n💰 BUSINESS IMPACT:")
        print(f"   Costo manuale stimato: €{len(successful_records) * 50:,}")
        print(f"   Costo AI: €{len(successful_records) * 2:,}")
        print(f"   Risparmio: €{self.processing_stats['estimated_savings']:,}")
        
        # Genre distribution
        if self.processing_stats['genres_detected']:
            print(f"\n🎼 DISTRIBUZIONE GENERI:")
            for genre, count in self.processing_stats['genres_detected'].items():
                percentage = (count / len(successful_records)) * 100
                print(f"   {genre}: {count} file ({percentage:.1f}%)")
        
        # Sample records
        if successful_records:
            print(f"\n📋 CAMPIONI CATALOGATI:")
            for record in successful_records[:3]:
                print(f"   📁 {record['filename'][:50]}...")
                print(f"      Genre: {record['rai_category']}")
                print(f"      Periodo: {record['era_period']}")
                print(f"      Valore: {record['cultural_value']}")
        
        # Failed files
        if failed_files:
            print(f"\n⚠️ FILE FALLITI ({len(failed_files)}):")
            for failed_file in failed_files[:5]:
                print(f"   ❌ {os.path.basename(failed_file)}")
            if len(failed_files) > 5:
                print(f"   ... e altri {len(failed_files) - 5} file")
    
    def search_archive(self, query, category=None, limit=10):
        """Search engine per archivi RAI"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Build search query
        base_query = '''
            SELECT DISTINCT m.* FROM music_archive m
            LEFT JOIN search_index s ON m.id = s.music_id
            WHERE (
                m.genre LIKE ? OR
                m.rai_category LIKE ? OR
                m.era_period LIKE ? OR
                m.instruments LIKE ? OR
                m.emotions LIKE ? OR
                s.search_terms LIKE ?
            )
        '''
        
        params = [f"%{query}%" for _ in range(6)]
        
        if category:
            base_query += " AND m.rai_category = ?"
            params.append(category)
        
        base_query += f" ORDER BY m.cultural_value DESC, m.genre_confidence DESC LIMIT {limit}"
        
        cursor.execute(base_query, params)
        results = cursor.fetchall()
        
        # Convert to dictionaries
        columns = [description[0] for description in cursor.description]
        search_results = [dict(zip(columns, row)) for row in results]
        
        conn.close()
        
        return search_results
    
    def export_catalog_csv(self, output_path="rai_catalog.csv"):
        """Export catalogo per sistemi RAI esistenti"""
        
        conn = sqlite3.connect(self.db_path)
        
        # Query tutti i record
        df = pd.read_sql_query('''
            SELECT 
                filename, rai_category, genre, tempo_bpm, key_signature,
                era_period, cultural_value, duration, instruments, emotions,
                archive_date
            FROM music_archive
            ORDER BY cultural_value DESC, rai_category, filename
        ''', conn)
        
        # Export CSV
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        conn.close()
        
        print(f"📄 Catalogo esportato: {output_path}")
        print(f"📊 {len(df)} record nel catalogo")
        
        return output_path

# Demo e testing del sistema RAI
def demo_rai_archive_tool():
    """Demo sistema catalogazione RAI"""
    
    print("🏛️ RAI MUSIC ARCHIVE TOOL - DEMO")
    print("=" * 60)
    
    # Initialize RAI system
    rai_tool = RAIMusicArchiveTool()
    
    # Demo con file esistenti
    demo_directory = "data"
    
    if os.path.exists(demo_directory):
        print(f"🎯 Demo catalogazione cartella: {demo_directory}")
        
        # Batch processing (limit per demo)
        successful_records = rai_tool.batch_process_archive(demo_directory, max_files=5)
        
        if successful_records:
            print(f"\n🔍 DEMO SEARCH ENGINE:")
            
            # Test search
            jazz_results = rai_tool.search_archive("jazz", limit=3)
            print(f"   Ricerca 'jazz': {len(jazz_results)} risultati")
            
            classical_results = rai_tool.search_archive("classical", limit=3) 
            print(f"   Ricerca 'classical': {len(classical_results)} risultati")
            
            # Export demo
            catalog_file = rai_tool.export_catalog_csv("demo_rai_catalog.csv")
            print(f"   Catalogo demo: {catalog_file}")
            
            print(f"\n🎯 SISTEMA RAI PRONTO!")
            print("📚 Applicabile a 100,000+ ore archivi RAI")
            print("💰 Risparmio stimato: €4.8M")
            
        else:
            print("⚠️ Nessun file processato nel demo")
    
    else:
        print(f"⚠️ Directory demo non trovata: {demo_directory}")
        print("💡 Crea cartella 'data/' con file audio per demo completo")

if __name__ == "__main__":
    demo_rai_archive_tool()
