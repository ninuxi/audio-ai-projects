"""
Audio AI Projects - Advanced Audio Processing System
Copyright (c) 2025 Antonino Mainenti (ninuxi)
Licensed under MIT License - see LICENSE file
GitHub: https://github.com/ninuxi/audio-ai-projects
"""
"""
Database Schema Manager
======================

Manages database connections, schemas, and operations for cultural heritage data.
Supports both SQL and NoSQL storage patterns.
"""

import asyncio
import logging
import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass, asdict
import uuid

try:
    import asyncpg
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    
try:
    import pymongo
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False

@dataclass
class CulturalHeritageItem:
    """Data class for cultural heritage items"""
    item_id: str
    title: str
    institution: str
    collection: str
    item_type: str  # audio, video, document, etc.
    created_date: datetime
    modified_date: datetime
    metadata: Dict[str, Any]
    cultural_context: Dict[str, Any]
    preservation_status: str
    access_level: str
    file_info: Dict[str, Any]

class DatabaseManager:
    """
    Comprehensive database manager for cultural heritage digitization.
    
    Supports multiple database backends:
    - SQLite for small to medium collections
    - PostgreSQL for large institutional deployments
    - MongoDB for flexible metadata storage
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize database manager"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Database configuration
        self.db_type = self.config.get('db_type', 'sqlite')
        self.connection_string = self.config.get('connection_string', 'heritage_collection.db')
        
        # Connection pools
        self.sqlite_conn = None
        self.postgres_pool = None
        self.mongo_client = None
        self.mongo_db = None
        
        # Schema version
        self.schema_version = "1.0.0"
        
        self.logger.info(f"DatabaseManager initialized with {self.db_type} backend")
    
    async def initialize(self):
        """Initialize database connection and schema"""
        
        try:
            if self.db_type == 'sqlite':
                await self._init_sqlite()
            elif self.db_type == 'postgresql' and POSTGRES_AVAILABLE:
                await self._init_postgresql()
            elif self.db_type == 'mongodb' and MONGO_AVAILABLE:
                await self._init_mongodb()
            else:
                # Fallback to SQLite
                self.logger.warning(f"Database type {self.db_type} not available, falling back to SQLite")
                self.db_type = 'sqlite'
                await self._init_sqlite()
            
            # Create schema if needed
            await self._ensure_schema()
            
            self.logger.info("Database initialization completed")
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            raise
    
    async def _init_sqlite(self):
        """Initialize SQLite database"""
        
        db_path = Path(self.connection_string)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.sqlite_conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.sqlite_conn.execute("PRAGMA foreign_keys = ON")
        self.sqlite_conn.execute("PRAGMA journal_mode = WAL")
        
        self.logger.info(f"SQLite database initialized: {db_path}")
    
    async def _init_postgresql(self):
        """Initialize PostgreSQL connection pool"""
        
        try:
            self.postgres_pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            self.logger.info("PostgreSQL connection pool initialized")
            
        except Exception as e:
            self.logger.error(f"PostgreSQL initialization failed: {e}")
            raise
    
    async def _init_mongodb(self):
        """Initialize MongoDB connection"""
        
        try:
            self.mongo_client = pymongo.MongoClient(self.connection_string)
            db_name = self.config.get('mongodb_database', 'cultural_heritage')
            self.mongo_db = self.mongo_client[db_name]
            
            # Test connection
            self.mongo_client.admin.command('ping')
            self.logger.info(f"MongoDB initialized: {db_name}")
            
        except Exception as e:
            self.logger.error(f"MongoDB initialization failed: {e}")
            raise
    
    async def _ensure_schema(self):
        """Ensure database schema exists"""
        
        if self.db_type == 'sqlite':
            await self._create_sqlite_schema()
        elif self.db_type == 'postgresql':
            await self._create_postgresql_schema()
        elif self.db_type == 'mongodb':
            await self._create_mongodb_collections()
    
    async def _create_sqlite_schema(self):
        """Create SQLite schema for cultural heritage data"""
        
        schema_sql = """
        -- Cultural Heritage Items table
        CREATE TABLE IF NOT EXISTS heritage_items (
            item_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            institution TEXT NOT NULL,
            collection TEXT NOT NULL,
            item_type TEXT NOT NULL,
            created_date TIMESTAMP NOT NULL,
            modified_date TIMESTAMP NOT NULL,
            preservation_status TEXT NOT NULL,
            access_level TEXT NOT NULL,
            schema_version TEXT DEFAULT '1.0.0'
        );
        
        -- Audio Files table
        CREATE TABLE IF NOT EXISTS audio_files (
            file_id TEXT PRIMARY KEY,
            item_id TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_size INTEGER,
            duration_seconds REAL,
            sample_rate INTEGER,
            bit_depth INTEGER,
            channels INTEGER,
            format TEXT,
            checksum TEXT,
            created_date TIMESTAMP NOT NULL,
            FOREIGN KEY (item_id) REFERENCES heritage_items (item_id)
        );
        
        -- Technical Metadata table
        CREATE TABLE IF NOT EXISTS technical_metadata (
            metadata_id TEXT PRIMARY KEY,
            file_id TEXT NOT NULL,
            snr_db REAL,
            thd_percent REAL,
            dynamic_range_db REAL,
            frequency_response_score REAL,
            quality_score REAL,
            quality_grade TEXT,
            restoration_needed BOOLEAN,
            analysis_date TIMESTAMP NOT NULL,
            FOREIGN KEY (file_id) REFERENCES audio_files (file_id)
        );
        
        -- Cultural Context table
        CREATE TABLE IF NOT EXISTS cultural_context (
            context_id TEXT PRIMARY KEY,
            item_id TEXT NOT NULL,
            genre TEXT,
            historical_period TEXT,
            language TEXT,
            cultural_significance TEXT,
            subject_keywords TEXT, -- JSON array
            geographic_origin TEXT,
            cultural_markers TEXT, -- JSON array
            institutional_relevance TEXT, -- JSON object
            FOREIGN KEY (item_id) REFERENCES heritage_items (item_id)
        );
        
        -- Processing History table
        CREATE TABLE IF NOT EXISTS processing_history (
            history_id TEXT PRIMARY KEY,
            item_id TEXT NOT NULL,
            process_type TEXT NOT NULL,
            process_date TIMESTAMP NOT NULL,
            parameters TEXT, -- JSON
            results TEXT, -- JSON
            status TEXT NOT NULL,
            processing_time_seconds REAL,
            FOREIGN KEY (item_id) REFERENCES heritage_items (item_id)
        );
        
        -- Preservation Actions table
        CREATE TABLE IF NOT EXISTS preservation_actions (
            action_id TEXT PRIMARY KEY,
            item_id TEXT NOT NULL,
            action_type TEXT NOT NULL,
            action_date TIMESTAMP NOT NULL,
            performed_by TEXT,
            description TEXT,
            results TEXT, -- JSON
            next_action_due DATE,
            FOREIGN KEY (item_id) REFERENCES heritage_items (item_id)
        );
        
        -- User Access Log table
        CREATE TABLE IF NOT EXISTS access_log (
            log_id TEXT PRIMARY KEY,
            item_id TEXT NOT NULL,
            user_id TEXT,
            access_type TEXT NOT NULL,
            access_date TIMESTAMP NOT NULL,
            ip_address TEXT,
            user_agent TEXT,
            FOREIGN KEY (item_id) REFERENCES heritage_items (item_id)
        );
        
        -- Search Index table (for full-text search)
        CREATE VIRTUAL TABLE IF NOT EXISTS search_index USING fts5(
            item_id UNINDEXED,
            title,
            description,
            keywords,
            content='heritage_items'
        );
        
        -- Create indexes for performance
        CREATE INDEX IF NOT EXISTS idx_heritage_items_institution ON heritage_items(institution);
        CREATE INDEX IF NOT EXISTS idx_heritage_items_collection ON heritage_items(collection);
        CREATE INDEX IF NOT EXISTS idx_heritage_items_type ON heritage_items(item_type);
        CREATE INDEX IF NOT EXISTS idx_heritage_items_created ON heritage_items(created_date);
        CREATE INDEX IF NOT EXISTS idx_audio_files_item_id ON audio_files(item_id);
        CREATE INDEX IF NOT EXISTS idx_technical_metadata_quality ON technical_metadata(quality_score);
        CREATE INDEX IF NOT EXISTS idx_cultural_context_genre ON cultural_context(genre);
        CREATE INDEX IF NOT EXISTS idx_processing_history_date ON processing_history(process_date);
        """
        
        # Execute schema creation
        cursor = self.sqlite_conn.cursor()
        cursor.executescript(schema_sql)
        self.sqlite_conn.commit()
        
        self.logger.info("SQLite schema created successfully")
    
    async def _create_postgresql_schema(self):
        """Create PostgreSQL schema for cultural heritage data"""
        
        schema_sql = """
        -- Create schema
        CREATE SCHEMA IF NOT EXISTS cultural_heritage;
        
        -- Heritage Items table
        CREATE TABLE IF NOT EXISTS cultural_heritage.heritage_items (
            item_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            title TEXT NOT NULL,
            institution TEXT NOT NULL,
            collection TEXT NOT NULL,
            item_type TEXT NOT NULL,
            created_date TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            modified_date TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            preservation_status TEXT NOT NULL,
            access_level TEXT NOT NULL,
            schema_version TEXT DEFAULT '1.0.0'
        );
        
        -- Audio Files table
        CREATE TABLE IF NOT EXISTS cultural_heritage.audio_files (
            file_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            item_id UUID NOT NULL,
            original_filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_size BIGINT,
            duration_seconds REAL,
            sample_rate INTEGER,
            bit_depth INTEGER,
            channels INTEGER,
            format TEXT,
            checksum TEXT,
            created_date TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            FOREIGN KEY (item_id) REFERENCES cultural_heritage.heritage_items (item_id) ON DELETE CASCADE
        );
        
        -- Technical Metadata table with JSONB
        CREATE TABLE IF NOT EXISTS cultural_heritage.technical_metadata (
            metadata_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            file_id UUID NOT NULL,
            technical_data JSONB NOT NULL,
            quality_score REAL,
            quality_grade TEXT,
            analysis_date TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            FOREIGN KEY (file_id) REFERENCES cultural_heritage.audio_files (file_id) ON DELETE CASCADE
        );
        
        -- Cultural Context table with JSONB
        CREATE TABLE IF NOT EXISTS cultural_heritage.cultural_context (
            context_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            item_id UUID NOT NULL,
            cultural_data JSONB NOT NULL,
            genre TEXT,
            historical_period TEXT,
            cultural_significance TEXT,
            FOREIGN KEY (item_id) REFERENCES cultural_heritage.heritage_items (item_id) ON DELETE CASCADE
        );
        
        -- Processing History table
        CREATE TABLE IF NOT EXISTS cultural_heritage.processing_history (
            history_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            item_id UUID NOT NULL,
            process_type TEXT NOT NULL,
            process_date TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            parameters JSONB,
            results JSONB,
            status TEXT NOT NULL,
            processing_time_seconds REAL,
            FOREIGN KEY (item_id) REFERENCES cultural_heritage.heritage_items (item_id) ON DELETE CASCADE
        );
        
        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_heritage_items_institution ON cultural_heritage.heritage_items(institution);
        CREATE INDEX IF NOT EXISTS idx_heritage_items_collection ON cultural_heritage.heritage_items(collection);
        CREATE INDEX IF NOT EXISTS idx_technical_metadata_quality ON cultural_heritage.technical_metadata(quality_score);
        CREATE INDEX IF NOT EXISTS idx_cultural_context_genre ON cultural_heritage.cultural_context(genre);
        CREATE INDEX IF NOT EXISTS idx_cultural_data_gin ON cultural_heritage.cultural_context USING GIN (cultural_data);
        CREATE INDEX IF NOT EXISTS idx_technical_data_gin ON cultural_heritage.technical_metadata USING GIN (technical_data);
        """
        
        async with self.postgres_pool.acquire() as conn:
            await conn.execute(schema_sql)
        
        self.logger.info("PostgreSQL schema created successfully")
    
    async def _create_mongodb_collections(self):
        """Create MongoDB collections and indexes"""
        
        # Collections will be created automatically, but we can set up indexes
        collections = {
            'heritage_items': [
                [('institution', 1), ('collection', 1)],
                [('item_type', 1)],
                [('created_date', -1)],
                [('title', 'text'), ('description', 'text')]
            ],
            'audio_files': [
                [('item_id', 1)],
                [('quality_score', -1)],
                [('created_date', -1)]
            ],
            'cultural_context': [
                [('item_id', 1)],
                [('genre', 1)],
                [('historical_period', 1)],
                [('cultural_significance', 1)]
            ],
            'processing_history': [
                [('item_id', 1)],
                [('process_date', -1)],
                [('process_type', 1)]
            ]
        }
        
        for collection_name, indexes in collections.items():
            collection = self.mongo_db[collection_name]
            
            for index_spec in indexes:
                try:
                    await collection.create_index(index_spec)
                except Exception as e:
                    self.logger.warning(f"Failed to create index {index_spec} on {collection_name}: {e}")
        
        self.logger.info("MongoDB collections and indexes created successfully")
    
    async def store_item(self, item_id: str, metadata: Dict[str, Any]) -> bool:
        """Store a cultural heritage item with its metadata"""
        
        try:
            if self.db_type == 'sqlite':
                return await self._store_item_sqlite(item_id, metadata)
            elif self.db_type == 'postgresql':
                return await self._store_item_postgresql(item_id, metadata)
            elif self.db_type == 'mongodb':
                return await self._store_item_mongodb(item_id, metadata)
            
        except Exception as e:
            self.logger.error(f"Failed to store item {item_id}: {e}")
            return False
    
    async def _store_item_sqlite(self, item_id: str, metadata: Dict[str, Any]) -> bool:
        """Store item in SQLite database"""
        
        cursor = self.sqlite_conn.cursor()
        
        try:
            # Extract main item information
            item_data = {
                'item_id': item_id,
                'title': metadata.get('file_metadata', {}).get('filename', 'Unknown'),
                'institution': metadata.get('institution', 'Unknown'),
                'collection': metadata.get('collection', 'Default'),
                'item_type': 'audio',
                'created_date': datetime.now(),
                'modified_date': datetime.now(),
                'preservation_status': metadata.get('preservation_priority', 'medium'),
                'access_level': metadata.get('access_level', 'restricted')
            }
            
            # Insert heritage item
            cursor.execute("""
                INSERT OR REPLACE INTO heritage_items 
                (item_id, title, institution, collection, item_type, created_date, 
                 modified_date, preservation_status, access_level)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                item_data['item_id'], item_data['title'], item_data['institution'],
                item_data['collection'], item_data['item_type'], item_data['created_date'],
                item_data['modified_date'], item_data['preservation_status'], item_data['access_level']
            ))
            
            # Insert audio file information if available
            if 'file_metadata' in metadata:
                file_metadata = metadata['file_metadata']
                file_id = str(uuid.uuid4())
                
                cursor.execute("""
                    INSERT OR REPLACE INTO audio_files
                    (file_id, item_id, original_filename, file_path, file_size,
                     duration_seconds, sample_rate, bit_depth, channels, format, created_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    file_id, item_id, 
                    file_metadata.get('filename', ''),
                    file_metadata.get('file_path', ''),
                    file_metadata.get('file_size_bytes', 0),
                    metadata.get('audio_metadata', {}).get('duration_seconds', 0),
                    metadata.get('audio_metadata', {}).get('sample_rate', 0),
                    16,  # Default bit depth
                    metadata.get('audio_metadata', {}).get('channels', 1),
                    file_metadata.get('file_extension', ''),
                    datetime.now()
                ))
                
                # Insert technical metadata if available
                if 'quality_assessment' in metadata:
                    quality_data = metadata['quality_assessment']
                    
                    cursor.execute("""
                        INSERT OR REPLACE INTO technical_metadata
                        (metadata_id, file_id, snr_db, thd_percent, dynamic_range_db,
                         frequency_response_score, quality_score, quality_grade,
                         restoration_needed, analysis_date)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        str(uuid.uuid4()), file_id,
                        quality_data.get('snr_db', 0),
                        quality_data.get('thd_plus_n_percent', 0),
                        quality_data.get('dynamic_range_db', 0),
                        quality_data.get('frequency_response_score', 0),
                        quality_data.get('overall_quality_score', 0),
                        quality_data.get('quality_grade', 'Unknown'),
                        quality_data.get('restoration_needed', False),
                        datetime.now()
                    ))
            
            # Insert cultural context if available
            if 'genre' in metadata or 'historical_period' in metadata:
                context_id = str(uuid.uuid4())
                
                cursor.execute("""
                    INSERT OR REPLACE INTO cultural_context
                    (context_id, item_id, genre, historical_period, language,
                     cultural_significance, subject_keywords, geographic_origin)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    context_id, item_id,
                    metadata.get('genre', {}).get('primary_genre', ''),
                    metadata.get('historical_period', {}).get('period', ''),
                    metadata.get('language', {}).get('primary_language', ''),
                    metadata.get('cultural_heritage', {}).get('cultural_significance', ''),
                    json.dumps(metadata.get('subjects', [])),
                    metadata.get('geographic_origin', '')
                ))
            
            self.sqlite_conn.commit()
            return True
            
        except Exception as e:
            self.sqlite_conn.rollback()
            self.logger.error(f"SQLite storage failed: {e}")
            return False
    
    async def _store_item_postgresql(self, item_id: str, metadata: Dict[str, Any]) -> bool:
        """Store item in PostgreSQL database"""
        
        async with self.postgres_pool.acquire() as conn:
            async with conn.transaction():
                try:
                    # Insert heritage item
                    await conn.execute("""
                        INSERT INTO cultural_heritage.heritage_items 
                        (item_id, title, institution, collection, item_type, 
                         preservation_status, access_level)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        ON CONFLICT (item_id) DO UPDATE SET
                        modified_date = NOW(),
                        preservation_status = EXCLUDED.preservation_status,
                        access_level = EXCLUDED.access_level
                    """, 
                        uuid.UUID(item_id) if len(item_id) == 36 else uuid.uuid4(),
                        metadata.get('file_metadata', {}).get('filename', 'Unknown'),
                        metadata.get('institution', 'Unknown'),
                        metadata.get('collection', 'Default'),
                        'audio',
                        metadata.get('preservation_priority', 'medium'),
                        metadata.get('access_level', 'restricted')
                    )
                    
                    # Insert technical metadata as JSONB
                    if 'quality_assessment' in metadata:
                        await conn.execute("""
                            INSERT INTO cultural_heritage.technical_metadata
                            (file_id, technical_data, quality_score, quality_grade)
                            VALUES (gen_random_uuid(), $1, $2, $3)
                        """,
                            json.dumps(metadata['quality_assessment']),
                            metadata['quality_assessment'].get('overall_quality_score', 0),
                            metadata['quality_assessment'].get('quality_grade', 'Unknown')
                        )
                    
                    return True
                    
                except Exception as e:
                    self.logger.error(f"PostgreSQL storage failed: {e}")
                    return False
    
    async def _store_item_mongodb(self, item_id: str, metadata: Dict[str, Any]) -> bool:
        """Store item in MongoDB"""
        
        try:
            # Prepare document
            document = {
                '_id': item_id,
                'title': metadata.get('file_metadata', {}).get('filename', 'Unknown'),
                'institution': metadata.get('institution', 'Unknown'),
                'collection': metadata.get('collection', 'Default'),
                'item_type': 'audio',
                'created_date': datetime.now(),
                'modified_date': datetime.now(),
                'preservation_status': metadata.get('preservation_priority', 'medium'),
                'access_level': metadata.get('access_level', 'restricted'),
                'metadata': metadata,
                'schema_version': self.schema_version
            }
            
            # Upsert document
            result = await self.mongo_db.heritage_items.replace_one(
                {'_id': item_id},
                document,
                upsert=True
            )
            
            return result.acknowledged
            
        except Exception as e:
            self.logger.error(f"MongoDB storage failed: {e}")
            return False
    
    async def get_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a cultural heritage item by ID"""
        
        try:
            if self.db_type == 'sqlite':
                return await self._get_item_sqlite(item_id)
            elif self.db_type == 'postgresql':
                return await self._get_item_postgresql(item_id)
            elif self.db_type == 'mongodb':
                return await self._get_item_mongodb(item_id)
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve item {item_id}: {e}")
            return None
    
    async def _get_item_sqlite(self, item_id: str) -> Optional[Dict[str, Any]]:
        """Get item from SQLite database"""
        
        cursor = self.sqlite_conn.cursor()
        
        # Get main item data
        cursor.execute("""
            SELECT * FROM heritage_items WHERE item_id = ?
        """, (item_id,))
        
        item_row = cursor.fetchone()
        if not item_row:
            return None
        
        # Get column names
        columns = [description[0] for description in cursor.description]
        item_data = dict(zip(columns, item_row))
        
        # Get technical metadata
        cursor.execute("""
            SELECT tm.* FROM technical_metadata tm
            JOIN audio_files af ON tm.file_id = af.file_id
            WHERE af.item_id = ?
        """, (item_id,))
        
        tech_row = cursor.fetchone()
        if tech_row:
            tech_columns = [description[0] for description in cursor.description]
            item_data['technical_metadata'] = dict(zip(tech_columns, tech_row))
        
        # Get cultural context
        cursor.execute("""
            SELECT * FROM cultural_context WHERE item_id = ?
        """, (item_id,))
        
        context_row = cursor.fetchone()
        if context_row:
            context_columns = [description[0] for description in cursor.description]
            context_data = dict(zip(context_columns, context_row))
            
            # Parse JSON fields
            if context_data.get('subject_keywords'):
                context_data['subject_keywords'] = json.loads(context_data['subject_keywords'])
            
            item_data['cultural_context'] = context_data
        
        return item_data
    
    async def _get_item_postgresql(self, item_id: str) -> Optional[Dict[str, Any]]:
        """Get item from PostgreSQL database"""
        
        async with self.postgres_pool.acquire() as conn:
            # Get main item
            row = await conn.fetchrow("""
                SELECT * FROM cultural_heritage.heritage_items 
                WHERE item_id = $1
            """, uuid.UUID(item_id))
            
            if not row:
                return None
            
            item_data = dict(row)
            
            # Get technical metadata
            tech_rows = await conn.fetch("""
                SELECT technical_data, quality_score, quality_grade 
                FROM cultural_heritage.technical_metadata tm
                JOIN cultural_heritage.audio_files af ON tm.file_id = af.file_id
                WHERE af.item_id = $1
            """, uuid.UUID(item_id))
            
            if tech_rows:
                item_data['technical_metadata'] = dict(tech_rows[0])
            
            return item_data
    
    async def _get_item_mongodb(self, item_id: str) -> Optional[Dict[str, Any]]:
        """Get item from MongoDB"""
        
        document = await self.mongo_db.heritage_items.find_one({'_id': item_id})
        return document
    
    async def search_items(self, query: Dict[str, Any], limit: int = 100) -> List[Dict[str, Any]]:
        """Search for cultural heritage items"""
        
        try:
            if self.db_type == 'sqlite':
                return await self._search_items_sqlite(query, limit)
            elif self.db_type == 'postgresql':
                return await self._search_items_postgresql(query, limit)
            elif self.db_type == 'mongodb':
                return await self._search_items_mongodb(query, limit)
                
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []
    
    async def _search_items_sqlite(self, query: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """Search items in SQLite database"""
        
        cursor = self.sqlite_conn.cursor()
        
        # Build WHERE clause from query
        where_conditions = []
        params = []
        
        if 'institution' in query:
            where_conditions.append("institution = ?")
            params.append(query['institution'])
        
        if 'collection' in query:
            where_conditions.append("collection = ?")
            params.append(query['collection'])
        
        if 'item_type' in query:
            where_conditions.append("item_type = ?")
            params.append(query['item_type'])
        
        if 'title_contains' in query:
            where_conditions.append("title LIKE ?")
            params.append(f"%{query['title_contains']}%")
        
        where_clause = ""
        if where_conditions:
            where_clause = "WHERE " + " AND ".join(where_conditions)
        
        sql = f"""
            SELECT * FROM heritage_items 
            {where_clause}
            ORDER BY created_date DESC
            LIMIT ?
        """
        params.append(limit)
        
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row)) for row in rows]
    
    async def _search_items_postgresql(self, query: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """Search items in PostgreSQL database"""
        
        # Build dynamic query - simplified version
        base_sql = "SELECT * FROM cultural_heritage.heritage_items"
        conditions = []
        params = []
        param_count = 0
        
        for key, value in query.items():
            if key in ['institution', 'collection', 'item_type']:
                param_count += 1
                conditions.append(f"{key} = ${param_count}")
                params.append(value)
        
        if conditions:
            sql = f"{base_sql} WHERE {' AND '.join(conditions)} ORDER BY created_date DESC LIMIT ${param_count + 1}"
            params.append(limit)
        else:
            sql = f"{base_sql} ORDER BY created_date DESC LIMIT $1"
            params = [limit]
        
        async with self.postgres_pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
            return [dict(row) for row in rows]
    
    async def _search_items_mongodb(self, query: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """Search items in MongoDB"""
        
        # Build MongoDB query
        mongo_query = {}
        
        if 'institution' in query:
            mongo_query['institution'] = query['institution']
        
        if 'collection' in query:
            mongo_query['collection'] = query['collection']
        
        if 'title_contains' in query:
            mongo_query['title'] = {'$regex': query['title_contains'], '$options': 'i'}
        
        cursor = self.mongo_db.heritage_items.find(mongo_query).limit(limit).sort('created_date', -1)
        
        return await cursor.to_list(length=limit)
    
    async def get_collection_statistics(self, institution: str = None) -> Dict[str, Any]:
        """Get statistics about the collection"""
        
        try:
            if self.db_type == 'sqlite':
                return await self._get_stats_sqlite(institution)
            elif self.db_type == 'postgresql':
                return await self._get_stats_postgresql(institution)
            elif self.db_type == 'mongodb':
                return await self._get_stats_mongodb(institution)
                
        except Exception as e:
            self.logger.error(f"Failed to get statistics: {e}")
            return {}
    
    async def _get_stats_sqlite(self, institution: str = None) -> Dict[str, Any]:
        """Get collection statistics from SQLite"""
        
        cursor = self.sqlite_conn.cursor()
        
        where_clause = ""
        params = []
        
        if institution:
            where_clause = "WHERE institution = ?"
            params.append(institution)
        
        # Basic counts
        cursor.execute(f"SELECT COUNT(*) FROM heritage_items {where_clause}", params)
        total_items = cursor.fetchone()[0]
        
        cursor.execute(f"""
            SELECT item_type, COUNT(*) FROM heritage_items 
            {where_clause} 
            GROUP BY item_type
        """, params)
        items_by_type = dict(cursor.fetchall())
        
        # Quality distribution
        cursor.execute(f"""
            SELECT tm.quality_grade, COUNT(*) 
            FROM technical_metadata tm
            JOIN audio_files af ON tm.file_id = af.file_id
            JOIN heritage_items hi ON af.item_id = hi.item_id
            {where_clause}
            GROUP BY tm.quality_grade
        """, params)
        quality_distribution = dict(cursor.fetchall())
        
        return {
            'total_items': total_items,
            'items_by_type': items_by_type,
            'quality_distribution': quality_distribution,
            'institution': institution or 'All institutions'
        }
    
    async def _get_stats_postgresql(self, institution: str = None) -> Dict[str, Any]:
        """Get collection statistics from PostgreSQL"""
        
        where_clause = ""
        params = []
        
        if institution:
            where_clause = "WHERE institution = $1"
            params.append(institution)
        
        async with self.postgres_pool.acquire() as conn:
            # Total items
            total = await conn.fetchval(f"SELECT COUNT(*) FROM cultural_heritage.heritage_items {where_clause}", *params)
            
            # Items by type
            rows = await conn.fetch(f"""
                SELECT item_type, COUNT(*) as count 
                FROM cultural_heritage.heritage_items 
                {where_clause}
                GROUP BY item_type
            """, *params)
            
            items_by_type = {row['item_type']: row['count'] for row in rows}
            
            return {
                'total_items': total,
                'items_by_type': items_by_type,
                'institution': institution or 'All institutions'
            }
    
    async def _get_stats_mongodb(self, institution: str = None) -> Dict[str, Any]:
        """Get collection statistics from MongoDB"""
        
        match_stage = {}
        if institution:
            match_stage['institution'] = institution
        
        # Aggregation pipeline
        pipeline = []
        
        if match_stage:
            pipeline.append({'$match': match_stage})
        
        pipeline.extend([
            {
                '$group': {
                    '_id': '$item_type',
                    'count': {'$sum': 1}
                }
            }
        ])
        
        cursor = self.mongo_db.heritage_items.aggregate(pipeline)
        type_counts = {doc['_id']: doc['count'] async for doc in cursor}
        
        # Total count
        total_items = sum(type_counts.values())
        
        return {
            'total_items': total_items,
            'items_by_type': type_counts,
            'institution': institution or 'All institutions'
        }
    
    async def close(self):
        """Close database connections"""
        
        try:
            if self.sqlite_conn:
                self.sqlite_conn.close()
                self.logger.info("SQLite connection closed")
            
            if self.postgres_pool:
                await self.postgres_pool.close()
                self.logger.info("PostgreSQL pool closed")
            
            if self.mongo_client:
                self.mongo_client.close()
                self.logger.info("MongoDB connection closed")
                
        except Exception as e:
            self.logger.error(f"Error closing database connections: {e}")
    
    async def backup_database(self, backup_path: str) -> bool:
        """Create database backup"""
        
        try:
            if self.db_type == 'sqlite':
                return await self._backup_sqlite(backup_path)
            elif self.db_type == 'postgresql':
                return await self._backup_postgresql(backup_path)
            elif self.db_type == 'mongodb':
                return await self._backup_mongodb(backup_path)
                
        except Exception as e:
            self.logger.error(f"Backup failed: {e}")
            return False
    
    async def _backup_sqlite(self, backup_path: str) -> bool:
        """Backup SQLite database"""
        
        try:
            import shutil
            
            backup_file = Path(backup_path) / f"heritage_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            backup_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Create backup copy
            shutil.copy2(self.connection_string, backup_file)
            
            self.logger.info(f"SQLite backup created: {backup_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"SQLite backup failed: {e}")
            return False
    
    async def _backup_postgresql(self, backup_path: str) -> bool:
        """Backup PostgreSQL database"""
        
        try:
            import subprocess
            
            backup_file = Path(backup_path) / f"heritage_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sql"
            backup_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Use pg_dump command
            cmd = [
                'pg_dump',
                self.connection_string,
                '-f', str(backup_file),
                '--schema=cultural_heritage'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"PostgreSQL backup created: {backup_file}")
                return True
            else:
                self.logger.error(f"pg_dump failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"PostgreSQL backup failed: {e}")
            return False
    
    async def _backup_mongodb(self, backup_path: str) -> bool:
        """Backup MongoDB database"""
        
        try:
            import subprocess
            
            backup_dir = Path(backup_path) / f"heritage_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Use mongodump command
            db_name = self.config.get('mongodb_database', 'cultural_heritage')
            
            cmd = [
                'mongodump',
                '--uri', self.connection_string,
                '--db', db_name,
                '--out', str(backup_dir)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"MongoDB backup created: {backup_dir}")
                return True
            else:
                self.logger.error(f"mongodump failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"MongoDB backup failed: {e}")
            return False
    
    async def migrate_schema(self, target_version: str) -> bool:
        """Migrate database schema to target version"""
        
        try:
            current_version = await self._get_schema_version()
            
            if current_version == target_version:
                self.logger.info(f"Schema already at version {target_version}")
                return True
            
            migration_path = self._get_migration_path(current_version, target_version)
            
            if not migration_path:
                self.logger.error(f"No migration path from {current_version} to {target_version}")
                return False
            
            for migration in migration_path:
                success = await self._execute_migration(migration)
                if not success:
                    self.logger.error(f"Migration {migration} failed")
                    return False
            
            await self._set_schema_version(target_version)
            self.logger.info(f"Schema migrated to version {target_version}")
            return True
            
        except Exception as e:
            self.logger.error(f"Schema migration failed: {e}")
            return False
    
    async def _get_schema_version(self) -> str:
        """Get current schema version"""
        
        try:
            if self.db_type == 'sqlite':
                cursor = self.sqlite_conn.cursor()
                cursor.execute("SELECT schema_version FROM heritage_items LIMIT 1")
                row = cursor.fetchone()
                return row[0] if row else "0.0.0"
                
            elif self.db_type == 'postgresql':
                async with self.postgres_pool.acquire() as conn:
                    row = await conn.fetchrow("SELECT schema_version FROM cultural_heritage.heritage_items LIMIT 1")
                    return row['schema_version'] if row else "0.0.0"
                    
            elif self.db_type == 'mongodb':
                doc = await self.mongo_db.heritage_items.find_one({}, {'schema_version': 1})
                return doc.get('schema_version', '0.0.0') if doc else "0.0.0"
                
        except Exception:
            return "0.0.0"
    
    async def _set_schema_version(self, version: str) -> bool:
        """Set schema version"""
        
        try:
            if self.db_type == 'sqlite':
                cursor = self.sqlite_conn.cursor()
                cursor.execute("UPDATE heritage_items SET schema_version = ?", (version,))
                self.sqlite_conn.commit()
                
            elif self.db_type == 'postgresql':
                async with self.postgres_pool.acquire() as conn:
                    await conn.execute("UPDATE cultural_heritage.heritage_items SET schema_version = $1", version)
                    
            elif self.db_type == 'mongodb':
                await self.mongo_db.heritage_items.update_many({}, {'$set': {'schema_version': version}})
            
            self.schema_version = version
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set schema version: {e}")
            return False
    
    def _get_migration_path(self, from_version: str, to_version: str) -> List[str]:
        """Get migration path between versions"""
        
        # Define available migrations
        migrations = {
            "0.0.0": ["1.0.0"],
            "1.0.0": ["1.1.0"],
            "1.1.0": ["2.0.0"]
        }
        
        # Simple linear migration path
        path = []
        current = from_version
        
        while current != to_version and current in migrations:
            next_versions = migrations[current]
            if to_version in next_versions:
                path.append(f"migrate_{current}_to_{to_version}")
                break
            else:
                # Take first available migration
                next_version = next_versions[0]
                path.append(f"migrate_{current}_to_{next_version}")
                current = next_version
        
        return path if current == to_version or to_version in migrations.get(current, []) else []
    
    async def _execute_migration(self, migration_name: str) -> bool:
        """Execute a specific migration"""
        
        try:
            migration_method = getattr(self, f"_{migration_name}", None)
            if migration_method:
                return await migration_method()
            else:
                self.logger.warning(f"Migration method {migration_name} not found")
                return True  # Assume success if method doesn't exist
                
        except Exception as e:
            self.logger.error(f"Migration {migration_name} failed: {e}")
            return False
    
    async def _migrate_0_0_0_to_1_0_0(self) -> bool:
        """Migration from version 0.0.0 to 1.0.0"""
        
        try:
            if self.db_type == 'sqlite':
                cursor = self.sqlite_conn.cursor()
                
                # Add new columns if they don't exist
                cursor.execute("""
                    ALTER TABLE heritage_items 
                    ADD COLUMN schema_version TEXT DEFAULT '1.0.0'
                """)
                
                self.sqlite_conn.commit()
                
            elif self.db_type == 'postgresql':
                async with self.postgres_pool.acquire() as conn:
                    await conn.execute("""
                        ALTER TABLE cultural_heritage.heritage_items 
                        ADD COLUMN IF NOT EXISTS schema_version TEXT DEFAULT '1.0.0'
                    """)
            
            # MongoDB doesn't need schema changes for this migration
            
            self.logger.info("Migration 0.0.0 to 1.0.0 completed")
            return True
            
        except Exception as e:
            # Column might already exist
            self.logger.warning(f"Migration 0.0.0 to 1.0.0 warning: {e}")
            return True  # Continue anyway
    
    async def optimize_database(self) -> bool:
        """Optimize database performance"""
        
        try:
            if self.db_type == 'sqlite':
                return await self._optimize_sqlite()
            elif self.db_type == 'postgresql':
                return await self._optimize_postgresql()
            elif self.db_type == 'mongodb':
                return await self._optimize_mongodb()
                
        except Exception as e:
            self.logger.error(f"Database optimization failed: {e}")
            return False
    
    async def _optimize_sqlite(self) -> bool:
        """Optimize SQLite database"""
        
        try:
            cursor = self.sqlite_conn.cursor()
            
            # Analyze tables for query optimization
            cursor.execute("ANALYZE")
            
            # Vacuum to reclaim space
            cursor.execute("VACUUM")
            
            # Update statistics
            cursor.execute("PRAGMA optimize")
            
            self.sqlite_conn.commit()
            
            self.logger.info("SQLite database optimized")
            return True
            
        except Exception as e:
            self.logger.error(f"SQLite optimization failed: {e}")
            return False
    
    async def _optimize_postgresql(self) -> bool:
        """Optimize PostgreSQL database"""
        
        try:
            async with self.postgres_pool.acquire() as conn:
                # Analyze tables
                await conn.execute("ANALYZE cultural_heritage.heritage_items")
                await conn.execute("ANALYZE cultural_heritage.technical_metadata")
                await conn.execute("ANALYZE cultural_heritage.cultural_context")
                
                # Vacuum if needed (requires appropriate permissions)
                try:
                    await conn.execute("VACUUM ANALYZE cultural_heritage.heritage_items")
                except Exception as e:
                    self.logger.warning(f"VACUUM failed (normal if no permissions): {e}")
            
            self.logger.info("PostgreSQL database optimized")
            return True
            
        except Exception as e:
            self.logger.error(f"PostgreSQL optimization failed: {e}")
            return False
    
    async def _optimize_mongodb(self) -> bool:
        """Optimize MongoDB database"""
        
        try:
            # Compact collections to reclaim space
            collections = ['heritage_items', 'audio_files', 'cultural_context', 'processing_history']
            
            for collection_name in collections:
                try:
                    await self.mongo_db.command("compact", collection_name)
                except Exception as e:
                    self.logger.warning(f"Compact failed for {collection_name}: {e}")
            
            # Reindex collections
            for collection_name in collections:
                try:
                    await self.mongo_db[collection_name].reindex()
                except Exception as e:
                    self.logger.warning(f"Reindex failed for {collection_name}: {e}")
            
            self.logger.info("MongoDB database optimized")
            return True
            
        except Exception as e:
            self.logger.error(f"MongoDB optimization failed: {e}")
            return False
