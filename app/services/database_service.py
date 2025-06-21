"""
Database service for job persistence using SQLite in-memory database.
Provides async CRUD operations for job management.
"""

import json
import sqlite3
import aiosqlite
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging
from contextlib import asynccontextmanager

from app.models.job_models import Job, JobCreate, JobUpdate, JobMetadata

logger = logging.getLogger(__name__)


class DatabaseService:
    """
    Async database service for job management using SQLite in-memory database.
    
    Uses :memory: database that persists only during application lifetime.
    All data is lost when the application restarts.
    """
    
    def __init__(self, database_url: str = ":memory:"):
        self.database_url = database_url
        self._db = None
        
    async def initialize(self):
        """Initialize the database connection and create tables."""
        try:
            self._db = await aiosqlite.connect(self.database_url)
            # Enable foreign key constraints
            await self._db.execute("PRAGMA foreign_keys = ON")
            await self.create_tables()
            logger.info(f"Database initialized successfully: {self.database_url}")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    async def create_tables(self):
        """Create the jobs table and indexes."""
        
        # Jobs table as per HLD specification
        create_jobs_table = """
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            original_filename TEXT NOT NULL,
            source_language TEXT,
            target_language TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'uploaded',
            input_file_path TEXT NOT NULL,
            output_file_path TEXT,
            input_file_size INTEGER,
            output_file_size INTEGER,
            processing_time_seconds INTEGER,
            error_message TEXT,
            
            -- Translation parameters
            stt_engine TEXT DEFAULT 'auto',
            stt_model TEXT DEFAULT 'medium', 
            translation_engine TEXT DEFAULT 'nllb',
            translation_model TEXT DEFAULT 'nllb-200-1.3B',
            tts_engine TEXT DEFAULT 'mms',
            
            -- Metadata and timestamps
            job_metadata TEXT,  -- JSON string
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            completed_at TEXT,
            
            -- Status constraint
            CHECK (status IN ('uploaded', 'processing', 'completed', 'failed', 'cancelled'))
        );
        """
        
        # Create indexes for performance
        create_indexes = [
            "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);",
            "CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at);", 
            "CREATE INDEX IF NOT EXISTS idx_jobs_updated_at ON jobs(updated_at);",
            "CREATE INDEX IF NOT EXISTS idx_jobs_target_language ON jobs(target_language);"
        ]
        
        try:
            await self._db.execute(create_jobs_table)
            
            for index_sql in create_indexes:
                await self._db.execute(index_sql)
                
            await self._db.commit()
            logger.info("Database tables and indexes created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise

    async def create_job(self, job_create: JobCreate) -> Job:
        """Create a new job in the database."""
        
        job = Job(
            original_filename=job_create.original_filename,
            source_language=job_create.source_language,
            target_language=job_create.target_language,
            input_file_path=job_create.input_file_path,
            input_file_size=job_create.input_file_size,
            stt_engine=job_create.stt_engine,
            stt_model=job_create.stt_model,
            translation_engine=job_create.translation_engine,
            translation_model=job_create.translation_model,
            tts_engine=job_create.tts_engine,
            job_metadata=job_create.job_metadata.dict() if job_create.job_metadata else None
        )
        
        insert_sql = """
        INSERT INTO jobs (
            id, original_filename, source_language, target_language, status,
            input_file_path, input_file_size, stt_engine, stt_model,
            translation_engine, translation_model, tts_engine,
            job_metadata, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        try:
            job_metadata_json = json.dumps(job.job_metadata) if job.job_metadata else None
            
            await self._db.execute(insert_sql, (
                job.id, job.original_filename, job.source_language, job.target_language,
                job.status, job.input_file_path, job.input_file_size,
                job.stt_engine, job.stt_model, job.translation_engine, 
                job.translation_model, job.tts_engine,
                job_metadata_json, job.created_at.isoformat(), job.updated_at.isoformat()
            ))
            
            await self._db.commit()
            logger.info(f"Created job: {job.id}")
            return job
            
        except Exception as e:
            logger.error(f"Failed to create job: {e}")
            raise

    async def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        
        select_sql = "SELECT * FROM jobs WHERE id = ?"
        
        try:
            async with self._db.execute(select_sql, (job_id,)) as cursor:
                row = await cursor.fetchone()
                
            if row is None:
                return None
                
            return self._row_to_job(row)
            
        except Exception as e:
            logger.error(f"Failed to get job {job_id}: {e}")
            raise

    async def update_job(self, job_id: str, job_update: JobUpdate) -> Optional[Job]:
        """Update a job with new data."""
        
        # Build dynamic update query
        update_fields = []
        values = []
        
        if job_update.status is not None:
            update_fields.append("status = ?")
            values.append(job_update.status)
            
        if job_update.source_language is not None:
            update_fields.append("source_language = ?")
            values.append(job_update.source_language)
            
        if job_update.output_file_path is not None:
            update_fields.append("output_file_path = ?")
            values.append(job_update.output_file_path)
            
        if job_update.output_file_size is not None:
            update_fields.append("output_file_size = ?")
            values.append(job_update.output_file_size)
            
        if job_update.processing_time_seconds is not None:
            update_fields.append("processing_time_seconds = ?")
            values.append(job_update.processing_time_seconds)
            
        if job_update.error_message is not None:
            update_fields.append("error_message = ?")
            values.append(job_update.error_message)
            
        if job_update.job_metadata is not None:
            update_fields.append("job_metadata = ?")
            values.append(json.dumps(job_update.job_metadata.dict()))
            
        if job_update.completed_at is not None:
            update_fields.append("completed_at = ?")
            values.append(job_update.completed_at.isoformat())
        
        # Always update the updated_at timestamp
        update_fields.append("updated_at = ?")
        values.append(datetime.utcnow().isoformat())
        
        if not update_fields:
            # No fields to update
            return await self.get_job(job_id)
        
        update_sql = f"UPDATE jobs SET {', '.join(update_fields)} WHERE id = ?"
        values.append(job_id)
        
        try:
            await self._db.execute(update_sql, values)
            await self._db.commit()
            
            logger.info(f"Updated job: {job_id}")
            return await self.get_job(job_id)
            
        except Exception as e:
            logger.error(f"Failed to update job {job_id}: {e}")
            raise

    async def list_jobs(
        self, 
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Job]:
        """List jobs with optional filtering and pagination."""
        
        base_sql = "SELECT * FROM jobs"
        where_conditions = []
        values = []
        
        if status:
            where_conditions.append("status = ?")
            values.append(status)
        
        if where_conditions:
            base_sql += " WHERE " + " AND ".join(where_conditions)
        
        base_sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        values.extend([limit, offset])
        
        try:
            async with self._db.execute(base_sql, values) as cursor:
                rows = await cursor.fetchall()
            
            jobs = [self._row_to_job(row) for row in rows]
            logger.info(f"Listed {len(jobs)} jobs")
            return jobs
            
        except Exception as e:
            logger.error(f"Failed to list jobs: {e}")
            raise

    async def count_jobs(self, status: Optional[str] = None) -> int:
        """Count total jobs with optional status filter."""
        
        base_sql = "SELECT COUNT(*) FROM jobs"
        values = []
        
        if status:
            base_sql += " WHERE status = ?"
            values.append(status)
        
        try:
            async with self._db.execute(base_sql, values) as cursor:
                row = await cursor.fetchone()
            
            return row[0] if row else 0
            
        except Exception as e:
            logger.error(f"Failed to count jobs: {e}")
            raise

    async def delete_job(self, job_id: str) -> bool:
        """Delete a job by ID."""
        
        delete_sql = "DELETE FROM jobs WHERE id = ?"
        
        try:
            cursor = await self._db.execute(delete_sql, (job_id,))
            await self._db.commit()
            
            deleted = cursor.rowcount > 0
            if deleted:
                logger.info(f"Deleted job: {job_id}")
            else:
                logger.warning(f"Job not found for deletion: {job_id}")
                
            return deleted
            
        except Exception as e:
            logger.error(f"Failed to delete job {job_id}: {e}")
            raise

    def _row_to_job(self, row) -> Job:
        """Convert database row to Job model."""
        
        # Map database columns to Job fields
        job_data = {
            'id': row[0],
            'original_filename': row[1],
            'source_language': row[2],
            'target_language': row[3],
            'status': row[4],
            'input_file_path': row[5],
            'output_file_path': row[6],
            'input_file_size': row[7],
            'output_file_size': row[8],
            'processing_time_seconds': row[9],
            'error_message': row[10],
            'stt_engine': row[11],
            'stt_model': row[12],
            'translation_engine': row[13],
            'translation_model': row[14],
            'tts_engine': row[15],
            'job_metadata': json.loads(row[16]) if row[16] else None,
            'created_at': datetime.fromisoformat(row[17]),
            'updated_at': datetime.fromisoformat(row[18]),
            'completed_at': datetime.fromisoformat(row[19]) if row[19] else None
        }
        
        return Job(**job_data)

    async def health_check(self) -> Dict[str, Any]:
        """Check database health and return status information."""
        
        try:
            # Test basic connectivity
            await self._db.execute("SELECT 1")
            
            # Get job statistics
            total_jobs = await self.count_jobs()
            processing_jobs = await self.count_jobs("processing")
            completed_jobs = await self.count_jobs("completed")
            failed_jobs = await self.count_jobs("failed")
            
            return {
                "status": "healthy",
                "database_type": "sqlite_memory",
                "connection": "active",
                "statistics": {
                    "total_jobs": total_jobs,
                    "processing_jobs": processing_jobs,
                    "completed_jobs": completed_jobs,
                    "failed_jobs": failed_jobs
                }
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "database_type": "sqlite_memory", 
                "error": str(e)
            }

    async def close(self):
        """Close the database connection."""
        if self._db:
            await self._db.close()
            logger.info("Database connection closed")


# Global database service instance
_db_service: Optional[DatabaseService] = None


async def get_database_service() -> DatabaseService:
    """Get the global database service instance."""
    global _db_service
    
    if _db_service is None:
        _db_service = DatabaseService()
        await _db_service.initialize()
    
    return _db_service


async def close_database_service():
    """Close the global database service."""
    global _db_service
    
    if _db_service:
        await _db_service.close()
        _db_service = None 