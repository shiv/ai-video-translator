"""
Job queue service for async video translation processing.
Handles job scheduling, progress tracking, and WebSocket notifications.
"""

import asyncio
import logging
import time
import os
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime

from app.models.job_models import Job, JobUpdate, ProgressUpdate
# Import the original TranslationRequest dataclass from translation_service
from app.services.translation_service import TranslationService, TranslationRequest as ServiceTranslationRequest
from app.services.database_service import get_database_service
from app.services.util import get_env_var

logger = logging.getLogger(__name__)


class JobProcessor:
    """
    Handles async processing of individual translation jobs.
    """
    
    def __init__(self, translation_service: TranslationService):
        self.translation_service = translation_service
    
    async def process_job(
        self, 
        job: Job, 
        progress_callback: Optional[Callable[[ProgressUpdate], None]] = None
    ) -> bool:
        """
        Process a single translation job with progress tracking.
        
        Args:
            job: Job to process
            progress_callback: Callback for progress updates
            
        Returns:
            bool: True if successful, False if failed
        """
        
        logger.info(f"Starting processing for job {job.id}")
        start_time = time.time()
        
        try:
            # Update job status to processing
            db_service = await get_database_service()
            await db_service.update_job(job.id, JobUpdate(status="processing"))
            
            # Send progress update: Starting
            if progress_callback:
                await self._send_progress_update(
                    progress_callback, job.id, "processing", "starting", 0.0,
                    "Initializing video translation process"
                )
            
            # Prepare translation request using the original dataclass
            output_dir = os.environ.get("OUTPUT_DIRECTORY", "output/")
            output_filename = f"translated_{job.original_filename}"
            output_path = os.path.join(output_dir, output_filename)
            
            # Create the correct dataclass TranslationRequest (not the Pydantic model)
            translation_request = ServiceTranslationRequest(
                input_file=job.input_file_path,
                source_language=job.source_language,
                target_language=job.target_language,
                tts=job.tts_engine,
                stt=job.stt_engine,
                translator=job.translation_engine,
                translator_model=job.translation_model,
                stt_model=job.stt_model,
                output_directory=output_dir
            )
            
            # Send progress update: Processing stages
            stages = [
                (10.0, "Speech recognition starting"),
                (30.0, "Translation processing"),
                (60.0, "Text-to-speech synthesis"),
                (80.0, "Video assembly"),
                (95.0, "Finalizing output")
            ]
            
            # Simulate progress updates during processing
            # In a real implementation, the translation service would provide callbacks
            progress_task = asyncio.create_task(
                self._simulate_progress_updates(progress_callback, job.id, stages)
            )
            
            # Perform the actual translation
            result = await asyncio.to_thread(
                self.translation_service.translate_video, translation_request
            )
            
            # Cancel progress simulation
            progress_task.cancel()
            
            # Calculate processing time
            processing_time = int(time.time() - start_time)
            
            if result.success:
                # Get output file path from result
                output_file_path = result.video_file or result.audio_file
                
                # Get output file size
                output_file_size = 0
                if output_file_path and os.path.exists(output_file_path):
                    output_file_size = os.path.getsize(output_file_path)
                
                # Update job as completed
                job_update = JobUpdate(
                    status="completed",
                    source_language=job.source_language,  # Use job's source language if no detection
                    output_file_path=output_file_path,
                    output_file_size=output_file_size,
                    processing_time_seconds=processing_time,
                    completed_at=datetime.utcnow()
                )
                
                await db_service.update_job(job.id, job_update)
                
                # Send final progress update
                if progress_callback:
                    await self._send_progress_update(
                        progress_callback, job.id, "completed", "completed", 100.0,
                        f"Translation completed successfully in {processing_time}s"
                    )
                
                logger.info(f"Job {job.id} completed successfully in {processing_time}s")
                return True
                
            else:
                # Handle translation failure
                job_update = JobUpdate(
                    status="failed",
                    error_message=result.error_message,
                    processing_time_seconds=processing_time,
                    completed_at=datetime.utcnow()
                )
                
                await db_service.update_job(job.id, job_update)
                
                # Send failure progress update
                if progress_callback:
                    await self._send_progress_update(
                        progress_callback, job.id, "failed", "failed", 0.0,
                        f"Translation failed: {result.error_message}"
                    )
                
                logger.error(f"Job {job.id} failed: {result.error_message}")
                return False
                
        except Exception as e:
            # Handle unexpected errors
            processing_time = int(time.time() - start_time)
            error_message = f"Unexpected error during processing: {str(e)}"
            
            try:
                job_update = JobUpdate(
                    status="failed",
                    error_message=error_message,
                    processing_time_seconds=processing_time,
                    completed_at=datetime.utcnow()
                )
                
                db_service = await get_database_service()
                await db_service.update_job(job.id, job_update)
                
                # Send failure progress update
                if progress_callback:
                    await self._send_progress_update(
                        progress_callback, job.id, "failed", "error", 0.0,
                        error_message
                    )
                    
            except Exception as update_error:
                logger.error(f"Failed to update job {job.id} after error: {update_error}")
            
            logger.error(f"Job {job.id} failed with exception: {e}")
            return False
    
    async def _send_progress_update(
        self, 
        callback: Callable[[ProgressUpdate], None],
        job_id: str,
        status: str,
        stage: str,
        percentage: float,
        message: str
    ):
        """Send a progress update via callback."""
        try:
            progress_update = ProgressUpdate(
                job_id=job_id,
                status=status,
                stage=stage,
                percentage=percentage,
                message=message
            )
            
            if asyncio.iscoroutinefunction(callback):
                await callback(progress_update)
            else:
                callback(progress_update)
                
        except Exception as e:
            logger.error(f"Failed to send progress update for job {job_id}: {e}")
    
    async def _simulate_progress_updates(
        self,
        callback: Optional[Callable[[ProgressUpdate], None]],
        job_id: str,
        stages: List[tuple]
    ):
        """Simulate progress updates during processing."""
        if not callback:
            return
            
        try:
            for percentage, message in stages:
                await self._send_progress_update(
                    callback, job_id, "processing", "processing", percentage, message
                )
                await asyncio.sleep(2)  # Delay between updates
                
        except asyncio.CancelledError:
            # Progress simulation was cancelled (processing completed)
            pass
        except Exception as e:
            logger.error(f"Error in progress simulation for job {job_id}: {e}")


class JobQueueService:
    """
    In-memory job queue service for managing async video translation jobs.
    """
    
    def __init__(self, max_concurrent_jobs: Optional[int] = None):
        # Use environment variable or default to 2
        if max_concurrent_jobs is None:
            max_concurrent_jobs = get_env_var("MAX_CONCURRENT_JOBS", 2, int)
        
        self.max_concurrent_jobs = max_concurrent_jobs
        self.job_queue: asyncio.Queue = asyncio.Queue()
        self.active_jobs: Dict[str, asyncio.Task] = {}
        self.progress_callbacks: Dict[str, List[Callable[[ProgressUpdate], None]]] = {}
        self.processor: Optional[JobProcessor] = None
        self._queue_task: Optional[asyncio.Task] = None
        self._semaphore = asyncio.Semaphore(max_concurrent_jobs)
        
    def initialize(self, translation_service: TranslationService):
        """Initialize the job queue service with translation service."""
        self.processor = JobProcessor(translation_service)
        self._queue_task = asyncio.create_task(self._process_queue())
        logger.info(f"Job queue service initialized with max {self.max_concurrent_jobs} concurrent jobs")
    
    async def submit_job(self, job: Job) -> str:
        """Submit a job for async processing."""
        logger.info(f"Submitting job {job.id} to queue")
        await self.job_queue.put(job)
        return job.id
    
    def add_progress_callback(self, job_id: str, callback: Callable[[ProgressUpdate], None]):
        """Add a progress callback for a specific job."""
        if job_id not in self.progress_callbacks:
            self.progress_callbacks[job_id] = []
        self.progress_callbacks[job_id].append(callback)
        logger.info(f"Added progress callback for job {job_id}")
    
    def remove_progress_callbacks(self, job_id: str):
        """Remove all progress callbacks for a job."""
        if job_id in self.progress_callbacks:
            del self.progress_callbacks[job_id]
            logger.info(f"Removed progress callbacks for job {job_id}")
    
    async def _process_queue(self):
        """Main queue processing loop."""
        logger.info("Job queue processor started")
        
        while True:
            try:
                # Get next job from queue
                job = await self.job_queue.get()
                
                # Wait for available slot
                await self._semaphore.acquire()
                
                # Create processing task
                task = asyncio.create_task(self._process_job_with_cleanup(job))
                self.active_jobs[job.id] = task
                
                logger.info(f"Started processing job {job.id} ({len(self.active_jobs)} active)")
                
            except Exception as e:
                logger.error(f"Error in queue processing: {e}")
                await asyncio.sleep(1)  # Brief pause before continuing
    
    async def _process_job_with_cleanup(self, job: Job):
        """Process a job and handle cleanup."""
        try:
            # Get progress callback for this job
            progress_callback = None
            if job.id in self.progress_callbacks:
                # Combine all callbacks for this job
                callbacks = self.progress_callbacks[job.id]
                progress_callback = lambda update: self._broadcast_progress(callbacks, update)
            
            # Process the job
            success = await self.processor.process_job(job, progress_callback)
            
            logger.info(f"Job {job.id} processing {'succeeded' if success else 'failed'}")
            
        except Exception as e:
            logger.error(f"Unexpected error processing job {job.id}: {e}")
            
        finally:
            # Cleanup
            if job.id in self.active_jobs:
                del self.active_jobs[job.id]
            
            # Remove progress callbacks
            self.remove_progress_callbacks(job.id)
            
            # Release semaphore
            self._semaphore.release()
            
            logger.info(f"Cleaned up job {job.id} ({len(self.active_jobs)} active)")
    
    def _broadcast_progress(self, callbacks: List[Callable], update: ProgressUpdate):
        """Broadcast progress update to all callbacks."""
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(update))
                else:
                    callback(update)
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        return {
            "queue_size": self.job_queue.qsize(),
            "active_jobs": len(self.active_jobs),
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "active_job_ids": list(self.active_jobs.keys())
        }
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job if it's currently active."""
        if job_id in self.active_jobs:
            task = self.active_jobs[job_id]
            task.cancel()
            
            # Update job status in database
            try:
                db_service = await get_database_service()
                await db_service.update_job(
                    job_id, 
                    JobUpdate(status="cancelled", completed_at=datetime.utcnow())
                )
            except Exception as e:
                logger.error(f"Failed to update cancelled job {job_id}: {e}")
            
            logger.info(f"Cancelled job {job_id}")
            return True
        
        return False
    
    async def shutdown(self):
        """Shutdown the job queue service."""
        logger.info("Shutting down job queue service...")
        
        # Cancel the queue processing task
        if self._queue_task:
            self._queue_task.cancel()
        
        # Cancel all active jobs
        for job_id, task in self.active_jobs.items():
            task.cancel()
            logger.info(f"Cancelled active job {job_id}")
        
        # Wait for all active jobs to complete
        if self.active_jobs:
            await asyncio.gather(*self.active_jobs.values(), return_exceptions=True)
        
        logger.info("Job queue service shut down")


# Global job queue service instance
_job_queue_service: Optional[JobQueueService] = None


def get_job_queue_service() -> JobQueueService:
    """Get the global job queue service instance."""
    global _job_queue_service
    
    if _job_queue_service is None:
        _job_queue_service = JobQueueService()
    
    return _job_queue_service


async def shutdown_job_queue_service():
    """Shutdown the global job queue service."""
    global _job_queue_service
    
    if _job_queue_service:
        await _job_queue_service.shutdown()
        _job_queue_service = None 