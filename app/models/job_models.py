"""Pydantic models for job management."""

from typing import Optional, Dict, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field
import uuid


class JobStatus(BaseModel):
    """Job status enumeration."""
    UPLOADED: Literal["uploaded"] = "uploaded"
    PROCESSING: Literal["processing"] = "processing" 
    COMPLETED: Literal["completed"] = "completed"
    FAILED: Literal["failed"] = "failed"
    CANCELLED: Literal["cancelled"] = "cancelled"


class JobMetadata(BaseModel):
    """Job metadata and processing parameters."""
    file_format: Optional[str] = None
    duration_seconds: Optional[float] = None
    video_codec: Optional[str] = None
    audio_codec: Optional[str] = None
    resolution: Optional[str] = None
    fps: Optional[float] = None
    bitrate: Optional[int] = None
    
    # Translation parameters
    stt_engine: Optional[str] = None
    stt_model: Optional[str] = None
    translation_engine: Optional[str] = None
    translation_model: Optional[str] = None
    tts_engine: Optional[str] = None
    
    # Processing metadata
    progress_stage: Optional[str] = None
    progress_percentage: Optional[float] = None


class JobCreate(BaseModel):
    """Model for creating a new job."""
    original_filename: str = Field(..., description="Original filename")
    source_language: Optional[str] = Field(None, description="Source language code")
    target_language: str = Field(..., description="Target language code")
    input_file_path: str = Field(..., description="Input file path")
    input_file_size: int = Field(..., description="File size in bytes")
    
    # Translation parameters
    stt_engine: str = Field("auto", description="Speech-to-text engine")
    stt_model: str = Field("medium", description="STT model")
    translation_engine: str = Field("nllb", description="Translation engine")
    translation_model: str = Field("nllb-200-1.3B", description="Translation model")
    tts_engine: str = Field("mms", description="Text-to-speech engine")
    
    job_metadata: Optional[JobMetadata] = None


class JobUpdate(BaseModel):
    """Model for updating job fields."""
    status: Optional[str] = None
    source_language: Optional[str] = None
    output_file_path: Optional[str] = None
    output_file_size: Optional[int] = None
    processing_time_seconds: Optional[int] = None
    error_message: Optional[str] = None
    job_metadata: Optional[JobMetadata] = None
    completed_at: Optional[datetime] = None


class Job(BaseModel):
    """Complete job model."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Job ID")
    original_filename: str = Field(..., description="Original filename")
    source_language: Optional[str] = Field(None, description="Source language")
    target_language: str = Field(..., description="Target language")
    status: str = Field("uploaded", description="Job status")
    
    # File paths and sizes
    input_file_path: str = Field(..., description="Input file path")
    output_file_path: Optional[str] = Field(None, description="Output file path")
    input_file_size: int = Field(..., description="Input file size")
    output_file_size: Optional[int] = Field(None, description="Output file size")
    
    # Processing information
    processing_time_seconds: Optional[int] = Field(None, description="Processing time")
    error_message: Optional[str] = Field(None, description="Error message")
    
    # Translation parameters
    stt_engine: str = Field("auto", description="STT engine")
    stt_model: str = Field("medium", description="STT model")
    translation_engine: str = Field("nllb", description="Translation engine")
    translation_model: str = Field("nllb-200-1.3B", description="Translation model")
    tts_engine: str = Field("mms", description="TTS engine")
    
    # Metadata and timestamps
    job_metadata: Optional[Dict[str, Any]] = Field(None, description="Job metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Created at")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Updated at")
    completed_at: Optional[datetime] = Field(None, description="Completed at")
    
    class Config:
        from_attributes = True


class JobResponse(BaseModel):
    """API response model for job information."""
    job_id: str = Field(..., description="Job ID")
    status: str = Field(..., description="Job status")
    original_filename: str = Field(..., description="Original filename")
    source_language: Optional[str] = Field(None, description="Source language")
    target_language: str = Field(..., description="Target language")
    
    # Progress information
    progress_stage: Optional[str] = Field(None, description="Processing stage")
    progress_percentage: Optional[float] = Field(None, description="Progress percentage")
    
    # File information
    input_file_size: int = Field(..., description="Input file size")
    output_file_size: Optional[int] = Field(None, description="Output file size")
    
    # Processing information
    processing_time_seconds: Optional[int] = Field(None, description="Processing time")
    error_message: Optional[str] = Field(None, description="Error message")
    
    # Timestamps
    created_at: datetime = Field(..., description="Created at")
    updated_at: datetime = Field(..., description="Updated at")
    completed_at: Optional[datetime] = Field(None, description="Completed at")
    
    # URLs
    download_url: Optional[str] = Field(None, description="Download URL")
    preview_url: Optional[str] = Field(None, description="Preview URL")


class ProgressUpdate(BaseModel):
    """Model for real-time progress updates."""
    job_id: str = Field(..., description="Job ID")
    status: str = Field(..., description="Status")
    stage: str = Field(..., description="Processing stage")
    percentage: float = Field(0.0, description="Progress percentage")
    message: Optional[str] = Field(None, description="Progress message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp")
    
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion")
    error_details: Optional[str] = Field(None, description="Error details") 