"""Pydantic models for translation requests and responses."""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime


class UploadRequest(BaseModel):
    """File upload request parameters."""
    target_language: str = Field(..., description="Target language code")
    source_language: Optional[str] = Field(None, description="Source language")
    
    # Engine configuration
    stt_engine: str = Field("auto", description="Speech-to-text engine")
    stt_model: str = Field("medium", description="Whisper model")
    translation_engine: str = Field("nllb", description="Translation engine")
    translation_model: str = Field("nllb-200-1.3B", description="Translation model")
    tts_engine: str = Field("mms", description="Text-to-speech engine")


class UploadResponse(BaseModel):
    """File upload response."""
    job_id: str = Field(..., description="Job ID for tracking")
    status: str = Field("uploaded", description="Job status")
    original_filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., description="File size in bytes")
    
    # URLs
    status_url: str = Field(..., description="Status URL")
    download_url: Optional[str] = Field(None, description="Download URL")
    websocket_url: str = Field(..., description="WebSocket URL")
    
    processing_config: Dict[str, Any] = Field(..., description="Processing configuration")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Upload timestamp")


class TranslationRequest(BaseModel):
    """Translation request for internal services."""
    source_language: Optional[str] = None
    target_language: str
    input_file_path: str
    output_file_path: str
    
    # Engine configuration
    stt_engine: str = "auto"
    stt_model: str = "medium"
    translation_engine: str = "nllb"
    translation_model: str = "nllb-200-1.3B"
    tts_engine: str = "mms"
    
    # Processing options
    device: Optional[str] = None
    cpu_threads: Optional[int] = None
    
    class Config:
        from_attributes = True


class TranslationResult(BaseModel):
    """Translation result from internal services."""
    success: bool
    output_file_path: Optional[str] = None
    error_message: Optional[str] = None
    processing_time_seconds: Optional[float] = None
    
    detected_language: Optional[str] = None
    output_file_size: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        from_attributes = True


class JobListResponse(BaseModel):
    """Job listing response."""
    jobs: List[Dict[str, Any]] = Field(..., description="List of jobs")
    total_count: int = Field(..., description="Total job count")
    page: int = Field(1, description="Current page")
    page_size: int = Field(50, description="Jobs per page")
    
    
class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID") 