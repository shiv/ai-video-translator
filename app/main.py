"""Main FastAPI application for AI Video Translation Service."""

import os
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import tempfile
import shutil

from app.api.routes import job_router, websocket_router
from app.services.database_service import get_database_service, close_database_service
from app.services.job_queue_service import get_job_queue_service, shutdown_job_queue_service
from app.services.translation_service import TranslationService
from app.services.ai_service_factory import get_ai_factory
from app.services.util import get_env_var

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Video Translation Service",
    description="""
    A comprehensive service for translating videos using AI models.
    
    **Features:**
    - Complete web frontend interface
    - Drag & drop file upload
    - Real-time progress tracking via WebSocket
    - Job status management with copyable Job IDs
    - Video preview and download capabilities
    - Responsive design for mobile and desktop
    - Complete job management API with persistence
    - Async video processing with queue system
    - Enhanced monitoring and health checks
    """,
    version="5.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

translation_service: Optional[TranslationService] = None


@app.on_event("startup")
async def startup_event():
    """Initialize all services on application startup."""
    global translation_service
    
    logger.info("Starting AI Video Translation Service...")
    
    try:
        os.makedirs("uploads", exist_ok=True)
        os.makedirs(get_env_var("OUTPUT_DIRECTORY", "output/"), exist_ok=True)
        os.makedirs("static", exist_ok=True)
        os.makedirs("templates", exist_ok=True)
        
        logger.info("Initializing database service...")
        db_service = await get_database_service()
        logger.info("Database service initialized successfully")
        
        logger.info("Initializing AI Service Factory...")
        ai_factory = get_ai_factory()
        
        try:
            ai_factory.preload_default_models()
            logger.info("Model preloading completed")
        except Exception as e:
            logger.warning(f"Model preloading failed (will load on-demand): {e}")
        
        logger.info("Initializing translation service...")
        translation_service = TranslationService()
        logger.info("Translation service initialized")
        
        logger.info("Initializing job queue service...")
        job_queue = get_job_queue_service()
        job_queue.initialize(translation_service)
        logger.info("Job queue service initialized")
        
        logger.info("✅ AI Video Translation Service started successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to start application: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on application shutdown."""
    logger.info("Shutting down AI Video Translation Service...")
    
    try:
        await shutdown_job_queue_service()
        logger.info("Job queue service shut down")
        
        await close_database_service()
        logger.info("Database service closed")
        
        try:
            ai_factory = get_ai_factory()
            ai_factory.clear_cache()
            logger.info("Model cache cleared")
        except Exception as e:
            logger.error(f"Error clearing model cache: {e}")
        
        logger.info("✅ AI Video Translation Service shut down successfully")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


app.include_router(job_router)
app.include_router(websocket_router)


@app.get("/", response_class=HTMLResponse)
async def frontend_home(request: Request):
    """Serve the main frontend interface."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/job/{job_id}", response_class=HTMLResponse)
async def frontend_job_status(request: Request, job_id: str):
    """Serve frontend with specific job ID pre-loaded."""
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "job_id": job_id
    })


@app.get("/app", response_class=HTMLResponse)
async def frontend_app(request: Request):
    """Alternative route to access the frontend application."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/status")
async def root():
    """Root endpoint with comprehensive service information."""
    return {
        "message": "AI Video Translation Service",
        "version": "5.0.0",
        "phase": "Phase 5 - Complete Frontend Interface",
        "features": [
            "✅ Complete web frontend interface",
            "✅ Drag & drop file upload with validation",
            "✅ Real-time progress tracking via WebSocket",
            "✅ Job status management with copyable Job IDs",
            "✅ Video preview and download capabilities",
            "✅ Responsive design for mobile and desktop",
            "✅ Job persistence with SQLite in-memory database",
            "✅ Async video processing with job queue",
            "✅ File upload and download capabilities",
            "✅ Model caching and preloading",
            "✅ Multi-language support (200+ languages)",
            "✅ Complete job lifecycle management",
            "✅ Enhanced monitoring and health checks"
        ],
        "frontend_urls": {
            "main_interface": "/",
            "job_tracking": "/job/{job_id}",
            "alternative_app": "/app"
        },
        "api_documentation": "/docs",
        "endpoints": {
            "upload": "POST /api/v1/upload",
            "job_status": "GET /api/v1/jobs/{job_id}/status", 
            "download": "GET /api/v1/jobs/{job_id}/download",
            "preview": "GET /api/v1/jobs/{job_id}/preview",
            "list_jobs": "GET /api/v1/jobs",
            "cancel_job": "DELETE /api/v1/jobs/{job_id}",
            "job_details": "GET /api/v1/jobs/{job_id}",
            "queue_status": "GET /api/v1/queue/status",
            "progress_websocket": "WS /api/v1/jobs/{job_id}/progress",
            "websocket_status": "GET /api/v1/websocket/status",
            "frontend_home": "GET /",
            "frontend_job": "GET /job/{job_id}",
            "health": "GET /health",
            "models": "GET /api/v1/models",
            "languages": "GET /api/v1/languages"
        },
        "websocket_support": {
            "real_time_progress": "WS /api/v1/jobs/{job_id}/progress",
            "message_types": ["progress_update", "ping", "pong", "status_response", "error"]
        },
        "frontend_features": {
            "file_upload": "Drag & drop or browse for MP4 files (max 200MB)",
            "real_time_tracking": "WebSocket-based progress updates",
            "job_management": "Copyable job IDs, status checking, cancellation",
            "download_preview": "Video preview and download capabilities",
            "responsive_design": "Mobile and desktop optimized",
            "error_handling": "User-friendly error messages and notifications"
        }
    }


@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint."""
    try:
        health_status = {
            "status": "healthy",
            "version": "5.0.0",
            "phase": "Phase 5",
            "timestamp": os.popen('date -u +"%Y-%m-%dT%H:%M:%S.%fZ"').read().strip(),
            "frontend": {
                "status": "active",
                "static_files": os.path.exists("static"),
                "templates": os.path.exists("templates"),
                "main_interface": "/"
            }
        }
        
        if translation_service:
            translation_health = translation_service.health_check()
            health_status["translation_service"] = translation_health
        else:
            health_status["translation_service"] = {"status": "not_initialized"}
        
        try:
            db_service = await get_database_service()
            db_health = await db_service.health_check()
            health_status["database_service"] = db_health
        except Exception as e:
            health_status["database_service"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        try:
            job_queue = get_job_queue_service()
            queue_status = job_queue.get_queue_status()
            health_status["job_queue_service"] = {
                "status": "healthy",
                **queue_status
            }
        except Exception as e:
            health_status["job_queue_service"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        service_statuses = [
            health_status["translation_service"].get("status"),
            health_status["database_service"].get("status"),
            health_status["job_queue_service"].get("status")
        ]
        
        if any(status == "unhealthy" for status in service_statuses):
            health_status["status"] = "unhealthy"
            status_code = 503
        elif any(status == "warning" for status in service_statuses):
            health_status["status"] = "warning"
            status_code = 200
        else:
            status_code = 200
            
        return JSONResponse(
            status_code=status_code,
            content=health_status
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": f"Health check failed: {str(e)}",
                "service": "AI Video Translation Service",
                "version": "5.0.0",
                "phase": "Phase 5"
            }
        )


@app.get("/api/v1/models")
async def get_model_status():
    """Get status information about cached models and AI services."""
    try:
        if translation_service:
            model_status = translation_service.get_model_status()
        else:
            model_status = {"error": "Translation service not initialized"}
        
        return {
            "status": "success",
            "timestamp": os.popen('date -u +"%Y-%m-%dT%H:%M:%S.%fZ"').read().strip(),
            "model_cache": model_status,
            "note": "This is a legacy Phase 2 endpoint. Use /health for comprehensive status."
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model status: {str(e)}"
        )


@app.get("/api/v1/languages")
async def get_supported_languages():
    """Get list of supported languages for translation."""
    try:
        languages = {
            "source_languages": [
                {"code": "eng", "name": "English"},
                {"code": "spa", "name": "Spanish"},
                {"code": "fra", "name": "French"},
                {"code": "deu", "name": "German"},
                {"code": "ita", "name": "Italian"},
                {"code": "por", "name": "Portuguese"},
                {"code": "jpn", "name": "Japanese"},
                {"code": "kor", "name": "Korean"},
                {"code": "cmn", "name": "Chinese (Mandarin)"},
                {"code": "hin", "name": "Hindi"}
            ],
            "target_languages": [
                {"code": "eng", "name": "English"},
                {"code": "spa", "name": "Spanish"},
                {"code": "fra", "name": "French"},
                {"code": "deu", "name": "German"},
                {"code": "ita", "name": "Italian"},
                {"code": "por", "name": "Portuguese"},
                {"code": "jpn", "name": "Japanese"},
                {"code": "kor", "name": "Korean"},
                {"code": "cmn", "name": "Chinese (Mandarin)"},
                {"code": "hin", "name": "Hindi"}
            ]
        }
        
        return {
            "status": "success",
            "languages": languages,
            "note": "This is a legacy Phase 2 endpoint. Use /docs for full API documentation."
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get supported languages: {str(e)}"
        )


@app.post("/api/v1/translate")
async def legacy_translate_video():
    """Legacy direct translation endpoint - deprecated."""
    return JSONResponse(
        status_code=410,
        content={
            "error": "Endpoint deprecated",
            "message": "Direct translation endpoint has been deprecated in Phase 5",
            "alternatives": {
                "job_based_api": "Use POST /api/v1/upload for job-based async processing",
                "frontend_interface": "Use / for complete web interface with drag & drop upload"
            },
            "documentation": "/docs",
            "migration_guide": {
                "old_flow": "POST /api/v1/translate → immediate response",
                "new_api_flow": "POST /api/v1/upload → GET /api/v1/jobs/{job_id}/status → GET /api/v1/jobs/{job_id}/download",
                "new_frontend_flow": "Visit / → upload file → track progress → download result"
            }
        }
    )


@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Custom 404 handler with helpful information."""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested endpoint was not found",
            "available_endpoints": "/docs",
            "frontend_interface": "/",
            "service": "AI Video Translation Service v5.0.0"
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Custom 500 handler with service information."""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "service": "AI Video Translation Service v5.0.0",
            "support": "Check /health endpoint for service status",
            "frontend": "Try the web interface at /"
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    host = get_env_var("HOST", "0.0.0.0")
    port = int(get_env_var("PORT", "8000"))
    
    logger.info(f"Starting Phase 5 development server on {host}:{port}")
    logger.info(f"Frontend interface will be available at: http://{host}:{port}/")
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )