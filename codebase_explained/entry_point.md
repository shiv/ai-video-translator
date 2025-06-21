# Entry Point and Initialization

This document explains how the AI Video Translation Service starts up, initializes its components, and prepares for operation. Understanding the initialization process is crucial for debugging, deployment, and system maintenance.

## 🚀 Application Entry Points

### Primary Entry Point: `app/main.py`

The main application entry point is the FastAPI application defined in `app/main.py`. This file serves as the central orchestrator for the entire service.

```python
# Main FastAPI application instance
app = FastAPI(
    title="AI Video Translation Service",
    description="A comprehensive service for translating videos using AI models",
    version="5.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)
```

### Development Server Entry Point

For development, the application can be started directly:

```python
if __name__ == "__main__":
    import uvicorn
    
    host = get_env_var("HOST", "0.0.0.0")
    port = int(get_env_var("PORT", "8000"))
    
    logger.info(f"Starting Phase 5 development server on {host}:{port}")
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
```

### Production Entry Point

For production deployment, the application is typically started via Docker or a process manager:

```bash
# Via Docker Compose
docker-compose up

# Via Uvicorn directly
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Via Gunicorn (for production)
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

## 🔧 Initialization Sequence

### 1. Application Startup Event

The initialization process begins with the FastAPI startup event handler:

```python
@app.on_event("startup")
async def startup_event():
    """Initialize all services on application startup."""
    global translation_service
    
    logger.info("Starting AI Video Translation Service...")
    
    try:
        # Step 1: Create necessary directories
        os.makedirs("uploads", exist_ok=True)
        os.makedirs(get_env_var("OUTPUT_DIRECTORY", "output/"), exist_ok=True)
        os.makedirs("static", exist_ok=True)
        os.makedirs("templates", exist_ok=True)
        
        # Step 2: Initialize database service
        logger.info("Initializing database service...")
        db_service = await get_database_service()
        logger.info("Database service initialized successfully")
        
        # Step 3: Initialize AI Service Factory
        logger.info("Initializing AI Service Factory...")
        ai_factory = get_ai_factory()
        
        # Step 4: Preload default models (optional)
        try:
            ai_factory.preload_default_models()
            logger.info("Model preloading completed")
        except Exception as e:
            logger.warning(f"Model preloading failed (will load on-demand): {e}")
        
        # Step 5: Initialize translation service
        logger.info("Initializing translation service...")
        translation_service = TranslationService()
        logger.info("Translation service initialized")
        
        # Step 6: Initialize job queue service
        logger.info("Initializing job queue service...")
        job_queue = get_job_queue_service()
        job_queue.initialize(translation_service)
        logger.info("Job queue service initialized")
        
        logger.info("✅ AI Video Translation Service started successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to start application: {e}")
        raise
```

### 2. Directory Structure Creation

The first step ensures all necessary directories exist:

```python
# Required directories for operation
directories = [
    "uploads",           # User uploaded files
    "output/",          # Processed video outputs
    "static",           # Frontend assets
    "templates",        # HTML templates
    "logs"              # Application logs (if configured)
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)
```

### 3. Database Service Initialization

The database service is initialized with automatic schema creation:

```python
async def get_database_service() -> DatabaseService:
    """Get or create database service instance."""
    global _database_service
    
    if _database_service is None:
        _database_service = DatabaseService()
        await _database_service.initialize()
    
    return _database_service

# In DatabaseService.initialize()
async def initialize(self):
    """Initialize database connection and create tables."""
    self._connection = await aiosqlite.connect(self._database_url)
    await self.create_tables()
    logger.info("Database initialized successfully")
```

### 4. AI Service Factory Initialization

The AI Service Factory is initialized as a singleton:

```python
def get_ai_factory() -> AIServiceFactory:
    """Get singleton AI Service Factory instance."""
    global _ai_factory
    
    if _ai_factory is None:
        _ai_factory = AIServiceFactory()
    
    return _ai_factory

# In AIServiceFactory.__init__()
def __init__(self):
    """Initialize the AI Service Factory."""
    self._model_cache: Dict[str, CachedModel] = {}
    self._cache_lock = threading.Lock()
    self._memory_threshold = 0.85
    self._max_cache_size = 5
    logger.info("AI Service Factory initialized")
```

### 5. Model Preloading (Optional)

Default models can be preloaded during startup to reduce first-request latency:

```python
def preload_default_models(self):
    """Preload commonly used models for faster first requests."""
    default_configs = [
        ModelConfig(ModelType.STT, "faster-whisper", "tiny"),
        ModelConfig(ModelType.TRANSLATION, "nllb", "nllb-200-distilled-600M"),
        ModelConfig(ModelType.TTS, "mms", "default")
    ]
    
    for config in default_configs:
        try:
            self.load_model(config)
            logger.info(f"Preloaded model: {config.model_type.value}/{config.model_name}")
        except Exception as e:
            logger.warning(f"Failed to preload {config.model_name}: {e}")
```

### 6. Translation Service Initialization

The translation service is initialized with logging configuration:

```python
class TranslationService:
    def __init__(self):
        """Initialize the translation service with AI Service Factory."""
        self._init_logging()
        self._ai_factory = get_ai_factory()
        logger.info("Translation service initialized")
    
    def _init_logging(self):
        """Initialize logging configuration."""
        logging.basicConfig(level=logging.ERROR)
        
        app_logger = logging.getLogger("app")
        log_level = get_env_var("LOG_LEVEL", "INFO")
        app_logger.setLevel(getattr(logging, log_level))
        
        # Configure handlers and formatters
        file_handler = logging.FileHandler("app.log")
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        app_logger.addHandler(file_handler)
        app_logger.addHandler(console_handler)
```

### 7. Job Queue Service Initialization

The job queue service is initialized with the translation service:

```python
def initialize(self, translation_service: TranslationService):
    """Initialize job queue with translation service."""
    self._translation_service = translation_service
    self._processor = JobProcessor(translation_service)
    
    # Start the queue processing task
    self._queue_task = asyncio.create_task(self._process_queue())
    logger.info("Job queue service initialized and processing started")
```

## 🔧 Configuration Management

### Environment Variables

The application uses environment variables for configuration:

```python
# Core configuration
HOST = get_env_var("HOST", "0.0.0.0")
PORT = get_env_var("PORT", "8000", int)
LOG_LEVEL = get_env_var("LOG_LEVEL", "INFO")

# Processing configuration
OUTPUT_DIRECTORY = get_env_var("OUTPUT_DIRECTORY", "output/")
MAX_FILE_SIZE_MB = get_env_var("MAX_FILE_SIZE_MB", 200, int)
DEVICE = get_env_var("DEVICE", "cpu")
CPU_THREADS = get_env_var("CPU_THREADS", 0, int)

# AI model configuration
HUGGING_FACE_TOKEN = get_env_var("HUGGING_FACE_TOKEN", required=True)

# Feature flags
CLEAN_INTERMEDIATE_FILES = get_env_var("CLEAN_INTERMEDIATE_FILES", False, bool)
VAD = get_env_var("VAD", False, bool)
```

### Configuration Validation

Configuration is validated during startup:

```python
def _get_hugging_face_token(self) -> str:
    """Get Hugging Face token from environment variable."""
    token = os.getenv("HUGGING_FACE_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        raise ConfigurationError(
            "Hugging Face token must be set via 'HUGGING_FACE_TOKEN' or 'HF_TOKEN' environment variable."
        )
    return token

def _check_ffmpeg(self) -> None:
    """Check if FFmpeg is installed."""
    if not FFmpeg.is_ffmpeg_installed():
        raise MissingDependencyError("FFmpeg (which includes ffprobe) must be installed.")
```

## 🌐 Middleware and Route Registration

### CORS Middleware

Cross-Origin Resource Sharing is configured for web frontend access:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Static Files and Templates

Static file serving and template engine setup:

```python
# Static file serving
app.mount("/static", StaticFiles(directory="static"), name="static")

# Template engine
templates = Jinja2Templates(directory="templates")
```

### Route Registration

API routes are registered with the main application:

```python
# Include API routers
app.include_router(job_router)      # Job management endpoints
app.include_router(websocket_router) # WebSocket endpoints

# Frontend routes
@app.get("/", response_class=HTMLResponse)
async def frontend_home(request: Request):
    """Serve the main frontend interface."""
    return templates.TemplateResponse("index.html", {"request": request})
```

## 🔍 Health Checks and Monitoring

### Startup Health Verification

During startup, the system performs health checks:

```python
@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint."""
    try:
        health_status = {
            "status": "healthy",
            "version": "5.0.0",
            "timestamp": datetime.utcnow().isoformat(),
            "frontend": {
                "status": "active",
                "static_files": os.path.exists("static"),
                "templates": os.path.exists("templates")
            }
        }
        
        # Check translation service health
        if translation_service:
            translation_health = translation_service.health_check()
            health_status["translation_service"] = translation_health
        
        # Check database service health
        db_service = await get_database_service()
        db_health = await db_service.health_check()
        health_status["database_service"] = db_health
        
        # Check job queue service health
        job_queue = get_job_queue_service()
        queue_status = job_queue.get_queue_status()
        health_status["job_queue_service"] = {
            "status": "healthy",
            **queue_status
        }
        
        return health_status
        
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "service": "AI Video Translation Service"
            }
        )
```

## 🛑 Shutdown Process

### Graceful Shutdown

The application implements graceful shutdown:

```python
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on application shutdown."""
    logger.info("Shutting down AI Video Translation Service...")
    
    try:
        # Shutdown job queue service
        await shutdown_job_queue_service()
        logger.info("Job queue service shut down")
        
        # Close database service
        await close_database_service()
        logger.info("Database service closed")
        
        # Clear model cache
        try:
            ai_factory = get_ai_factory()
            ai_factory.clear_cache()
            logger.info("Model cache cleared")
        except Exception as e:
            logger.error(f"Error clearing model cache: {e}")
        
        logger.info("✅ AI Video Translation Service shut down successfully")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
```

### Resource Cleanup

Each service implements proper cleanup:

```python
# Job Queue Service cleanup
async def shutdown(self):
    """Shutdown job queue and cancel active jobs."""
    if self._queue_task and not self._queue_task.done():
        self._queue_task.cancel()
    
    for job_id, task in self._active_jobs.items():
        if not task.done():
            task.cancel()
            logger.info(f"Cancelled job {job_id}")

# Database Service cleanup
async def close(self):
    """Close database connection."""
    if self._connection:
        await self._connection.close()
        logger.info("Database connection closed")

# AI Service Factory cleanup
def clear_cache(self):
    """Clear all cached models and free memory."""
    with self._cache_lock:
        for model in self._model_cache.values():
            del model.instance
        self._model_cache.clear()
        logger.info("Model cache cleared")
```

## 🐳 Docker Initialization

### Container Startup

When running in Docker, the initialization follows the same pattern but with containerized environment:

```dockerfile
# Dockerfile entry point
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose Orchestration

```yaml
# docker-compose.yml
services:
  ai-video-translation:
    build: .
    ports:
      - "8000:8000"
    environment:
      - HUGGING_FACE_TOKEN=${HUGGING_FACE_TOKEN}
      - HOST=0.0.0.0
      - PORT=8000
    volumes:
      - ./uploads:/app/uploads
      - ./output:/app/output
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
```

## 🔧 Troubleshooting Initialization Issues

### Common Startup Problems

1. **Missing Dependencies**:
   ```bash
   # Check FFmpeg installation
   ffmpeg -version
   
   # Install if missing
   apt-get update && apt-get install -y ffmpeg
   ```

2. **Missing Environment Variables**:
   ```bash
   # Check required environment variables
   echo $HUGGING_FACE_TOKEN
   
   # Set if missing
   export HUGGING_FACE_TOKEN="your_token_here"
   ```

3. **Port Conflicts**:
   ```bash
   # Check if port is in use
   lsof -i :8000
   
   # Use different port
   export PORT=8001
   ```

4. **Permission Issues**:
   ```bash
   # Ensure directories are writable
   chmod 755 uploads output static templates
   ```

###
