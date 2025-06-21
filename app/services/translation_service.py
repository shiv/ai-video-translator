import logging
import os
import sys
import warnings
from typing import Optional, Dict, Any
from dataclasses import dataclass

import transformers
from iso639 import Lang

from app import logger
from app.services.processing.dubbing import Dubber
from app.services.processing.ffmpeg import FFmpeg
from app.services.util import get_env_var
from app.services.ai_service_factory import get_ai_factory, ModelLoadingError

@dataclass
class TranslationRequest:
    """Request parameters for video translation."""
    input_file: str
    source_language: Optional[str]
    target_language: str
    tts: str = "mms"
    stt: str = "auto"
    translator: str = "nllb"
    translator_model: str = "nllb-200-distilled-600M"
    stt_model: str = "tiny"
    output_directory: Optional[str] = None


@dataclass
class TranslationResult:
    """Result of video translation operation."""
    success: bool
    audio_file: Optional[str] = None
    video_file: Optional[str] = None
    error_message: Optional[str] = None
    processing_time_seconds: Optional[float] = None


class TranslationServiceError(Exception):
    """Base exception for translation service errors."""
    pass


class InvalidLanguageError(TranslationServiceError):
    """Raised when language validation fails."""
    pass


class InvalidFileFormatError(TranslationServiceError):
    """Raised when file format is not supported."""
    pass


class MissingDependencyError(TranslationServiceError):
    """Raised when required dependencies are missing."""
    pass


class ConfigurationError(TranslationServiceError):
    """Raised when configuration is invalid."""
    pass


class TranslationService:
    """Service for translating videos using AI models with performance optimizations."""
    
    ACCEPTED_VIDEO_FORMATS = ["mp4"]
    
    def __init__(self):
        """Initialize the translation service with AI Service Factory."""
        self._init_logging()
        self._ai_factory = get_ai_factory()
        
    def _init_logging(self):
        """Initialize logging configuration."""
        logging.basicConfig(level=logging.ERROR)  # Suppress third-party loggers
        
        # Create application logger
        app_logger = logging.getLogger("app")
        log_level = get_env_var("LOG_LEVEL", "INFO", str, ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        app_logger.setLevel(getattr(logging, log_level))
        app_logger.propagate = False
        
        # File handler for logging to a file
        file_handler = logging.FileHandler("app.log")
        console_handler = logging.StreamHandler()
        
        # Formatter for log messages
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        app_logger.addHandler(file_handler)
        app_logger.addHandler(console_handler)
        
        transformers.logging.set_verbosity_error()
        warnings.filterwarnings("ignore", category=FutureWarning)
    
    def _get_env_var(self, var_name: str, default_value, var_type=str, choices=None):
        """Get environment variable with type conversion and validation."""
        return get_env_var(var_name, default_value, var_type, choices)
    
    def _get_hugging_face_token(self) -> str:
        """Get Hugging Face token from environment variable."""
        token = os.getenv("HUGGING_FACE_TOKEN") or os.getenv("HF_TOKEN")
        if not token:
            raise ConfigurationError("Hugging Face token must be set via 'HUGGING_FACE_TOKEN' or 'HF_TOKEN' environment variable.")
        return token
    
    def _check_is_video(self, input_file: str) -> None:
        """Validate that the input file is a supported video format."""
        if not os.path.exists(input_file):
            raise InvalidFileFormatError(f"Input file does not exist: {input_file}")
        
        _, file_extension = os.path.splitext(input_file)
        file_extension = file_extension.lower().lstrip(".")
        
        if file_extension not in self.ACCEPTED_VIDEO_FORMATS:
            raise InvalidFileFormatError(f"Unsupported file format: {file_extension}. Supported formats: {self.ACCEPTED_VIDEO_FORMATS}")
    
    def _check_ffmpeg(self) -> None:
        """Check if FFmpeg is installed."""
        if not FFmpeg.is_ffmpeg_installed():
            raise MissingDependencyError("FFmpeg (which includes ffprobe) must be installed.")
    
    def _get_selected_tts(self, selected_tts: str, device: str):
        """Get the selected TTS service using AI Service Factory."""
        try:
            return self._ai_factory.get_tts_service(selected_tts)
        except Exception as e:
            raise ConfigurationError(f"Failed to create TTS service '{selected_tts}': {str(e)}")
    
    def _get_selected_translator(self, translator: str, translator_model: str, device: str):
        """Get the selected translator service using AI Service Factory."""
        try:
            return self._ai_factory.get_translation_service(translator, translator_model)
        except Exception as e:
            raise ConfigurationError(f"Failed to create translation service '{translator}': {str(e)}")
    
    def _check_languages(self, source_language: str, target_language: str, tts, translation, stt) -> None:
        """Validate language support across all services."""
        spt = stt.get_languages()
        translation_languages = translation.get_language_pairs()
        tts_languages = tts.get_languages()
        
        if source_language not in spt:
            raise InvalidLanguageError(
                f"Source language '{source_language}' is not supported by the speech recognition system. "
                f"Supported languages: {spt}"
            )
        
        pair = (source_language, target_language)
        if pair not in translation_languages:
            raise InvalidLanguageError(f"Language pair '{pair}' is not supported by the translation system.")
        
        if target_language not in tts_languages:
            raise InvalidLanguageError(
                f"Target language '{target_language}' is not supported by the text to speech system. "
                f"Supported languages: {tts_languages}"
            )
        
        voices = tts.get_available_voices(language_code=target_language)
        if len(voices) == 0:
            raise InvalidLanguageError(f"No voices available for language '{target_language}' in the text to speech system")
    
    def translate_video(self, request: TranslationRequest) -> TranslationResult:
        """
        Translate a video from source language to target language using cached models.
        
        Args:
            request: TranslationRequest containing all necessary parameters
            
        Returns:
            TranslationResult with success status and output file paths or error message
        """
        import time
        start_time = time.time()
        
        try:
            # Validate inputs
            self._check_is_video(request.input_file)
            self._check_ffmpeg()
            
            # Get configuration from environment variables
            output_directory = request.output_directory or get_env_var("OUTPUT_DIRECTORY", "output/")
            device = get_env_var("DEVICE", "cpu", str, ["cpu", "cuda"])
            cpu_threads = get_env_var("CPU_THREADS", 0, int)
            vad = get_env_var("VAD", False, bool)
            clean_intermediate_files = get_env_var("CLEAN_INTERMEDIATE_FILES", False, bool)
            
            hugging_face_token = self._get_hugging_face_token()
            
            # Initialize services using AI Service Factory (with caching)
            logger().info(f"Initializing AI services with caching...")
            
            try:
                tts = self._get_selected_tts(request.tts, device)
                logger().info(f"TTS service '{request.tts}' initialized")
            except Exception as e:
                raise ConfigurationError(f"Failed to initialize TTS service: {str(e)}")
            
            # Platform-specific configuration
            if sys.platform == "darwin":
                os.environ["TOKENIZERS_PARALLELISM"] = "false"
            
            # Initialize STT service using AI Service Factory (with caching)
            try:
                stt = self._ai_factory.get_stt_service(request.stt, request.stt_model)
                logger().info(f"STT service '{request.stt}' with model '{request.stt_model}' initialized")
                
                # Ensure model is loaded for STT services that need explicit loading
                if hasattr(stt, 'load_model') and stt._model is None:
                    stt.load_model()
                    
            except Exception as e:
                raise ConfigurationError(f"Failed to initialize STT service: {str(e)}")
            
            # Auto-detect source language if not provided
            source_language = request.source_language
            if not source_language:
                try:
                    source_language = stt.detect_language(request.input_file)
                    logger().info(f"Auto-detected source language: '{source_language}'")
                except Exception as e:
                    raise ConfigurationError(f"Failed to auto-detect source language: {str(e)}")
            
            # Initialize translation service using AI Service Factory (with caching)
            try:
                translation = self._get_selected_translator(request.translator, request.translator_model, device)
                logger().info(f"Translation service '{request.translator}' with model '{request.translator_model}' initialized")
            except Exception as e:
                raise ConfigurationError(f"Failed to initialize translation service: {str(e)}")
            
            # Validate language support
            self._check_languages(source_language, request.target_language, tts, translation, stt)
            
            # Create output directory
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
            
            # Initialize dubber and process video
            dubber = Dubber(
                input_file=request.input_file,
                output_directory=output_directory,
                source_language=source_language,
                target_language=request.target_language,
                hugging_face_token=hugging_face_token,
                tts=tts,
                translation=translation,
                stt=stt,
                device=device,
                cpu_threads=cpu_threads,
                clean_intermediate_files=clean_intermediate_files,
            )
            
            logger().info(f"Processing '{request.input_file}' file with STT '{request.stt}', TTS '{request.tts}' and device '{device}'")
            result = dubber.dub()
            
            processing_time = time.time() - start_time
            
            logger().info(f"Video translation completed successfully in {processing_time:.2f}s")
            
            return TranslationResult(
                success=True,
                audio_file=result.audio_file,
                video_file=result.video_file,
                processing_time_seconds=processing_time
            )
            
        except (InvalidLanguageError, InvalidFileFormatError, MissingDependencyError, ConfigurationError) as e:
            # Known service errors
            processing_time = time.time() - start_time
            error_message = str(e)
            logger().error(f"Translation service error: {error_message}")
            
            return TranslationResult(
                success=False,
                error_message=error_message,
                processing_time_seconds=processing_time
            )
            
        except ModelLoadingError as e:
            # Model loading specific errors
            processing_time = time.time() - start_time
            error_message = f"Model loading failed: {str(e)}"
            logger().error(error_message)
            
            return TranslationResult(
                success=False,
                error_message=error_message,
                processing_time_seconds=processing_time
            )
            
        except Exception as e:
            # Unexpected errors
            processing_time = time.time() - start_time
            error_message = f"Unexpected error: {str(e)}"
            logger().error(f"Translation failed with unexpected error: {error_message}")
            
            return TranslationResult(
                success=False,
                error_message=error_message,
                processing_time_seconds=processing_time
            )
    
    def get_model_status(self) -> Dict[str, Any]:
        """
        Get status information about cached models.
        
        Returns:
            Dictionary containing model cache statistics and health information
        """
        return self._ai_factory.get_model_status()
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a comprehensive health check of the translation service.
        
        Returns:
            Dictionary containing health status and diagnostic information
        """
        try:
            # Check AI Service Factory health
            factory_health = self._ai_factory.health_check()
            
            # Check basic service health
            service_health = {
                "translation_service": "healthy",
                "ffmpeg_available": FFmpeg.is_ffmpeg_installed(),
                "hugging_face_token_configured": bool(os.getenv("HUGGING_FACE_TOKEN") or os.getenv("HF_TOKEN")),
                "accepted_video_formats": self.ACCEPTED_VIDEO_FORMATS
            }
            
            # Combine health checks
            overall_status = "healthy"
            if factory_health.get("status") != "healthy":
                overall_status = factory_health.get("status", "unhealthy")
            elif not service_health["ffmpeg_available"]:
                overall_status = "warning"
                service_health["warning"] = "FFmpeg not available"
            elif not service_health["hugging_face_token_configured"]:
                overall_status = "warning"
                service_health["warning"] = "Hugging Face token not configured"
            
            return {
                "status": overall_status,
                "ai_service_factory": factory_health,
                "translation_service": service_health
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "ai_service_factory": {"status": "unknown"},
                "translation_service": {"status": "error"}
            }