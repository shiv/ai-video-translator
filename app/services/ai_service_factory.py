"""AI Service Factory with model caching and resource management."""

import os
import sys
import time
import threading
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, VitsModel
from faster_whisper import WhisperModel

from app import logger
from app.services.util import get_env_var
from app.services.stt.speech_to_text_faster_whisper import SpeechToTextFasterWhisper
from app.services.stt.speech_to_text_whisper_transformers import SpeechToTextWhisperTransformers
from app.services.tts.text_to_speech_mms import TextToSpeechMMS
from app.services.translation.translation_nllb import TranslationNLLB


class ModelType(Enum):
    """Supported model types."""
    STT_WHISPER = "stt_whisper"
    STT_WHISPER_TRANSFORMERS = "stt_whisper_transformers" 
    TTS_MMS = "tts_mms"
    TRANSLATION_NLLB = "translation_nllb"


@dataclass
class ModelConfig:
    """Model loading and caching configuration."""
    model_type: ModelType
    model_name: str
    device: str
    cache_key: str
    cpu_threads: int = 0
    vad: bool = False
    compute_type: Optional[str] = None


@dataclass
class CachedModel:
    """Cached model container."""
    model: Any
    tokenizer: Optional[Any] = None
    config: Optional[ModelConfig] = None
    load_time: float = 0.0
    memory_usage_mb: Optional[float] = None


class ModelLoadingError(Exception):
    """Model loading failure."""
    pass


class AIServiceFactory:
    """Factory for AI services with model caching and resource management."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize AI Service Factory."""
        if not hasattr(self, '_initialized'):
            self._model_cache: Dict[str, CachedModel] = {}
            self._device = get_env_var("DEVICE", "cpu", str, ["cpu", "cuda"])
            self._cpu_threads = get_env_var("CPU_THREADS", 0, int)
            self._vad = get_env_var("VAD", False, bool)
            self._hugging_face_token = self._get_hugging_face_token()
            self._cache_enabled = get_env_var("MODEL_CACHE_ENABLED", True, bool)
            self._preload_models = get_env_var("PRELOAD_MODELS", False, bool)
            self._initialized = True
            
            # Platform-specific optimizations
            if sys.platform == "darwin":
                os.environ["TOKENIZERS_PARALLELISM"] = "false"
            
            logger().info("AI Service Factory initialized")
    
    def _get_hugging_face_token(self) -> str:
        """Get Hugging Face token from environment."""
        token = os.getenv("HUGGING_FACE_TOKEN") or os.getenv("HF_TOKEN")
        if not token:
            logger().warning("No Hugging Face token found. Some models may not load properly.")
        return token
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def _get_cache_key(self, config: ModelConfig) -> str:
        """Generate unique cache key for model configuration."""
        return f"{config.model_type.value}_{config.model_name}_{config.device}_{config.cpu_threads}_{config.vad}"
    
    def _load_whisper_model(self, config: ModelConfig) -> CachedModel:
        """Load and cache Whisper model."""
        logger().info(f"Loading Whisper model: {config.model_name} on {config.device}")
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            compute_type = "float16" if config.device == "cuda" else "int8"
            model = WhisperModel(
                model_size_or_path=config.model_name,
                device=config.device,
                cpu_threads=config.cpu_threads,
                compute_type=compute_type,
            )
            
            load_time = time.time() - start_time
            memory_usage = self._get_memory_usage() - start_memory
            
            logger().info(f"Whisper model loaded in {load_time:.2f}s, memory: {memory_usage:.1f}MB")
            
            return CachedModel(
                model=model,
                config=config,
                load_time=load_time,
                memory_usage_mb=memory_usage
            )
            
        except Exception as e:
            raise ModelLoadingError(f"Failed to load Whisper model {config.model_name}: {str(e)}")
    
    def _load_whisper_transformers_model(self, config: ModelConfig) -> CachedModel:
        """Load and cache Whisper Transformers model."""
        logger().info(f"Loading Whisper Transformers model: {config.model_name} on {config.device}")
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            # Transformers implementation handles model loading internally
            model = {"model_name": config.model_name, "device": config.device}
            
            load_time = time.time() - start_time
            memory_usage = self._get_memory_usage() - start_memory
            
            logger().info(f"Whisper Transformers model prepared in {load_time:.2f}s")
            
            return CachedModel(
                model=model,
                config=config,
                load_time=load_time,
                memory_usage_mb=memory_usage
            )
            
        except Exception as e:
            raise ModelLoadingError(f"Failed to prepare Whisper Transformers model {config.model_name}: {str(e)}")
    
    def _load_nllb_model(self, config: ModelConfig) -> CachedModel:
        """Load and cache NLLB translation model."""
        logger().info(f"Loading NLLB model: {config.model_name} on {config.device}")
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            model_name = f"facebook/{config.model_name}"
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            try:
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(config.device)
            except RuntimeError as e:
                if config.device == "cuda":
                    logger().warning(f"Loading NLLB model {model_name} on CPU due to GPU error")
                    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cpu")
                else:
                    raise e
            
            load_time = time.time() - start_time
            memory_usage = self._get_memory_usage() - start_memory
            
            logger().info(f"NLLB model loaded in {load_time:.2f}s, memory: {memory_usage:.1f}MB")
            
            return CachedModel(
                model=model,
                tokenizer=tokenizer,
                config=config,
                load_time=load_time,
                memory_usage_mb=memory_usage
            )
            
        except Exception as e:
            raise ModelLoadingError(f"Failed to load NLLB model {config.model_name}: {str(e)}")
    
    def _load_mms_model_info(self, config: ModelConfig) -> CachedModel:
        """Cache MMS model information (models loaded on-demand per language)."""
        logger().info(f"Preparing MMS TTS model cache for device: {config.device}")
        start_time = time.time()
        
        # MMS models are loaded on-demand per language
        model_info = {
            "device": config.device,
            "model_pattern": "facebook/mms-tts-{language}",
            "supported_languages": TextToSpeechMMS(config.device).get_languages()
        }
        
        load_time = time.time() - start_time
        
        logger().info(f"MMS TTS model info cached in {load_time:.2f}s")
        
        return CachedModel(
            model=model_info,
            config=config,
            load_time=load_time,
            memory_usage_mb=0.0
        )
    
    def load_model(self, config: ModelConfig) -> CachedModel:
        """Load and cache model based on configuration."""
        cache_key = self._get_cache_key(config)
        
        if self._cache_enabled and cache_key in self._model_cache:
            logger().debug(f"Using cached model: {cache_key}")
            return self._model_cache[cache_key]
        
        if config.model_type == ModelType.STT_WHISPER:
            cached_model = self._load_whisper_model(config)
        elif config.model_type == ModelType.STT_WHISPER_TRANSFORMERS:
            cached_model = self._load_whisper_transformers_model(config)
        elif config.model_type == ModelType.TRANSLATION_NLLB:
            cached_model = self._load_nllb_model(config)
        elif config.model_type == ModelType.TTS_MMS:
            cached_model = self._load_mms_model_info(config)
        else:
            raise ModelLoadingError(f"Unsupported model type: {config.model_type}")
        
        if self._cache_enabled:
            self._model_cache[cache_key] = cached_model
            logger().debug(f"Model cached with key: {cache_key}")
        
        return cached_model
    
    def get_stt_service(self, stt_type: str, model_name: str) -> Any:
        """Get Speech-to-Text service with cached model."""
        
        if stt_type == "auto":
            actual_stt_type = "faster-whisper" if sys.platform != "darwin" else "transformers"
        else:
            actual_stt_type = stt_type
        
        if actual_stt_type == "faster-whisper":
            config = ModelConfig(
                model_type=ModelType.STT_WHISPER,
                model_name=model_name,
                device=self._device,
                cache_key="",
                cpu_threads=self._cpu_threads,
                vad=self._vad
            )
            config.cache_key = self._get_cache_key(config)
            
            cached_model = self.load_model(config)
            
            service = SpeechToTextFasterWhisper(
                model_name=model_name,
                device=self._device,
                cpu_threads=self._cpu_threads,
                vad=self._vad
            )
            service._model = cached_model.model
            
            return service
            
        elif actual_stt_type == "transformers":
            config = ModelConfig(
                model_type=ModelType.STT_WHISPER_TRANSFORMERS,
                model_name=model_name,
                device=self._device,
                cache_key="",
                cpu_threads=self._cpu_threads
            )
            config.cache_key = self._get_cache_key(config)
            
            cached_model = self.load_model(config)
            
            service = SpeechToTextWhisperTransformers(
                model_name=model_name,
                device=self._device,
                cpu_threads=self._cpu_threads
            )
            
            return service
        
        else:
            raise ValueError(f"Unsupported STT type: {stt_type}")
    
    def get_translation_service(self, translator_type: str, model_name: str) -> Any:
        """Get Translation service with cached model."""
        if translator_type == "nllb":
            config = ModelConfig(
                model_type=ModelType.TRANSLATION_NLLB,
                model_name=model_name,
                device=self._device,
                cache_key=""
            )
            config.cache_key = self._get_cache_key(config)
            
            cached_model = self.load_model(config)
            
            service = TranslationNLLB(self._device)
            service.model_name = f"facebook/{model_name}"
            service.tokenizer = cached_model.tokenizer
            service._cached_model = cached_model.model
            
            return service
        
        else:
            raise ValueError(f"Unsupported translation type: {translator_type}")
    
    def get_tts_service(self, tts_type: str) -> Any:
        """Get Text-to-Speech service with cached model info."""
        if tts_type == "mms":
            config = ModelConfig(
                model_type=ModelType.TTS_MMS,
                model_name="mms",
                device=self._device,
                cache_key=""
            )
            config.cache_key = self._get_cache_key(config)
            
            cached_model = self.load_model(config)
            
            service = TextToSpeechMMS(self._device)
            service._cached_model_info = cached_model.model
            
            return service
        
        else:
            raise ValueError(f"Unsupported TTS type: {tts_type}")
    
    def preload_default_models(self):
        """Preload commonly used models at startup for better performance."""
        if not self._preload_models:
            logger().info("Model preloading disabled")
            return
        
        logger().info("Starting model preloading...")
        start_time = time.time()
        
        default_stt_model = get_env_var("DEFAULT_STT_MODEL", "tiny")
        default_translation_model = get_env_var("DEFAULT_TRANSLATION_MODEL", "nllb-200-distilled-600M")
        
        default_models = [
            ModelConfig(
                model_type=ModelType.STT_WHISPER,
                model_name=default_stt_model,
                device=self._device,
                cache_key="",
                cpu_threads=self._cpu_threads,
                vad=self._vad
            ),
            ModelConfig(
                model_type=ModelType.TRANSLATION_NLLB,
                model_name=default_translation_model,
                device=self._device,
                cache_key=""
            ),
            ModelConfig(
                model_type=ModelType.TTS_MMS,
                model_name="mms",
                device=self._device,
                cache_key=""
            )
        ]
        
        loaded_count = 0
        for config in default_models:
            try:
                config.cache_key = self._get_cache_key(config)
                self.load_model(config)
                loaded_count += 1
            except Exception as e:
                logger().error(f"Failed to preload model {config.model_name}: {str(e)}")
        
        total_time = time.time() - start_time
        logger().info(f"Model preloading completed: {loaded_count}/{len(default_models)} models loaded in {total_time:.2f}s")
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get model cache statistics and health information."""
        total_memory = sum(
            model.memory_usage_mb or 0 
            for model in self._model_cache.values()
        )
        
        models_info = []
        for cache_key, cached_model in self._model_cache.items():
            models_info.append({
                "cache_key": cache_key,
                "model_type": cached_model.config.model_type.value if cached_model.config else "unknown",
                "model_name": cached_model.config.model_name if cached_model.config else "unknown",
                "device": cached_model.config.device if cached_model.config else "unknown",
                "load_time_seconds": cached_model.load_time,
                "memory_usage_mb": cached_model.memory_usage_mb
            })
        
        return {
            "cache_enabled": self._cache_enabled,
            "preload_enabled": self._preload_models,
            "total_cached_models": len(self._model_cache),
            "total_memory_mb": total_memory,
            "device": self._device,
            "cpu_threads": self._cpu_threads,
            "models": models_info
        }
    
    def clear_cache(self):
        """Clear all cached models to free memory."""
        logger().info("Clearing model cache...")
        
        if self._device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        cleared_count = len(self._model_cache)
        self._model_cache.clear()
        
        logger().info(f"Model cache cleared: {cleared_count} models removed")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on AI Service Factory."""
        try:
            status = {
                "status": "healthy",
                "cache_enabled": self._cache_enabled,
                "device": self._device,
                "cached_models": len(self._model_cache),
                "memory_usage_mb": self._get_memory_usage(),
                "torch_cuda_available": torch.cuda.is_available() if hasattr(torch.cuda, 'is_available') else False
            }
            
            if self._device == "cuda":
                if torch.cuda.is_available():
                    status["cuda_device_count"] = torch.cuda.device_count()
                    status["cuda_current_device"] = torch.cuda.current_device()
                else:
                    status["status"] = "warning"
                    status["warning"] = "CUDA device requested but not available"
            
            return status
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "cache_enabled": self._cache_enabled,
                "device": self._device
            }


_ai_factory = None

def get_ai_factory() -> AIServiceFactory:
    """Get the global AI Service Factory instance."""
    global _ai_factory
    if _ai_factory is None:
        _ai_factory = AIServiceFactory()
    return _ai_factory