"""
Whisper ASR Implementation
参考 voice_benchmark/asr/whisper.py
"""

import logging
import numpy as np
import os
from pathlib import Path
from typing import Optional
from .base import BaseASR

logger = logging.getLogger(__name__)


class Whisper(BaseASR):
    """
    Whisper-based ASR implementation using faster-whisper
    
    Features:
    - Multi-language support
    - Multiple model sizes (tiny to large)
    - GPU acceleration support
    - Good general-purpose performance
    """
    
    def __init__(
        self,
        model_name: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        model_cache_dir: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Whisper ASR
        
        Args:
            model_name: Model size (tiny, base, small, medium, large)
            device: Device to run on (cpu, cuda)
            compute_type: Computation type (int8, float16, float32)
            model_cache_dir: Base directory for model cache (default: D:\\models)
            **kwargs: Additional arguments
        """
        super().__init__(name=f"Whisper-{model_name}", **kwargs)
        
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.model_cache_dir = model_cache_dir or os.getenv("MODEL_CACHE_DIR", "D:\\models")
        self.model = None
        
        # Set HuggingFace cache directory
        self._setup_huggingface_cache()
        
        # Try to load model
        self.load_model()
    
    def _setup_huggingface_cache(self):
        """Setup HuggingFace cache directory"""
        try:
            hf_cache = os.path.join(self.model_cache_dir, "huggingface")
            os.makedirs(hf_cache, exist_ok=True)
            
            # Set HuggingFace cache directory via environment variable
            os.environ["HF_HOME"] = hf_cache
            os.environ["HUGGINGFACE_HUB_CACHE"] = hf_cache
            
            logger.info(f"[{self.name}] HuggingFace cache directory: {hf_cache}")
        except Exception as e:
            logger.warning(f"[{self.name}] Failed to setup HuggingFace cache: {e}")
    
    def _get_model_size(self) -> str:
        """Get approximate model size"""
        sizes = {
            "tiny": "~75MB",
            "base": "~150MB",
            "small": "~500MB",
            "medium": "~1.5GB",
            "large": "~3GB"
        }
        return sizes.get(self.model_name, "unknown")
    
    def load_model(self) -> bool:
        """Load Whisper model"""
        if self._is_initialized:
            return True
        
        try:
            from faster_whisper import WhisperModel
            
            logger.info(f"[{self.name}] Loading Whisper model '{self.model_name}'...")
            logger.info(f"[{self.name}] First-time download: {self._get_model_size()}")
            logger.info(f"[{self.name}] Downloading model... Please wait (this may take several minutes)")
            
            # Use HuggingFace cache directory
            hf_cache = os.path.join(self.model_cache_dir, "huggingface")
            download_root = os.path.join(hf_cache, "hub")
            
            self.model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                download_root=download_root
            )
            
            self._is_initialized = True
            logger.info(f"[{self.name}] ✓ Model loaded successfully!")
            return True
            
        except ImportError:
            logger.error(f"[{self.name}] faster-whisper not installed. Install with: pip install faster-whisper")
            return False
        except Exception as e:
            logger.error(f"[{self.name}] Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def transcribe(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        language: str = "zh"
    ) -> str:
        """
        Transcribe audio using Whisper
        
        Args:
            audio_data: Raw audio bytes (16-bit PCM)
            sample_rate: Sample rate
            language: Language code (zh, en, etc.)
            
        Returns:
            Transcribed text
        """
        if not self._is_initialized or not self.model:
            logger.error(f"[{self.name}] Model not initialized")
            return ""
        
        if not audio_data or len(audio_data) == 0:
            return ""
        
        try:
            # Convert bytes to float32 numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            # Transcribe
            segments, info = self.model.transcribe(
                audio_float,
                language=language if language != "zh" else None,  # Auto-detect Chinese
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            # Collect text from segments
            text_parts = []
            for segment in segments:
                text_parts.append(segment.text.strip())
            
            text = " ".join(text_parts).strip()
            
            if text:
                logger.debug(f"[{self.name}] Transcribed: {text[:50]}...")
            
            return text
            
        except Exception as e:
            logger.error(f"[{self.name}] Transcription failed: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def get_info(self) -> dict:
        """Get Whisper-specific information"""
        info = super().get_info()
        info.update({
            "model_name": self.model_name,
            "device": self.device,
            "compute_type": self.compute_type,
            "model_size": self._get_model_size()
        })
        return info

