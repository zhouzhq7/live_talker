"""
FunASR Implementation - Chinese-optimized ASR
参考 Eva/perception/audio/asr_funasr.py
"""

import logging
import numpy as np
import os
from pathlib import Path
from typing import Optional
from .base import BaseASR

logger = logging.getLogger(__name__)


class FunASR(BaseASR):
    """
    FunASR-based ASR implementation (Alibaba DAMO Academy)
    
    Features:
    - Excellent Chinese recognition (95%+ accuracy)
    - Automatic punctuation restoration
    - Fast inference speed
    - Streaming support
    """
    
    def __init__(
        self,
        model_name: str = "paraformer-zh",
        device: str = "cpu",
        enable_vad: bool = False,
        model_cache_dir: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize FunASR
        
        Args:
            model_name: Model name (paraformer-zh, paraformer-zh-streaming)
            device: Device to run on (cpu, cuda)
            enable_vad: Enable built-in VAD (usually handled separately)
            model_cache_dir: Base directory for model cache (default: D:\\models)
            **kwargs: Additional arguments
        """
        super().__init__(name=f"FunASR-{model_name}", **kwargs)
        
        self.model_name = model_name
        self.device = device
        self.enable_vad = enable_vad
        self.model_cache_dir = model_cache_dir or os.getenv("MODEL_CACHE_DIR", "D:\\models")
        self.model = None
        self.vad_model = None
        
        # Set ModelScope cache directory
        self._setup_modelscope_cache()
        
        # Try to load model
        self.load_model()
    
    def _setup_modelscope_cache(self):
        """Setup ModelScope cache directory"""
        try:
            modelscope_cache = os.path.join(self.model_cache_dir, "modelscope")
            os.makedirs(modelscope_cache, exist_ok=True)
            
            # Set ModelScope cache directory via environment variable
            os.environ["MODELSCOPE_CACHE"] = modelscope_cache
            os.environ["MODELSCOPE_HOME"] = modelscope_cache
            
            logger.info(f"[{self.name}] ModelScope cache directory: {modelscope_cache}")
        except Exception as e:
            logger.warning(f"[{self.name}] Failed to setup ModelScope cache: {e}")
    
    def load_model(self) -> bool:
        """Load FunASR model"""
        if self._is_initialized:
            return True
        
        try:
            from funasr import AutoModel
            
            logger.info(f"[{self.name}] Loading FunASR model '{self.model_name}'...")
            logger.info(f"[{self.name}] First-time download: ~500MB-1GB from ModelScope")
            logger.info(f"[{self.name}] Downloading model... Please wait (this may take several minutes)")
            
            self.model = AutoModel(
                model=self.model_name,
                device=self.device,
                disable_pbar=False,  # Enable progress bar
                disable_log=True,
                disable_update=True
            )
            
            logger.info(f"[{self.name}] ✓ ASR model loaded successfully!")
            
            # Load VAD model if enabled
            if self.enable_vad:
                logger.info(f"[{self.name}] Loading VAD model (fsmn-vad)...")
                logger.info(f"[{self.name}] Downloading VAD model... (~50MB)")
                try:
                    self.vad_model = AutoModel(
                        model="fsmn-vad",
                        device=self.device,
                        disable_pbar=False
                    )
                    logger.info(f"[{self.name}] ✓ VAD model loaded successfully!")
                except Exception as e:
                    logger.warning(f"[{self.name}] ✗ VAD model loading failed: {e}")
                    self.vad_model = None
            
            self._is_initialized = True
            logger.info(f"[{self.name}] === All models ready! ===")
            return True
            
        except ImportError:
            logger.error(f"[{self.name}] funasr not installed. Install with: pip install funasr modelscope")
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
        Transcribe audio using FunASR
        
        Args:
            audio_data: Raw audio bytes (16-bit PCM)
            sample_rate: Sample rate (should be 16000)
            language: Language code (zh recommended)
            
        Returns:
            Transcribed text with punctuation
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
            
            # VAD check if enabled
            if self.enable_vad and self.vad_model:
                try:
                    vad_result = self.vad_model.generate(
                        input=audio_float,
                        cache={},
                        is_final=True
                    )
                    
                    # Check if speech detected
                    if not vad_result or len(vad_result[0].get("value", [])) == 0:
                        logger.debug(f"[{self.name}] No speech detected by VAD")
                        return ""
                except Exception as e:
                    logger.debug(f"[{self.name}] VAD check failed: {e}")
            
            # Perform ASR
            result = self.model.generate(
                input=audio_float,
                cache={},
                is_final=True,
                language=language
            )
            
            # Extract text
            if result and len(result) > 0:
                text = result[0].get("text", "")
                text = text.strip()
                
                if text:
                    logger.debug(f"[{self.name}] Transcribed: {text[:50]}...")
                
                return text
            
            return ""
            
        except Exception as e:
            logger.error(f"[{self.name}] Transcription failed: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def get_info(self) -> dict:
        """Get FunASR-specific information"""
        info = super().get_info()
        info.update({
            "model_name": self.model_name,
            "device": self.device,
            "enable_vad": self.enable_vad,
            "vad_loaded": self.vad_model is not None
        })
        return info

