"""
SenseVoice ASR Implementation
阿里最新语音理解模型 - 支持语音识别、情感识别、音频事件检测
"""

import logging
import numpy as np
import os
import re
import io
import tempfile
from typing import Optional, Dict, Any
from .base import BaseASR

logger = logging.getLogger(__name__)


class SenseVoice(BaseASR):
    """
    SenseVoice ASR - 阿里最新语音理解模型
    支持：语音识别 + 情感识别 + 音频事件检测
    
    Features:
    - 50+ 语言支持
    - 情感识别 (SER)
    - 音频事件检测 (AED)
    - 70ms 处理 10s 音频 (比 Whisper 快 15 倍)
    """
    
    def __init__(
        self,
        model_name: str = "iic/SenseVoiceSmall",
        device: str = "cpu",
        language: str = "auto",
        enable_vad: bool = True,
        model_cache_dir: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize SenseVoice
        
        Args:
            model_name: Model name (iic/SenseVoiceSmall)
            device: Device to run on (cpu, cuda)
            language: Language code (auto, zh, en, yue, ja, ko, nospeech)
            enable_vad: Enable built-in VAD
            model_cache_dir: Base directory for model cache
            **kwargs: Additional arguments
        """
        super().__init__(name="SenseVoice", **kwargs)
        
        self.model_name = model_name
        self.device = device
        self.language = language
        self.enable_vad = enable_vad
        self.model_cache_dir = model_cache_dir or os.getenv("MODEL_CACHE_DIR", os.path.join(tempfile.gettempdir(), "models"))
        
        self.model = None
        self.vad_model = None
        
        # Setup cache directory
        self._setup_cache()
        
        # Try to load model
        self.load_model()
    
    def _setup_cache(self):
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
        """
        Load SenseVoice model
        
        Returns:
            True if successful, False otherwise
        """
        if self._is_initialized:
            return True
        
        try:
            from funasr import AutoModel
            
            logger.info(f"[{self.name}] Loading SenseVoice model '{self.model_name}'...")
            logger.info(f"[{self.name}] First-time download: ~1GB from ModelScope")
            logger.info(f"[{self.name}] Downloading model... Please wait (this may take several minutes)")
            
            # Load SenseVoice model
            self.model = AutoModel(
                model=self.model_name,
                vad_model="fsmn-vad" if self.enable_vad else None,
                vad_kwargs={"max_single_segment_time": 30000} if self.enable_vad else None,
                device=self.device,
                trust_remote_code=True,
                disable_pbar=False,
                disable_log=True
            )
            
            logger.info(f"[{self.name}] ✓ SenseVoice model loaded successfully!")
            
            self._is_initialized = True
            logger.info(f"[{self.name}] === Model ready! ===")
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
        language: str = None
    ) -> str:
        """
        Transcribe audio to text
        
        Args:
            audio_data: Raw audio bytes (16-bit PCM)
            sample_rate: Audio sample rate in Hz
            language: Language code (auto, zh, en, yue, ja, ko, nospeech)
                     If None, uses self.language
        
        Returns:
            Transcribed text with emotion tags removed
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
            
            # Use specified language or default
            lang = language or self.language
            
            # Inference
            result = self.model.generate(
                input=audio_float,
                language=lang,
                use_itn=True,
                batch_size_s=0,
                ban_emo_unk=False
            )
            
            if result and len(result) > 0:
                text = result[0].get("text", "")
                # Parse emotion tags if present
                text = self._clean_emotion_tags(text)
                return text.strip()
            
            return ""
            
        except Exception as e:
            logger.error(f"[{self.name}] Transcription failed: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def _clean_emotion_tags(self, text: str) -> str:
        """
        Clean emotion and event tags from text
        
        Args:
            text: Raw text with tags like <|EMO_UNKNOWN|><|Event_unknow|>
        
        Returns:
            Clean text without tags
        """
        if not text:
            return ""
        
        # Remove emotion and event tags
        # Pattern matches <|...|>
        text = re.sub(r'<\|[^|]+\|>', '', text)
        
        return text.strip()
    
    def transcribe_with_emotion(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        language: str = None
    ) -> Dict[str, Any]:
        """
        Transcribe audio with emotion detection
        
        Args:
            audio_data: Raw audio bytes (16-bit PCM)
            sample_rate: Audio sample rate in Hz
            language: Language code
        
        Returns:
            Dictionary with:
                - text: Transcribed text (without tags)
                - raw_text: Raw text with tags
                - emotion: Detected emotion
                - event: Detected audio event
        """
        if not self._is_initialized or not self.model:
            logger.error(f"[{self.name}] Model not initialized")
            return {"text": "", "raw_text": "", "emotion": "", "event": ""}
        
        if not audio_data or len(audio_data) == 0:
            return {"text": "", "raw_text": "", "emotion": "", "event": ""}
        
        try:
            # Convert bytes to float32 numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            # Use specified language or default
            lang = language or self.language
            
            # Inference
            result = self.model.generate(
                input=audio_float,
                language=lang,
                use_itn=True,
                batch_size_s=0,
                ban_emo_unk=False
            )
            
            if result and len(result) > 0:
                raw_text = result[0].get("text", "")
                
                # Parse emotion and event tags
                emotion = self._extract_emotion(raw_text)
                event = self._extract_event(raw_text)
                
                # Clean text
                text = self._clean_emotion_tags(raw_text)
                
                return {
                    "text": text.strip(),
                    "raw_text": raw_text,
                    "emotion": emotion,
                    "event": event
                }
            
            return {"text": "", "raw_text": "", "emotion": "", "event": ""}
            
        except Exception as e:
            logger.error(f"[{self.name}] Transcription with emotion failed: {e}")
            import traceback
            traceback.print_exc()
            return {"text": "", "raw_text": "", "emotion": "", "event": ""}
    
    def _extract_emotion(self, text: str) -> str:
        """
        Extract emotion from tagged text
        
        Args:
            text: Text with emotion tags
        
        Returns:
            Emotion string
        """
        if not text:
            return ""
        
        # Emotion patterns
        emotion_patterns = [
            r'<\|(HAPPY)\|>',
            r'<\|(SAD)\|>',
            r'<\|(ANGRY)\|>',
            r'<\|(NEUTRAL)\|>',
            r'<\|(EMO_UNKNOWN)\|>',
        ]
        
        for pattern in emotion_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        return ""
    
    def _extract_event(self, text: str) -> str:
        """
        Extract audio event from tagged text
        
        Args:
            text: Text with event tags
        
        Returns:
            Event string
        """
        if not text:
            return ""
        
        # Event patterns
        event_patterns = [
            r'<\|(Speech)\|>',
            r'<\|(Applause)\|>',
            r'<\|(Laughter)\|>',
            r'<\|(Music)\|>',
            r'<\|(Event_unknow)\|>',
        ]
        
        for pattern in event_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        return ""
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get model information
        
        Returns:
            Dictionary with engine information
        """
        info = super().get_info()
        info.update({
            "model_name": self.model_name,
            "device": self.device,
            "language": self.language,
            "supports_emotion": True,
            "supports_event_detection": True,
            "enable_vad": self.enable_vad
        })
        return info
