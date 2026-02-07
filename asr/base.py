"""
Base ASR Interface
参考 voice_benchmark/asr/base.py
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, NamedTuple, AsyncGenerator
import time
import logging

logger = logging.getLogger(__name__)


class ASRResult(NamedTuple):
    """ASR 识别结果"""
    text: str                      # 识别的文本
    is_final: bool                # 是否是最终结果
    confidence: float = 0.0       # 置信度 0-1
    timestamp_start: int = 0      # 开始时间戳 (ms)
    timestamp_end: int = 0        # 结束时间戳 (ms)


class BaseASR(ABC):
    """
    Abstract base class for ASR engines
    
    All ASR implementations should inherit from this class and implement
    the abstract methods.
    """
    
    def __init__(self, name: str, **kwargs):
        """
        Initialize ASR engine
        
        Args:
            name: Engine name for identification
            **kwargs: Engine-specific configuration
        """
        self.name = name
        self.config = kwargs
        self._is_initialized = False
        self._warmup_done = False
        
    @abstractmethod
    def load_model(self) -> bool:
        """
        Load the ASR model
        
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def transcribe(
        self, 
        audio_data: bytes,
        sample_rate: int = 16000,
        language: str = "zh"
    ) -> str:
        """
        Transcribe audio to text
        
        Args:
            audio_data: Raw audio bytes (16-bit PCM)
            sample_rate: Audio sample rate in Hz
            language: Language code (zh, en, etc.)
            
        Returns:
            Transcribed text
        """
        pass
    
    def transcribe_with_timing(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        language: str = "zh"
    ) -> Dict[str, Any]:
        """
        Transcribe audio and measure timing metrics
        
        Args:
            audio_data: Raw audio bytes
            sample_rate: Sample rate
            language: Language code
            
        Returns:
            Dictionary containing:
                - text: Transcribed text
                - latency_ms: Processing time in milliseconds
                - audio_duration_ms: Audio duration in milliseconds
                - rtf: Real-time factor (processing_time / audio_duration)
        """
        import numpy as np
        
        # Calculate audio duration
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        audio_duration_s = len(audio_array) / sample_rate
        audio_duration_ms = audio_duration_s * 1000
        
        # Measure transcription time
        start_time = time.time()
        text = self.transcribe(audio_data, sample_rate, language)
        end_time = time.time()
        
        latency_ms = (end_time - start_time) * 1000
        rtf = (end_time - start_time) / audio_duration_s if audio_duration_s > 0 else 0
        
        return {
            "text": text,
            "latency_ms": latency_ms,
            "audio_duration_ms": audio_duration_ms,
            "rtf": rtf,
            "engine": self.name
        }
    
    def warmup(self, num_iterations: int = 1) -> None:
        """
        Warm up the model with dummy inference
        
        Args:
            num_iterations: Number of warmup iterations
        """
        if self._warmup_done:
            return
            
        logger.info(f"[{self.name}] Warming up model...")
        import numpy as np
        
        # Create dummy audio (1 second of silence)
        dummy_audio = np.zeros(16000, dtype=np.int16).tobytes()
        
        for i in range(num_iterations):
            try:
                self.transcribe(dummy_audio, sample_rate=16000)
            except Exception as e:
                logger.warning(f"[{self.name}] Warmup iteration {i+1} failed: {e}")
        
        self._warmup_done = True
        logger.info(f"[{self.name}] Warmup completed")
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the ASR engine
        
        Returns:
            Dictionary with engine information
        """
        return {
            "name": self.name,
            "initialized": self._is_initialized,
            "warmup_done": self._warmup_done,
            "config": self.config
        }
    
    def is_available(self) -> bool:
        """
        Check if the ASR engine is available and initialized
        
        Returns:
            True if available, False otherwise
        """
        return self._is_initialized
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


# 流式 ASR 接口 (v2.0)
class StreamingASRMixin(ABC):
    """流式 ASR 混合类

    提供流式识别接口，增量输出识别结果
    """

    @abstractmethod
    def stream_recognize(
        self,
        audio_stream: AsyncGenerator[bytes, None]
    ) -> AsyncGenerator[ASRResult, None]:
        """流式识别

        Args:
            audio_stream: 音频流生成器

        Yields:
            ASRResult: 识别结果
        """
        pass

    @abstractmethod
    def start_stream(self) -> None:
        """开始流式识别会话"""
        pass

    @abstractmethod
    def accept_audio(self, audio_data: bytes) -> None:
        """接收音频数据

        Args:
            audio_data: 音频数据
        """
        pass

    @abstractmethod
    def get_result(self) -> Optional[ASRResult]:
        """获取当前结果

        Returns:
            Optional[ASRResult]: 当前识别结果，None 表示无结果
        """
        pass

    @abstractmethod
    def end_stream(self) -> Optional[ASRResult]:
        """结束流式识别

        Returns:
            Optional[ASRResult]: 最终识别结果
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """重置流式识别状态"""
        pass

