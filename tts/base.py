"""
Base TTS Interface
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, NamedTuple, AsyncGenerator
import logging

logger = logging.getLogger(__name__)


class TTSResult(NamedTuple):
    """TTS 合成结果"""
    audio_chunk: bytes               # 音频数据
    text: str                        # 对应的文本
    is_final: bool                   # 是否是最终结果
    sample_rate: int = 16000         # 采样率


class BaseTTS(ABC):
    """
    Abstract base class for TTS engines
    
    All TTS implementations should inherit from this class and implement
    the abstract methods.
    """
    
    def __init__(self, name: str, **kwargs):
        """
        Initialize TTS engine
        
        Args:
            name: Engine name for identification
            **kwargs: Engine-specific configuration
        """
        self.name = name
        self.config = kwargs
        self._is_initialized = False
        
    @abstractmethod
    def synthesize(
        self,
        text: str,
        output_file: Optional[str] = None
    ) -> bytes:
        """
        Synthesize speech from text

        Args:
            text: Text to synthesize
            output_file: Optional file path to save audio

        Returns:
            Audio data as bytes (16-bit PCM)
        """
        pass

    async def synthesize_stream(
        self,
        text_stream: AsyncGenerator[str, None]
    ) -> AsyncGenerator[TTSResult, None]:
        """
        Stream synthesis - synthesize text chunks as they arrive

        Args:
            text_stream: Async generator of text chunks

        Yields:
            TTSResult: Audio chunk with metadata
        """
        # Default implementation: collect all text and synthesize
        full_text = ""
        async for text_chunk in text_stream:
            full_text += text_chunk

        if full_text:
            audio = self.synthesize(full_text)
            if audio:
                yield TTSResult(
                    audio_chunk=audio,
                    text=full_text,
                    is_final=True
                )

    def _split_sentences(self, text: str) -> list:
        """
        Split text into sentence-like chunks for streaming

        Args:
            text: Input text

        Returns:
            List of text chunks
        """
        import re
        # Split on common sentence boundaries
        sentences = re.split(r'([。！？；\n]+)', text)
        chunks = []
        current = ""

        for part in sentences:
            current += part
            if len(current) >= 50 or (part in '。！？；\n' and current.strip()):
                if current.strip():
                    chunks.append(current.strip())
                current = ""

        if current.strip():
            chunks.append(current.strip())

        return chunks
    
    def synthesize_to_file(
        self,
        text: str,
        output_file: str
    ) -> bool:
        """
        Synthesize speech and save to file
        
        Args:
            text: Text to synthesize
            output_file: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            audio_data = self.synthesize(text, output_file=output_file)
            if output_file and audio_data:
                with open(output_file, 'wb') as f:
                    f.write(audio_data)
                return True
            return False
        except Exception as e:
            logger.error(f"[{self.name}] Failed to synthesize to file: {e}")
            return False
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the TTS engine
        
        Returns:
            Dictionary with engine information
        """
        return {
            "name": self.name,
            "initialized": self._is_initialized,
            "config": self.config
        }
    
    def is_available(self) -> bool:
        """
        Check if the TTS engine is available and initialized
        
        Returns:
            True if available, False otherwise
        """
        return self._is_initialized
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"

