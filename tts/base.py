"""
Base TTS Interface
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


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

