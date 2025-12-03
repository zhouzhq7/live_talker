"""
Pyttsx3 TTS Implementation
Offline TTS engine (cross-platform)
"""

import logging
import io
import wave
import numpy as np
from typing import Optional
from .base import BaseTTS

logger = logging.getLogger(__name__)


class Pyttsx3TTS(BaseTTS):
    """
    Pyttsx3 TTS implementation
    
    Features:
    - Offline TTS (no internet required)
    - Cross-platform (Windows, Linux, macOS)
    - Multiple voices available
    - Fast synthesis
    """
    
    def __init__(
        self,
        rate: int = 200,
        volume: float = 1.0,
        voice_id: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize Pyttsx3 TTS
        
        Args:
            rate: Speech rate (words per minute)
            volume: Volume (0.0 to 1.0)
            voice_id: Optional voice ID (None = default)
            **kwargs: Additional arguments
        """
        super().__init__(name="Pyttsx3", **kwargs)
        
        self.rate = rate
        self.volume = volume
        self.voice_id = voice_id
        self.engine = None
        
        # Try to initialize
        self.load_model()
    
    def load_model(self) -> bool:
        """Initialize Pyttsx3 engine"""
        try:
            import pyttsx3
            
            self.engine = pyttsx3.init()
            
            # Set properties
            self.engine.setProperty('rate', self.rate)
            self.engine.setProperty('volume', self.volume)
            
            # Set voice if specified
            if self.voice_id is not None:
                voices = self.engine.getProperty('voices')
                if self.voice_id < len(voices):
                    self.engine.setProperty('voice', voices[self.voice_id].id)
            
            self._is_initialized = True
            logger.info(f"[{self.name}] Initialized (rate={self.rate}, volume={self.volume})")
            return True
            
        except ImportError:
            logger.error(f"[{self.name}] pyttsx3 not installed. Install with: pip install pyttsx3")
            return False
        except Exception as e:
            logger.error(f"[{self.name}] Initialization failed: {e}")
            return False
    
    def synthesize(
        self,
        text: str,
        output_file: Optional[str] = None
    ) -> bytes:
        """
        Synthesize speech using Pyttsx3
        
        Args:
            text: Text to synthesize
            output_file: Optional file path to save audio
            
        Returns:
            Audio data as bytes (16-bit PCM, 16kHz)
        """
        if not self._is_initialized or not self.engine:
            logger.error(f"[{self.name}] Not initialized")
            return b''
        
        if not text:
            return b''
        
        try:
            # Pyttsx3 doesn't directly return bytes, so we need to save to temp file
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            # Save to temp file
            self.engine.save_to_file(text, tmp_path)
            self.engine.runAndWait()
            
            # Read audio data
            with open(tmp_path, 'rb') as f:
                audio_data = f.read()
            
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass
            
            # If output_file specified, save there too
            if output_file:
                with open(output_file, 'wb') as f:
                    f.write(audio_data)
            
            return audio_data
            
        except Exception as e:
            logger.error(f"[{self.name}] Synthesis failed: {e}")
            import traceback
            traceback.print_exc()
            return b''
    
    def list_voices(self):
        """List available voices"""
        if not self.engine:
            return []
        
        try:
            voices = self.engine.getProperty('voices')
            return [{"id": v.id, "name": v.name} for v in voices]
        except Exception as e:
            logger.error(f"Failed to list voices: {e}")
            return []
    
    def get_info(self) -> dict:
        """Get Pyttsx3-specific information"""
        info = super().get_info()
        info.update({
            "rate": self.rate,
            "volume": self.volume,
            "voice_id": self.voice_id,
            "available_voices": len(self.list_voices()) if self.engine else 0
        })
        return info

