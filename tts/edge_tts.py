"""
Edge-TTS Implementation
Microsoft Edge TTS (free, high quality)
"""

import logging
import asyncio
import io
from typing import Optional
from .base import BaseTTS

logger = logging.getLogger(__name__)


class EdgeTTS(BaseTTS):
    """
    Edge-TTS implementation
    
    Features:
    - Free Microsoft Edge TTS API
    - High quality voices
    - Multiple languages and voices
    - Fast synthesis
    """
    
    def __init__(
        self,
        voice: str = "zh-CN-XiaoxiaoNeural",
        rate: str = "+0%",
        volume: str = "+0%",
        **kwargs
    ):
        """
        Initialize Edge-TTS
        
        Args:
            voice: Voice name (e.g., zh-CN-XiaoxiaoNeural)
            rate: Speech rate adjustment (+/-XX%)
            volume: Volume adjustment (+/-XX%)
            **kwargs: Additional arguments
        """
        super().__init__(name="Edge-TTS", **kwargs)
        
        self.voice = voice
        self.rate = rate
        self.volume = volume
        
        # Try to initialize
        self.load_model()
    
    def load_model(self) -> bool:
        """Initialize Edge-TTS (no model loading needed)"""
        try:
            import edge_tts
            self._edge_tts = edge_tts
            self._is_initialized = True
            logger.info(f"[{self.name}] Initialized with voice: {self.voice}")
            return True
        except ImportError:
            logger.error(f"[{self.name}] edge-tts not installed. Install with: pip install edge-tts")
            return False
    
    def synthesize(
        self,
        text: str,
        output_file: Optional[str] = None
    ) -> bytes:
        """
        Synthesize speech using Edge-TTS
        
        Args:
            text: Text to synthesize
            output_file: Optional file path to save audio
            
        Returns:
            Audio data as bytes (16-bit PCM, 16kHz)
        """
        if not self._is_initialized:
            logger.error(f"[{self.name}] Not initialized")
            return b''
        
        if not text:
            return b''
        
        try:
            # Run async synthesis (returns MP3)
            mp3_data = asyncio.run(self._synthesize_async(text))
            
            if output_file:
                with open(output_file, 'wb') as f:
                    f.write(mp3_data)
            
            # Convert MP3 to PCM
            audio_data = self._mp3_to_pcm(mp3_data)
            
            return audio_data
            
        except Exception as e:
            logger.error(f"[{self.name}] Synthesis failed: {e}")
            import traceback
            traceback.print_exc()
            return b''
    
    async def _synthesize_async(self, text: str) -> bytes:
        """Async synthesis helper (returns MP3)"""
        communicate = self._edge_tts.Communicate(
            text=text,
            voice=self.voice,
            rate=self.rate,
            volume=self.volume
        )
        
        audio_data = b''
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        
        return audio_data
    
    def _mp3_to_pcm(self, mp3_data: bytes) -> bytes:
        """Convert MP3 to 16-bit PCM"""
        try:
            import pydub
            from pydub import AudioSegment
            import io
            import subprocess
            
            # Check if ffprobe is available
            try:
                subprocess.run(['ffprobe', '-version'], 
                             capture_output=True, 
                             check=True, 
                             timeout=2)
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                logger.error(
                    f"[{self.name}] FFmpeg not found. FFmpeg is required for MP3 to PCM conversion.\n"
                    "Install FFmpeg:\n"
                    "  macOS: brew install ffmpeg\n"
                    "  Ubuntu/Debian: sudo apt-get install ffmpeg\n"
                    "  Windows: Download from https://ffmpeg.org/download.html\n"
                    "  Or: conda install -c conda-forge ffmpeg"
                )
                return mp3_data
            
            # Load MP3
            audio = AudioSegment.from_mp3(io.BytesIO(mp3_data))
            
            # Convert to 16kHz mono
            audio = audio.set_frame_rate(16000)
            audio = audio.set_channels(1)
            
            # Export as raw PCM
            pcm_data = audio.raw_data
            
            return pcm_data
            
        except ImportError:
            logger.warning(f"[{self.name}] pydub not installed, returning MP3 data. Install with: pip install pydub")
            return mp3_data
        except Exception as e:
            error_msg = str(e)
            if 'ffprobe' in error_msg.lower() or 'no such file' in error_msg.lower():
                logger.error(
                    f"[{self.name}] FFmpeg not found. FFmpeg is required for MP3 to PCM conversion.\n"
                    "Install FFmpeg:\n"
                    "  macOS: brew install ffmpeg\n"
                    "  Ubuntu/Debian: sudo apt-get install ffmpeg\n"
                    "  Windows: Download from https://ffmpeg.org/download.html\n"
                    "  Or: conda install -c conda-forge ffmpeg"
                )
            else:
                logger.error(f"[{self.name}] MP3 to PCM conversion failed: {e}")
            return mp3_data
    
    @staticmethod
    async def list_voices():
        """List available voices"""
        try:
            import edge_tts
            voices = await edge_tts.list_voices()
            return voices
        except Exception as e:
            logger.error(f"Failed to list voices: {e}")
            return []
    
    def get_info(self) -> dict:
        """Get Edge-TTS-specific information"""
        info = super().get_info()
        info.update({
            "voice": self.voice,
            "rate": self.rate,
            "volume": self.volume
        })
        return info

