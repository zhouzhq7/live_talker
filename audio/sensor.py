"""
Audio Sensor - Captures audio input from microphone
参考 Eva/perception/audio/sensor.py
"""

import logging
from typing import Optional
from config import AudioConfig

logger = logging.getLogger(__name__)


class AudioSensor:
    """
    Audio sensor implementation for capturing microphone input
    
    Supports:
    - Real-time audio capture via PyAudio
    - Configurable sample rate and chunk size
    - Mock mode for testing without hardware
    """
    
    def __init__(self, config: Optional[AudioConfig] = None):
        """
        Initialize audio sensor
        
        Args:
            config: Audio configuration (sample rate, channels, etc.)
        """
        self.config = config or AudioConfig()
        self.is_running = False
        self.stream = None
        self._audio_interface = None
        
        # Initialize PyAudio if available
        try:
            import pyaudio
            self._pyaudio = pyaudio
            self._has_pyaudio = True
            logger.info("[AudioSensor] PyAudio available")
        except ImportError:
            self._pyaudio = None
            self._has_pyaudio = False
            logger.warning("[AudioSensor] PyAudio not available, using mock mode")
    
    def start(self):
        """Start capturing audio from microphone"""
        if self.is_running:
            logger.warning("[AudioSensor] Already running")
            return
        
        if self._has_pyaudio:
            try:
                self._audio_interface = self._pyaudio.PyAudio()
                self.stream = self._audio_interface.open(
                    format=self.config.format,
                    channels=self.config.channels,
                    rate=self.config.sample_rate,
                    input=True,
                    frames_per_buffer=self.config.chunk_size
                )
                self.is_running = True
                logger.info(f"[AudioSensor] Started - {self.config.sample_rate}Hz, {self.config.channels}ch")
            except Exception as e:
                logger.error(f"[AudioSensor] Failed to start: {e}")
                self.is_running = False
        else:
            # Mock mode
            self.is_running = True
            logger.info("[AudioSensor] Started (Mock mode)")
    
    def stop(self):
        """Stop capturing audio"""
        if not self.is_running:
            return
        
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                logger.error(f"[AudioSensor] Error stopping stream: {e}")
        
        if self._audio_interface:
            try:
                self._audio_interface.terminate()
            except Exception as e:
                logger.error(f"[AudioSensor] Error terminating audio interface: {e}")
        
        self.is_running = False
        logger.info("[AudioSensor] Stopped")
    
    def get_frame(self) -> Optional[bytes]:
        """
        Get current audio frame
        
        Returns:
            Audio data as bytes, or None if not available
        """
        if not self.is_running:
            return None
        
        try:
            if self.stream:
                # Read audio data from stream
                audio_data = self.stream.read(
                    self.config.chunk_size,
                    exception_on_overflow=False
                )
                return audio_data
            else:
                # Mock mode - return empty audio data
                return b''
        except Exception as e:
            logger.error(f"[AudioSensor] Error reading frame: {e}")
            return None
    
    def __del__(self):
        """Cleanup on deletion"""
        if self.is_running:
            self.stop()

