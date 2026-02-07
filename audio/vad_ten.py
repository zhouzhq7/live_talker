"""
TEN-VAD Implementation
TEN 框架的轻量级 VAD - 库大小仅 306KB，比 Silero 快 32%
"""

import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class TENVAD:
    """
    TEN-VAD Voice Activity Detector
    
    Features:
    - Library size: 306KB (vs 2.16MB Silero)
    - RTF: 32% faster than Silero
    - Lower end-of-sentence detection latency
    - Cross-platform: Linux/macOS/Windows
    
    Reference: https://github.com/TEN-framework/ten-vad
    """
    
    def __init__(
        self,
        hop_size: int = 256,        # 16ms at 16kHz
        threshold: float = 0.5,     # Detection threshold
        sample_rate: int = 16000,   # Sample rate
        min_speech_duration: float = 0.25,   # 250ms
        min_silence_duration: float = 0.5,    # 500ms
    ):
        """
        Initialize TEN-VAD
        
        Args:
            hop_size: Frame hop size (256 = 16ms at 16kHz)
            threshold: Detection threshold (0.0 - 1.0)
            sample_rate: Audio sample rate
            min_speech_duration: Minimum speech duration to trigger
            min_silence_duration: Minimum silence to end speech
        """
        self.hop_size = hop_size
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.min_speech_duration = min_speech_duration
        self.min_silence_duration = min_silence_duration
        
        self.method = "ten"
        
        # TEN-VAD instance
        self._vad = None
        
        # State tracking
        self.speech_start_time = None
        self.silence_start_time = None
        self.is_speaking = False
        
        # Initialize
        self._init_vad()
    
    def _init_vad(self):
        """Initialize TEN-VAD"""
        try:
            from ten_vad import TenVad
            
            logger.info(f"[{self.method}] Initializing TEN-VAD...")
            logger.info(f"[{self.method}] Hop size: {self.hop_size}, Threshold: {self.threshold}")
            
            # Create TEN-VAD instance
            self._vad = TenVad(
                hop_size=self.hop_size,
                threshold=self.threshold
            )
            
            # Create and initialize handler
            self._vad.create_and_init_handler()
            
            logger.info(f"[{self.method}] ✓ TEN-VAD initialized successfully")
            
        except ImportError:
            logger.error(f"[{self.method}] ten_vad not installed. Install with: pip install git+https://github.com/TEN-framework/ten-vad.git")
            self._vad = None
        except Exception as e:
            logger.error(f"[{self.method}] Failed to initialize TEN-VAD: {e}")
            self._vad = None
    
    def detect(self, audio_data: bytes) -> bool:
        """
        Detect if speech is present in audio
        
        Args:
            audio_data: Raw audio bytes (16-bit PCM)
        
        Returns:
            True if speech detected, False otherwise
        """
        if not audio_data or len(audio_data) == 0:
            return False
        
        if self._vad is None:
            logger.warning(f"[{self.method}] VAD not initialized, returning False")
            return False
        
        try:
            # Convert bytes to numpy array (float32)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            # Process with TEN-VAD
            result = self._vad.process(audio_float)
            
            # result is a probability-like score, compare with threshold
            is_speech = result > self.threshold
            
            logger.debug(f"[{self.method}] Detection result: {result:.3f}, is_speech: {is_speech}")
            
            return is_speech
            
        except Exception as e:
            logger.error(f"[{self.method}] Detection failed: {e}")
            return False
    
    def update_state(self, has_speech: bool) -> dict:
        """
        Update speech state based on detection
        
        Args:
            has_speech: Whether speech was detected
        
        Returns:
            State dict with speech_started, speech_ended flags
        """
        import time
        
        current_time = time.time()
        state = {
            "speech_started": False,
            "speech_ended": False,
            "is_speaking": self.is_speaking
        }
        
        if has_speech:
            # Speech detected
            if not self.is_speaking:
                # Check if speech duration is long enough
                if self.speech_start_time is None:
                    self.speech_start_time = current_time
                    logger.debug(f"[{self.method}] Speech detected, starting timer")
                elif current_time - self.speech_start_time >= self.min_speech_duration:
                    # Speech started!
                    speech_duration = current_time - self.speech_start_time
                    self.is_speaking = True
                    state["speech_started"] = True
                    state["is_speaking"] = True
                    logger.info(f"[{self.method}] ✅ Speech started (duration: {speech_duration:.2f}s)")
            
            # Reset silence timer
            self.silence_start_time = None
        
        else:
            # No speech detected
            if self.is_speaking:
                # Check if silence duration is long enough
                if self.silence_start_time is None:
                    self.silence_start_time = current_time
                    logger.debug(f"[{self.method}] Silence detected, starting timer")
                elif current_time - self.silence_start_time >= self.min_silence_duration:
                    # Speech ended!
                    silence_duration = current_time - self.silence_start_time
                    self.is_speaking = False
                    state["speech_ended"] = True
                    state["is_speaking"] = False
                    logger.info(f"[{self.method}] ✅ Speech ended (silence: {silence_duration:.2f}s)")
            
            # Reset speech timer
            self.speech_start_time = None
        
        return state
    
    def reset(self):
        """Reset VAD state"""
        self.speech_start_time = None
        self.silence_start_time = None
        self.is_speaking = False
        logger.debug(f"[{self.method}] State reset")
    
    def is_available(self) -> bool:
        """Check if TEN-VAD is available"""
        return self._vad is not None
    
    def get_info(self) -> dict:
        """Get TEN-VAD information"""
        return {
            "method": self.method,
            "hop_size": self.hop_size,
            "threshold": self.threshold,
            "sample_rate": self.sample_rate,
            "min_speech_duration": self.min_speech_duration,
            "min_silence_duration": self.min_silence_duration,
            "available": self.is_available(),
            "library_size_kb": 306,  # TEN-VAD library size
            "is_speaking": self.is_speaking
        }
    
    @staticmethod
    def compare_with_silero():
        """
        Compare TEN-VAD with Silero VAD
        
        Returns:
            Comparison table
        """
        comparison = {
            "library_size": {
                "ten_vad": "306 KB",
                "silero": "2.16 MB",
                "improvement": "86% smaller"
            },
            "speed": {
                "ten_vad": "Baseline",
                "silero": "32% slower",
                "improvement": "32% faster"
            },
            "latency": {
                "ten_vad": "Lower",
                "silero": "Higher",
                "improvement": "Better end-of-sentence detection"
            }
        }
        return comparison
