"""
Voice Activity Detection (VAD)
Detect speech presence and support user interruption
参考 Eva/perception/audio/vad_detector.py
"""

import logging
import numpy as np
import os
from pathlib import Path
from typing import Optional, Callable
import time

logger = logging.getLogger(__name__)


class VADDetector:
    """
    Voice Activity Detector
    
    Features:
    - Detect speech presence in audio
    - Support user interruption detection
    - Energy-based and model-based VAD
    """
    
    def __init__(
        self,
        method: str = "silero",  # "silero", "webrtc", or "energy"
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_speech_duration: float = 0.25,  # 250ms
        min_silence_duration: float = 0.5,   # 500ms
        model_cache_dir: Optional[str] = None
    ):
        """
        Initialize VAD detector
        
        Args:
            method: VAD method (silero, webrtc, energy)
            sample_rate: Audio sample rate
            threshold: Detection threshold
            min_speech_duration: Minimum speech duration to trigger
            min_silence_duration: Minimum silence to end speech
            model_cache_dir: Base directory for model cache (default: D:\\models)
        """
        self.method = method
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_speech_duration = min_speech_duration
        self.min_silence_duration = min_silence_duration
        self.model_cache_dir = model_cache_dir or os.getenv("MODEL_CACHE_DIR", "D:\\models")
        
        self.model = None
        self.speech_start_time = None
        self.silence_start_time = None
        self.is_speaking = False
        
        # Setup Torch Hub cache directory
        self._setup_torch_cache()
        
        # Initialize VAD model
        self._init_vad()
    
    def _setup_torch_cache(self):
        """Setup Torch Hub cache directory"""
        try:
            torch_cache = os.path.join(self.model_cache_dir, "torch")
            os.makedirs(torch_cache, exist_ok=True)
            
            # Set Torch Hub cache directory via environment variable
            os.environ["TORCH_HOME"] = torch_cache
            
            logger.info(f"[VAD] Torch Hub cache directory: {torch_cache}")
        except Exception as e:
            logger.warning(f"[VAD] Failed to setup Torch cache: {e}")
    
    def _init_vad(self):
        """Initialize VAD model based on method"""
        if self.method == "silero":
            try:
                import torch
                # Load Silero VAD model
                self.model, utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False
                )
                self._has_silero = True
                logger.info("[VAD] Silero VAD loaded")
            except Exception as e:
                self._has_silero = False
                logger.warning(f"[VAD] Silero VAD not available: {e}")
                logger.info("[VAD] Falling back to energy-based VAD")
                self.method = "energy"
        
        elif self.method == "webrtc":
            try:
                import webrtcvad
                self.model = webrtcvad.Vad(3)  # Aggressiveness: 0-3
                self._has_webrtc = True
                logger.info("[VAD] WebRTC VAD loaded")
            except Exception as e:
                self._has_webrtc = False
                logger.warning(f"[VAD] WebRTC VAD not available: {e}")
                logger.info("[VAD] Falling back to energy-based VAD")
                self.method = "energy"
        
        else:
            # Energy-based VAD (always available)
            logger.info("[VAD] Using energy-based VAD")
    
    def detect(self, audio_data: bytes) -> bool:
        """
        Detect if speech is present in audio
        
        Args:
            audio_data: Raw audio bytes
        
        Returns:
            True if speech detected, False otherwise
        """
        if not audio_data:
            return False
        
        if self.method == "silero":
            result = self._detect_silero(audio_data)
        elif self.method == "webrtc":
            result = self._detect_webrtc(audio_data)
        else:
            result = self._detect_energy(audio_data)
        
        logger.debug(f"[VAD] 检测结果: {'有语音' if result else '无语音'} (方法: {self.method}, 音频长度: {len(audio_data)} bytes)")
        return result
    
    def _detect_silero(self, audio_data: bytes) -> bool:
        """Silero VAD detection"""
        if not hasattr(self, '_has_silero') or not self._has_silero or not self.model:
            return self._detect_energy(audio_data)
        
        try:
            import torch
            
            # Convert to float32
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            # Silero VAD requires exactly 512 samples for 16kHz (or 256 for 8kHz)
            required_samples = 512 if self.sample_rate == 16000 else 256
            
            # If audio is shorter than required, pad with zeros
            if len(audio_float) < required_samples:
                audio_float = np.pad(audio_float, (0, required_samples - len(audio_float)), mode='constant')
            
            # Process in chunks of required_samples
            max_prob = 0.0
            for i in range(0, len(audio_float), required_samples):
                chunk = audio_float[i:i + required_samples]
                
                # Pad last chunk if needed
                if len(chunk) < required_samples:
                    chunk = np.pad(chunk, (0, required_samples - len(chunk)), mode='constant')
                
                # Convert to tensor
                audio_tensor = torch.from_numpy(chunk)
                
                # Get speech probability
                speech_prob = self.model(audio_tensor, self.sample_rate).item()
                max_prob = max(max_prob, speech_prob)
            
            return max_prob > self.threshold
            
        except Exception as e:
            logger.error(f"[VAD] Silero detection failed: {e}")
            import traceback
            traceback.print_exc()
            return self._detect_energy(audio_data)
    
    def _detect_webrtc(self, audio_data: bytes) -> bool:
        """WebRTC VAD detection"""
        if not hasattr(self, '_has_webrtc') or not self._has_webrtc or not self.model:
            return self._detect_energy(audio_data)
        
        try:
            # WebRTC VAD requires specific frame sizes
            # 10, 20, or 30 ms frames at 8000, 16000, 32000, or 48000 Hz
            frame_duration = 30  # ms
            frame_size = int(self.sample_rate * frame_duration / 1000) * 2  # bytes
            
            # Process in chunks
            is_speech = False
            for i in range(0, len(audio_data), frame_size):
                chunk = audio_data[i:i + frame_size]
                if len(chunk) == frame_size:
                    if self.model.is_speech(chunk, self.sample_rate):
                        is_speech = True
                        break
            
            return is_speech
            
        except Exception as e:
            logger.error(f"[VAD] WebRTC detection failed: {e}")
            return self._detect_energy(audio_data)
    
    def _detect_energy(self, audio_data: bytes) -> bool:
        """Energy-based VAD detection (fallback)"""
        try:
            # Convert to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            # Calculate RMS energy
            energy = np.sqrt(np.mean(audio_float ** 2))
            
            # Simple threshold-based detection
            return energy > (self.threshold * 0.02)  # Adjust threshold for energy
            
        except Exception as e:
            logger.error(f"[VAD] Energy detection failed: {e}")
            return False
    
    def update_state(self, has_speech: bool) -> dict:
        """
        Update speech state based on detection
        
        Args:
            has_speech: Whether speech was detected
        
        Returns:
            State dict with speech_started, speech_ended flags
        """
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
                    logger.debug(f"[VAD] 检测到语音，开始计时 (min_duration={self.min_speech_duration}s)")
                elif current_time - self.speech_start_time >= self.min_speech_duration:
                    # Speech started!
                    speech_duration = current_time - self.speech_start_time
                    self.is_speaking = True
                    state["speech_started"] = True
                    state["is_speaking"] = True
                    logger.info(f"[VAD] ✅ 语音开始 (持续时长: {speech_duration:.2f}s, 方法: {self.method})")
            
            # Reset silence timer
            self.silence_start_time = None
        
        else:
            # No speech detected
            if self.is_speaking:
                # Check if silence duration is long enough
                if self.silence_start_time is None:
                    self.silence_start_time = current_time
                    logger.debug(f"[VAD] 检测到静音，开始计时 (min_silence={self.min_silence_duration}s)")
                elif current_time - self.silence_start_time >= self.min_silence_duration:
                    # Speech ended!
                    silence_duration = current_time - self.silence_start_time
                    self.is_speaking = False
                    state["speech_ended"] = True
                    state["is_speaking"] = False
                    logger.info(f"[VAD] ✅ 语音结束 (静音时长: {silence_duration:.2f}s, 方法: {self.method})")
            
            # Reset speech timer
            self.speech_start_time = None
        
        return state
    
    def reset(self):
        """Reset VAD state"""
        self.speech_start_time = None
        self.silence_start_time = None
        self.is_speaking = False
        logger.debug("[VAD] State reset")


class InterruptionDetector:
    """
    Detect user interruption during system speech
    """
    
    def __init__(
        self,
        vad_detector: VADDetector,
        interruption_threshold: float = 0.3,  # 300ms of speech triggers interruption
        on_interrupt: Optional[Callable] = None
    ):
        """
        Initialize interruption detector
        
        Args:
            vad_detector: VAD detector instance
            interruption_threshold: Time threshold for interruption
            on_interrupt: Callback function when interruption detected
        """
        self.vad = vad_detector
        self.interruption_threshold = interruption_threshold
        self.on_interrupt = on_interrupt
        
        self.is_system_speaking = False
        self.interrupt_start_time = None
    
    def set_system_speaking(self, is_speaking: bool):
        """
        Set whether system is currently speaking
        
        Args:
            is_speaking: True if system is speaking
        """
        self.is_system_speaking = is_speaking
        if not is_speaking:
            self.interrupt_start_time = None
    
    def check_interruption(self, audio_data: bytes) -> bool:
        """
        Check if user is interrupting system speech
        
        Args:
            audio_data: Audio data to check
        
        Returns:
            True if interruption detected
        """
        if not self.is_system_speaking:
            return False
        
        # Detect speech
        has_speech = self.vad.detect(audio_data)
        
        if has_speech:
            current_time = time.time()
            
            if self.interrupt_start_time is None:
                self.interrupt_start_time = current_time
            elif current_time - self.interrupt_start_time >= self.interruption_threshold:
                # Interruption detected!
                logger.info("[Interruption] User interrupted system speech")
                
                if self.on_interrupt:
                    self.on_interrupt()
                
                self.interrupt_start_time = None
                return True
        else:
            self.interrupt_start_time = None
        
        return False

