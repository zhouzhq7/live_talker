"""
Real-time Audio Recorder
Continuous recording with VAD-based segmentation
参考 Eva/perception/audio/realtime_recorder.py
"""

import logging
import threading
import queue
import time
from typing import Optional, Callable
import numpy as np

logger = logging.getLogger(__name__)


class RealtimeRecorder:
    """
    Real-time audio recorder with VAD-based automatic segmentation
    
    Features:
    - Continuous recording in background thread
    - Automatic speech segmentation using VAD
    - Callback when complete utterance detected
    - Support for user interruption
    """
    
    def __init__(
        self,
        vad_detector,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        on_utterance_complete: Optional[Callable] = None,
        on_interrupt: Optional[Callable] = None
    ):
        """
        Initialize real-time recorder
        
        Args:
            vad_detector: VAD detector instance
            sample_rate: Audio sample rate
            chunk_size: Audio chunk size
            on_utterance_complete: Callback(audio_data) when utterance complete
            on_interrupt: Callback() when user interrupts
        """
        self.vad = vad_detector
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.on_utterance_complete = on_utterance_complete
        self.on_interrupt = on_interrupt
        
        self.audio_queue = queue.Queue()
        self.current_utterance = []
        self.pre_speech_buffer = []  # Buffer to store audio before speech_started
        self.pre_speech_buffer_size = max(20, int(1.0 * sample_rate / chunk_size))  # ~1 second of audio
        self.is_recording = False
        self.is_system_speaking = False
        
        self.process_thread = None
        
        # Initialize PyAudio
        try:
            import pyaudio
            self._pyaudio = pyaudio
            self._audio_interface = None
            self._stream = None
            self._has_pyaudio = True
            logger.info("[RealtimeRecorder] PyAudio available")
        except ImportError:
            self._has_pyaudio = False
            logger.warning("[RealtimeRecorder] PyAudio not available")
    
    def start(self):
        """Start real-time recording"""
        if self.is_recording:
            logger.warning("[RealtimeRecorder] Already recording")
            return
        
        if not self._has_pyaudio:
            logger.error("[RealtimeRecorder] Cannot start without PyAudio")
            return
        
        try:
            # Initialize audio stream
            self._audio_interface = self._pyaudio.PyAudio()
            self._stream = self._audio_interface.open(
                format=self._pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            self.is_recording = True
            
            # Start processing thread
            self.process_thread = threading.Thread(
                target=self._process_audio,
                daemon=True
            )
            self.process_thread.start()
            
            self._stream.start_stream()
            
            logger.info("[RealtimeRecorder] Started")
            
        except Exception as e:
            logger.error(f"[RealtimeRecorder] Failed to start: {e}")
            self.is_recording = False
    
    def stop(self):
        """Stop real-time recording"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        # Stop stream
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception as e:
                logger.error(f"[RealtimeRecorder] Error stopping stream: {e}")
        
        # Terminate audio interface
        if self._audio_interface:
            try:
                self._audio_interface.terminate()
            except Exception as e:
                logger.error(f"[RealtimeRecorder] Error terminating audio: {e}")
        
        # Wait for processing thread
        if self.process_thread and self.process_thread.is_alive():
            self.process_thread.join(timeout=2.0)
        
        logger.info("[RealtimeRecorder] Stopped")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for audio chunks"""
        if self.is_recording:
            self.audio_queue.put(in_data)
        
        return (None, self._pyaudio.paContinue)
    
    def _process_audio(self):
        """Process audio chunks in background thread"""
        logger.info("[RealtimeRecorder] Processing thread started")
        
        while self.is_recording:
            try:
                # Get audio chunk (with timeout)
                audio_chunk = self.audio_queue.get(timeout=0.1)
                
                # Check for interruption if system is speaking
                if self.is_system_speaking:
                    has_speech = self.vad.detect(audio_chunk)
                    if has_speech:
                        logger.warning("=" * 70)
                        logger.warning("⚠️  [VAD打断] 检测到用户打断系统语音！")
                        logger.warning(f"   - 音频块大小: {len(audio_chunk)} bytes")
                        logger.warning(f"   - VAD方法: {self.vad.method}")
                        logger.warning("=" * 70)
                        if self.on_interrupt:
                            self.on_interrupt()
                        self.is_system_speaking = False
                        continue
                
                # Normal VAD processing
                has_speech = self.vad.detect(audio_chunk)
                state = self.vad.update_state(has_speech)
                
                # Always maintain pre-speech buffer (circular buffer)
                self.pre_speech_buffer.append(audio_chunk)
                if len(self.pre_speech_buffer) > self.pre_speech_buffer_size:
                    self.pre_speech_buffer.pop(0)  # Remove oldest chunk
                
                if state["is_speaking"] or state["speech_started"]:
                    # When speech starts, include pre-speech buffer
                    if state["speech_started"] and self.pre_speech_buffer:
                        logger.debug(f"[RealtimeRecorder] Speech started, adding {len(self.pre_speech_buffer)} pre-speech chunks")
                        self.current_utterance.extend(self.pre_speech_buffer)
                        self.pre_speech_buffer = []  # Clear buffer after using
                    
                    # Collect current speech chunk
                    self.current_utterance.append(audio_chunk)
                
                if state["speech_ended"] and self.current_utterance:
                    # Complete utterance detected
                    utterance_data = b''.join(self.current_utterance)
                    utterance_duration = len(utterance_data) / (self.sample_rate * 2)  # 16-bit = 2 bytes per sample
                    num_chunks = len(self.current_utterance)
                    
                    logger.info("=" * 70)
                    logger.info("🎤 [VAD] 完整语音片段检测完成")
                    logger.info(f"   - 音频时长: {utterance_duration:.2f}s")
                    logger.info(f"   - 音频大小: {len(utterance_data)} bytes")
                    logger.info(f"   - 音频块数: {num_chunks}")
                    logger.info(f"   - VAD方法: {self.vad.method}")
                    logger.info("=" * 70)
                    
                    # Call callback
                    if self.on_utterance_complete:
                        try:
                            self.on_utterance_complete(utterance_data)
                        except Exception as e:
                            logger.error(f"[RealtimeRecorder] Callback error: {e}")
                    
                    # Reset
                    self.current_utterance = []
                    self.pre_speech_buffer = []  # Also clear pre-speech buffer
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[RealtimeRecorder] Processing error: {e}")
        
        logger.info("[RealtimeRecorder] Processing thread stopped")
    
    def set_system_speaking(self, is_speaking: bool):
        """
        Set whether system is currently speaking
        
        Args:
            is_speaking: True if system is speaking
        """
        self.is_system_speaking = is_speaking
        logger.debug(f"[RealtimeRecorder] System speaking: {is_speaking}")
    
    def clear_buffer(self):
        """Clear current utterance buffer"""
        self.current_utterance = []
        self.pre_speech_buffer = []
        self.vad.reset()
        logger.debug("[RealtimeRecorder] Buffer cleared")

