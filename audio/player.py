"""
Audio Player for TTS Output
参考 voice_benchmark/realtime/audio_player.py
"""

import logging
import threading
import queue
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


class AudioPlayer:
    """
    Audio player for real-time TTS output
    
    Features:
    - Play audio from bytes or file
    - Support for streaming audio
    - Interruptible playback
    - Background playback thread
    """
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        """
        Initialize audio player
        
        Args:
            sample_rate: Audio sample rate
            channels: Number of audio channels
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.is_playing = False
        self.is_initialized = False
        
        self._audio_queue = queue.Queue()
        self._play_thread = None
        self._stop_event = threading.Event()
        
        # Try to initialize PyAudio
        try:
            import pyaudio
            self._pyaudio = pyaudio
            self._audio_interface = pyaudio.PyAudio()
            self._stream = None
            self.is_initialized = True
            logger.info("[AudioPlayer] Initialized")
        except ImportError:
            logger.error("[AudioPlayer] PyAudio not installed")
            self._pyaudio = None
            self._audio_interface = None
        except Exception as e:
            logger.error(f"[AudioPlayer] Initialization failed: {e}")
            self._pyaudio = None
            self._audio_interface = None
    
    def play_bytes(self, audio_data: bytes, blocking: bool = False) -> None:
        """
        Play audio from bytes
        
        Args:
            audio_data: Audio data as bytes (16-bit PCM)
            blocking: If True, wait for playback to complete
        """
        if not self.is_initialized:
            logger.warning("[AudioPlayer] Not initialized")
            return
        
        if not audio_data:
            return
        
        if blocking:
            self._play_sync(audio_data)
        else:
            self._play_async(audio_data)
    
    def play_file(self, audio_file: str, blocking: bool = False) -> None:
        """
        Play audio from file
        
        Args:
            audio_file: Path to audio file
            blocking: If True, wait for playback to complete
        """
        try:
            import soundfile as sf
            audio_data, sr = sf.read(audio_file)
            
            # Convert to int16 bytes
            audio_int16 = (audio_data * 32768).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            
            self.play_bytes(audio_bytes, blocking=blocking)
            
        except Exception as e:
            logger.error(f"[AudioPlayer] Failed to play file: {e}")
    
    def _play_sync(self, audio_data: bytes) -> None:
        """Play audio synchronously"""
        if not self._audio_interface:
            return
        
        try:
            # Open stream
            stream = self._audio_interface.open(
                format=self._pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                output=True
            )
            
            self.is_playing = True
            
            # Play audio
            chunk_size = 1024
            for i in range(0, len(audio_data), chunk_size):
                if self._stop_event.is_set():
                    break
                chunk = audio_data[i:i + chunk_size]
                stream.write(chunk)
            
            # Close stream
            stream.stop_stream()
            stream.close()
            
            self.is_playing = False
            
        except Exception as e:
            logger.error(f"[AudioPlayer] Playback failed: {e}")
            self.is_playing = False
    
    def _play_async(self, audio_data: bytes) -> None:
        """Play audio asynchronously"""
        # Add to queue
        self._audio_queue.put(audio_data)
        
        # Start playback thread if not running
        if self._play_thread is None or not self._play_thread.is_alive():
            self._stop_event.clear()
            self._play_thread = threading.Thread(target=self._playback_worker)
            self._play_thread.daemon = True
            self._play_thread.start()
    
    def _playback_worker(self) -> None:
        """Background playback worker thread"""
        if not self._audio_interface:
            return
        
        try:
            # Open stream
            stream = self._audio_interface.open(
                format=self._pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=1024
            )
            
            self.is_playing = True
            
            # Process queue
            while not self._stop_event.is_set():
                try:
                    # Get audio data with timeout
                    audio_data = self._audio_queue.get(timeout=0.1)
                    
                    # Play audio
                    chunk_size = 1024
                    for i in range(0, len(audio_data), chunk_size):
                        if self._stop_event.is_set():
                            break
                        chunk = audio_data[i:i + chunk_size]
                        stream.write(chunk)
                    
                    self._audio_queue.task_done()
                    
                except queue.Empty:
                    # No more audio in queue, check if should exit
                    if self._audio_queue.empty():
                        break
            
            # Close stream
            stream.stop_stream()
            stream.close()
            
            self.is_playing = False
            
        except Exception as e:
            logger.error(f"[AudioPlayer] Playback worker failed: {e}")
            self.is_playing = False
    
    def stop(self) -> None:
        """Stop playback"""
        self._stop_event.set()
        
        # Clear queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
                self._audio_queue.task_done()
            except queue.Empty:
                break
        
        # Wait for thread
        if self._play_thread and self._play_thread.is_alive():
            self._play_thread.join(timeout=1.0)
        
        self.is_playing = False
        logger.debug("[AudioPlayer] Stopped")
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        self.stop()
        
        if self._audio_interface:
            try:
                self._audio_interface.terminate()
            except:
                pass
    
    def __del__(self):
        """Cleanup on deletion"""
        self.cleanup()

