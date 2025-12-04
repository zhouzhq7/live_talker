"""
Talker Thread - Background thread running LiveTalker
Communicates with UI through signal-slot mechanism
"""

import sys
import logging
from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import load_config_from_env
from core.talker import LiveTalker

logger = logging.getLogger(__name__)


class TalkerThread(QThread):
    """Background thread running LiveTalker"""
    
    # Signal definitions
    user_speech_detected = pyqtSignal()  # User started speaking
    asr_result = pyqtSignal(str)  # ASR recognition result
    llm_thinking = pyqtSignal()  # LLM thinking
    llm_response = pyqtSignal(str)  # LLM response
    tts_synthesizing = pyqtSignal()  # TTS synthesizing
    audio_playing = pyqtSignal()  # Audio playing
    audio_finished = pyqtSignal()  # Audio finished
    error_occurred = pyqtSignal(str)  # Error occurred
    state_changed = pyqtSignal(str)  # State changed: idle, listening, thinking, speaking
    models_loaded = pyqtSignal()  # Models loaded and ready
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.talker = None
        self._should_stop = False
    
    def run(self):
        """Run thread"""
        try:
            # Emit initializing state
            self.state_changed.emit("initializing")
            
            # Load configuration
            config = load_config_from_env()
            
            # Create LiveTalker instance (this will load all models)
            logger.info("Loading models...")
            self.talker = LiveTalker(config=config)
            
            # Save original methods
            original_on_utterance = self.talker._on_utterance_complete
            original_on_interrupt = self.talker._on_interrupt
            
            # Create wrapper methods
            def wrapped_on_utterance(audio_data: bytes):
                """Wrapped utterance completion callback"""
                try:
                    self.user_speech_detected.emit()
                    self.state_changed.emit("thinking")
                    
                    # Call original method, which handles ASR, LLM, TTS
                    original_on_utterance(audio_data)
                    
                    # After processing, state returns to listening (set in original method)
                    # But we update state by monitoring is_speaking
                except Exception as e:
                    logger.error(f"Error in wrapped_on_utterance: {e}")
                    self.error_occurred.emit(str(e))
            
            def wrapped_on_interrupt():
                """Wrapped interrupt callback"""
                self.state_changed.emit("listening")
                original_on_interrupt()
            
            # Replace callbacks
            self.talker._on_utterance_complete = wrapped_on_utterance
            self.talker._on_interrupt = wrapped_on_interrupt
            
            # Models are now loaded, emit signal
            logger.info("Models loaded successfully")
            self.models_loaded.emit()
            
            # Start system (recorder will start listening)
            self.state_changed.emit("listening")
            self.talker.recorder.start()
            self.talker.is_running = True
            
            # Play welcome message
            self.talker._play_welcome_message()
            
            # Monitor state changes
            import time
            last_state = "listening"
            while not self._should_stop and self.talker.is_running:
                # Monitor talker state and emit corresponding signals
                if self.talker.is_speaking:
                    if last_state != "speaking":
                        self.state_changed.emit("speaking")
                        self.audio_playing.emit()
                        last_state = "speaking"
                elif self.talker.is_processing:
                    if last_state != "thinking":
                        self.state_changed.emit("thinking")
                        self.llm_thinking.emit()
                        last_state = "thinking"
                else:
                    if last_state != "listening":
                        self.state_changed.emit("listening")
                        self.audio_finished.emit()
                        last_state = "listening"
                
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"TalkerThread error: {e}")
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(str(e))
        finally:
            if self.talker:
                self.talker.stop()
            self.state_changed.emit("idle")
    
    def stop_talker(self):
        """Stop talker"""
        self._should_stop = True
        if self.talker:
            self.talker.stop()
    
    def interrupt(self):
        """Interrupt current playback"""
        if self.talker:
            self.talker._on_interrupt()

