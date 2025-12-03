"""
Live Talker - Main Conversation Engine
整合 ASR + TTS + LLM + VAD 实现完整对话流程
参考 Eva/perception/audio/example_voice_assistant.py
"""

import logging
import threading
from typing import Optional
from config import TalkerConfig
from audio import RealtimeRecorder, VADDetector, AudioPlayer
from asr import BaseASR, FunASR, Whisper, FireRedASR
from tts import BaseTTS, EdgeTTS, Pyttsx3TTS
from llm import BaseLLM, DeepseekLLM
from core.conversation import ConversationManager

logger = logging.getLogger(__name__)


class LiveTalker:
    """
    Main conversation engine
    
    Features:
    - Real-time speech recognition
    - LLM-based conversation
    - Text-to-speech synthesis
    - Interruption handling
    - Conversation history management
    """
    
    def __init__(self, config: Optional[TalkerConfig] = None):
        """
        Initialize Live Talker
        
        Args:
            config: Configuration object
        """
        self.config = config or TalkerConfig()
        
        # Initialize components
        logger.info("=" * 70)
        logger.info("Initializing Live Talker...")
        logger.info("=" * 70)
        
        # VAD
        logger.info("Initializing VAD...")
        self.vad = VADDetector(
            method=self.config.vad.method,
            sample_rate=self.config.vad.sample_rate,
            threshold=self.config.vad.threshold,
            min_speech_duration=self.config.vad.min_speech_duration,
            min_silence_duration=self.config.vad.min_silence_duration,
            model_cache_dir=self.config.model_cache_dir
        )
        
        # ASR
        logger.info(f"Initializing ASR engine: {self.config.asr.engine}...")
        self.asr = self._create_asr()
        
        # TTS
        logger.info(f"Initializing TTS engine: {self.config.tts.engine}...")
        self.tts = self._create_tts()
        
        # LLM
        logger.info(f"Initializing LLM: {self.config.llm.provider}...")
        self.llm = self._create_llm()
        
        # Audio components
        logger.info("Initializing audio components...")
        self.recorder = RealtimeRecorder(
            vad_detector=self.vad,
            sample_rate=self.config.audio.sample_rate,
            chunk_size=self.config.audio.chunk_size,
            on_utterance_complete=self._on_utterance_complete,
            on_interrupt=self._on_interrupt
        )
        
        self.player = AudioPlayer(
            sample_rate=self.config.audio.sample_rate,
            channels=self.config.audio.channels
        )
        
        # Conversation manager
        self.conversation = ConversationManager(
            max_history=self.config.max_conversation_history
        )
        
        # State
        self.is_running = False
        self.is_processing = False
        self.is_speaking = False
        
        # Set system prompt
        self.conversation.set_system_prompt(
            "你是一个友好的AI助手。请用简洁、自然的语言回答问题。"
        )
        
        logger.info("=" * 70)
        logger.info("Live Talker initialized successfully!")
        logger.info("=" * 70)
    
    def _create_asr(self) -> BaseASR:
        """Create ASR engine based on config"""
        engine = self.config.asr.engine.lower()
        
        if engine == "funasr":
            return FunASR(
                model_name=self.config.asr.funasr_model,
                device=self.config.asr.funasr_device,
                enable_vad=self.config.asr.funasr_enable_vad,
                model_cache_dir=self.config.model_cache_dir
            )
        elif engine == "whisper":
            return Whisper(
                model_name=self.config.asr.whisper_model,
                device=self.config.asr.whisper_device,
                model_cache_dir=self.config.model_cache_dir
            )
        elif engine == "fireredasr":
            return FireRedASR(
                model_name=self.config.asr.fireredasr_model,
                device=self.config.asr.fireredasr_device
            )
        else:
            logger.warning(f"Unknown ASR engine: {engine}, using FunASR")
            return FunASR(model_cache_dir=self.config.model_cache_dir)
    
    def _create_tts(self) -> BaseTTS:
        """Create TTS engine based on config"""
        engine = self.config.tts.engine.lower()
        
        if engine == "edge":
            return EdgeTTS(
                voice=self.config.tts.edge_voice,
                rate=self.config.tts.edge_rate,
                volume=self.config.tts.edge_volume
            )
        elif engine == "pyttsx3":
            return Pyttsx3TTS(
                rate=self.config.tts.pyttsx3_rate,
                volume=self.config.tts.pyttsx3_volume
            )
        else:
            logger.warning(f"Unknown TTS engine: {engine}, using Edge-TTS")
            return EdgeTTS()
    
    def _create_llm(self) -> BaseLLM:
        """Create LLM provider based on config"""
        provider = self.config.llm.provider.lower()
        
        if provider == "deepseek":
            return DeepseekLLM(
                api_key=self.config.llm.api_key,
                api_base=self.config.llm.api_base,
                model=self.config.llm.model,
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens
            )
        else:
            logger.warning(f"Unknown LLM provider: {provider}, using Deepseek")
            return DeepseekLLM()
    
    def start(self):
        """Start the conversation system"""
        if self.is_running:
            logger.warning("[LiveTalker] Already running")
            return
        
        # Check components
        if not self.asr.is_available():
            logger.error("[LiveTalker] ASR not available")
            return
        
        if not self.tts.is_available():
            logger.error("[LiveTalker] TTS not available")
            return
        
        if not self.llm.is_available():
            logger.error("[LiveTalker] LLM not available")
            return
        
        logger.info("[LiveTalker] Starting conversation system...")
        
        # Start recorder
        self.recorder.start()
        self.is_running = True
        
        logger.info("[LiveTalker] System started! Listening for speech...")
        logger.info("[LiveTalker] Press Ctrl+C to stop")
        
        # Play welcome message
        self._play_welcome_message()
        
        # Keep running until interrupted
        try:
            import time
            while self.is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("[LiveTalker] Interrupted by user")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the conversation system"""
        if not self.is_running:
            return
        
        logger.info("[LiveTalker] Stopping...")
        
        self.is_running = False
        
        # Stop recorder
        self.recorder.stop()
        
        # Stop player
        self.player.stop()
        
        logger.info("[LiveTalker] Stopped")
    
    def _on_utterance_complete(self, audio_data: bytes):
        """
        Callback when user utterance is complete
        
        Args:
            audio_data: Complete audio utterance
        """
        if self.is_processing:
            logger.warning("[LiveTalker] Already processing, skipping")
            return
        
        self.is_processing = True
        
        try:
            logger.info("[LiveTalker] Processing user utterance...")
            
            # ASR: Speech to text
            text = self.asr.transcribe(
                audio_data,
                sample_rate=self.config.audio.sample_rate,
                language=self.config.asr.language
            )
            
            if not text or not text.strip():
                logger.warning("[LiveTalker] No text recognized")
                self.is_processing = False
                return
            
            # Log user message
            print("\n" + "=" * 70)
            print("👤 用户:", text)
            print("=" * 70)
            logger.info(f"[对话] 用户: {text}")
            
            # LLM: Generate response
            logger.info("[LiveTalker] Generating response...")
            response = self.llm.chat(
                user_message=text,
                system_prompt=self.conversation.system_prompt,
                stream=False
            )
            
            if not response or not response.strip():
                logger.warning("[LiveTalker] No response generated")
                self.is_processing = False
                return
            
            # Log assistant response
            print("\n" + "=" * 70)
            print("🤖 助手:", response)
            print("=" * 70 + "\n")
            logger.info(f"[对话] 助手: {response}")
            
            # TTS: Text to speech
            logger.info("[LiveTalker] Synthesizing speech...")
            audio_output = self.tts.synthesize(response)
            
            if not audio_output:
                logger.warning("[LiveTalker] TTS synthesis failed")
                self.is_processing = False
                return
            
            # Play audio
            logger.info("[LiveTalker] Playing response...")
            self.is_speaking = True
            self.recorder.set_system_speaking(True)
            
            self.player.play_bytes(audio_output, blocking=True)
            
            self.is_speaking = False
            self.recorder.set_system_speaking(False)
            
            logger.info("[LiveTalker] Response complete")
            
        except Exception as e:
            logger.error(f"[LiveTalker] Error processing utterance: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_processing = False
    
    def _play_welcome_message(self):
        """Play welcome message when system starts"""
        welcome_text = self.config.welcome_message
        
        if not welcome_text or not welcome_text.strip():
            return
        
        try:
            # Log welcome message
            print("\n" + "=" * 70)
            print("🤖 助手:", welcome_text)
            print("=" * 70 + "\n")
            logger.info(f"[对话] 助手(欢迎语): {welcome_text}")
            
            # TTS: Synthesize welcome message
            audio_output = self.tts.synthesize(welcome_text)
            
            if not audio_output:
                logger.warning("[LiveTalker] Welcome message TTS synthesis failed")
                return
            
            # Play audio
            self.is_speaking = True
            self.recorder.set_system_speaking(True)
            
            self.player.play_bytes(audio_output, blocking=True)
            
            self.is_speaking = False
            self.recorder.set_system_speaking(False)
            
            logger.info("[LiveTalker] Welcome message played")
            
        except Exception as e:
            logger.error(f"[LiveTalker] Error playing welcome message: {e}")
            import traceback
            traceback.print_exc()
            self.is_speaking = False
            self.recorder.set_system_speaking(False)
    
    def _on_interrupt(self):
        """Callback when user interrupts system speech"""
        logger.info("[LiveTalker] User interruption detected!")
        
        # Stop current playback
        self.player.stop()
        self.is_speaking = False
        self.recorder.set_system_speaking(False)
        
        # Clear buffer
        self.recorder.clear_buffer()

