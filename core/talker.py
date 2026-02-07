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
from asr import BaseASR, SenseVoice, FunASR, Whisper, FireRedASR
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
            "你是一个友好的AI语音助手，你的任务是生成自然、流畅的对话文本，这些文本将被转换为语音播放给用户。\n"
            "重要要求：\n"
            "1. 使用自然、口语化的表达方式，就像日常对话一样\n"
            "2. 保持简洁明了，避免冗长的句子\n"
            "3. 语气友好、亲切，符合AI助手的角色定位\n"
            "4. 绝对不要使用任何emoji表情符号、特殊符号或标记符号\n"
            "5. 只输出纯文本内容，适合直接用于语音合成\n"
            "6. 回答要自然流畅，符合中文口语习惯"
        )
        
        logger.info("=" * 70)
        logger.info("Live Talker initialized successfully!")
        logger.info("=" * 70)
    
    def _create_asr(self) -> BaseASR:
        """Create ASR engine based on config"""
        engine = self.config.asr.engine.lower()
        
        if engine == "sensevoice":
            return SenseVoice(
                model_name=self.config.asr.sensevoice_model,
                device=self.config.asr.sensevoice_device,
                language=self.config.asr.sensevoice_language,
                enable_vad=self.config.asr.sensevoice_enable_vad,
                model_cache_dir=self.config.model_cache_dir
            )
        elif engine == "funasr":
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
            logger.warning(f"Unknown ASR engine: {engine}, using SenseVoice")
            return SenseVoice(model_cache_dir=self.config.model_cache_dir)
    
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
            import time
            process_start_time = time.time()
            
            logger.info("[LiveTalker] Processing user utterance...")
            
            # ASR: Speech to text
            asr_start_time = time.time()
            text = self.asr.transcribe(
                audio_data,
                sample_rate=self.config.audio.sample_rate,
                language=self.config.asr.language
            )
            asr_elapsed = time.time() - asr_start_time
            
            if not text or not text.strip():
                logger.warning(f"[LiveTalker] No text recognized (ASR耗时: {asr_elapsed:.2f}s)")
                self.is_processing = False
                return
            
            # Log user message
            print("\n" + "=" * 70)
            print("👤 用户:", text)
            print("=" * 70)
            logger.info(f"[对话] 用户: {text}")
            
            # Log ASR details
            asr_duration = len(audio_data) / (self.config.audio.sample_rate * 2)  # 16-bit = 2 bytes
            logger.info(f"[ASR] ✅ 识别完成 (耗时: {asr_elapsed:.2f}s, 音频时长: {asr_duration:.2f}s, 文本长度: {len(text)} 字符)")
            
            # LLM: Generate response
            logger.info("[LiveTalker] 调用LLM生成回复...")
            llm_start_time = time.time()
            
            response = self.llm.chat(
                user_message=text,
                system_prompt=self.conversation.system_prompt,
                stream=False
            )
            
            llm_elapsed = time.time() - llm_start_time
            
            if not response or not response.strip():
                logger.warning(f"[LiveTalker] LLM未生成有效回复 (耗时: {llm_elapsed:.2f}s)")
                self.is_processing = False
                return
            
            logger.info(f"[LiveTalker] LLM生成完成 (耗时: {llm_elapsed:.2f}s)")
            
            # Log assistant response
            print("\n" + "=" * 70)
            print("🤖 助手:", response)
            print("=" * 70 + "\n")
            logger.info(f"[对话] 助手: {response}")
            
            # TTS: Text to speech
            logger.info("[LiveTalker] 开始TTS语音合成...")
            tts_start_time = time.time()
            
            audio_output = self.tts.synthesize(response)
            
            tts_elapsed = time.time() - tts_start_time
            
            if not audio_output:
                logger.warning(f"[LiveTalker] TTS合成失败 (耗时: {tts_elapsed:.2f}s)")
                self.is_processing = False
                return
            
            audio_duration = len(audio_output) / (self.config.audio.sample_rate * 2)  # 16-bit = 2 bytes
            logger.info(f"[TTS] ✅ 合成完成 (耗时: {tts_elapsed:.2f}s, 音频时长: {audio_duration:.2f}s, 大小: {len(audio_output)} bytes)")
            
            # Validate audio output format
            if len(audio_output) == 0:
                logger.error("[LiveTalker] ❌ 音频数据为空，无法播放")
                self.is_processing = False
                return
            
            # Check if audio data looks like PCM (should be divisible by 2 for 16-bit)
            if len(audio_output) % 2 != 0:
                logger.warning(f"[LiveTalker] ⚠️ 音频数据大小异常 (不是16-bit对齐): {len(audio_output)} bytes")
            
            # Play audio
            logger.info("[LiveTalker] 开始播放回复...")
            logger.debug(f"[LiveTalker] 播放参数 - 采样率: {self.config.audio.sample_rate}Hz, 声道数: {self.config.audio.channels}, 数据大小: {len(audio_output)} bytes")
            play_start_time = time.time()
            self.is_speaking = True
            self.recorder.set_system_speaking(True)
            
            try:
                self.player.play_bytes(audio_output, blocking=True)
            except Exception as e:
                logger.error(f"[LiveTalker] ❌ 播放失败: {e}")
                import traceback
                traceback.print_exc()
                self.is_speaking = False
                self.recorder.set_system_speaking(False)
                self.is_processing = False
                return
            
            play_elapsed = time.time() - play_start_time
            self.is_speaking = False
            self.recorder.set_system_speaking(False)
            
            logger.info(f"[LiveTalker] ✅ 回复播放完成 (播放耗时: {play_elapsed:.2f}s)")
            
            # Log total processing time
            total_time = time.time() - process_start_time
            logger.info("=" * 70)
            logger.info(f"✅ [LiveTalker] 完整处理流程完成")
            logger.info(f"   - 总耗时: {total_time:.2f}s")
            logger.info(f"   - ASR: {asr_elapsed:.2f}s")
            logger.info(f"   - LLM: {llm_elapsed:.2f}s")
            logger.info(f"   - TTS: {tts_elapsed:.2f}s")
            logger.info(f"   - 播放: {play_elapsed:.2f}s")
            logger.info("=" * 70)
            
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
        logger.warning("=" * 70)
        logger.warning("🛑 [打断处理] 用户打断系统语音")
        logger.warning("   - 停止当前播放")
        logger.warning("   - 清空音频缓冲区")
        logger.warning("   - 重置VAD状态")
        logger.warning("=" * 70)
        
        # Stop current playback
        self.player.stop()
        self.is_speaking = False
        self.recorder.set_system_speaking(False)
        
        # Clear buffer
        self.recorder.clear_buffer()
        
        logger.info("[打断处理] 系统已准备好接收新的用户输入")

