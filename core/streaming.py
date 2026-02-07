"""
Streaming Pipeline for Live Talker

实现流式处理以降低端到端延迟：
- 增量ASR：滑动窗口识别
- 流式LLM：实时输出
- 分段TTS：逐句合成播放

目标：端到端延迟 < 1秒
"""

import logging
import threading
import queue
import time
import re
from typing import Optional, Callable, Iterator, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum, auto

logger = logging.getLogger(__name__)


class StreamState(Enum):
    """Streaming pipeline states"""
    IDLE = auto()
    LISTENING = auto()      # 正在听用户说话
    RECOGNIZING = auto()    # ASR识别中
    THINKING = auto()       # LLM生成中
    SYNTHESIZING = auto()   # TTS合成中
    SPEAKING = auto()       # 正在播放语音
    INTERRUPTED = auto()    # 被打断


@dataclass
class StreamingMetrics:
    """Performance metrics for streaming pipeline"""
    # Timing
    first_asr_latency_ms: float = 0
    first_llm_latency_ms: float = 0
    first_tts_latency_ms: float = 0
    total_latency_ms: float = 0
    
    # Counters
    audio_chunks_processed: int = 0
    text_chunks_generated: int = 0
    audio_segments_played: int = 0
    
    # Rates
    asr_rtf: float = 0
    tts_speed: float = 0
    
    def log_summary(self):
        """Log performance summary"""
        logger.info("="*60)
        logger.info("📊 Streaming Pipeline Metrics")
        logger.info("="*60)
        logger.info(f"延迟指标:")
        logger.info(f"   首字ASR: {self.first_asr_latency_ms:.0f}ms")
        logger.info(f"   首token LLM: {self.first_llm_latency_ms:.0f}ms")
        logger.info(f"   首帧TTS: {self.first_tts_latency_ms:.0f}ms")
        logger.info(f"   端到端总计: {self.total_latency_ms:.0f}ms")
        logger.info(f"\n处理统计:")
        logger.info(f"   ASR chunks: {self.audio_chunks_processed}")
        logger.info(f"   LLM chunks: {self.text_chunks_generated}")
        logger.info(f"   TTS segments: {self.audio_segments_played}")
        logger.info(f"\n性能指标:")
        logger.info(f"   ASR RTF: {self.asr_rtf:.2f}")
        logger.info(f"   TTS速度: {self.tts_speed:.1f}x")
        logger.info("="*60)


class IncrementalASR:
    """
    增量ASR处理器
    
    使用滑动窗口进行增量识别，降低延迟
    """
    
    def __init__(
        self,
        asr_engine,
        window_size_ms: int = 500,      # 窗口大小
        hop_size_ms: int = 200,         # 滑动步长
        sample_rate: int = 16000,
        stable_threshold: int = 3       # 稳定阈值（连续几次相同）
    ):
        self.asr_engine = asr_engine
        self.window_size_samples = int(window_size_ms * sample_rate / 1000)
        self.hop_size_samples = int(hop_size_ms * sample_rate / 1000)
        self.sample_rate = sample_rate
        self.stable_threshold = stable_threshold
        
        # Audio buffer
        self.audio_buffer = bytearray()
        self.buffer_lock = threading.Lock()
        
        # Recognition state
        self.full_text = ""
        self.pending_text = ""
        self.stable_count = 0
        self.last_window_text = ""
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
    def start(self):
        """Start incremental recognition"""
        self._running = True
        self._thread = threading.Thread(target=self._recognition_loop, daemon=True)
        self._thread.start()
        logger.debug("[IncrementalASR] Started")
        
    def stop(self):
        """Stop incremental recognition"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        logger.debug("[IncrementalASR] Stopped")
        
    def feed_audio(self, audio_chunk: bytes):
        """Feed audio chunk for processing"""
        with self.buffer_lock:
            self.audio_buffer.extend(audio_chunk)
            
    def get_partial_result(self) -> str:
        """Get current partial recognition result"""
        return self.full_text + self.pending_text
    
    def finalize(self) -> str:
        """Finalize recognition and return full text"""
        self._running = False
        
        # Process remaining audio
        with self.buffer_lock:
            remaining = bytes(self.audio_buffer)
        
        if remaining:
            text = self._recognize_chunk(remaining)
            if text:
                self.full_text += text
                
        return self.full_text.strip()
    
    def reset(self):
        """Reset state for new utterance"""
        with self.buffer_lock:
            self.audio_buffer.clear()
        self.full_text = ""
        self.pending_text = ""
        self.stable_count = 0
        self.last_window_text = ""
        
    def _recognition_loop(self):
        """Background recognition loop"""
        while self._running:
            # Check if we have enough audio
            with self.buffer_lock:
                buffer_size = len(self.audio_buffer)
                
            if buffer_size >= self.window_size_samples * 2:  # 16-bit
                self._process_window()
            else:
                time.sleep(0.05)  # 50ms sleep
                
    def _process_window(self):
        """Process one window of audio"""
        with self.buffer_lock:
            # Extract window
            window_bytes = bytes(self.audio_buffer[:self.window_size_samples * 2])
            
        # Recognize
        text = self._recognize_chunk(window_bytes)
        
        if text != self.last_window_text:
            # Text changed, update pending
            self.pending_text = text
            self.stable_count = 0
            self.last_window_text = text
        else:
            # Text stable
            self.stable_count += 1
            if self.stable_count >= self.stable_threshold:
                # Commit stable text
                if self.pending_text:
                    self.full_text += self.pending_text + " "
                    self.pending_text = ""
                    
        # Slide window
        with self.buffer_lock:
            hop_bytes = self.hop_size_samples * 2
            self.audio_buffer = self.audio_buffer[hop_bytes:]
            
    def _recognize_chunk(self, audio_bytes: bytes) -> str:
        """Recognize a single chunk"""
        try:
            text = self.asr_engine.transcribe(
                audio_bytes,
                sample_rate=self.sample_rate
            )
            return text.strip()
        except Exception as e:
            logger.error(f"[IncrementalASR] Recognition error: {e}")
            return ""


class SentenceSplitter:
    """
    句子分割器
    
    将流式文本分割成适合TTS的句子单元
    """
    
    # 句子结束标记
    SENTENCE_ENDINGS = r'[。！？.!?；;]'
    
    # 最小句子长度（避免太短的片段）
    MIN_SENTENCE_LENGTH = 5
    
    def __init__(self, min_length: int = 5):
        self.min_length = min_length
        self.buffer = ""
        self.completed_sentences: List[str] = []
        
    def feed_text(self, text: str) -> List[str]:
        """
        Feed text and return any completed sentences
        
        Returns:
            List of complete sentences ready for TTS
        """
        self.buffer += text
        
        # Find sentence boundaries
        sentences = []
        remaining = self.buffer
        
        while True:
            match = re.search(self.SENTENCE_ENDINGS, remaining)
            if not match:
                break
                
            end_pos = match.end()
            sentence = remaining[:end_pos].strip()
            
            if len(sentence) >= self.min_length:
                sentences.append(sentence)
                remaining = remaining[end_pos:]
            else:
                # Too short, keep in buffer
                break
                
        self.buffer = remaining
        return sentences
    
    def finalize(self) -> List[str]:
        """Get remaining text as final sentence"""
        sentences = []
        if self.buffer.strip():
            sentences.append(self.buffer.strip())
        self.buffer = ""
        return sentences
    
    def reset(self):
        """Reset for new utterance"""
        self.buffer = ""
        self.completed_sentences = []


class StreamingTTS:
    """
    流式TTS处理器
    
    将长文本分段合成，实现边生成边播放
    """
    
    def __init__(
        self,
        tts_engine,
        audio_player,
        sentence_splitter: Optional[SentenceSplitter] = None
    ):
        self.tts_engine = tts_engine
        self.audio_player = audio_player
        self.splitter = sentence_splitter or SentenceSplitter()
        
        # Audio queue
        self.audio_queue: queue.Queue[Optional[bytes]] = queue.Queue()
        
        # State
        self._running = False
        self._synthesis_thread: Optional[threading.Thread] = None
        self._playback_thread: Optional[threading.Thread] = None
        
        # Callbacks
        self.on_sentence_start: Optional[Callable[[str], None]] = None
        self.on_sentence_end: Optional[Callable[[str], None]] = None
        
    def start(self):
        """Start streaming TTS"""
        self._running = True
        
        self._synthesis_thread = threading.Thread(
            target=self._synthesis_loop, daemon=True
        )
        self._playback_thread = threading.Thread(
            target=self._playback_loop, daemon=True
        )
        
        self._synthesis_thread.start()
        self._playback_thread.start()
        
        logger.debug("[StreamingTTS] Started")
        
    def stop(self):
        """Stop streaming TTS"""
        self._running = False
        
        # Signal end
        self.audio_queue.put(None)
        
        # Wait for threads
        if self._synthesis_thread:
            self._synthesis_thread.join(timeout=1.0)
        if self._playback_thread:
            self._playback_thread.join(timeout=1.0)
            
        logger.debug("[StreamingTTS] Stopped")
        
    def feed_text(self, text: str):
        """Feed text for synthesis"""
        sentences = self.splitter.feed_text(text)
        
        for sentence in sentences:
            if self.on_sentence_start:
                self.on_sentence_start(sentence)
                
            # Synthesize in background
            audio = self.tts_engine.synthesize(sentence)
            if audio:
                self.audio_queue.put(audio)
                
            if self.on_sentence_end:
                self.on_sentence_end(sentence)
                
    def finalize(self):
        """Finalize and play remaining text"""
        sentences = self.splitter.finalize()
        
        for sentence in sentences:
            audio = self.tts_engine.synthesize(sentence)
            if audio:
                self.audio_queue.put(audio)
                
        # Signal end
        self.audio_queue.put(None)
        
    def reset(self):
        """Reset for new utterance"""
        self.splitter.reset()
        
        # Clear queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
                
    def _synthesis_loop(self):
        """Background synthesis loop"""
        # Synthesis is done in feed_text for simplicity
        pass
        
    def _playback_loop(self):
        """Background playback loop"""
        while self._running:
            try:
                audio = self.audio_queue.get(timeout=0.1)
                if audio is None:
                    break
                    
                # Play audio
                self.audio_player.play_bytes(audio, blocking=True)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[StreamingTTS] Playback error: {e}")


class StreamingPipeline:
    """
    流式处理Pipeline
    
    协调ASR、LLM、TTS的流式处理
    """
    
    def __init__(
        self,
        asr_engine,
        llm_manager,
        tts_engine,
        audio_player,
        enable_incremental_asr: bool = True,
        enable_streaming_tts: bool = True
    ):
        self.asr_engine = asr_engine
        self.llm_manager = llm_manager
        self.tts_engine = tts_engine
        self.audio_player = audio_player
        
        # Components
        self.incremental_asr = IncrementalASR(asr_engine) if enable_incremental_asr else None
        self.streaming_tts = StreamingTTS(tts_engine, audio_player) if enable_streaming_tts else None
        
        # State
        self.state = StreamState.IDLE
        self.state_lock = threading.Lock()
        
        # Metrics
        self.metrics = StreamingMetrics()
        
        # Callbacks
        self.on_state_change: Optional[Callable[[StreamState], None]] = None
        self.on_partial_asr: Optional[Callable[[str], None]] = None
        self.on_partial_llm: Optional[Callable[[str], None]] = None
        
    def start(self):
        """Start streaming pipeline"""
        if self.incremental_asr:
            self.incremental_asr.start()
        if self.streaming_tts:
            self.streaming_tts.start()
            
        self._set_state(StreamState.LISTENING)
        logger.info("[StreamingPipeline] Started")
        
    def stop(self):
        """Stop streaming pipeline"""
        if self.incremental_asr:
            self.incremental_asr.stop()
        if self.streaming_tts:
            self.streaming_tts.stop()
            
        self._set_state(StreamState.IDLE)
        logger.info("[StreamingPipeline] Stopped")
        
    def feed_audio(self, audio_chunk: bytes):
        """Feed audio chunk"""
        if self.incremental_asr and self.state == StreamState.LISTENING:
            self.incremental_asr.feed_audio(audio_chunk)
            
    def process_utterance(self, audio_data: bytes) -> str:
        """
        Process complete utterance with streaming
        
        This is the main entry point for processing user speech
        """
        start_time = time.time()
        
        # Step 1: ASR (with incremental for partial results)
        self._set_state(StreamState.RECOGNIZING)
        asr_start = time.time()
        
        if self.incremental_asr:
            # Use incremental ASR
            self.incremental_asr.reset()
            self.incremental_asr.feed_audio(audio_data)
            text = self.incremental_asr.finalize()
        else:
            # Use standard ASR
            text = self.asr_engine.transcribe(audio_data)
            
        asr_latency = (time.time() - asr_start) * 1000
        self.metrics.first_asr_latency_ms = asr_latency
        
        if not text:
            self._set_state(StreamState.IDLE)
            return ""
            
        logger.info(f"[StreamingPipeline] ASR: '{text}' ({asr_latency:.0f}ms)")
        
        # Step 2 & 3: LLM + TTS (streaming)
        self._process_llm_tts_streaming(text)
        
        # Metrics
        total_latency = (time.time() - start_time) * 1000
        self.metrics.total_latency_ms = total_latency
        
        logger.info(f"[StreamingPipeline] Total latency: {total_latency:.0f}ms")
        
        self._set_state(StreamState.IDLE)
        return text
        
    def _process_llm_tts_streaming(self, user_text: str):
        """Process LLM and TTS with streaming"""
        self._set_state(StreamState.THINKING)
        
        # Prepare messages
        messages = [
            {"role": "system", "content": "你是一个友好的AI助手。回答简洁自然。"},
            {"role": "user", "content": user_text}
        ]
        
        # Reset TTS
        if self.streaming_tts:
            self.streaming_tts.reset()
            
        llm_start = time.time()
        first_token = True
        
        # Stream LLM output
        try:
            for chunk in self.llm_manager.chat_completion_stream(messages):
                if first_token:
                    first_latency = (time.time() - llm_start) * 1000
                    self.metrics.first_llm_latency_ms = first_latency
                    self._set_state(StreamState.SYNTHESIZING)
                    first_token = False
                    
                # Update partial result
                if self.on_partial_llm:
                    self.on_partial_llm(chunk)
                    
                # Feed to TTS
                if self.streaming_tts:
                    self.streaming_tts.feed_text(chunk)
                    
            # Finalize TTS
            if self.streaming_tts:
                self.streaming_tts.finalize()
                self._set_state(StreamState.SPEAKING)
                
        except Exception as e:
            logger.error(f"[StreamingPipeline] Streaming error: {e}")
            
    def _set_state(self, new_state: StreamState):
        """Update pipeline state"""
        with self.state_lock:
            old_state = self.state
            self.state = new_state
            
        if old_state != new_state:
            logger.debug(f"[StreamingPipeline] State: {old_state.name} -> {new_state.name}")
            if self.on_state_change:
                self.on_state_change(new_state)
                
    def get_state(self) -> StreamState:
        """Get current state"""
        with self.state_lock:
            return self.state
        
    def is_speaking(self) -> bool:
        """Check if currently speaking"""
        return self.state == StreamState.SPEAKING
        
    def interrupt(self):
        """Interrupt current playback"""
        if self.state == StreamState.SPEAKING:
            self._set_state(StreamState.INTERRUPTED)
            
            # Stop TTS
            if self.streaming_tts:
                self.streaming_tts.reset()
                
            # Stop audio
            self.audio_player.stop()
            
            logger.info("[StreamingPipeline] Interrupted")
