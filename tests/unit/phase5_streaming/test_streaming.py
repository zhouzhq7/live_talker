"""
Unit tests for Streaming Pipeline (Phase 5)

Tests:
- IncrementalASR
- SentenceSplitter
- StreamingTTS
- StreamingPipeline
"""

import pytest
import time
import threading
from unittest.mock import Mock, MagicMock

from core.streaming import (
    StreamState,
    StreamingMetrics,
    IncrementalASR,
    SentenceSplitter,
    StreamingTTS,
    StreamingPipeline
)


class TestStreamingMetrics:
    """Test StreamingMetrics dataclass"""
    
    def test_default_initialization(self):
        """Test default metric values"""
        metrics = StreamingMetrics()
        assert metrics.first_asr_latency_ms == 0
        assert metrics.first_llm_latency_ms == 0
        assert metrics.first_tts_latency_ms == 0
        assert metrics.total_latency_ms == 0


class TestIncrementalASR:
    """Test IncrementalASR"""
    
    def test_initialization(self):
        """Test initialization"""
        asr_engine = Mock()
        incremental = IncrementalASR(
            asr_engine=asr_engine,
            window_size_ms=500,
            hop_size_ms=200
        )
        
        assert incremental.window_size_samples == 8000  # 500ms * 16kHz
        assert incremental.hop_size_samples == 3200     # 200ms * 16kHz
        assert incremental.sample_rate == 16000
        
    def test_feed_audio(self):
        """Test feeding audio"""
        asr_engine = Mock()
        incremental = IncrementalASR(asr_engine=asr_engine)
        
        audio_chunk = b'\x00\x00' * 1600  # 100ms
        incremental.feed_audio(audio_chunk)
        
        assert len(incremental.audio_buffer) == len(audio_chunk)
        
    def test_get_partial_result(self):
        """Test getting partial result"""
        asr_engine = Mock()
        incremental = IncrementalASR(asr_engine=asr_engine)
        
        incremental.full_text = "Hello "
        incremental.pending_text = "world"
        
        result = incremental.get_partial_result()
        assert result == "Hello world"
        
    def test_finalize(self):
        """Test finalization"""
        asr_engine = Mock()
        asr_engine.transcribe = Mock(return_value="Final result")
        
        incremental = IncrementalASR(asr_engine=asr_engine)
        incremental.feed_audio(b'\x00\x00' * 1600)
        
        result = incremental.finalize()
        assert result == "Final result"
        
    def test_reset(self):
        """Test reset"""
        asr_engine = Mock()
        incremental = IncrementalASR(asr_engine=asr_engine)
        
        incremental.feed_audio(b'\x00\x00' * 1600)
        incremental.full_text = "Some text"
        
        incremental.reset()
        
        assert len(incremental.audio_buffer) == 0
        assert incremental.full_text == ""


class TestSentenceSplitter:
    """Test SentenceSplitter"""
    
    def test_initialization(self):
        """Test initialization"""
        splitter = SentenceSplitter(min_length=5)
        assert splitter.min_length == 5
        assert splitter.buffer == ""
        
    def test_feed_text_complete_sentence(self):
        """Test feeding text that forms a complete sentence"""
        splitter = SentenceSplitter(min_length=5)
        
        sentences = splitter.feed_text("Hello world.")
        
        assert len(sentences) == 1
        assert sentences[0] == "Hello world."
        
    def test_feed_text_incomplete_sentence(self):
        """Test feeding incomplete sentence"""
        splitter = SentenceSplitter(min_length=5)
        
        sentences = splitter.feed_text("Hello")
        
        assert len(sentences) == 0
        assert splitter.buffer == "Hello"
        
    def test_feed_text_multiple_sentences(self):
        """Test feeding multiple sentences"""
        splitter = SentenceSplitter(min_length=5)
        
        sentences = splitter.feed_text("First sentence. Second sentence.")
        
        assert len(sentences) == 2
        assert "First sentence." in sentences
        assert "Second sentence." in sentences
        
    def test_feed_text_chinese(self):
        """Test feeding Chinese text"""
        splitter = SentenceSplitter(min_length=5)
        
        sentences = splitter.feed_text("你好世界。今天天气不错。")
        
        assert len(sentences) == 2
        
    def test_finalize(self):
        """Test finalization"""
        splitter = SentenceSplitter(min_length=5)
        
        splitter.feed_text("Incomplete")
        sentences = splitter.finalize()
        
        assert len(sentences) == 1
        assert sentences[0] == "Incomplete"
        
    def test_reset(self):
        """Test reset"""
        splitter = SentenceSplitter(min_length=5)
        
        splitter.feed_text("Hello world.")
        splitter.reset()
        
        assert splitter.buffer == ""
        assert len(splitter.completed_sentences) == 0


class TestStreamingTTS:
    """Test StreamingTTS"""
    
    def test_initialization(self):
        """Test initialization"""
        tts_engine = Mock()
        audio_player = Mock()
        
        streaming_tts = StreamingTTS(
            tts_engine=tts_engine,
            audio_player=audio_player
        )
        
        assert streaming_tts.tts_engine == tts_engine
        assert streaming_tts.audio_player == audio_player
        
    def test_start_stop(self):
        """Test start and stop"""
        tts_engine = Mock()
        audio_player = Mock()
        
        streaming_tts = StreamingTTS(tts_engine, audio_player)
        streaming_tts.start()
        
        assert streaming_tts._running is True
        assert streaming_tts._synthesis_thread is not None
        assert streaming_tts._playback_thread is not None
        
        streaming_tts.stop()
        
        assert streaming_tts._running is False
        
    def test_feed_text(self):
        """Test feeding text"""
        tts_engine = Mock()
        tts_engine.synthesize = Mock(return_value=b'audio_data')
        audio_player = Mock()
        
        streaming_tts = StreamingTTS(tts_engine, audio_player)
        streaming_tts.start()
        
        streaming_tts.feed_text("Hello world.")
        
        # Give time for synthesis
        time.sleep(0.1)
        
        tts_engine.synthesize.assert_called_once()
        
        streaming_tts.stop()
        
    def test_feed_text_multiple_sentences(self):
        """Test feeding multiple sentences"""
        tts_engine = Mock()
        tts_engine.synthesize = Mock(return_value=b'audio_data')
        audio_player = Mock()
        
        streaming_tts = StreamingTTS(tts_engine, audio_player)
        streaming_tts.start()
        
        streaming_tts.feed_text("First sentence. Second sentence.")
        
        time.sleep(0.1)
        
        assert tts_engine.synthesize.call_count == 2
        
        streaming_tts.stop()
        
    def test_callbacks(self):
        """Test sentence callbacks"""
        tts_engine = Mock()
        tts_engine.synthesize = Mock(return_value=b'audio_data')
        audio_player = Mock()
        
        streaming_tts = StreamingTTS(tts_engine, audio_player)
        
        on_start = Mock()
        on_end = Mock()
        streaming_tts.on_sentence_start = on_start
        streaming_tts.on_sentence_end = on_end
        
        streaming_tts.start()
        streaming_tts.feed_text("Hello world.")
        time.sleep(0.1)
        streaming_tts.stop()
        
        on_start.assert_called_once()
        on_end.assert_called_once()


class TestStreamingPipeline:
    """Test StreamingPipeline"""
    
    def create_mock_components(self):
        """Create mock components for pipeline"""
        asr_engine = Mock()
        asr_engine.transcribe = Mock(return_value="Test transcription")
        
        llm_manager = Mock()
        llm_manager.chat_completion_stream = Mock(return_value=iter(["Hello", " world"]))
        
        tts_engine = Mock()
        tts_engine.synthesize = Mock(return_value=b'audio_data')
        
        audio_player = Mock()
        
        return asr_engine, llm_manager, tts_engine, audio_player
        
    def test_initialization(self):
        """Test initialization"""
        asr, llm, tts, player = self.create_mock_components()
        
        pipeline = StreamingPipeline(
            asr_engine=asr,
            llm_manager=llm,
            tts_engine=tts,
            audio_player=player
        )
        
        assert pipeline.state == StreamState.IDLE
        assert pipeline.metrics.first_asr_latency_ms == 0
        
    def test_start_stop(self):
        """Test start and stop"""
        asr, llm, tts, player = self.create_mock_components()
        
        pipeline = StreamingPipeline(asr, llm, tts, player)
        pipeline.start()
        
        assert pipeline.state == StreamState.LISTENING
        
        pipeline.stop()
        
        assert pipeline.state == StreamState.IDLE
        
    def test_process_utterance(self):
        """Test processing utterance"""
        asr, llm, tts, player = self.create_mock_components()
        
        pipeline = StreamingPipeline(asr, llm, tts, player)
        pipeline.start()
        
        audio_data = b'\x00\x00' * 1600
        result = pipeline.process_utterance(audio_data)
        
        assert result == "Test transcription"
        assert pipeline.metrics.first_asr_latency_ms > 0
        
        pipeline.stop()
        
    def test_state_transitions(self):
        """Test state transitions during processing"""
        asr, llm, tts, player = self.create_mock_components()
        
        states = []
        def on_state_change(state):
            states.append(state)
        
        pipeline = StreamingPipeline(asr, llm, tts, player)
        pipeline.on_state_change = on_state_change
        pipeline.start()
        
        audio_data = b'\x00\x00' * 1600
        pipeline.process_utterance(audio_data)
        
        # Should have gone through multiple states
        assert len(states) >= 3
        assert StreamState.LISTENING in states
        assert StreamState.RECOGNIZING in states
        
        pipeline.stop()
        
    def test_interrupt(self):
        """Test interruption"""
        asr, llm, tts, player = self.create_mock_components()
        
        pipeline = StreamingPipeline(asr, llm, tts, player)
        pipeline.start()
        
        # Set speaking state
        pipeline._set_state(StreamState.SPEAKING)
        
        pipeline.interrupt()
        
        assert pipeline.state == StreamState.INTERRUPTED
        player.stop.assert_called_once()
        
        pipeline.stop()
        
    def test_is_speaking(self):
        """Test is_speaking check"""
        asr, llm, tts, player = self.create_mock_components()
        
        pipeline = StreamingPipeline(asr, llm, tts, player)
        
        assert pipeline.is_speaking() is False
        
        pipeline._set_state(StreamState.SPEAKING)
        assert pipeline.is_speaking() is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
