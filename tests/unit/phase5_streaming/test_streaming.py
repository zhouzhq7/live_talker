"""
Unit tests for streaming pipeline
"""

import pytest
from unittest.mock import MagicMock, patch


@pytest.mark.phase5
@pytest.mark.unit
@pytest.mark.streaming
class TestStreamingASR:
    """Test Streaming ASR"""
    
    @pytest.fixture
    def streaming_asr_class(self):
        """Import StreamingASR class"""
        try:
            from asr.streaming import StreamingASR
            return StreamingASR
        except ImportError:
            pytest.skip("StreamingASR not implemented yet")
    
    def test_streaming_asr_init(self, streaming_asr_class):
        """Test Streaming ASR initialization"""
        asr = streaming_asr_class()
        assert asr.is_streaming is False
    
    def test_start_stream(self, streaming_asr_class):
        """Test starting stream"""
        asr = streaming_asr_class()
        asr.start_stream()
        
        assert asr.is_streaming is True
    
    def test_feed_audio(self, streaming_asr_class):
        """Test feeding audio chunks"""
        asr = streaming_asr_class()
        asr.start_stream()
        
        audio_chunk = b'\x00\x01' * 512
        asr.feed_audio(audio_chunk)
        
        assert len(asr.audio_buffer) > 0
    
    def test_get_partial_result(self, streaming_asr_class):
        """Test getting partial transcription result"""
        asr = streaming_asr_class()
        asr.start_stream()
        
        # Feed some audio
        audio_chunk = b'\x00\x01' * 512
        asr.feed_audio(audio_chunk)
        
        # Get partial result
        result = asr.get_partial_result()
        assert isinstance(result, str)
    
    def test_finalize(self, streaming_asr_class):
        """Test finalizing stream"""
        asr = streaming_asr_class()
        asr.start_stream()
        
        # Feed audio
        audio_chunk = b'\x00\x01' * 512
        asr.feed_audio(audio_chunk)
        
        # Finalize
        result = asr.finalize()
        
        assert isinstance(result, str)
        assert asr.is_streaming is False


@pytest.mark.phase5
@pytest.mark.unit
@pytest.mark.streaming
class TestStreamingTTS:
    """Test Streaming TTS"""
    
    @pytest.fixture
    def streaming_tts_class(self):
        """Import StreamingTTS class"""
        try:
            from tts.streaming import StreamingTTS
            return StreamingTTS
        except ImportError:
            pytest.skip("StreamingTTS not implemented yet")
    
    def test_streaming_tts_init(self, streaming_tts_class):
        """Test Streaming TTS initialization"""
        tts = streaming_tts_class()
        assert tts.is_streaming is False
    
    def test_start_stream(self, streaming_tts_class):
        """Test starting stream"""
        tts = streaming_tts_class()
        tts.start_stream()
        
        assert tts.is_streaming is True
    
    def test_feed_text(self, streaming_tts_class):
        """Test feeding text"""
        tts = streaming_tts_class()
        tts.start_stream()
        
        tts.feed_text("你好")
        
        assert len(tts.text_buffer) > 0
    
    def test_get_audio_chunk(self, streaming_tts_class):
        """Test getting audio chunks"""
        tts = streaming_tts_class()
        tts.start_stream()
        
        tts.feed_text("你好")
        
        chunk = tts.get_audio_chunk()
        # May be None if not enough text to synthesize
        assert chunk is None or isinstance(chunk, bytes)


@pytest.mark.phase5
@pytest.mark.unit
@pytest.mark.streaming
class TestSentenceSplitter:
    """Test sentence splitting for streaming"""
    
    def test_simple_sentence_split(self):
        """Test simple sentence splitting"""
        try:
            from core.streaming import SentenceSplitter
        except ImportError:
            pytest.skip("SentenceSplitter not implemented yet")
        
        splitter = SentenceSplitter()
        
        text = "这是第一句。这是第二句。"
        sentences = list(splitter.split(text))
        
        assert len(sentences) == 2
        assert sentences[0] == "这是第一句。"
        assert sentences[1] == "这是第二句。"
    
    def test_multiple_punctuation(self):
        """Test splitting with multiple punctuation marks"""
        try:
            from core.streaming import SentenceSplitter
        except ImportError:
            pytest.skip("SentenceSplitter not implemented yet")
        
        splitter = SentenceSplitter()
        
        text = "你好！最近怎么样？我很好。"
        sentences = list(splitter.split(text))
        
        assert len(sentences) == 3
    
    def test_incomplete_sentence(self):
        """Test handling incomplete sentence"""
        try:
            from core.streaming import SentenceSplitter
        except ImportError:
            pytest.skip("SentenceSplitter not implemented yet")
        
        splitter = SentenceSplitter()
        
        text = "这句话还没说完"
        sentences = list(splitter.split(text))
        
        assert len(sentences) == 1
        assert sentences[0] == "这句话还没说完"
    
    def test_is_complete_sentence(self):
        """Test complete sentence detection"""
        try:
            from core.streaming import is_complete_sentence
        except ImportError:
            pytest.skip("is_complete_sentence not implemented yet")
        
        assert is_complete_sentence("这是完整的句子。") is True
        assert is_complete_sentence("这是完整的句子！") is True
        assert is_complete_sentence("这是完整的句子？") is True
        assert is_complete_sentence("这句话不完整") is False


@pytest.mark.phase5
@pytest.mark.unit
@pytest.mark.streaming
class TestStreamingPipeline:
    """Test Streaming Pipeline"""
    
    @pytest.fixture
    def streaming_pipeline_class(self):
        """Import StreamingPipeline class"""
        try:
            from core.streaming import StreamingPipeline
            return StreamingPipeline
        except ImportError:
            pytest.skip("StreamingPipeline not implemented yet")
    
    def test_pipeline_init(self, streaming_pipeline_class):
        """Test pipeline initialization"""
        pipeline = streaming_pipeline_class()
        assert pipeline.is_running is False
    
    def test_pipeline_start_stop(self, streaming_pipeline_class):
        """Test starting and stopping pipeline"""
        pipeline = streaming_pipeline_class()
        
        pipeline.start()
        assert pipeline.is_running is True
        
        pipeline.stop()
        assert pipeline.is_running is False
    
    def test_interrupt_handling(self, streaming_pipeline_class):
        """Test interruption handling"""
        pipeline = streaming_pipeline_class()
        pipeline.start()
        
        # Simulate processing
        pipeline.interrupt()
        
        # Should clear buffers and reset state
        assert pipeline.is_interrupted is True


@pytest.mark.phase5
@pytest.mark.unit
@pytest.mark.streaming
class TestStreamUtils:
    """Test streaming utilities"""
    
    def test_buffer_management(self):
        """Test audio buffer management"""
        try:
            from core.streaming import AudioBuffer
        except ImportError:
            pytest.skip("AudioBuffer not implemented yet")
        
        buffer = AudioBuffer(max_size=1024)
        
        # Add data
        buffer.write(b'\x00\x01' * 512)
        assert buffer.size == 512
        
        # Read data
        data = buffer.read(256)
        assert len(data) == 256
        assert buffer.size == 256
        
        # Clear buffer
        buffer.clear()
        assert buffer.size == 0
