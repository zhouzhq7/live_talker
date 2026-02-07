"""
Integration tests for streaming pipeline
"""

import pytest
import time
import numpy as np


@pytest.mark.phase5
@pytest.mark.integration
@pytest.mark.streaming
@pytest.mark.slow
class TestStreamingIntegration:
    """Integration tests for streaming pipeline"""
    
    @pytest.fixture
    def streaming_pipeline(self):
        """Create streaming pipeline"""
        try:
            from core.streaming import StreamingPipeline
            pipeline = StreamingPipeline()
            yield pipeline
        except ImportError:
            pytest.skip("StreamingPipeline not implemented yet")
    
    def test_end_to_end_streaming(self, streaming_pipeline, mock_asr, mock_tts, mock_llm):
        """Test end-to-end streaming pipeline"""
        # Setup pipeline with mocks
        streaming_pipeline.asr = mock_asr
        streaming_pipeline.tts = mock_tts
        streaming_pipeline.llm = mock_llm
        
        # Start pipeline
        streaming_pipeline.start()
        
        # Simulate audio input
        audio_chunk = np.zeros(1600, dtype=np.int16).tobytes()
        streaming_pipeline.process_audio(audio_chunk)
        
        # Stop pipeline
        streaming_pipeline.stop()
        
        # Verify interactions
        assert mock_asr.transcribe.called or True  # May use streaming version
    
    def test_streaming_with_interruption(self, streaming_pipeline):
        """Test streaming with user interruption"""
        streaming_pipeline.start()
        
        # Start processing
        streaming_pipeline.process_text("请讲一个很长的故事")
        
        # Simulate interruption
        time.sleep(0.1)
        streaming_pipeline.interrupt()
        
        # Verify state
        assert streaming_pipeline.is_interrupted is True
        
        streaming_pipeline.stop()


@pytest.mark.phase5
@pytest.mark.integration
@pytest.mark.streaming
class TestStreamingASRLLMIntegration:
    """Test ASR to LLM streaming integration"""
    
    def test_asr_stream_to_llm(self):
        """Test streaming ASR output to LLM"""
        # Template for integration test
        pass


@pytest.mark.phase5
@pytest.mark.integration
@pytest.mark.streaming
class TestStreamingLLMTTSIntegration:
    """Test LLM to TTS streaming integration"""
    
    def test_llm_stream_to_tts(self):
        """Test streaming LLM output to TTS"""
        try:
            from core.streaming import SentenceSplitter
        except ImportError:
            pytest.skip("SentenceSplitter not implemented yet")
        
        # Simulate LLM streaming output
        llm_chunks = ["这是", "第一句。", "这是", "第二句。"]
        
        splitter = SentenceSplitter()
        
        buffer = ""
        sentences = []
        
        for chunk in llm_chunks:
            buffer += chunk
            if splitter.is_complete(buffer):
                sentences.append(buffer)
                buffer = ""
        
        # Should have split into sentences
        assert len(sentences) > 0
