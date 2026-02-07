"""
Acceptance tests for Streaming Pipeline

Based on CHECKLIST.md requirements:
- End-to-end latency < 1s
- ASR first word latency < 200ms
- LLM first token latency < 500ms
- TTS first audio latency < 500ms
"""

import pytest
import time
import numpy as np


@pytest.mark.phase5
@pytest.mark.acceptance
@pytest.mark.streaming
@pytest.mark.slow
class TestStreamingAcceptance:
    """Acceptance tests for streaming pipeline"""
    
    @pytest.fixture
    def streaming_pipeline(self):
        """Create streaming pipeline"""
        try:
            from core.streaming import StreamingPipeline
            pipeline = StreamingPipeline()
            yield pipeline
        except ImportError:
            pytest.skip("StreamingPipeline not implemented yet")
    
    def test_end_to_end_latency(self, streaming_pipeline, performance_monitor):
        """Test end-to-end latency < 1s"""
        monitor = performance_monitor()
        
        streaming_pipeline.start()
        monitor.start()
        
        # Simulate user input
        audio_input = np.zeros(16000, dtype=np.int16).tobytes()  # 1 second
        streaming_pipeline.process_audio(audio_input)
        
        # Wait for output (or timeout)
        max_wait = 5  # seconds
        start = time.time()
        while time.time() - start < max_wait:
            if streaming_pipeline.has_output():
                break
            time.sleep(0.01)
        
        metrics = monitor.stop()
        streaming_pipeline.stop()
        
        assert metrics["duration_ms"] < 1000, \
            f"End-to-end latency too high: {metrics['duration_ms']:.2f}ms"
    
    def test_interruption_handling(self, streaming_pipeline):
        """Test interruption handling"""
        streaming_pipeline.start()
        
        # Start a long response
        streaming_pipeline.process_text("请讲一个很长的故事" * 10)
        time.sleep(0.1)
        
        # Interrupt
        streaming_pipeline.interrupt()
        
        # Should be interrupted
        assert streaming_pipeline.is_interrupted is True
        
        streaming_pipeline.stop()
    
    def test_continuous_conversation(self, streaming_pipeline):
        """Test continuous conversation"""
        streaming_pipeline.start()
        
        # Multiple turns
        for i in range(3):
            audio_input = np.zeros(8000, dtype=np.int16).tobytes()
            streaming_pipeline.process_audio(audio_input)
            time.sleep(0.2)
        
        streaming_pipeline.stop()
        
        # Should complete without errors
        assert True


@pytest.mark.phase5
@pytest.mark.acceptance
@pytest.mark.streaming
class TestStreamingLatencyBreakdown:
    """Test individual component latencies"""
    
    def test_asr_streaming_latency(self, performance_monitor):
        """Test ASR streaming latency < 200ms"""
        try:
            from asr.streaming import StreamingASR
            asr = StreamingASR()
        except ImportError:
            pytest.skip("StreamingASR not implemented yet")
        
        asr.start_stream()
        
        monitor = performance_monitor()
        monitor.start()
        
        audio_chunk = np.zeros(3200, dtype=np.int16).tobytes()
        asr.feed_audio(audio_chunk)
        result = asr.get_partial_result()
        
        metrics = monitor.stop()
        
        assert metrics["duration_ms"] < 200, \
            f"ASR latency too high: {metrics['duration_ms']:.2f}ms"
    
    def test_tts_streaming_latency(self, performance_monitor):
        """Test TTS streaming latency < 500ms"""
        try:
            from tts.streaming import StreamingTTS
            tts = StreamingTTS()
        except ImportError:
            pytest.skip("StreamingTTS not implemented yet")
        
        tts.start_stream()
        
        monitor = performance_monitor()
        monitor.start()
        
        tts.feed_text("你好")
        audio = tts.get_audio_chunk()
        
        metrics = monitor.stop()
        
        assert metrics["duration_ms"] < 500, \
            f"TTS latency too high: {metrics['duration_ms']:.2f}ms"


@pytest.mark.phase5
@pytest.mark.acceptance
@pytest.mark.streaming
class TestStreamingReliability:
    """Test streaming pipeline reliability"""
    
    def test_long_running_stability(self):
        """Test long-running stability"""
        try:
            from core.streaming import StreamingPipeline
            pipeline = StreamingPipeline()
        except ImportError:
            pytest.skip("StreamingPipeline not implemented yet")
        
        pipeline.start()
        
        # Run for a while
        for i in range(100):
            audio_input = np.zeros(1600, dtype=np.int16).tobytes()
            pipeline.process_audio(audio_input)
            time.sleep(0.01)
        
        pipeline.stop()
        
        # Should not crash
        assert True
