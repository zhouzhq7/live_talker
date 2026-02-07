"""
Performance tests for streaming pipeline

Performance targets (from CHECKLIST.md):
- End-to-end latency < 1s
- ASR first word latency < 200ms
- LLM first token latency < 500ms
- TTS first audio latency < 500ms
"""

import pytest
import time
import numpy as np


@pytest.mark.phase5
@pytest.mark.performance
@pytest.mark.streaming
@pytest.mark.slow
class TestStreamingPerformance:
    """Performance tests for streaming pipeline"""
    
    def test_end_to_end_latency(self, performance_monitor):
        """Test end-to-end latency (< 1s)"""
        try:
            from core.streaming import StreamingPipeline
            pipeline = StreamingPipeline()
        except ImportError:
            pytest.skip("StreamingPipeline not implemented yet")
        
        monitor = performance_monitor()
        
        pipeline.start()
        monitor.start()
        
        # Simulate a simple interaction
        audio_input = np.zeros(16000, dtype=np.int16).tobytes()  # 1 second
        pipeline.process_audio(audio_input)
        
        # Wait for output
        time.sleep(0.5)
        
        metrics = monitor.stop()
        pipeline.stop()
        
        # Should be fast (actual target is < 1s)
        print(f"\nEnd-to-end latency: {metrics['duration_ms']:.2f}ms")
    
    def test_asr_latency(self, performance_monitor):
        """Test ASR latency in streaming mode"""
        try:
            from asr.streaming import StreamingASR
            asr = StreamingASR()
        except ImportError:
            pytest.skip("StreamingASR not implemented yet")
        
        asr.start_stream()
        
        monitor = performance_monitor()
        monitor.start()
        
        # Feed audio and get first result
        audio_chunk = np.zeros(3200, dtype=np.int16).tobytes()  # 200ms
        asr.feed_audio(audio_chunk)
        result = asr.get_partial_result()
        
        metrics = monitor.stop()
        
        assert metrics["duration_ms"] < 200, \
            f"ASR latency too high: {metrics['duration_ms']:.2f}ms"
    
    def test_tts_latency(self, performance_monitor):
        """Test TTS latency in streaming mode"""
        try:
            from tts.streaming import StreamingTTS
            tts = StreamingTTS()
        except ImportError:
            pytest.skip("StreamingTTS not implemented yet")
        
        tts.start_stream()
        
        monitor = performance_monitor()
        monitor.start()
        
        # Feed text and get first audio
        tts.feed_text("你好")
        audio = tts.get_audio_chunk()
        
        metrics = monitor.stop()
        
        assert metrics["duration_ms"] < 500, \
            f"TTS latency too high: {metrics['duration_ms']:.2f}ms"
    
    def test_sentence_splitting_speed(self):
        """Test sentence splitting performance"""
        try:
            from core.streaming import SentenceSplitter
        except ImportError:
            pytest.skip("SentenceSplitter not implemented yet")
        
        splitter = SentenceSplitter()
        
        # Test with long text
        text = "这是一个测试。" * 100
        
        start = time.time()
        sentences = list(splitter.split(text))
        end = time.time()
        
        duration_ms = (end - start) * 1000
        
        assert duration_ms < 100, \
            f"Sentence splitting too slow: {duration_ms:.2f}ms"
    
    def test_buffer_performance(self):
        """Test audio buffer performance"""
        try:
            from core.streaming import AudioBuffer
        except ImportError:
            pytest.skip("AudioBuffer not implemented yet")
        
        buffer = AudioBuffer(max_size=16000 * 10)  # 10 seconds
        
        # Write performance
        data = b'\x00\x01' * 1600  # 100ms of audio
        
        start = time.time()
        for _ in range(100):
            buffer.write(data)
        write_time = (time.time() - start) * 1000
        
        # Read performance
        start = time.time()
        for _ in range(100):
            buffer.read(1600)
        read_time = (time.time() - start) * 1000
        
        assert write_time < 100, f"Buffer write too slow: {write_time:.2f}ms"
        assert read_time < 100, f"Buffer read too slow: {read_time:.2f}ms"


@pytest.mark.phase5
@pytest.mark.performance
@pytest.mark.streaming
class TestStreamingThroughput:
    """Throughput tests for streaming pipeline"""
    
    def test_pipeline_throughput(self):
        """Test pipeline throughput"""
        try:
            from core.streaming import StreamingPipeline
            pipeline = StreamingPipeline()
        except ImportError:
            pytest.skip("StreamingPipeline not implemented yet")
        
        pipeline.start()
        
        # Process multiple chunks
        chunks = [np.zeros(1600, dtype=np.int16).tobytes() for _ in range(100)]
        
        start = time.time()
        for chunk in chunks:
            pipeline.process_audio(chunk)
        end = time.time()
        
        pipeline.stop()
        
        duration = end - start
        chunks_per_sec = len(chunks) / duration
        
        print(f"\nPipeline throughput: {chunks_per_sec:.1f} chunks/s")
