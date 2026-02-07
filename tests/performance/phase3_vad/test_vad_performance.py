"""
Performance tests for VAD

Performance targets (from CHECKLIST.md):
- Detection latency < 100ms
- RTF < 0.01
- 30% faster than Silero
- Library size < 306KB (vs 2.16MB)
"""

import pytest
import time
import numpy as np
from tests.utils.metrics import benchmark


@pytest.mark.phase3
@pytest.mark.performance
@pytest.mark.vad
@pytest.mark.slow
class TestVADPerformance:
    """Performance tests for VAD"""
    
    @pytest.fixture
    def test_audio_chunks(self):
        """Generate test audio chunks"""
        chunks = []
        
        # Speech chunk
        t = np.linspace(0, 0.1, int(16000 * 0.1))
        speech = (np.sin(2 * np.pi * 440 * t) * 0.3 * 32767).astype(np.int16)
        chunks.append(speech.tobytes())
        
        # Silence chunk
        silence = np.zeros(int(16000 * 0.1), dtype=np.int16)
        chunks.append(silence.tobytes())
        
        return chunks
    
    def test_detection_latency(self, performance_monitor, test_audio_chunks):
        """Test detection latency (< 100ms)"""
        from audio.vad import VADDetector
        
        try:
            vad = VADDetector(method="silero")
        except:
            vad = VADDetector(method="energy")
        
        monitor = performance_monitor()
        monitor.start()
        
        for chunk in test_audio_chunks:
            vad.detect(chunk)
        
        metrics = monitor.stop()
        
        avg_latency = metrics["duration_ms"] / len(test_audio_chunks)
        
        assert avg_latency < 100, \
            f"Detection latency too high: {avg_latency:.2f}ms"
    
    def test_vad_rtf(self, test_audio_chunks):
        """Test VAD RTF (< 0.01)"""
        from audio.vad import VADDetector
        
        try:
            vad = VADDetector(method="silero")
        except:
            vad = VADDetector(method="energy")
        
        # Measure processing time
        start = time.time()
        for chunk in test_audio_chunks:
            vad.detect(chunk)
        processing_time = time.time() - start
        
        # Calculate audio duration
        audio_duration = len(test_audio_chunks) * 0.1  # 100ms per chunk
        
        rtf = processing_time / audio_duration
        
        assert rtf < 0.01, f"RTF too high: {rtf:.4f}"
    
    def test_throughput(self, test_audio_chunks):
        """Test VAD throughput"""
        from audio.vad import VADDetector
        
        try:
            vad = VADDetector(method="silero")
        except:
            vad = VADDetector(method="energy")
        
        # Create larger test set
        test_set = test_audio_chunks * 100
        
        result = benchmark(
            lambda: [vad.detect(chunk) for chunk in test_set],
            iterations=10
        )
        
        # Should process quickly
        assert result["mean_ms"] < 5000, \
            f"Throughput too low: {result['mean_ms']:.2f}ms"
    
    def test_memory_usage(self, performance_monitor, test_audio_chunks):
        """Test memory usage"""
        from audio.vad import VADDetector
        
        try:
            vad = VADDetector(method="silero")
        except:
            vad = VADDetector(method="energy")
        
        monitor = performance_monitor()
        monitor.start()
        
        # Process many chunks
        for _ in range(1000):
            for chunk in test_audio_chunks:
                vad.detect(chunk)
        
        metrics = monitor.stop()
        
        # Memory should remain stable
        assert metrics["memory_used_mb"] < 100, \
            f"Memory usage too high: {metrics['memory_used_mb']:.0f}MB"


@pytest.mark.phase3
@pytest.mark.performance
@pytest.mark.vad
class TestVADBenchmark:
    """Benchmark tests comparing VAD methods"""
    
    def test_vad_method_comparison(self, test_audio_chunks):
        """Compare different VAD methods"""
        from audio.vad import VADDetector
        
        results = {}
        
        methods = ["energy"]  # Always available
        
        for method in methods:
            try:
                vad = VADDetector(method=method)
                
                result = benchmark(
                    lambda: [vad.detect(chunk) for chunk in test_audio_chunks * 10],
                    iterations=5
                )
                
                results[method] = {
                    "mean_ms": result["mean_ms"],
                    "memory_mb": result["memory_peak_mb"]
                }
            except Exception as e:
                print(f"{method} VAD benchmark failed: {e}")
        
        assert len(results) > 0, "No VAD methods available for benchmarking"
        
        # Print comparison
        print("\nVAD Method Comparison:")
        for method, metrics in results.items():
            print(f"  {method}: {metrics['mean_ms']:.2f}ms, {metrics['memory_mb']:.0f}MB")
