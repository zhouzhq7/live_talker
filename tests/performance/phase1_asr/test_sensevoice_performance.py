"""
Performance tests for SenseVoice ASR

Performance targets (from CHECKLIST.md):
- Cold start time < 30s
- 10s audio processing time < 100ms
- RTF (Real-Time Factor) < 0.1
- Memory < 2GB
- CPU < 50%
"""

import pytest
import time
import numpy as np
from tests.utils.metrics import calculate_rtf


@pytest.mark.phase1
@pytest.mark.performance
@pytest.mark.asr
@pytest.mark.slow
class TestSenseVoicePerformance:
    """Performance tests for SenseVoice"""
    
    @pytest.fixture
    def test_audio_10s(self):
        """Generate 10 seconds of test audio"""
        return np.zeros(16000 * 10, dtype=np.int16).tobytes()
    
    @pytest.fixture
    def test_audio_1s(self):
        """Generate 1 second of test audio"""
        t = np.linspace(0, 1, 16000)
        audio = np.sin(2 * np.pi * 440 * t) * 0.3
        return (audio * 32767).astype(np.int16).tobytes()
    
    def test_model_loading_time(self, performance_monitor):
        """Test model loading time (< 30s)"""
        try:
            from asr.sensevoice import SenseVoice
        except ImportError:
            pytest.skip("SenseVoice not installed")
        
        monitor = performance_monitor()
        monitor.start()
        
        engine = SenseVoice(device="cpu")
        engine.load_model()
        
        metrics = monitor.stop()
        
        # Assert: Cold start < 30s
        assert metrics["duration_ms"] < 30000, \
            f"Model loading too slow: {metrics['duration_ms']/1000:.2f}s"
    
    def test_transcription_latency_10s(self, performance_monitor, test_audio_10s):
        """Test transcription latency for 10s audio (< 100ms)"""
        try:
            from asr.sensevoice import SenseVoice
            engine = SenseVoice(device="cpu")
            engine.load_model()
        except ImportError:
            pytest.skip("SenseVoice not installed")
        
        # Warmup
        engine.warmup()
        
        monitor = performance_monitor()
        monitor.start()
        
        # Transcribe
        result = engine.transcribe_with_timing(test_audio_10s, sample_rate=16000)
        
        monitor.stop()
        
        latency_ms = result["latency_ms"]
        
        # Assert: 10s audio processing < 100ms
        assert latency_ms < 100, \
            f"Transcription too slow: {latency_ms:.2f}ms"
    
    def test_rtf_requirement(self, test_audio_1s):
        """Test RTF requirement (< 0.1)"""
        try:
            from asr.sensevoice import SenseVoice
            engine = SenseVoice(device="cpu")
            engine.load_model()
        except ImportError:
            pytest.skip("SenseVoice not installed")
        
        # Warmup
        engine.warmup()
        
        result = engine.transcribe_with_timing(test_audio_1s, sample_rate=16000)
        
        rtf = result["rtf"]
        
        # Assert: RTF < 0.1
        assert rtf < 0.1, f"RTF too high: {rtf:.4f}"
    
    def test_memory_usage(self, performance_monitor, test_audio_1s):
        """Test memory usage during transcription (< 2GB)"""
        try:
            from asr.sensevoice import SenseVoice
            engine = SenseVoice(device="cpu")
            engine.load_model()
        except ImportError:
            pytest.skip("SenseVoice not installed")
        
        monitor = performance_monitor()
        monitor.start()
        
        # Run multiple transcriptions
        for _ in range(10):
            engine.transcribe(test_audio_1s)
        
        metrics = monitor.stop()
        
        # Assert: Memory < 2GB
        assert metrics["peak_memory_mb"] < 2048, \
            f"Memory usage too high: {metrics['peak_memory_mb']:.0f}MB"
    
    def test_concurrent_transcription(self, test_audio_1s):
        """Test concurrent transcription performance"""
        import concurrent.futures
        
        try:
            from asr.sensevoice import SenseVoice
            engine = SenseVoice(device="cpu")
            engine.load_model()
        except ImportError:
            pytest.skip("SenseVoice not installed")
        
        def transcribe():
            return engine.transcribe_with_timing(test_audio_1s)
        
        start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(transcribe) for _ in range(4)]
            results = [f.result() for f in futures]
        end = time.time()
        
        total_time = (end - start) * 1000
        
        # Assert: Concurrent processing is efficient
        assert total_time < 4000, \
            f"Concurrent processing too slow: {total_time:.0f}ms"


@pytest.mark.phase1
@pytest.mark.performance
@pytest.mark.asr
class TestASRBenchmark:
    """Benchmark tests comparing ASR engines"""
    
    def test_benchmark_asr_engines(self, sample_audio_bytes_16k):
        """Benchmark different ASR engines"""
        # This is a template for comparing engines
        results = {}
        
        # Test FunASR
        try:
            from asr.funasr import FunASR
            from tests.utils.metrics import benchmark
            
            engine = FunASR(device="cpu")
            engine.load_model()
            
            result = benchmark(
                engine.transcribe,
                sample_audio_bytes_16k,
                iterations=5
            )
            
            results["funasr"] = {
                "mean_ms": result["mean_ms"],
                "memory_mb": result["memory_peak_mb"]
            }
        except ImportError:
            pass
        
        # Test Whisper
        try:
            from asr.whisper import WhisperASR
            from tests.utils.metrics import benchmark
            
            engine = WhisperASR(model="base", device="cpu")
            engine.load_model()
            
            result = benchmark(
                engine.transcribe,
                sample_audio_bytes_16k,
                iterations=5
            )
            
            results["whisper"] = {
                "mean_ms": result["mean_ms"],
                "memory_mb": result["memory_peak_mb"]
            }
        except ImportError:
            pass
        
        # Assert: Results were collected
        assert len(results) > 0, "No ASR engines available for benchmarking"
