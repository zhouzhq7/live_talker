"""
Performance tests for MeloTTS

Performance targets (from CHECKLIST.md):
- First chunk latency < 500ms
- 10 char text synthesis < 1s
- 100 char text synthesis < 3s
- Memory < 1GB
- No FFmpeg required
"""

import pytest
import time
from tests.utils.metrics import benchmark


@pytest.mark.phase2
@pytest.mark.performance
@pytest.mark.tts
@pytest.mark.slow
class TestMeloTTSPerformance:
    """Performance tests for MeloTTS"""
    
    @pytest.fixture
    def test_texts(self):
        """Test texts of different lengths"""
        return {
            "10_chars": "你好，世界。",
            "50_chars": "这是一个测试文本，用于测试MeloTTS的语音合成性能。",
            "100_chars": "这是一个很长的测试文本，用于测试MeloTTS的语音合成性能。" * 2,
        }
    
    def test_model_loading_time(self, performance_monitor):
        """Test model loading time"""
        try:
            from tts.melotts import MeloTTS
        except ImportError:
            pytest.skip("MeloTTS not installed")
        
        monitor = performance_monitor()
        monitor.start()
        
        engine = MeloTTS(language="ZH")
        engine.load_model()
        
        metrics = monitor.stop()
        
        # Model loading should be reasonable (< 30s)
        assert metrics["duration_ms"] < 30000, \
            f"Model loading too slow: {metrics['duration_ms']/1000:.2f}s"
    
    def test_first_chunk_latency(self, performance_monitor):
        """Test first chunk latency (< 500ms)"""
        try:
            from tts.melotts import MeloTTS
            engine = MeloTTS(language="ZH")
            engine.load_model()
        except ImportError:
            pytest.skip("MeloTTS not installed")
        
        monitor = performance_monitor()
        monitor.start()
        
        audio = engine.synthesize("你好")
        
        metrics = monitor.stop()
        
        assert metrics["duration_ms"] < 500, \
            f"First chunk too slow: {metrics['duration_ms']:.2f}ms"
    
    def test_synthesis_10_chars(self, performance_monitor, test_texts):
        """Test 10 char text synthesis (< 1s)"""
        try:
            from tts.melotts import MeloTTS
            engine = MeloTTS(language="ZH")
            engine.load_model()
        except ImportError:
            pytest.skip("MeloTTS not installed")
        
        monitor = performance_monitor()
        monitor.start()
        
        audio = engine.synthesize(test_texts["10_chars"])
        
        metrics = monitor.stop()
        
        assert metrics["duration_ms"] < 1000, \
            f"10 char synthesis too slow: {metrics['duration_ms']:.2f}ms"
    
    def test_synthesis_100_chars(self, performance_monitor, test_texts):
        """Test 100 char text synthesis (< 3s)"""
        try:
            from tts.melotts import MeloTTS
            engine = MeloTTS(language="ZH")
            engine.load_model()
        except ImportError:
            pytest.skip("MeloTTS not installed")
        
        monitor = performance_monitor()
        monitor.start()
        
        audio = engine.synthesize(test_texts["100_chars"])
        
        metrics = monitor.stop()
        
        assert metrics["duration_ms"] < 3000, \
            f"100 char synthesis too slow: {metrics['duration_ms']:.2f}ms"
    
    def test_memory_usage(self, performance_monitor, test_texts):
        """Test memory usage during synthesis (< 1GB)"""
        try:
            from tts.melotts import MeloTTS
            engine = MeloTTS(language="ZH")
            engine.load_model()
        except ImportError:
            pytest.skip("MeloTTS not installed")
        
        monitor = performance_monitor()
        monitor.start()
        
        # Synthesize multiple texts
        for _ in range(10):
            engine.synthesize(test_texts["50_chars"])
        
        metrics = monitor.stop()
        
        assert metrics["peak_memory_mb"] < 1024, \
            f"Memory too high: {metrics['peak_memory_mb']:.0f}MB"
    
    def test_concurrent_synthesis(self, test_texts):
        """Test concurrent synthesis performance"""
        import concurrent.futures
        
        try:
            from tts.melotts import MeloTTS
            engine = MeloTTS(language="ZH")
            engine.load_model()
        except ImportError:
            pytest.skip("MeloTTS not installed")
        
        def synthesize():
            return engine.synthesize(test_texts["10_chars"])
        
        start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(synthesize) for _ in range(4)]
            results = [f.result() for f in futures]
        end = time.time()
        
        total_time = (end - start) * 1000
        
        assert total_time < 6000, \
            f"Concurrent synthesis too slow: {total_time:.0f}ms"
    
    def test_throughput(self, test_texts):
        """Test synthesis throughput"""
        try:
            from tts.melotts import MeloTTS
            engine = MeloTTS(language="ZH")
            engine.load_model()
        except ImportError:
            pytest.skip("MeloTTS not installed")
        
        # Benchmark synthesis
        result = benchmark(
            engine.synthesize,
            test_texts["10_chars"],
            iterations=10
        )
        
        # Should be reasonably fast
        assert result["mean_ms"] < 1000, \
            f"Throughput too low: {result['mean_ms']:.2f}ms per synthesis"


@pytest.mark.phase2
@pytest.mark.performance
@pytest.mark.tts
class TestTTSBenchmark:
    """Benchmark tests comparing TTS engines"""
    
    def test_benchmark_tts_engines(self):
        """Benchmark different TTS engines"""
        results = {}
        test_text = "你好，世界。"
        
        # Test Edge-TTS
        try:
            from tts.edge_tts import EdgeTTS
            from tests.utils.metrics import benchmark
            
            engine = EdgeTTS(voice="zh-CN-XiaoxiaoNeural")
            
            result = benchmark(
                engine.synthesize,
                test_text,
                iterations=3
            )
            
            results["edge"] = {
                "mean_ms": result["mean_ms"],
                "memory_mb": result["memory_peak_mb"]
            }
        except ImportError:
            pass
        
        # Test Pyttsx3
        try:
            from tts.pyttsx3_tts import Pyttsx3TTS
            from tests.utils.metrics import benchmark
            
            engine = Pyttsx3TTS()
            
            result = benchmark(
                engine.synthesize,
                test_text,
                iterations=3
            )
            
            results["pyttsx3"] = {
                "mean_ms": result["mean_ms"],
                "memory_mb": result["memory_peak_mb"]
            }
        except ImportError:
            pass
        
        # Assert: Results were collected
        assert len(results) > 0, "No TTS engines available for benchmarking"
