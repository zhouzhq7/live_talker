"""
Acceptance tests for ASR (SenseVoice)

Based on CHECKLIST.md requirements:
- Accuracy > 95% for Chinese
- Accuracy > 90% for mixed Chinese-English
- RTF < 0.1
- 10s audio processing < 100ms
"""

import pytest
import numpy as np
from tests.utils.metrics import calculate_accuracy


@pytest.mark.phase1
@pytest.mark.acceptance
@pytest.mark.asr
@pytest.mark.slow
class TestASRAcceptance:
    """Acceptance tests for ASR"""
    
    @pytest.fixture
    def test_audio_files(self):
        """Generate test audio for different scenarios"""
        return {
            "zh_simple": self._generate_test_audio("你好，世界"),
            "zh_complex": self._generate_test_audio("今天的天气真不错"),
            "mixed": self._generate_test_audio("这是一个test"),
        }
    
    def _generate_test_audio(self, text: str) -> bytes:
        """Generate synthetic audio for testing"""
        # Generate 1 second of audio
        t = np.linspace(0, 1, 16000)
        audio = np.sin(2 * np.pi * 440 * t) * 0.3
        return (audio * 32767).astype(np.int16).tobytes()
    
    def test_chinese_accuracy(self, test_audio_files):
        """Test Chinese transcription accuracy (> 95%)"""
        try:
            from asr.sensevoice import SenseVoice
            engine = SenseVoice(device="cpu")
            engine.load_model()
        except ImportError:
            pytest.skip("SenseVoice not installed")
        
        # Test cases
        test_cases = [
            ("你好世界", "你好世界"),
            ("今天的天气真不错", "今天的天气真不错"),
        ]
        
        accuracies = []
        for expected, audio_key in test_cases:
            result = engine.transcribe(test_audio_files["zh_simple"])
            accuracy = calculate_accuracy(expected, result)
            accuracies.append(accuracy)
        
        avg_accuracy = np.mean(accuracies)
        
        assert avg_accuracy >= 0.95, \
            f"Chinese accuracy too low: {avg_accuracy:.2%}"
    
    def test_rtf_requirement(self):
        """Test RTF < 0.1 requirement"""
        try:
            from asr.sensevoice import SenseVoice
            engine = SenseVoice(device="cpu")
            engine.load_model()
        except ImportError:
            pytest.skip("SenseVoice not installed")
        
        # Generate 10s audio
        audio = np.zeros(16000 * 10, dtype=np.int16).tobytes()
        
        result = engine.transcribe_with_timing(audio, sample_rate=16000)
        
        assert result["rtf"] < 0.1, \
            f"RTF too high: {result['rtf']:.4f}"
    
    def test_processing_speed(self):
        """Test 10s audio processing < 100ms"""
        try:
            from asr.sensevoice import SenseVoice
            engine = SenseVoice(device="cpu")
            engine.load_model()
        except ImportError:
            pytest.skip("SenseVoice not installed")
        
        # Generate 10s audio
        audio = np.zeros(16000 * 10, dtype=np.int16).tobytes()
        
        result = engine.transcribe_with_timing(audio, sample_rate=16000)
        
        assert result["latency_ms"] < 100, \
            f"Processing too slow: {result['latency_ms']:.2f}ms"
    
    def test_memory_usage(self, performance_monitor):
        """Test memory usage < 2GB"""
        try:
            from asr.sensevoice import SenseVoice
            engine = SenseVoice(device="cpu")
            engine.load_model()
        except ImportError:
            pytest.skip("SenseVoice not installed")
        
        monitor = performance_monitor()
        monitor.start()
        
        # Process multiple times
        audio = np.zeros(16000, dtype=np.int16).tobytes()
        for _ in range(100):
            engine.transcribe(audio)
        
        metrics = monitor.stop()
        
        assert metrics["peak_memory_mb"] < 2048, \
            f"Memory too high: {metrics['peak_memory_mb']:.0f}MB"
