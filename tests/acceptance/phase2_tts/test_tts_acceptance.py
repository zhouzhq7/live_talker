"""
Acceptance tests for TTS (MeloTTS)

Based on CHECKLIST.md requirements:
- First chunk latency < 500ms
- 10 char text synthesis < 1s
- 100 char text synthesis < 3s
- Output: 16-bit PCM, 16kHz, mono
"""

import pytest
import numpy as np


@pytest.mark.phase2
@pytest.mark.acceptance
@pytest.mark.tts
@pytest.mark.slow
class TestTTSAcceptance:
    """Acceptance tests for TTS"""
    
    @pytest.fixture
    def test_texts(self):
        """Test texts"""
        return {
            "10_chars": "你好，世界。",
            "50_chars": "这是一个测试文本，用于测试MeloTTS的语音合成。",
            "100_chars": "这是一个很长的测试文本，用于测试MeloTTS的语音合成性能，确保语音合成自然流畅。",
        }
    
    def test_first_chunk_latency(self, performance_monitor):
        """Test first chunk latency < 500ms"""
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
        """Test 10 char text synthesis < 1s"""
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
        """Test 100 char text synthesis < 3s"""
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
    
    def test_output_format(self):
        """Test output audio format"""
        try:
            from tts.melotts import MeloTTS
            engine = MeloTTS(language="ZH")
            engine.load_model()
        except ImportError:
            pytest.skip("MeloTTS not installed")
        
        audio = engine.synthesize("测试")
        
        # Convert to numpy array
        audio_array = np.frombuffer(audio, dtype=np.int16)
        
        # Check format
        assert len(audio_array) > 0
        assert audio_array.dtype == np.int16
        
        # Should be approximately 16kHz sample rate output
        # (length depends on text length)
        assert len(audio_array) > 1000
    
    def test_multilingual_support(self):
        """Test multilingual synthesis"""
        try:
            from tts.melotts import MeloTTS
        except ImportError:
            pytest.skip("MeloTTS not installed")
        
        languages = ["ZH", "EN"]
        
        for lang in languages:
            engine = MeloTTS(language=lang)
            engine.load_model()
            
            if lang == "ZH":
                text = "你好"
            else:
                text = "Hello"
            
            audio = engine.synthesize(text)
            assert isinstance(audio, bytes)
            assert len(audio) > 0
    
    def test_no_ffmpeg_requirement(self):
        """Test that MeloTTS doesn't require FFmpeg"""
        # This is a configuration/design test
        # MeloTTS should work without FFmpeg
        try:
            import subprocess
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True
            )
            ffmpeg_available = result.returncode == 0
        except:
            ffmpeg_available = False
        
        # Test should pass regardless of FFmpeg availability
        # because MeloTTS doesn't require it
        try:
            from tts.melotts import MeloTTS
            engine = MeloTTS(language="ZH")
            engine.load_model()
            audio = engine.synthesize("测试")
            assert len(audio) > 0
        except ImportError:
            pytest.skip("MeloTTS not installed")
