"""
Unit Tests for ASR Streaming Module
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asr.base import BaseASR, ASRResult, StreamingASRMixin


class TestASRResult:
    """Test cases for ASRResult"""

    def test_creation(self):
        """Test ASR result creation"""
        result = ASRResult(
            text="你好",
            is_final=True,
            confidence=0.95,
            timestamp_start=0,
            timestamp_end=1000
        )

        assert result.text == "你好"
        assert result.is_final is True
        assert result.confidence == 0.95
        assert result.timestamp_start == 0
        assert result.timestamp_end == 1000

    def test_defaults(self):
        """Test default values"""
        result = ASRResult(text="测试", is_final=False)

        assert result.confidence == 0.0
        assert result.timestamp_start == 0
        assert result.timestamp_end == 0


class TestBaseASR:
    """Test cases for BaseASR"""

    def test_init(self):
        """Test BaseASR initialization"""
        class TestASR(BaseASR):
            def load_model(self):
                return True

            def transcribe(self, audio_data, sample_rate=16000, language="zh"):
                return "test"

        asr = TestASR(name="test", model_size="small")
        assert asr.name == "test"
        assert asr.config["model_size"] == "small"
        assert asr._is_initialized is False

    def test_is_available_default(self):
        """Test default availability"""
        class TestASR(BaseASR):
            def load_model(self):
                return True

            def transcribe(self, audio_data, sample_rate=16000, language="zh"):
                return ""

        asr = TestASR(name="test")
        assert asr.is_available() is False

    def test_warmup(self):
        """Test warmup"""
        class TestASR(BaseASR):
            def __init__(self):
                super().__init__("test")
                self.warmup_count = 0

            def load_model(self):
                return True

            def transcribe(self, audio_data, sample_rate=16000, language="zh"):
                return "test"

        asr = TestASR()
        asr.warmup(num_iterations=2)
        assert asr._warmup_done is True


class TestStreamingASRMixin:
    """Test cases for StreamingASRMixin"""

    def test_interface_methods(self):
        """Test that mixin has required methods"""
        methods = [
            'stream_recognize',
            'start_stream',
            'accept_audio',
            'get_result',
            'end_stream',
            'reset'
        ]

        for method in methods:
            assert hasattr(StreamingASRMixin, method)


class TestSherpaONNXASR:
    """Test cases for SherpaONNXASR"""

    @pytest.fixture
    def sherpa_asr(self):
        """Create SherpaONNXASR instance"""
        pytest.importorskip("sherpa_onnx")
        from asr.sherpa_onnx import SherpaONNXASR

        config = {
            "num_threads": 1,
            "sample_rate": 16000,
        }

        asr = SherpaONNXASR(model_type="paraformer", config=config)
        return asr

    def test_init(self, sherpa_asr):
        """Test initialization"""
        assert sherpa_asr.name.startswith("SherpaONNX")
        assert sherpa_asr.model_type == "paraformer"

    def test_get_info(self, sherpa_asr):
        """Test get_info method"""
        info = sherpa_asr.get_info()

        assert "name" in info
        assert "model_type" in info
        assert "sample_rate" in info
        assert "streaming" in info

    def test_streaming_state(self, sherpa_asr):
        """Test streaming state management"""
        assert sherpa_asr._streaming is False

        sherpa_asr.start_stream()
        assert sherpa_asr._streaming is True

        sherpa_asr.reset()
        assert sherpa_asr._streaming is False

    def test_load_model_unavailable(self, sherpa_asr):
        """Test load model when sherpa not available"""
        if not sherpa_asr._sherpa_available:
            result = sherpa_asr.load_model()
            assert result is False


class TestFunASRStreaming:
    """Test cases for FunASRStreaming"""

    @pytest.fixture
    def funasr_streaming(self):
        """Create FunASRStreaming instance"""
        pytest.importorskip("funasr")
        from asr.funasr import FunASRStreaming

        asr = FunASRStreaming(
            model_name="paraformer-zh-streaming",
            device="cpu"
        )
        return asr

    def test_init(self, funasr_streaming):
        """Test initialization"""
        assert funasr_streaming.name.startswith("FunASR")
        assert funasr_streaming.model_name == "paraformer-zh-streaming"

    def test_get_info(self, funasr_streaming):
        """Test get_info method"""
        info = funasr_streaming.get_info()

        assert "name" in info
        assert "model_name" in info
        assert "device" in info
        assert "streaming" in info

    def test_streaming_state(self, funasr_streaming):
        """Test streaming state management"""
        assert funasr_streaming._streaming is False

        funasr_streaming.start_stream()
        assert funasr_streaming._streaming is True

        funasr_streaming.reset()
        assert funasr_streaming._streaming is False


class TestASRIntegration:
    """Integration tests for ASR modules"""

    def test_asr_interface_compatibility(self):
        """Test that all ASR implementations have compatible interfaces"""
        # Skip if dependencies not available
        try:
            from asr.sherpa_onnx import SherpaONNXASR
        except ImportError:
            pytest.skip("sherpa_onnx not available")

        try:
            from asr.funasr import FunASR, FunASRStreaming
        except ImportError:
            pytest.skip("funasr not available")

        # All should have these methods
        for asr_class in [SherpaONNXASR, FunASR, FunASRStreaming]:
            assert hasattr(asr_class, 'load_model')
            assert hasattr(asr_class, 'transcribe')
            assert hasattr(asr_class, 'get_info')
            assert hasattr(asr_class, 'is_available')

    def test_transcribe_compatibility(self):
        """Test transcribe method signature compatibility"""
        from asr.base import BaseASR

        # Create test audio
        test_audio = np.zeros(16000, dtype=np.int16).tobytes()

        class TestASR(BaseASR):
            def __init__(self):
                super().__init__("test")

            def load_model(self):
                return True

            def transcribe(self, audio_data, sample_rate=16000, language="zh"):
                return "test result"

        asr = TestASR()
        result = asr.transcribe(test_audio, sample_rate=16000, language="zh")

        assert isinstance(result, str)
        assert result == "test result"

    def test_asr_result_namedtuple(self):
        """Test ASRResult as NamedTuple"""
        result = ASRResult(
            text="识别文本",
            is_final=True,
            confidence=0.92,
            timestamp_start=500,
            timestamp_end=2500
        )

        # Test unpacking
        text, is_final, confidence, ts_start, ts_end = result
        assert text == "识别文本"
        assert is_final is True
        assert confidence == 0.92
        assert ts_start == 500
        assert ts_end == 2500

        # Test indexing
        assert result[0] == "识别文本"
        assert result[1] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
