"""
Unit Tests for TEN-VAD Module
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vad.base import VADState, VADResult
from vad.ten_vad import TENVAD, SimpleVAD, create_vad_engine


class TestSimpleVAD:
    """Test cases for SimpleVAD implementation"""

    @pytest.fixture
    def simple_vad(self, mock_config):
        """Create SimpleVAD instance"""
        return SimpleVAD(mock_config)

    def test_init(self, simple_vad):
        """Test SimpleVAD initialization"""
        assert simple_vad.threshold == 0.5
        assert simple_vad.sample_rate == 16000
        assert simple_vad._energy_threshold == 0.02

    def test_load_model(self, simple_vad):
        """Test model loading (always succeeds for SimpleVAD)"""
        assert simple_vad.load_model() is True

    def test_is_available(self, simple_vad):
        """Test availability check"""
        assert simple_vad.is_available() is True

    def test_detect_silence(self, simple_vad, sample_audio_100ms):
        """Test detecting silence"""
        result = simple_vad.detect(sample_audio_100ms)

        assert isinstance(result, VADResult)
        # Silent audio should not be detected as speech
        assert result.is_speech is False or result.state == VADState.SILENCE

    def test_detect_speech(self, simple_vad, sample_audio_speech):
        """Test detecting speech-like audio"""
        result = simple_vad.detect(sample_audio_speech)

        assert isinstance(result, VADResult)
        # Audio with energy should be detected
        assert result.confidence > 0

    def test_reset(self, simple_vad):
        """Test reset"""
        simple_vad._speech_start_time = 1.0
        simple_vad._is_speaking = True

        simple_vad.reset()

        assert simple_vad._speech_start_time is None
        assert simple_vad._silence_start_time is None
        assert simple_vad._is_speaking is False

    def test_get_info(self, simple_vad):
        """Test get_info method"""
        info = simple_vad.get_info()

        assert "type" in info
        assert "energy_threshold" in info
        assert info["type"] == "SimpleVAD"


class TestTENVAD:
    """Test cases for TENVAD implementation"""

    @pytest.fixture
    def ten_vad(self, mock_config):
        """Create TENVAD instance"""
        return TENVAD(mock_config)

    def test_init(self, ten_vad):
        """Test TENVAD initialization"""
        assert ten_vad.threshold == 0.5
        assert ten_vad.sample_rate == 16000

    def test_check_dependencies(self, ten_vad):
        """Test dependency checking"""
        # Should have checked ONNX availability
        assert isinstance(ten_vad._onnx_available, bool)

    def test_load_model(self, ten_vad):
        """Test model loading"""
        # May or may not load depending on dependencies
        result = ten_vad.load_model()
        assert isinstance(result, bool)

    def test_is_available(self, ten_vad):
        """Test availability check"""
        # Depends on whether model loaded successfully
        # Should not crash regardless
        try:
            available = ten_vad.is_available()
            assert isinstance(available, bool)
        except Exception:
            # If dependencies not available, should still handle gracefully
            pass

    def test_detect_silence(self, ten_vad, sample_audio_100ms):
        """Test detecting silence"""
        if not ten_vad._model_loaded:
            pytest.skip("Model not loaded")

        result = ten_vad.detect(sample_audio_100ms)
        assert isinstance(result, VADResult)

    def test_reset(self, ten_vad):
        """Test reset"""
        ten_vad.reset()

        assert ten_vad._speech_start_time is None
        assert ten_vad._silence_start_time is None
        assert ten_vad._is_speaking is False

    def test_get_info(self, ten_vad):
        """Test get_info method"""
        info = ten_vad.get_info()

        assert "model_loaded" in info
        assert "onnx_available" in info


class TestVADFactory:
    """Test cases for VAD factory function"""

    def test_create_silero(self):
        """Test creating SileroVAD"""
        from vad.silero import SileroVAD
        vad = create_vad_engine("silero")
        assert isinstance(vad, SileroVAD)

    def test_create_simple(self):
        """Test creating SimpleVAD"""
        vad = create_vad_engine("simple")
        assert isinstance(vad, SimpleVAD)

    def test_create_ten_vad(self):
        """Test creating TENVAD"""
        vad = create_vad_engine("ten_vad")
        assert isinstance(vad, TENVAD)

    def test_create_unknown(self):
        """Test creating with unknown type falls back to Silero"""
        from vad.silero import SileroVAD
        vad = create_vad_engine("unknown")
        assert isinstance(vad, SileroVAD)

    def test_create_with_config(self):
        """Test creating with custom config"""
        config = {
            "threshold": 0.7,
            "min_speech_duration": 0.5,
        }
        vad = create_vad_engine("simple", config)

        assert vad.threshold == 0.7
        assert vad.min_speech_duration == 0.5


class TestVADIntegration:
    """Integration tests for VAD modules"""

    def test_vad_compatibility(self):
        """Test that all VAD implementations have compatible interfaces"""
        from vad.silero import SileroVAD
        from vad.ten_vad import SimpleVAD, TENVAD

        # All should have these methods
        for vad_class in [SileroVAD, SimpleVAD, TENVAD]:
            assert hasattr(vad_class, 'load_model')
            assert hasattr(vad_class, 'detect')
            assert hasattr(vad_class, 'reset')
            assert hasattr(vad_class, 'get_info')
            assert hasattr(vad_class, 'is_available')

    def test_result_type_compatibility(self):
        """Test that all VADs return compatible results"""
        from vad.silero import SileroVAD
        from vad.ten_vad import SimpleVAD

        silero = SileroVAD()
        simple = SimpleVAD()

        # Both should return VADResult or dict-like objects
        for vad in [silero, simple]:
            if vad.is_available():
                result = vad.detect(b'\x00' * 320)  # 20ms of silence
                assert hasattr(result, 'is_speech')
                assert hasattr(result, 'confidence')
                assert hasattr(result, 'state')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
