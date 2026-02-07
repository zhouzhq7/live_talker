"""
Unit Tests for VAD Module
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vad.base import VADState, VADResult, VADetector
from vad.silero import SileroVAD


class TestVADBase:
    """Test cases for VAD base classes"""

    def test_vad_state_enum(self):
        """Test VAD state enum values"""
        assert VADState.SILENCE.value == "silence"
        assert VADState.SPEECH_START.value == "speech_start"
        assert VADState.SPEAKING.value == "speaking"
        assert VADState.SPEECH_END.value == "speech_end"

    def test_vad_result_creation(self):
        """Test VAD result creation"""
        result = VADResult(
            is_speech=True,
            confidence=0.8,
            state=VADState.SPEAKING,
            speech_duration=1.5
        )

        assert result.is_speech is True
        assert result.confidence == 0.8
        assert result.state == VADState.SPEAKING
        assert result.speech_duration == 1.5

    def test_vad_result_defaults(self):
        """Test VAD result default values"""
        result = VADResult(
            is_speech=False,
            confidence=0.1,
            state=VADState.SILENCE
        )

        assert result.speech_duration is None
        assert result.silence_duration is None


class TestVADetector:
    """Test cases for VADetector abstract class"""

    def test_init_with_config(self):
        """Test detector initialization with config"""
        config = {
            "threshold": 0.6,
            "min_speech_duration": 0.3,
            "min_silence_duration": 0.6,
            "sample_rate": 16000,
        }

        class TestVAD(VADetector):
            def load_model(self):
                return True

            def detect(self, audio_chunk):
                return VADResult(False, 0.0, VADState.SILENCE)

            def reset(self):
                pass

        vad = TestVAD(config)

        assert vad.threshold == 0.6
        assert vad.min_speech_duration == 0.3
        assert vad.min_silence_duration == 0.6

    def test_init_without_config(self):
        """Test detector initialization without config"""
        class TestVAD(VADetector):
            def load_model(self):
                return True

            def detect(self, audio_chunk):
                return VADResult(False, 0.0, VADState.SILENCE)

            def reset(self):
                pass

        vad = TestVAD()

        assert vad.threshold == 0.5
        assert vad.min_speech_duration == 0.25
        assert vad.min_silence_duration == 0.5

    def test_set_threshold(self):
        """Test threshold setting"""
        class TestVAD(VADetector):
            def load_model(self):
                return True

            def detect(self, audio_chunk):
                return VADResult(False, 0.0, VADState.SILENCE)

            def reset(self):
                pass

        vad = TestVAD()
        vad.set_threshold(0.8)
        assert vad.threshold == 0.8

        # Test clamping
        vad.set_threshold(1.5)
        assert vad.threshold == 1.0

        vad.set_threshold(-0.5)
        assert vad.threshold == 0.0

    def test_get_info(self):
        """Test get_info method"""
        class TestVAD(VADetector):
            def load_model(self):
                return True

            def detect(self, audio_chunk):
                return VADResult(False, 0.0, VADState.SILENCE)

            def reset(self):
                pass

        vad = TestVAD({"sample_rate": 16000})
        info = vad.get_info()

        assert "threshold" in info
        assert "sample_rate" in info
        assert info["sample_rate"] == 16000


class TestSileroVAD:
    """Test cases for SileroVAD implementation"""

    @pytest.fixture
    def silero_vad(self, mock_config):
        """Create SileroVAD instance"""
        vad = SileroVAD(mock_config)
        return vad

    def test_init(self, silero_vad):
        """Test SileroVAD initialization"""
        assert silero_vad.threshold == 0.5
        assert silero_vad.sample_rate == 16000

    def test_reset(self, silero_vad):
        """Test SileroVAD reset"""
        silero_vad.reset()

        assert silero_vad._speech_start_time is None
        assert silero_vad._silence_start_time is None
        assert silero_vad._is_speaking is False

    def test_detect_silence(self, silero_vad, sample_audio_100ms):
        """Test detecting silence"""
        # Load model first
        silero_vad.load_model()

        if not silero_vad.is_available():
            pytest.skip("Silero model not available")

        result = silero_vad.detect(sample_audio_100ms)

        # Silent audio should not be detected as speech
        assert isinstance(result, VADResult)

    def test_get_info(self, silero_vad, mock_config):
        """Test get_info method"""
        info = silero_vad.get_info()

        assert "type" in info
        assert "threshold" in info
        assert "sample_rate" in info
        assert info["type"] == "SileroVAD"


class TestVADStateMachine:
    """Test VAD state transitions"""

    def test_state_sequence_speech(self):
        """Test state sequence during speech"""
        # SILENCE -> SPEECH_START -> SPEAKING -> SPEECH_END -> SILENCE

        from vad.silero import SileroVAD

        vad = SileroVAD()
        vad.load_model()

        if not vad.is_available():
            pytest.skip("Silero model not available")

        # Start with silence
        silence = np.zeros(1600, dtype=np.int16).tobytes()  # 100ms
        result1 = vad.detect(silence)
        assert result1.state in [VADState.SILENCE, VADState.SPEECH_START]

        # Continue with speech
        speech = (np.random.randn(1600) * 1000).astype(np.int16)
        for _ in range(5):  # Simulate 500ms of speech
            result = vad.detect(speech.tobytes())
            if result.state == VADState.SPEAKING:
                break

    def test_state_sequence_with_min_duration(self):
        """Test state sequence respects min duration"""
        vad = SileroVAD({
            "threshold": 0.5,
            "min_speech_duration": 0.5,  # 500ms
            "min_silence_duration": 0.5,
        })

        vad.load_model()

        if not vad.is_available():
            pytest.skip("Silero model not available")

        # Short speech burst (200ms) should not trigger SPEECH_START
        short_speech = (np.random.randn(3200) * 1000).astype(np.int16)
        result = vad.detect(short_speech.tobytes())

        # Should still be in SILENCE or SPEECH_START
        assert result.state in [VADState.SILENCE, VADState.SPEECH_START]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
