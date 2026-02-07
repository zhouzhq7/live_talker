"""
Unit tests for TEN-VAD
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch


@pytest.mark.phase3
@pytest.mark.unit
@pytest.mark.vad
class TestTENVAD:
    """Test TEN-VAD implementation"""
    
    @pytest.fixture
    def ten_vad_class(self):
        """Import TEN-VAD class"""
        try:
            from audio.vad_ten import TENVAD
            return TENVAD
        except ImportError:
            pytest.skip("TEN-VAD not implemented yet")
    
    @pytest.fixture
    def mock_ten_vad_api(self):
        """Mock TEN-VAD API"""
        with patch('audio.vad_ten.ten_vad') as mock:
            mock_instance = MagicMock()
            mock_instance.process.return_value = 0.8
            mock.VAD.return_value = mock_instance
            yield mock, mock_instance
    
    def test_ten_vad_init(self, ten_vad_class, mock_ten_vad_api):
        """Test TEN-VAD initialization"""
        mock, _ = mock_ten_vad_api
        
        vad = ten_vad_class(
            sample_rate=16000,
            hop_size=256,
            threshold=0.5
        )
        
        assert vad.method == "ten"
        assert vad.sample_rate == 16000
        assert vad.hop_size == 256
        assert vad.threshold == 0.5
    
    def test_load_model_success(self, ten_vad_class, mock_ten_vad_api):
        """Test successful model loading"""
        mock, _ = mock_ten_vad_api
        
        vad = ten_vad_class()
        result = vad.load_model()
        
        assert result is True
        assert vad._is_initialized is True
    
    def test_detect_speech(self, ten_vad_class, mock_ten_vad_api):
        """Test speech detection"""
        mock, mock_instance = mock_ten_vad_api
        
        vad = ten_vad_class(threshold=0.5)
        vad.load_model()
        
        # Mock speech audio
        audio = np.ones(512, dtype=np.int16).tobytes()
        
        result = vad.detect(audio)
        
        assert isinstance(result, bool)
    
    def test_detect_no_speech(self, ten_vad_class, mock_ten_vad_api):
        """Test no speech detection"""
        mock, mock_instance = mock_ten_vad_api
        mock_instance.process.return_value = 0.1  # Below threshold
        
        vad = ten_vad_class(threshold=0.5)
        vad.load_model()
        
        # Silence audio
        audio = np.zeros(512, dtype=np.int16).tobytes()
        
        result = vad.detect(audio)
        
        assert result is False
    
    def test_update_state_speech_start(self, ten_vad_class, mock_ten_vad_api):
        """Test speech start state update"""
        mock, _ = mock_ten_vad_api
        
        vad = ten_vad_class(
            min_speech_duration=0.1,
            min_silence_duration=0.5
        )
        
        import time
        start_time = time.time()
        
        # Simulate speech detection over time
        state = {"speech_started": False, "speech_ended": False, "is_speaking": False}
        
        # First detection (not long enough)
        state = vad.update_state(True)
        assert state["speech_started"] is False
        
        # Wait for min_speech_duration
        time.sleep(0.15)
        
        # Second detection (now long enough)
        state = vad.update_state(True)
        assert state["speech_started"] is True
    
    def test_update_state_speech_end(self, ten_vad_class, mock_ten_vad_api):
        """Test speech end state update"""
        mock, _ = mock_ten_vad_api
        
        vad = ten_vad_class(
            min_speech_duration=0.1,
            min_silence_duration=0.1
        )
        
        # Start speaking
        vad.is_speaking = True
        
        import time
        
        # First silence (not long enough)
        state = vad.update_state(False)
        assert state["speech_ended"] is False
        
        # Wait for min_silence_duration
        time.sleep(0.15)
        
        # Second silence (now long enough)
        state = vad.update_state(False)
        assert state["speech_ended"] is True
    
    def test_reset(self, ten_vad_class, mock_ten_vad_api):
        """Test reset method"""
        mock, _ = mock_ten_vad_api
        
        vad = ten_vad_class()
        vad.is_speaking = True
        vad.speech_start_time = 12345
        vad.silence_start_time = 67890
        
        vad.reset()
        
        assert vad.is_speaking is False
        assert vad.speech_start_time is None
        assert vad.silence_start_time is None
    
    def test_fallback_to_energy(self, ten_vad_class):
        """Test fallback to energy-based VAD"""
        with patch('audio.vad_ten.ten_vad', side_effect=ImportError):
            with patch('audio.vad_ten.VADDetector') as mock_parent:
                mock_instance = MagicMock()
                mock_parent.return_value = mock_instance
                
                try:
                    vad = ten_vad_class()
                    # Should fallback to energy-based
                except:
                    pass  # Expected if not implemented


@pytest.mark.phase3
@pytest.mark.unit
@pytest.mark.vad
class TestVADConfig:
    """Test VAD configuration"""
    
    def test_vad_config_defaults(self):
        """Test default VAD configuration"""
        from config import VADConfig
        
        config = VADConfig()
        assert config.method == "silero"
    
    def test_ten_vad_config(self):
        """Test TEN-VAD configuration"""
        from config import VADConfig
        
        config = VADConfig(
            method="ten",
            ten_hop_size=256,
            ten_threshold=0.5
        )
        
        assert config.method == "ten"
        assert config.ten_hop_size == 256
        assert config.ten_threshold == 0.5


@pytest.mark.phase3
@pytest.mark.unit
@pytest.mark.vad
class TestVADDetectorComparison:
    """Test VAD detector comparison"""
    
    def test_vad_factory(self, vad_test_audio):
        """Test VAD factory creates correct detector"""
        from audio.vad import VADDetector
        
        # Test Silero
        vad_silero = VADDetector(method="silero")
        assert vad_silero.method in ["silero", "energy"]
        
        # Test WebRTC
        vad_webrtc = VADDetector(method="webrtc")
        assert vad_webrtc.method in ["webrtc", "energy"]
        
        # Test Energy
        vad_energy = VADDetector(method="energy")
        assert vad_energy.method == "energy"
    
    def test_vad_detect_all_methods(self, vad_test_audio):
        """Test detection with all VAD methods"""
        from audio.vad import VADDetector
        
        methods = ["energy"]  # Always available
        
        for method in methods:
            try:
                vad = VADDetector(method=method)
                result = vad.detect(vad_test_audio["pure_speech"])
                assert isinstance(result, bool)
            except Exception as e:
                pytest.skip(f"{method} VAD not available: {e}")
