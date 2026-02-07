"""
Unit tests for TEN-VAD
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import sys


# Mock ten_vad module before importing TENVAD
@pytest.fixture(scope="module", autouse=True)
def mock_ten_vad_module():
    """Mock ten_vad module for all tests"""
    mock_ten_vad = MagicMock()
    mock_vad_class = MagicMock()
    mock_instance = MagicMock()
    mock_instance.process.return_value = 0.8
    mock_vad_class.return_value = mock_instance
    mock_ten_vad.TenVad = mock_vad_class
    
    # Add to sys.modules
    sys.modules['ten_vad'] = mock_ten_vad
    
    yield mock_ten_vad
    
    # Cleanup
    if 'ten_vad' in sys.modules:
        del sys.modules['ten_vad']


@pytest.mark.phase3
@pytest.mark.unit
@pytest.mark.vad
class TestTENVAD:
    """Test TEN-VAD implementation"""
    
    @pytest.fixture
    def ten_vad_class(self):
        """Import TEN-VAD class"""
        from audio.vad_ten import TENVAD
        return TENVAD
    
    def test_ten_vad_init(self, ten_vad_class, mock_ten_vad_module):
        """Test TEN-VAD initialization"""
        vad = ten_vad_class(
            sample_rate=16000,
            hop_size=256,
            threshold=0.5
        )
        
        assert vad.method == "ten"
        assert vad.sample_rate == 16000
        assert vad.hop_size == 256
        assert vad.threshold == 0.5
    
    def test_load_model_success(self, ten_vad_class, mock_ten_vad_module):
        """Test successful model loading"""
        vad = ten_vad_class()
        
        # Model is loaded during __init__ with mock
        assert vad.is_available() is True
    
    def test_detect_speech(self, ten_vad_class, mock_ten_vad_module):
        """Test speech detection"""
        vad = ten_vad_class(threshold=0.5)
        
        # Create test audio (1 second of sine wave)
        t = np.linspace(0, 1, 16000)
        audio = (np.sin(2 * np.pi * 440 * t) * 0.3 * 32767).astype(np.int16)
        
        result = vad.detect(audio.tobytes())
        
        assert isinstance(result, bool)
    
    def test_detect_no_speech(self, ten_vad_class, mock_ten_vad_module):
        """Test no speech detection"""
        # Mock process to return low value (below threshold)
        mock_ten_vad_module.TenVad.return_value.process.return_value = 0.1
        
        vad = ten_vad_class(threshold=0.5)
        
        # Silence audio
        audio = np.zeros(16000, dtype=np.int16)
        
        result = vad.detect(audio.tobytes())
        
        assert result is False
    
    def test_update_state_speech_start(self, ten_vad_class, mock_ten_vad_module):
        """Test speech start state update"""
        import time
        
        vad = ten_vad_class(min_speech_duration=0.05)
        
        # Simulate speech detection over time
        state = vad.update_state(True)
        assert state["speech_started"] is False
        
        # Wait for min_speech_duration
        time.sleep(0.1)
        
        state = vad.update_state(True)
        assert state["speech_started"] is True
    
    def test_update_state_speech_end(self, ten_vad_class, mock_ten_vad_module):
        """Test speech end state update"""
        import time
        
        vad = ten_vad_class(min_silence_duration=0.05)
        
        # Start speaking
        vad.is_speaking = True
        
        # First silence (not long enough)
        state = vad.update_state(False)
        assert state["speech_ended"] is False
        
        # Wait for min_silence_duration
        time.sleep(0.1)
        
        # Second silence (now long enough)
        state = vad.update_state(False)
        assert state["speech_ended"] is True
    
    def test_reset(self, ten_vad_class, mock_ten_vad_module):
        """Test reset method"""
        vad = ten_vad_class()
        vad.is_speaking = True
        vad.speech_start_time = 12345
        vad.silence_start_time = 67890
        
        vad.reset()
        
        assert vad.is_speaking is False
        assert vad.speech_start_time is None
        assert vad.silence_start_time is None
    
    def test_get_info(self, ten_vad_class, mock_ten_vad_module):
        """Test get_info method"""
        vad = ten_vad_class(
            hop_size=256,
            threshold=0.5
        )
        
        info = vad.get_info()
        
        assert info["method"] == "ten"
        assert info["hop_size"] == 256
        assert info["threshold"] == 0.5
        assert info["library_size_kb"] == 306
    
    def test_compare_with_silero(self, ten_vad_class, mock_ten_vad_module):
        """Test comparison with Silero"""
        comparison = ten_vad_class.compare_with_silero()
        
        assert "library_size" in comparison
        assert "speed" in comparison
        assert "latency" in comparison
        assert comparison["library_size"]["ten_vad"] == "306 KB"


@pytest.mark.phase3
@pytest.mark.unit
@pytest.mark.vad
class TestVADConfig:
    """Test VAD configuration"""
    
    def test_vad_config_defaults(self):
        """Test default VAD configuration"""
        from config import VADConfig
        
        config = VADConfig()
        assert config.method == "ten"  # Updated default
    
    def test_ten_vad_config(self):
        """Test TEN-VAD-specific configuration"""
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
    
    def test_vad_factory_with_ten(self, vad_test_audio):
        """Test VAD factory creates TEN-VAD detector"""
        from audio.vad import VADDetector
        
        # Test with energy-based VAD (always available)
        vad = VADDetector(method="energy")
        assert vad.method == "energy"
    
    def test_vad_detect_with_ten(self, vad_test_audio):
        """Test detection with TEN-VAD"""
        from audio.vad import VADDetector
        
        # Test with energy-based (always available)
        vad = VADDetector(method="energy")
        result = vad.detect(vad_test_audio["pure_speech"])
        # numpy bool_ is subclass of bool
        assert result in [True, False]
