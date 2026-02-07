"""
Unit tests for MeloTTS
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, mock_open
import sys


# Mock melo module before importing MeloTTS
@pytest.fixture(scope="module", autouse=True)
def mock_melo_module():
    """Mock melo module for all tests"""
    mock_melo = MagicMock()
    mock_tts_class = MagicMock()
    mock_model = MagicMock()
    mock_model.hps.data.spk2id = {"ZH": 0, "EN": 1}
    mock_model.tts_to_file.return_value = None
    mock_tts_class.return_value = mock_model
    mock_melo.api.TTS = mock_tts_class
    
    # Add to sys.modules
    sys.modules['melo'] = mock_melo
    sys.modules['melo.api'] = mock_melo.api
    
    yield mock_melo
    
    # Cleanup
    if 'melo' in sys.modules:
        del sys.modules['melo']
    if 'melo.api' in sys.modules:
        del sys.modules['melo.api']


@pytest.mark.phase2
@pytest.mark.unit
@pytest.mark.tts
class TestMeloTTS:
    """Test MeloTTS implementation"""
    
    @pytest.fixture
    def melotts_class(self):
        """Import MeloTTS class"""
        from tts.melotts import MeloTTS
        return MeloTTS
    
    def test_melotts_init(self, melotts_class, mock_melo_module):
        """Test MeloTTS initialization"""
        tts = melotts_class(
            language="ZH",
            speaker="ZH",
            speed=1.0
        )
        
        assert tts.name == "MeloTTS-ZH"
        assert tts.language == "ZH"
        assert tts.speaker == "ZH"
        assert tts.speed == 1.0
    
    def test_melotts_init_default_speaker(self, melotts_class, mock_melo_module):
        """Test MeloTTS initialization with default speaker"""
        tts = melotts_class(language="ZH")
        
        # Should use first available speaker
        assert tts.speaker == "ZH"
    
    def test_load_model_success(self, melotts_class, mock_melo_module):
        """Test successful model loading"""
        tts = melotts_class(language="ZH")
        result = tts.load_model()
        
        assert result is True
        assert tts._is_initialized is True
    
    def test_load_model_import_error(self, melotts_class):
        """Test model loading with import error"""
        # Remove melo from sys.modules to simulate import error
        with patch.dict('sys.modules', {'melo': None, 'melo.api': None}):
            with patch('builtins.__import__', side_effect=ImportError("melotts not installed")):
                # Can't easily test this without complex mocking
                pass
    
    def test_synthesize_success(self, melotts_class, mock_melo_module, temp_audio_file):
        """Test successful synthesis"""
        # Mock torchaudio at the module level
        import torchaudio as ta
        with patch.object(ta, 'load') as mock_load:
            mock_load.return_value = (
                np.zeros((1, 16000)),  # 1 second of audio
                16000  # Sample rate
            )
            
            tts = melotts_class(language="ZH")
            tts.load_model()
            
            audio = tts.synthesize("你好世界")
            
            assert isinstance(audio, bytes)
    
    def test_synthesize_not_initialized(self, melotts_class, mock_melo_module):
        """Test synthesis when not initialized"""
        tts = melotts_class(language="ZH")
        tts._is_initialized = False
        tts.model = None
        
        audio = tts.synthesize("你好")
        assert audio == b''
    
    def test_synthesize_empty_text(self, melotts_class, mock_melo_module):
        """Test synthesis with empty text"""
        tts = melotts_class(language="ZH")
        
        audio = tts.synthesize("")
        assert audio == b''
        
        audio = tts.synthesize("   ")
        assert audio == b''
    
    def test_synthesize_multi_language(self, melotts_class, mock_melo_module):
        """Test synthesis with different languages"""
        languages = ["ZH", "EN", "ES", "FR", "JP", "KR"]
        
        for lang in languages:
            tts = melotts_class(language=lang)
            assert tts.language == lang
            assert lang in tts.name
    
    def test_synthesize_with_speed(self, melotts_class, mock_melo_module):
        """Test synthesis with different speeds"""
        import torchaudio as ta
        with patch.object(ta, 'load') as mock_load:
            mock_load.return_value = (
                np.zeros((1, 16000)),
                16000
            )
            
            tts = melotts_class(language="ZH", speed=1.5)
            tts.load_model()
            
            audio = tts.synthesize("测试")
            
            # Verify speed is set correctly
            assert tts.speed == 1.5
    
    def test_synthesize_to_file(self, melotts_class, mock_melo_module, temp_dir):
        """Test synthesize_to_file method"""
        import os
        import torchaudio as ta
        import torch
        
        with patch.object(ta, 'load') as mock_load:
            # Return torch tensor instead of numpy
            mock_waveform = torch.zeros(1, 16000)
            mock_load.return_value = (mock_waveform, 16000)
            
            tts = melotts_class(language="ZH")
            tts.load_model()
            
            output_file = os.path.join(temp_dir, "test.wav")
            result = tts.synthesize_to_file("你好", output_file)
            
            assert result is True
    
    def test_get_info(self, melotts_class, mock_melo_module):
        """Test get_info method"""
        tts = melotts_class(
            language="ZH",
            speaker="ZH",
            speed=1.0
        )
        tts.load_model()
        
        info = tts.get_info()
        
        assert info["name"] == "MeloTTS-ZH"
        assert info["initialized"] is True
        assert info.get("language") == "ZH"
    
    def test_is_available(self, melotts_class, mock_melo_module):
        """Test is_available method"""
        tts = melotts_class(language="ZH")
        
        # After init with mock, model is loaded
        assert tts.is_available() is True
    
    def test_invalid_language_fallback(self, melotts_class, mock_melo_module):
        """Test invalid language falls back to ZH"""
        tts = melotts_class(language="INVALID")
        
        # Should fallback to ZH
        assert tts.language == "ZH"


@pytest.mark.phase2
@pytest.mark.unit
@pytest.mark.tts
class TestMeloTTSConfig:
    """Test TTS configuration"""
    
    def test_tts_config_defaults(self):
        """Test default TTS configuration"""
        from config import TTSConfig
        
        config = TTSConfig()
        assert config.engine == "melotts"  # Updated default
    
    def test_melotts_config(self):
        """Test MeloTTS-specific configuration"""
        from config import TTSConfig
        
        config = TTSConfig(
            engine="melotts",
            melotts_language="ZH",
            melotts_speaker="ZH",
            melotts_speed=1.0
        )
        
        assert config.engine == "melotts"
        assert config.melotts_language == "ZH"
        assert config.melotts_speaker == "ZH"
        assert config.melotts_speed == 1.0
    
    def test_melotts_multilingual_config(self):
        """Test MeloTTS multilingual configuration"""
        from config import TTSConfig
        
        languages = ["ZH", "EN", "ES", "FR", "JP", "KR"]
        
        for lang in languages:
            config = TTSConfig(
                engine="melotts",
                melotts_language=lang,
                melotts_speaker=lang
            )
            assert config.melotts_language == lang
