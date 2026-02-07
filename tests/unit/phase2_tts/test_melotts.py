"""
Unit tests for MeloTTS
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, mock_open


@pytest.mark.phase2
@pytest.mark.unit
@pytest.mark.tts
class TestMeloTTS:
    """Test MeloTTS implementation"""
    
    @pytest.fixture
    def melotts_class(self):
        """Import MeloTTS class"""
        try:
            from tts.melotts import MeloTTS
            return MeloTTS
        except ImportError:
            pytest.skip("MeloTTS not implemented yet")
    
    @pytest.fixture
    def mock_melotts_api(self):
        """Mock MeloTTS API"""
        with patch('tts.melotts.TTS') as mock:
            model_instance = MagicMock()
            model_instance.hps.data.spk2id = {"ZH": 0, "EN": 1}
            model_instance.tts_to_file.return_value = None
            mock.return_value = model_instance
            yield mock, model_instance
    
    def test_melotts_init(self, melotts_class, mock_melotts_api):
        """Test MeloTTS initialization"""
        mock, _ = mock_melotts_api
        
        tts = melotts_class(
            language="ZH",
            speaker="ZH",
            speed=1.0
        )
        
        assert tts.name == "MeloTTS-ZH"
        assert tts.language == "ZH"
        assert tts.speaker == "ZH"
        assert tts.speed == 1.0
    
    def test_melotts_init_default_speaker(self, melotts_class, mock_melotts_api):
        """Test MeloTTS initialization with default speaker"""
        mock, model_instance = mock_melotts_api
        
        tts = melotts_class(language="ZH")
        
        # Should use first available speaker
        assert tts.speaker == "ZH"
    
    def test_load_model_success(self, melotts_class, mock_melotts_api):
        """Test successful model loading"""
        mock, _ = mock_melotts_api
        
        tts = melotts_class(language="ZH")
        result = tts.load_model()
        
        assert result is True
        assert tts._is_initialized is True
        mock.assert_called_once_with(language="ZH", device='auto')
    
    def test_load_model_import_error(self, melotts_class):
        """Test model loading with import error"""
        with patch('tts.melotts.TTS', side_effect=ImportError("melotts not installed")):
            tts = melotts_class(language="ZH")
            result = tts.load_model()
            
            assert result is False
            assert tts._is_initialized is False
    
    def test_synthesize_success(self, melotts_class, mock_melotts_api, temp_audio_file):
        """Test successful synthesis"""
        mock, model_instance = mock_melotts_api
        
        # Mock torchaudio
        with patch('tts.melotts.torchaudio') as mock_torchaudio:
            mock_torchaudio.load.return_value = (
                np.zeros((1, 16000)),  # 1 second of audio
                16000  # Sample rate
            )
            
            tts = melotts_class(language="ZH")
            tts.load_model()
            
            audio = tts.synthesize("你好世界")
            
            assert isinstance(audio, bytes)
            assert len(audio) > 0
    
    def test_synthesize_not_initialized(self, melotts_class):
        """Test synthesis when not initialized"""
        with patch('tts.melotts.TTS', side_effect=Exception("Not installed")):
            tts = melotts_class(language="ZH")
            audio = tts.synthesize("你好")
            
            assert audio == b''
    
    def test_synthesize_empty_text(self, melotts_class, mock_melotts_api):
        """Test synthesis with empty text"""
        tts = melotts_class(language="ZH")
        tts.load_model()
        
        audio = tts.synthesize("")
        assert audio == b''
        
        audio = tts.synthesize("   ")
        assert audio == b''
    
    def test_synthesize_multi_language(self, melotts_class, mock_melotts_api):
        """Test synthesis with different languages"""
        mock, model_instance = mock_melotts_api
        
        languages = ["ZH", "EN", "ES", "FR", "JP", "KR"]
        
        for lang in languages:
            tts = melotts_class(language=lang)
            assert tts.language == lang
            assert lang in tts.name
    
    def test_synthesize_with_speed(self, melotts_class, mock_melotts_api):
        """Test synthesis with different speeds"""
        mock, model_instance = mock_melotts_api
        
        with patch('tts.melotts.torchaudio') as mock_torchaudio:
            mock_torchaudio.load.return_value = (
                np.zeros((1, 16000)),
                16000
            )
            
            tts = melotts_class(language="ZH", speed=1.5)
            tts.load_model()
            
            audio = tts.synthesize("测试")
            
            # Verify speed was passed to tts_to_file
            call_args = model_instance.tts_to_file.call_args
            assert call_args[1]['speed'] == 1.5
    
    def test_synthesize_to_file(self, melotts_class, mock_melotts_api, temp_dir):
        """Test synthesize_to_file method"""
        import os
        
        mock, model_instance = mock_melotts_api
        
        with patch('tts.melotts.torchaudio') as mock_torchaudio:
            mock_torchaudio.load.return_value = (
                np.zeros((1, 16000)),
                16000
            )
            
            tts = melotts_class(language="ZH")
            tts.load_model()
            
            output_file = os.path.join(temp_dir, "test.wav")
            result = tts.synthesize_to_file("你好", output_file)
            
            assert result is True
            # File would be created in actual implementation
    
    def test_get_info(self, melotts_class, mock_melotts_api):
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
        assert info["config"]["language"] == "ZH"
    
    def test_is_available(self, melotts_class, mock_melotts_api):
        """Test is_available method"""
        tts = melotts_class(language="ZH")
        
        assert tts.is_available() is False
        
        tts.load_model()
        assert tts.is_available() is True


@pytest.mark.phase2
@pytest.mark.unit
@pytest.mark.tts
class TestMeloTTSConfig:
    """Test TTS configuration"""
    
    def test_tts_config_defaults(self):
        """Test default TTS configuration"""
        from config import TTSConfig
        
        config = TTSConfig()
        assert config.engine == "edge"  # Current default
    
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
